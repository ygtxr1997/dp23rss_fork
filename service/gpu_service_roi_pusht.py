import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict
import copy
import shutil
import yaml

import hydra
from omegaconf import OmegaConf
import numpy as np
import pydantic
from PIL import Image
from fastapi import FastAPI

import json
import torch
from torchvision.transforms import transforms

# from diffusion_policy.dataset.libero_dataset import LiberoFTDataset
# from robokit.service.service_connector import ServiceConnector
from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from robokit.debug_utils.printer import print_batch


""" How to use me?
conda activate robodiff
cd code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork
export CKPT_INDEX=-2
LOG_TIME=2026.05.31-12.58.06 CUDA_VISIBLE_DEVICES=6 uvicorn service.gpu_service_roi_pusht:gpu_app --port 7076
"""
gpu_app = FastAPI()
SERVICE_CONFIG_PATH = os.environ.get("ROI_DP_SERVICE_CONFIG", "configs/pusht256_roi_dp.yaml")
ROI_DP_LOG_TIME = os.environ.get("LOG_TIME", os.environ.get("ROI_DP_LOG_TIME", "")).strip()
ROI_DP_USE_EMA = os.environ.get("ROI_DP_USE_EMA", "1").strip().lower() not in {"0", "false", "no"}
ROI_DP_CKPT_INDEX = int(os.environ.get("CKPT_INDEX", os.environ.get("ROI_DP_CKPT_INDEX", "-2")))
max_cache_action = int(os.environ.get("ROI_DP_MAX_CACHE_ACTION", "8"))


def _resolve_indexed_ckpt(ckpt_dir: Path) -> str:
    ckpt_paths = sorted(p for p in ckpt_dir.iterdir() if p.suffix == ".ckpt")
    if not ckpt_paths:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    idx = ROI_DP_CKPT_INDEX
    if idx < 0:
        idx = len(ckpt_paths) + idx
    if idx < 0 or idx >= len(ckpt_paths):
        idx = len(ckpt_paths) - 1
    return str(ckpt_paths[idx])


def _resolve_checkpoint_path() -> str:
    if ROI_DP_LOG_TIME:
        log_time_dir = str(ROI_DP_LOG_TIME).replace('-', '/')
        train_dir = Path(f"/home/geyuan/code/dp23rss_fork/data/outputs/{log_time_dir}_train_diffusion_transformer_hybrid_pusht256")
        ckpt_dir = train_dir / "checkpoints"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"ROI_DP_LOG_TIME resolved checkpoint dir does not exist: {ckpt_dir}")
        return _resolve_indexed_ckpt(ckpt_dir)

    raise FileNotFoundError(
        "Please set one of `ROI_DP_CKPT_PATH`, `ROI_DP_LOG_TIME`, or `ROI_DP_TRAIN_DIR` for the ROI PushT service."
    )

# NO Need to: Load dataset statistics and save to train_project_dir if not exists

# No Need to: Load statistics from train project dir (to be compatible with ITX deployment)
"""
[DEBUG] PushTImageDataset: Dict, keys=['obs', 'action']
--obs: Dict, keys=['image', 'agent_pos']
----image, <class 'torch.Tensor'>, shape=torch.Size([10, 3, 256, 256]), min=-0.4902, max=1.0000, dtype=torch.float64
----agent_pos, <class 'torch.Tensor'>, shape=torch.Size([10, 2]), min=-0.1186, max=0.6088, dtype=torch.float32
--action, <class 'torch.Tensor'>, shape=torch.Size([10, 2]), min=-0.2812, max=0.6367, dtype=torch.float32
"""
dataset_action_max = np.array([512, 512], dtype=np.float32)  # hardcoded for pusht
dataset_action_min = np.array([0, 0], dtype=np.float32)
dataset_states_max = np.array([512, 512], dtype=np.float32)
dataset_states_min = np.array([0, 0], dtype=np.float32)


@lru_cache()
def get_agent(device: str, use_ema: bool = ROI_DP_USE_EMA):
    # 1. Load hydra config
    hydra_config = OmegaConf.load(SERVICE_CONFIG_PATH)
    model = hydra.utils.instantiate(hydra_config.policy)
    print(type(model))

    # 2. Load weights
    weight_path = hydra_config.get("pretrained_ckpt", None)
    if weight_path is None or str(weight_path).strip() == "":
        weight_path = _resolve_checkpoint_path()
    weight_path = str(weight_path)

    model.load_weight_from_ckpt(weight_path, use_ema=use_ema)
    model = model.to(device).eval()
    print(f"[get_agent] model loaded from: {weight_path}, use_ema={use_ema}")

    # 3. Other settings
    model.infer_frame_idx = 0

    return model, hydra_config, weight_path


def _get_effective_max_cache_action(hydra_config) -> int:
    return min(max_cache_action, int(hydra_config.horizon))


@gpu_app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@gpu_app.get("/init")
def model_init():
    _, hydra_config, _ = get_agent("cuda")
    return {"message": "Hello, World!", "max_cache_action": _get_effective_max_cache_action(hydra_config)}


@gpu_app.get("/reset")
def model_reset():
    agent, hydra_config, _ = get_agent("cuda")
    agent.reset()
    return {"max_cache_action": _get_effective_max_cache_action(hydra_config)}


@gpu_app.post("/step")
def model_step(step_request: StepRequestFromEvaluator):
    agent, hydra_config, weight_path = get_agent("cuda")  # shape:[C,H,W]
    print("[gpu_service] Using cached ckpt from:", weight_path, "Model type:", type(agent))

    # 1. Decode observation from received request
    image_shape = hydra_config.image_shape

    # [] Parse input observation
    step_data = step_request.decode_to_raw()
    instruction_text = step_data["instruction"]
    stage_flag = step_data["stage_flag"]
    gt_video = step_data["gt_video"]  # (B,V*Ts,H,W,3) uint8, Ts can be larger than v1
    tcp_state = step_data["tcp_state"]  # (B,Ts,2) float32 or None, NOTE: includes force data
    if tcp_state is None:
        raise ValueError("`tcp_state` is required by the ROI PushT service.")

    B, Ts, D_state = tcp_state.shape
    T_cond = int(hydra_config.n_obs_steps)
    gt_video = torch.from_numpy(gt_video).float() / 127.5 - 1.  # (B,V*Ts,H,W,3), in [-1,1]
    gt_view0_B_T_C_H_W = gt_video[:, -T_cond:].permute(0, 1, 4, 2, 3)  # (B,T,C,H,W), single view for now

    tcp_pose_B_T_D = tcp_state[:, -T_cond:, :].astype(np.float32)  # (B,T,2), in [0,512]

    # Norm input states
    tcp_pose_B_T_D = (tcp_pose_B_T_D - dataset_states_min) / (dataset_states_max - dataset_states_min) * 2. - 1.  # in [-1,1]

    # Should be consistent with the config yaml file
    obs_dict = {
        "agent_pos": torch.from_numpy(tcp_pose_B_T_D).to("cuda"),  # should be (B,T,2)
    }

    if True:  # always enter
        primary_img = gt_view0_B_T_C_H_W.to("cuda")
        # primary_img = gt_view0_B_T_C_H_W[:, :, :, :, :].unsqueeze(1)  # (B,1,C,H,W)
        print("[DEBUG] primary_img shape:", primary_img.shape, f"gt_video:{gt_video.shape}")

        obs_dict["image"] = primary_img  # should be (B,T,C,H,W)


    # 3.a Model inference
    with torch.no_grad():  # always enter
        action_pred = agent.predict_action(obs_dict)["action_pred"]
        action = action_pred[:, :]  # keep batch_dim, (B,T,D)

    # 3.b Postprocess
    # print(action.shape, action.min(dim=0)[0], action.max(dim=0)[0])
    action = (action * 0.5 + 0.5).cpu()  # in [0,1]
    action = action.clamp(0., 1.)
    action = action * (dataset_action_max[None, :] -
                       dataset_action_min[None, :]) + dataset_action_min[None, :]
    # # print(action.shape, action.min(dim=0), action.max(dim=0))
    cache_action = action

    agent.infer_frame_idx += 1
    # return {"action": cache_action.detach().numpy().tolist()}
    out_action = cache_action[:, :, :].numpy()  # (B,T,D)
    print("[DEBUG] gpu_service output action shape:", out_action.shape)
    request_to_evaluator = StepRequestFromPolicy.encode_from_raw(action=out_action)
    return request_to_evaluator.model_dump(mode="json")


if __name__ == "__main__":
    agent = get_agent("cuda")
    print(f"[gpu_service_roi_pusht] ready with max_cache_action={max_cache_action}")
"""
Usage:
  export ROI_DP_LOG_TIME=2026.04.16-17.04.04
  export ROI_DP_CKPT_INDEX=-2
  CUDA_VISIBLE_DEVICES=6 uvicorn service.gpu_service_roi_pusht:gpu_app --port 7076
"""
