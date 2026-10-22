import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import torch
from fastapi import FastAPI
from omegaconf import OmegaConf

from diffusion_policy.dataset.libero_dataset import LiberoFTDataset
from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy


""" How to use me?
conda activate robodiff
cd code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork
export CKPT_INDEX=-2
LOG_TIME=2026.06.07-15.05.23 CUDA_VISIBLE_DEVICES=6 uvicorn service.gpu_service_roi_libero:gpu_app --port 7077
"""

gpu_app = FastAPI()

ROI_DP_LOG_TIME = os.environ.get("LOG_TIME", os.environ.get("ROI_DP_LOG_TIME", "")).strip()
ROI_DP_CKPT_INDEX = int(os.environ.get("CKPT_INDEX", os.environ.get("ROI_DP_CKPT_INDEX", "-2")))
ROI_DP_OUTPUT_ROOT = Path(
    os.environ.get("ROI_DP_OUTPUT_ROOT", "/home/geyuan/code/dp23rss_fork/data/outputs")
).expanduser()
ROI_DP_TRAIN_DIR = os.environ.get("ROI_DP_TRAIN_DIR", "").strip()
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


def _resolve_train_dir_from_log_time() -> Path:
    if ROI_DP_TRAIN_DIR:
        return Path(ROI_DP_TRAIN_DIR).expanduser()

    if ROI_DP_LOG_TIME:
        log_time_dir = str(ROI_DP_LOG_TIME).replace("-", "/")
        return ROI_DP_OUTPUT_ROOT / f"{log_time_dir}_train_diffusion_transformer_hybrid_liberoft"

    raise FileNotFoundError(
        "Please set `LOG_TIME`, `ROI_DP_LOG_TIME`, or `ROI_DP_TRAIN_DIR` for the ROI Libero service."
    )


def _resolve_hydra_config_path() -> Path:
    train_dir = _resolve_train_dir_from_log_time()
    config_path = train_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"ROI Libero resolved hydra config does not exist: {config_path}"
        )
    return config_path


def _resolve_checkpoint_path() -> str:
    train_dir = _resolve_train_dir_from_log_time()
    ckpt_dir = train_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ROI Libero resolved checkpoint dir does not exist: {ckpt_dir}")
    return _resolve_indexed_ckpt(ckpt_dir)


def _np_stats_from_jsonable(statistics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        key: {sub_key: np.array(sub_value) for sub_key, sub_value in stat_dict.items()}
        for key, stat_dict in statistics.items()
    }


def _jsonable_stats(statistics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for key, stat_dict in statistics.items():
        out[key] = {}
        for sub_key, sub_value in stat_dict.items():
            out[key][sub_key] = sub_value.tolist() if hasattr(sub_value, "tolist") else sub_value
    return out


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _build_statistics_from_dataset(hydra_config) -> Dict[str, Dict[str, np.ndarray]]:
    dataset_cfg = hydra_config.task.dataset
    dataset_root = Path(str(dataset_cfg.dataset_root)).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"statistics.json is missing and dataset_root does not exist: {dataset_root}"
        )

    shape_meta = _to_plain(dataset_cfg.shape_meta)
    dataset = LiberoFTDataset(
        hdf5_fns=_to_plain(dataset_cfg.hdf5_fns),
        dataset_root=str(dataset_cfg.dataset_root),
        dataset_subname=_to_plain(dataset_cfg.dataset_subname),
        horizon=1,
        pad_before=0,
        pad_after=0,
        shape_meta=shape_meta,
        norm_force_type=str(dataset_cfg.get("norm_force_type", "quantile")),
        transform_color_jitter=False,
        zero_force=bool(dataset_cfg.get("zero_force", False)),
    )
    return dataset.dataset_stats


@lru_cache()
def get_dataset_stats() -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    hydra_config_path = _resolve_hydra_config_path()
    hydra_config = OmegaConf.load(hydra_config_path)
    train_dir = _resolve_train_dir_from_log_time()
    statistics_path = train_dir / "statistics.json"

    if statistics_path.exists():
        with statistics_path.open("r") as f:
            dataset_stats = _np_stats_from_jsonable(json.load(f))
    else:
        dataset_stats = _build_statistics_from_dataset(hydra_config)
        with statistics_path.open("w") as f:
            json.dump(_jsonable_stats(dataset_stats), f, indent=4)
        print(f"[INFO] Dumped Libero statistics to: {statistics_path}")

    required_keys = ("obs.ee_states", "action.actions")
    missing_keys = [key for key in required_keys if key not in dataset_stats]
    if missing_keys:
        raise KeyError(f"Libero statistics missing required keys: {missing_keys}")

    dataset_action_max = np.asarray(dataset_stats["action.actions"]["max"], dtype=np.float32)
    dataset_action_min = np.asarray(dataset_stats["action.actions"]["min"], dtype=np.float32)
    return dataset_stats, dataset_action_min, dataset_action_max


@lru_cache()
def get_agent(device: str):
    hydra_config_path = _resolve_hydra_config_path()
    hydra_config = OmegaConf.load(hydra_config_path)
    model = hydra.utils.instantiate(hydra_config.policy)
    print(type(model))

    weight_path = hydra_config.get("pretrained_ckpt", None)
    if weight_path is None or str(weight_path).strip() == "":
        weight_path = _resolve_checkpoint_path()
    weight_path = str(weight_path)

    model.load_weight_from_ckpt(weight_path, use_ema=True)
    model = model.to(device).eval()
    print(
        f"[get_agent] config loaded from: {hydra_config_path}; "
        f"model loaded from: {weight_path}, use_ema=True"
    )

    model.infer_frame_idx = 0
    get_dataset_stats()
    return model, hydra_config, weight_path


def _get_effective_max_cache_action(hydra_config) -> int:
    action_steps = hydra_config.get("n_action_steps", None)
    if action_steps is None:
        action_steps = hydra_config.get("horizon", max_cache_action)
    return min(max_cache_action, int(action_steps))


def _split_primary_video(gt_video: np.ndarray, tcp_state: np.ndarray) -> torch.Tensor:
    if gt_video.ndim != 5:
        raise ValueError(f"`gt_video` must be [B,V*Ts,H,W,3], got shape={gt_video.shape}.")
    if tcp_state.ndim != 3:
        raise ValueError(f"`tcp_state` must be [B,Ts,D], got shape={tcp_state.shape}.")

    batch_size, state_steps, _ = tcp_state.shape
    if gt_video.shape[0] != batch_size:
        raise ValueError(f"`gt_video` B={gt_video.shape[0]} does not match tcp_state B={batch_size}.")
    if gt_video.shape[1] < state_steps:
        raise ValueError(
            f"`gt_video` must contain at least one view of Ts frames, got video_T={gt_video.shape[1]} "
            f"and state_T={state_steps}."
        )

    num_views = max(gt_video.shape[1] // state_steps, 1)
    primary_video = gt_video[:, :state_steps]
    if num_views > 1:
        print(f"[DEBUG] ROI Libero received {num_views} views; using the first view for current config.")
    return torch.from_numpy(primary_video).float() / 127.5 - 1.0


def _match_time_length(vector_B_T_D: np.ndarray, target_t: int) -> np.ndarray:
    if vector_B_T_D.shape[1] == target_t:
        return vector_B_T_D
    if vector_B_T_D.shape[1] > target_t:
        return vector_B_T_D[:, :target_t, :]
    if vector_B_T_D.shape[1] == 0:
        raise ValueError("Cannot expand an empty time dimension.")

    pad = np.repeat(vector_B_T_D[:, -1:, :], target_t - vector_B_T_D.shape[1], axis=1)
    return np.concatenate([vector_B_T_D, pad], axis=1)


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
    agent, hydra_config, weight_path = get_agent("cuda")
    dataset_stats, dataset_action_min, dataset_action_max = get_dataset_stats()
    print("[gpu_service_roi_libero] Using cached ckpt from:", weight_path, "Model type:", type(agent))

    step_data = step_request.decode_to_raw()
    gt_video = step_data["gt_video"]  # (B,V*Ts,H,W,3) uint8
    tcp_state = step_data["tcp_state"]  # (B,Ts,D) float32
    if tcp_state is None:
        raise ValueError("`tcp_state` is required by the ROI Libero service.")
    if tcp_state.shape[-1] < 6:
        raise ValueError(f"`tcp_state` must contain at least 6 ee_state dims, got D={tcp_state.shape[-1]}.")

    t_cond = int(hydra_config.n_obs_steps)
    if tcp_state.shape[1] < t_cond:
        raise ValueError(f"`tcp_state` time length must be >= n_obs_steps={t_cond}, got {tcp_state.shape[1]}.")

    primary_video_B_T_H_W_C = _split_primary_video(gt_video, tcp_state)
    primary_img = primary_video_B_T_H_W_C[:, -t_cond:].permute(0, 1, 4, 2, 3).contiguous().to("cuda")

    tcp_pose_B_T_D = tcp_state[:, -t_cond:, :6].astype(np.float32)
    tcp_pose_B_T_D = LiberoFTDataset.norm_state_or_force(
        tcp_pose_B_T_D,
        norm_type="mean",
        meta_data=dataset_stats["obs.ee_states"],
    ).astype(np.float32)

    obs_dict = {
        "image": primary_img,
        "joint_state": torch.from_numpy(tcp_pose_B_T_D).to("cuda"),
    }
    print("[DEBUG] ROI Libero primary_img shape:", primary_img.shape, f"tcp_state:{tcp_state.shape}")

    with torch.no_grad():
        result = agent.predict_action(obs_dict)
        action = result["action_pred"]
        roi_param = result.get("roi_params_B_T_3", None)
        if roi_param is None:
            roi_param = result["roi_param"]
        roi_param = roi_param.detach().cpu().numpy().astype(np.float32)

    action_min = torch.from_numpy(dataset_action_min).to(device=action.device, dtype=action.dtype)
    action_max = torch.from_numpy(dataset_action_max).to(device=action.device, dtype=action.dtype)
    action = (action * 0.5 + 0.5)
    action = action.clamp(0.0, 1.0)
    action = action * (action_max - action_min) + action_min
    action = action.cpu()

    agent.infer_frame_idx += 1
    out_action = action.numpy().astype(np.float32)
    roi_param = _match_time_length(roi_param, out_action.shape[1])

    if roi_param.shape[:2] != out_action.shape[:2]:
        raise ValueError(
            f"roi_param B,T should match action B,T for StepRequestFromPolicy, "
            f"got {roi_param.shape[:2]} vs {out_action.shape[:2]}"
        )

    print("[DEBUG] ROI Libero output action shape:", out_action.shape)
    print("[DEBUG] ROI Libero output roi_param shape:", roi_param.shape)
    request_to_evaluator = StepRequestFromPolicy.encode_from_raw(
        action=out_action,
        extra_vector={"roi_param": roi_param},
        max_cache_action=_get_effective_max_cache_action(hydra_config),
    )
    return request_to_evaluator.model_dump(mode="json")


if __name__ == "__main__":
    get_agent("cuda")
    print(f"[gpu_service_roi_libero] ready with max_cache_action={max_cache_action}")
