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

from diffusion_policy.dataset.libero_dataset import LiberoFTDataset
# from robokit.service.service_connector import ServiceConnector
from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from robokit.debug_utils.printer import print_batch


""" How to use me?
conda activate robodiff
cd code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork
CUDA_VISIBLE_DEVICES=0 uvicorn gpu_service_libero:gpu_app --port 6070
"""
gpu_app = FastAPI()
max_cache_action = 32

# log_time = "2026.01.26-16.44.01"
# log_time = "2026.01.26-20.59.54"
# log_time = "2026.01.28-10.43.39"  # KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench.hdf5
# log_time = "2026.01.28-17.01.07"  # KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo_wrench
log_time = "2026.01.28-10.44.52"  # KITCHEN_SCENE6_close_the_microwave_demo_wrench.hdf5
# log_time = "2026.01.28-10.34.35"  # STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo_wrench.hdf5
w_idx = -2


def load_dataset_fields(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    shape_meta = cfg.get("shape_meta", None)
    ds = cfg["task"]["dataset"]
    hdf5_fns = ds.get("hdf5_fns", [])
    dataset_root = ds.get("dataset_root", None)
    dataset_subname = ds.get("dataset_subname", None)
    return hdf5_fns, dataset_root, dataset_subname, shape_meta


train_project_dir = f"/home/geyuan/code/dp23rss_fork/data/outputs/{log_time}_train_diffusion_transformer_hybrid_pusht_images"
train_project_dir = train_project_dir.replace('-', '/')
train_yaml_path = os.path.join(train_project_dir, ".hydra/config.yaml")
assert os.path.exists(train_yaml_path), f"[gpu_service_libero] train_yaml_path not found: {train_yaml_path}"

hdf5_fns, dataset_root, dataset_subname, shape_meta = load_dataset_fields(train_yaml_path)
print(f"[INFO] Loaded dataset fields from {train_yaml_path}:")
print("  hdf5_fns:", hdf5_fns)
print("  dataset_root:", dataset_root)
print("  dataset_subname:", dataset_subname)
print("  shape_meta:", shape_meta)

# Load dataset statistics and save to train_project_dir if not exists
train_project_statistics_file = os.path.join(train_project_dir, "statistics.json")
if os.path.exists(dataset_root):
    train_dataset = LiberoFTDataset(
        hdf5_fns=hdf5_fns,
        dataset_root=dataset_root,
        dataset_subname=dataset_subname,
        horizon=1,  # just for loading statistics
        pad_before=0,
        pad_after=0,
        shape_meta=shape_meta,
        norm_force_type="quantile",
        transform_color_jitter=False,
    )
    statistics = train_dataset.dataset_stats
    for k, stat_dict in statistics.items():
        statistics[k] = {sub_k: sub_v.tolist() for sub_k, sub_v in stat_dict.items()}

    # Dump updated statistics back to the JSON file
    if not os.path.exists(train_project_statistics_file):
        with open(train_project_statistics_file, 'w') as json_file:
            json.dump(statistics, json_file, indent=4)
        print("[INFO] Dumped updated statistics.json to train_project_dir.")

# Load statistics from train project dir (to be compatible with ITX deployment)
assert os.path.exists(train_project_statistics_file), "[gpu_service_libero] statistics.json not found in train_project_dir."
with open(train_project_statistics_file, 'r') as json_file:
    statistics = json.load(json_file)
    for k, stat_dict in statistics.items():
        statistics[k] = {sub_k: np.array(sub_v) for sub_k, sub_v in stat_dict.items()}
    dataset_stats = statistics
    '''
    dataset_stats: Dict, keys=['obs.wrenches', 'obs.ee_states', 'action.actions']
    --obs.wrenches: Dict, keys=['count', 'mean', 'std', 'min', 'max', 'p01', 'p99']
    ----count, <class 'numpy.ndarray'>, shape=(), value=11723
    ----mean, <class 'numpy.ndarray'>, shape=(6,), min=-0.2352, max=4.8427, dtype=float64
    ----std, <class 'numpy.ndarray'>, shape=(6,), min=1.5180, max=23.3075, dtype=float64
    ----min, <class 'numpy.ndarray'>, shape=(6,), min=-408.8553, max=-18.2925, dtype=float64
    ----max, <class 'numpy.ndarray'>, shape=(6,), min=27.7514, max=444.5059, dtype=float64
    ----p01, <class 'numpy.ndarray'>, shape=(6,), min=-32.5228, max=-5.5354, dtype=float64
    ----p99, <class 'numpy.ndarray'>, shape=(6,), min=3.2230, max=91.8009, dtype=float64
    '''
    dataset_action_max = dataset_stats['action.actions']['max']
    dataset_action_min = dataset_stats['action.actions']['min']
    dataset_states_max = dataset_stats['obs.ee_states']['max']
    dataset_states_min = dataset_stats['obs.ee_states']['min']


@lru_cache()
def get_agent(device: str, use_ema: bool = True):
    # 1. Load hydra config
    train_dir = train_project_dir
    hydra_config_path = os.path.join(train_dir, ".hydra/config.yaml")
    hydra_config = OmegaConf.load(hydra_config_path)
    model = hydra.utils.instantiate(hydra_config.policy)
    print(type(model))

    # 2. Load weights
    weight_paths = os.listdir(os.path.join(train_dir, "checkpoints"))
    weight_paths = list(filter(lambda x: x.endswith(".ckpt"), weight_paths))
    weight_paths.sort()
    print(weight_paths)
    weight_path = os.path.join(train_dir, "checkpoints", weight_paths[w_idx])
    weight = torch.load(weight_path, map_location="cpu", weights_only=False)['state_dicts']
    if not use_ema:
        weight = weight['model']
    else:
        weight = weight['ema_model']
    # for k, v in weight.items():
    #     print(k, v.shape)

    model.load_state_dict(weight)
    model = model.to(device).eval()
    print(f"[get_agent] model loaded from: {weight_path}, use_ema={use_ema}")

    # 3. Other settings
    model.infer_frame_idx = 0

    return model, hydra_config, weight_path


@gpu_app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@gpu_app.get("/init")
def model_init():
    return {"message": "Hello, World!", "max_cache_action": max_cache_action}


@gpu_app.get("/reset")
def model_reset():
    agent, image_shape, _ = get_agent("cuda")
    agent.reset()
    return {"max_cache_action": max_cache_action}


@gpu_app.post("/step")
def model_step(step_request: StepRequestFromEvaluator):
    agent, hydra_config, weight_path = get_agent("cuda")  # shape:[C,H,W]
    print("[gpu_service] Using cached ckpt from: None. Model type:", type(agent), weight_path)

    # 1. Decode observation from received request
    image_shape = hydra_config.image_shape

    # [] Parse input observation
    step_data = step_request.decode_to_raw()
    instruction_text = step_data["instruction"]
    stage_flag = step_data["stage_flag"]
    gt_video = step_data["gt_video"]  # (B,V*Ts,H,W,3) uint8, Ts can be larger than v1
    tcp_state = step_data["tcp_state"]  # (B,Ts,D+6) float32 or None, NOTE: includes force data

    B, Ts, D_plus6 = tcp_state.shape
    gt_video = torch.from_numpy(gt_video).float() / 127.5 - 1.  # (B,V*Ts,H,W,3), in [-1,1]
    gt_view0_B_T_C_H_W = gt_video[:, :Ts].permute(0, 1, 4, 2, 3)  # (B,T,C,H,W)
    gt_view1_B_T_C_H_W = gt_video[:, Ts:].permute(0, 1, 4, 2, 3)  # (B,T,C,H,W)

    # instruction_text = step_request.instruction
    # joint_state = step_request.joint_state
    force_B_T_D = tcp_state[:, :, -6:].astype(np.float32)  # (B,T,6)
    tcp_pose_B_T_D = tcp_state[:, :, :6].astype(np.float32)  # (B,T,6)

    # Norm input states
    force_B_T_D = LiberoFTDataset.norm_state_or_force(
        force_B_T_D, norm_type="quantile", meta_data=dataset_stats['obs.wrenches']
    )
    tcp_pose_B_T_D = LiberoFTDataset.norm_state_or_force(
        tcp_pose_B_T_D, norm_type="mean", meta_data=dataset_stats['obs.ee_states']
    )

    # joint_state = torch.from_numpy(np.array(joint_state)).to("cuda").unsqueeze(0)  # (B,T,6)
    obs_dict = {
        "joint_state": torch.from_numpy(tcp_pose_B_T_D).to("cuda"),  # should be (B,T,6)
        "force": torch.from_numpy(force_B_T_D).to("cuda"),  # should be (B,T,6)
    }

    if True or agent.infer_frame_idx % max_cache_action == 0:  # always enter
        # primary_imgs = []
        # for idx, primary_img in enumerate(step_request.primary_rgb):
        #     primary_img = base64.b64decode(primary_img)
        #     primary_img = Image.open(io.BytesIO(primary_img), formats=["JPEG"])
        #     primary_img.save(f"tmp_primary_{idx}.jpg")
        #
        #     rgb_transform = transforms.Compose([
        #         transforms.Resize(image_shape[1:]),
        #         transforms.ToTensor(),
        #     ])
        #     primary_img = rgb_transform(primary_img)  # (C,H,W), in [0,1]
        #     primary_img = primary_img * 2. - 1.  # in [-1,1]
        #     primary_imgs.append(primary_img)

        # gripper_imgs = []
        # for idx, gripper_img in enumerate(step_request.gripper_rgb):
        #     gripper_img = base64.b64decode(gripper_img)
        #     gripper_img = Image.open(io.BytesIO(gripper_img), formats=["JPEG"])
        #     gripper_img.save(f"tmp_gripper_{idx}.jpg")
        #
        #     rgb_transform = transforms.Compose([
        #         transforms.Resize(image_shape[1:]),
        #         transforms.ToTensor(),
        #     ])
        #     gripper_img = rgb_transform(gripper_img)  # (C,H,W), in [0,1]
        #     gripper_img = gripper_img * 2. - 1.  # in [-1,1]
        #     gripper_imgs.append(gripper_img)

        # 2. Preprocess, e.g resize, normalize, to_tensor, to_device
        # primary_img = torch.stack(primary_imgs, dim=0)  # (T,C,H,W)
        # primary_img = primary_img.to("cuda").unsqueeze(0)  # (B,T,C,H,W)
        # gripper_img = torch.stack(gripper_imgs, dim=0)
        # gripper_img = gripper_img.to("cuda").unsqueeze(0)
        primary_img = gt_view0_B_T_C_H_W.to("cuda")
        gripper_img = gt_view1_B_T_C_H_W.to("cuda")

        primary_img = gt_view0_B_T_C_H_W[:, -1, :, :, :].unsqueeze(1)  # (B,1,C,H,W)
        gripper_img = gt_view1_B_T_C_H_W[:, -1, :, :, :].unsqueeze(1)  # (B,1,C,H,W)

        print("[DEBUG] primary_img shape:", primary_img.shape)

        obs_dict["image"] = primary_img  # should be (B,T,C,H,W)
        if "gripper" in hydra_config.shape_meta["obs"]:
            obs_dict["gripper"] = gripper_img

    # cond = {
    #     "lang_text": instruction_text,
    #     "proprioception": joint_state,
    # }

    # 3.a Model inference
    # action_idx = agent.infer_frame_idx % max_cache_action
    # if action_idx == 0:
    with torch.no_grad():  # always enter
        action = agent.predict_action(obs_dict)['action_pred']
        action = action[0, :]  # remove batch_dim, (T,D)

    # 3.b Postprocess
    # print(action.shape, action.min(dim=0)[0], action.max(dim=0)[0])
    action = (action * 0.5 + 0.5).cpu()  # in [0,1]
    action = action.clamp(0., 1.)
    action = action * (dataset_action_max - dataset_action_min) + dataset_action_min
    # # print(action.shape, action.min(dim=0), action.max(dim=0))
    # agent.cache_action = action
    # else:
    #     action = agent.cache_action

    # 4. Return results
    # frame_action = action[action_idx].numpy().tolist()
    # if frame_action[6] > 0.5:
    #     frame_action[6] = 1.
    # else:
    #     frame_action[6] = 0.
    # print("[gpu_service] Action:", len(frame_action), frame_action, obs_dict.keys())

    # cache_action = copy.deepcopy(agent.cache_action[0])  # remove batch dim
    # cache_action = (cache_action * 0.5 + 0.5).cpu()  # in [0,1]
    # cache_action = cache_action * (data_max - data_min) + data_min
    cache_action = action
    # for act_idx in range(cache_action.shape[0]):  # NOTE: different gripper control
    #     if cache_action[act_idx, 6:] >= 0.5:
    #         cache_action[act_idx, 6:] = 1
    #     else:
    #         cache_action[act_idx, 6:] = 0

    agent.infer_frame_idx += 1
    # return {"action": cache_action.detach().numpy().tolist()}
    out_action = cache_action[None, :, :].numpy()  # (1,T,D)
    print("[DEBUG] gpu_service output action shape:", out_action.shape)
    request_to_evaluator = StepRequestFromPolicy.encode_from_raw(action=out_action)
    return request_to_evaluator.model_dump(mode="json")


if __name__ == "__main__":
    import time
    agent = get_agent("cuda")

    zero_rgb = np.zeros((2, 480, 848, 3), dtype=np.uint8)  # (T,H,W,C)

    debug_request = StepRequestWithObservation(
        primary_rgb=ServiceConnector.img_np_to_base64(zero_rgb),
        gripper_rgb=ServiceConnector.img_np_to_base64(zero_rgb),
        instruction="none",
        joint_state=[[0.] * 6] * 2,
    )
    pred_action = model_step(debug_request)
