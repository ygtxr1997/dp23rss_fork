import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict
import copy

import numpy as np
import pydantic
from PIL import Image
from fastapi import FastAPI

import json
import torch
from torchvision.transforms import transforms

from diffusion_policy.dataset.tcl_dataset import TCLImageDataset, TCLDatasetHDF5
# from robokit.service.service_connector import ServiceConnector
from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy


""" How to use me?
conda activate robodiff
cd code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork
CUDA_VISIBLE_DEVICES=6 uvicorn gpu_service:gpu_app --port 6070
"""
gpu_app = FastAPI()
max_cache_action = 16

log_time = "2025.11.08-10.25.33"
w_idx = -1

map_time_to_dataset = {
    "2025.11.08-10.27.18": "1021_sweep_bean",
    "2025.11.08-01.55.12": "1024_eggs_pick_place",
    "2025.11.08-10.28.20": "1024_pour_water",
    "2025.11.08-10.25.33": "1024_wipe_white_board",
}
dataset_name = "pot_object"  # shovel; pot, pot_light; pepper
dataset_name = map_time_to_dataset.get(log_time, dataset_name)

if "2025.05.11" in log_time or "2025.05.13" in log_time:
    dataset_dir = "collected_data_0507"
elif dataset_name == "shovel":
    dataset_dir = "collected_data_0514_shovel_source"
elif dataset_name == "pot":
    dataset_dir = "0627_pot_source"
elif dataset_name == "pot_light":
    dataset_dir = "0627_pot_light"
elif dataset_name == "pot_object":
    dataset_dir = "0627_pot_object"
elif dataset_name == "pepper":
    dataset_dir = "0704_pepper_source"
else:
    dataset_dir = dataset_name
    print(f"[Warning] Using {dataset_name} for log_time={log_time}.")

with open(f"/home/geyuan/datasets/TCL/{dataset_dir}/statistics.json", 'r') as json_file:
    # statistics = json.load(json_file)
    # data_min = torch.from_numpy(np.array(statistics['min']))
    # data_max = torch.from_numpy(np.array(statistics['max']))
    statistics = json.load(json_file)
    dataset_stats = statistics["stats"]
    datasets_total_len = statistics["total_len"]
    dataset_action_min = np.array(dataset_stats["rel_actions"]["min"])
    dataset_action_max = np.array(dataset_stats["rel_actions"]["max"])

    data_root = f"/home/geyuan/datasets/TCL/{dataset_dir}"
    h5_path = f"/home/geyuan/datasets/TCL/hdf5/{dataset_dir}_240p.h5"
    tcl_hdf5_dataset = TCLDatasetHDF5(
        data_root, h5_path,
        use_extracted=True,
        load_keys=["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text", "force_torque"]
    )
    all_force_torques = tcl_hdf5_dataset.dsets["force_torque"]

    # Calculate p01 and p99 for force_torque if available
    if 'force_torque' in dataset_stats:
        # Calculate p01 (1%) and p99 (99%) quantiles along the sample dimension (axis=0)
        p01 = np.quantile(all_force_torques, q=0.01, axis=0)
        p99 = np.quantile(all_force_torques, q=0.99, axis=0)

        # Add the calculated quantiles to the merged statistics dictionary
        dataset_stats['force_torque']['p01'] = p01
        dataset_stats['force_torque']['p99'] = p99


# class StepRequestWithObservation(pydantic.BaseModel):
#     primary_rgb: List[str]
#     gripper_rgb: List[str]
#     instruction: str
#     joint_state: List[List[float]]


@lru_cache()
def get_agent(device: str):
    ## Op1. Debug model, sleep only
    # model = DebugModel(sleep_duration=100)
    ## Op2. Replay model, load action data and sleep
    # model = ReplayModel(sleep_duration=25,
    #                     replay_root="/home/geyuan/datasets/TCL/collected_data")

    import hydra
    from omegaconf import OmegaConf

    # 1. Load hydra config
    train_dir = f"/home/geyuan/code/dp23rss_fork/data/outputs/{log_time}_train_diffusion_transformer_hybrid_pusht_images"
    train_dir = train_dir.replace('-', '/')
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
    weight = weight['model']
    # for k, v in weight.items():
    #     print(k, v.shape)

    model.load_state_dict(weight)
    model = model.to(device).eval()
    print(f"[get_agent] model loaded from: {weight_path}")

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
    force_B_T_D = TCLImageDataset.norm_state_or_force(
        force_B_T_D, norm_type="quantile", meta_data=dataset_stats['force_torque']
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
    for act_idx in range(cache_action.shape[0]):
        if cache_action[act_idx, 6:] >= 0.5:
            cache_action[act_idx, 6:] = 1
        else:
            cache_action[act_idx, 6:] = 0

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
