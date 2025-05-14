import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pydantic
from PIL import Image
from fastapi import FastAPI

import json
import torch
from torchvision.transforms import transforms
from robokit.data.data_handler import DataHandler
from robokit.service.service_connector import ServiceConnector


""" How to use me?
$ CUDA_VISIBLE_DEVICES=9 uvicorn gpu_service:gpu_app --port 6060
"""
gpu_app = FastAPI()
max_cache_action = 24
with open("/home/geyuan/local_soft/TCL/collected_data_0507/statistics.json", 'r') as json_file:
    statistics = json.load(json_file)
    data_min = torch.from_numpy(np.array(statistics['min']))
    data_max = torch.from_numpy(np.array(statistics['max']))


class StepRequestWithObservation(pydantic.BaseModel):
    primary_rgb: List[str]
    gripper_rgb: List[str]
    instruction: str
    joint_state: List[List[float]]


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
    train_dir = "/home/geyuan/Documents/ckpt/dp/2025.05.11-01.08.40/"
    hydra_config_path = os.path.join(train_dir, ".hydra/config.yaml")
    hydra_config = OmegaConf.load(hydra_config_path)
    model = hydra.utils.instantiate(hydra_config.policy)
    print(type(model))

    # 2. Load weights
    weight_paths = os.listdir(os.path.join(train_dir, "checkpoints"))
    weight_paths = list(filter(lambda x: x.endswith(".ckpt"), weight_paths))
    weight_paths.sort()
    print(weight_paths)
    weight_path = os.path.join(train_dir, "checkpoints", weight_paths[-2])
    weight = torch.load(weight_path, map_location="cpu")['state_dicts']
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


@gpu_app.get("/reset")
def model_reset():
    agent, image_shape, _ = get_agent("cuda")
    agent.reset()
    return {"max_cache_action": max_cache_action}


@gpu_app.post("/step")
def model_step(step_request: StepRequestWithObservation):
    agent, hydra_config, weight_path = get_agent("cuda")  # shape:[C,H,W]
    print("[gpu_service] Using cached ckpt from: None. Model type:", type(agent), weight_path)

    # 1. Decode observation from received request
    image_shape = hydra_config.image_shape

    instruction_text = step_request.instruction
    joint_state = step_request.joint_state
    joint_state = torch.from_numpy(np.array(joint_state)).to("cuda").unsqueeze(0)  # (B,T,6)
    obs_dict = {
        "joint_state": joint_state,  # should be (B,T,6)
    }

    if agent.infer_frame_idx % max_cache_action == 0:
        primary_imgs = []
        for idx, primary_img in enumerate(step_request.primary_rgb):
            primary_img = base64.b64decode(primary_img)
            primary_img = Image.open(io.BytesIO(primary_img), formats=["JPEG"])
            primary_img.save(f"tmp_primary_{idx}.jpg")

            rgb_transform = transforms.Compose([
                transforms.Resize(image_shape[1:]),
                transforms.ToTensor(),
            ])
            primary_img = rgb_transform(primary_img)  # (C,H,W), in [0,1]
            primary_img = primary_img * 2. - 1.  # in [-1,1]
            primary_imgs.append(primary_img)

        gripper_imgs = []
        for idx, gripper_img in enumerate(step_request.gripper_rgb):
            gripper_img = base64.b64decode(gripper_img)
            gripper_img = Image.open(io.BytesIO(gripper_img), formats=["JPEG"])
            gripper_img.save(f"tmp_gripper_{idx}.jpg")

            rgb_transform = transforms.Compose([
                transforms.Resize(image_shape[1:]),
                transforms.ToTensor(),
            ])
            gripper_img = rgb_transform(gripper_img)  # (C,H,W), in [0,1]
            gripper_img = gripper_img * 2. - 1.  # in [-1,1]
            gripper_imgs.append(gripper_img)

        # 2. Preprocess, e.g resize, normalize, to_tensor, to_device
        primary_img = torch.stack(primary_imgs, dim=0)  # (T,C,H,W)
        primary_img = primary_img.to("cuda").unsqueeze(0)  # (B,T,C,H,W)
        gripper_img = torch.stack(gripper_imgs, dim=0)
        gripper_img = gripper_img.to("cuda").unsqueeze(0)

        obs_dict["image"] = primary_img  # should be (B,T,C,H,W)
        if "gripper" in hydra_config.shape_meta["obs"]:
            obs_dict["gripper"] = gripper_img

    # cond = {
    #     "lang_text": instruction_text,
    #     "proprioception": joint_state,
    # }

    # 3.a Model inference
    action_idx = agent.infer_frame_idx % max_cache_action
    if action_idx == 0:
        with torch.no_grad():
            action = agent.predict_action(obs_dict)['action_pred']
            action = action[0, :]  # remove batch_dim

        # 3.b Postprocess
        print(action.shape, action.min(dim=0)[0], action.max(dim=0)[0])
        action = (action * 0.5 + 0.5).cpu()  # in [0,1]
        action = action * (data_max - data_min) + data_min
        # print(action.shape, action.min(dim=0), action.max(dim=0))
        agent.cache_action = action
    else:
        action = agent.cache_action

    # 4. Return results
    frame_action = action[action_idx].numpy().tolist()
    if frame_action[6] > 0.5:
        frame_action[6] = 1.
    else:
        frame_action[6] = 0.
    print("[gpu_service] Action:", len(frame_action), frame_action, obs_dict.keys())

    agent.infer_frame_idx += 1
    return {"action": frame_action}


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
