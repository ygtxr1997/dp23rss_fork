"""FastAPI GPU service for real-robot Pi0.5 TCL inference.

The service consumes the existing RoboKit evaluator request.  It keeps the
policy's self-contained EMA checkpoint as the source of preprocessing,
normalization, and migrated foveation/router weights.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from diffusion_policy.common.tcl_task_util import load_tcl_task_mapping
from diffusion_policy.policy.pi05_policy import Pi05Policy
from robokit.connects.protocols import (
    StepRequestEvent,
    StepRequestFromEvaluator,
    StepRequestFromPolicy,
)


"""How to use me?

export PI05_CHECKPOINT=outputs/pi05_tcl_stageB/.../checkpoints/latest_ema.pt
export PI05_TASK_MAPPING_PATH=meta/tcl_tasks_10.yaml
CUDA_VISIBLE_DEVICES=0 uvicorn gpu_service_pi05:gpu_app --host 0.0.0.0 --port 6081
"""

gpu_app = FastAPI()


def _resolve_path(value: str) -> Path:
    """Resolve a user-provided path relative to the repository when needed."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _resolve_checkpoint() -> Path:
    """Resolve a self-contained Pi0.5 EMA checkpoint from environment settings."""
    value = os.environ.get("PI05_CHECKPOINT", "").strip()
    if not value:
        raise FileNotFoundError("Set PI05_CHECKPOINT to a Pi0.5 EMA checkpoint or run directory.")
    path = _resolve_path(value)
    if path.is_dir():
        checkpoint_type = os.environ.get("PI05_CHECKPOINT_TYPE", "latest").strip()
        if checkpoint_type not in ("best", "latest"):
            raise ValueError("PI05_CHECKPOINT_TYPE must be best or latest")
        path = path / "checkpoints" / f"{checkpoint_type}_ema.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Pi0.5 checkpoint does not exist: {path}")
    return path.resolve()


def _resolve_device() -> torch.device:
    """Resolve and validate the inference device."""
    device = torch.device(os.environ.get("PI05_DEVICE", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Pi0.5 GPU service requires CUDA")
    return device


def _mapping_path() -> Path:
    """Return the explicit TCL task mapping used by the service."""
    return _resolve_path(os.environ.get(
        "PI05_TASK_MAPPING_PATH", "meta/tcl_tasks_10.yaml"))


@lru_cache(maxsize=1)
def get_task_mapping() -> dict[str, int]:
    """Load the user-maintained TCL task mapping and require at least one task."""
    path = _mapping_path()
    mapping = load_tcl_task_mapping(str(path))
    if not mapping:
        raise ValueError(f"TCL task mapping is empty: {path}")
    return mapping


@lru_cache(maxsize=2)
def get_agent(device_name: str = "cuda") -> tuple[Pi05Policy, Path]:
    """Load and cache the self-contained Pi0.5 EMA policy on one GPU."""
    device = torch.device(device_name)
    checkpoint = _resolve_checkpoint()
    get_task_mapping()
    policy = Pi05Policy.from_checkpoint(checkpoint, device=device)
    policy.eval()
    policy.infer_frame_idx = 0
    print(f"[gpu_service_pi05] loaded checkpoint={checkpoint} device={device}", flush=True)
    return policy, checkpoint


def _task_ids_from_tcp_state(tcp_state_B_T_D: np.ndarray) -> np.ndarray:
    """Extract one stable integer task ID per batch from the final state dimension."""
    if tcp_state_B_T_D.ndim != 3 or tcp_state_B_T_D.shape[-1] < 7:
        raise ValueError("tcp_state must have shape [B, Ts, D] with D >= 7")
    task_values_B_T = tcp_state_B_T_D[..., -1]
    if not np.isfinite(task_values_B_T).all():
        raise ValueError("tcp_state task ID must be finite")
    task_ids_B = np.rint(task_values_B_T[:, -1]).astype(np.int64)
    if not np.allclose(task_values_B_T, task_ids_B[:, None], atol=1e-5):
        raise ValueError("tcp_state task ID must be an integer stable across the request time window")
    if np.any(task_ids_B < 0) or np.any(task_ids_B >= 10):
        raise ValueError("task ID must be in [0, 9]")
    mapping_ids = set(get_task_mapping().values())
    if any(int(task_id) not in mapping_ids for task_id in task_ids_B):
        raise ValueError("task ID is not present in PI05_TASK_MAPPING_PATH")
    return task_ids_B


def _build_observation(step_data: dict[str, Any]) -> dict[str, Any]:
    """Build a single-frame Pi0.5 raw observation from a RoboKit request."""
    gt_video_B_V_T_H_W_C = step_data["gt_video"]
    tcp_state_B_T_D = step_data["tcp_state"]
    if tcp_state_B_T_D is None:
        raise ValueError("tcp_state is required by the Pi0.5 TCL service")
    if gt_video_B_V_T_H_W_C.ndim != 5 or gt_video_B_V_T_H_W_C.shape[-1] != 3:
        raise ValueError("gt_video must have shape [B, V*Ts, H, W, 3]")
    if gt_video_B_V_T_H_W_C.dtype != np.uint8:
        raise ValueError("gt_video must contain uint8 RGB frames")
    if tcp_state_B_T_D.ndim != 3:
        raise ValueError("tcp_state must have shape [B, Ts, D]")
    batch_size, time_steps, _ = tcp_state_B_T_D.shape
    if gt_video_B_V_T_H_W_C.shape[0] != batch_size:
        raise ValueError("gt_video and tcp_state batch sizes must match")
    if int(step_data.get("num_camera_views", 0)) != 2:
        raise ValueError("Pi0.5 TCL service requires exactly two camera views")
    if gt_video_B_V_T_H_W_C.shape[1] != 2 * time_steps:
        raise ValueError("gt_video must contain exactly two Ts-length camera sequences")

    task_ids_B = _task_ids_from_tcp_state(tcp_state_B_T_D)
    return {
        "image": gt_video_B_V_T_H_W_C[:, time_steps - 1],
        "gripper": gt_video_B_V_T_H_W_C[:, 2 * time_steps - 1],
        "joint_state": tcp_state_B_T_D[:, -1, :6].astype(np.float32),
        "prompt": [str(step_data["instruction"])] * batch_size,
        "task_id": task_ids_B,
    }


def _effective_cache_action(request: StepRequestFromEvaluator, action_horizon: int,
                            default_steps: int) -> int:
    """Bound evaluator caching to the predicted action horizon."""
    requested = request.max_cache_action
    value = default_steps if requested is None else int(requested)
    if value <= 0:
        raise ValueError("max_cache_action must be positive for a step request")
    return min(value, action_horizon)


@gpu_app.get("/")
def read_root():
    """Return a lightweight service health response."""
    return {"message": "Pi0.5 TCL GPU service"}


@gpu_app.get("/init")
def model_init():
    """Load the policy and report its deployment contract."""
    device = _resolve_device()
    agent, checkpoint = get_agent(str(device))
    mapping = get_task_mapping()
    return {
        "message": "Initialized.",
        "checkpoint": str(checkpoint),
        "device": str(device),
        "action_horizon": int(agent.config.action_horizon),
        "n_action_steps": int(agent.config.n_action_steps),
        "max_cache_action": int(os.environ.get("PI05_MAX_CACHE_ACTION",
                                               agent.config.n_action_steps)),
        "task_mapping_path": str(_mapping_path()),
        "task_ids": sorted(set(mapping.values())),
    }


@gpu_app.get("/reset")
def model_reset():
    """Reset policy state and return the configured execution prefix."""
    device = _resolve_device()
    agent, _ = get_agent(str(device))
    agent.reset()
    agent.infer_frame_idx = 0
    return {"max_cache_action": int(os.environ.get("PI05_MAX_CACHE_ACTION",
                                                   agent.config.n_action_steps))}


@gpu_app.post("/step")
def model_step(step_request: StepRequestFromEvaluator):
    """Decode one evaluator request, infer a Pi0.5 action chunk, and encode it."""
    device = _resolve_device()
    agent, checkpoint = get_agent(str(device))
    if step_request.event == StepRequestEvent.ROLLOUT_END:
        agent.reset()
        agent.infer_frame_idx = 0
        return StepRequestFromPolicy.encode_from_raw(
            action=np.zeros((1, 1, 7), dtype=np.float32), max_cache_action=0
        ).model_dump(mode="json")

    try:
        step_data = step_request.decode_to_raw()
        obs = _build_observation(step_data)
        with torch.no_grad():
            result = agent.predict_action(obs)
        action_B_T_D = result["action_pred"].detach().cpu().numpy().astype(np.float32)
        if action_B_T_D.ndim != 3 or action_B_T_D.shape[-1] != 7:
            raise ValueError(f"Pi0.5 action must have shape [B, T, 7], got {action_B_T_D.shape}")
        if not np.isfinite(action_B_T_D).all():
            raise ValueError("Pi0.5 produced non-finite actions")
        cache_steps = _effective_cache_action(
            step_request, action_B_T_D.shape[1],
            int(os.environ.get("PI05_MAX_CACHE_ACTION", agent.config.n_action_steps)))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    agent.infer_frame_idx += 1
    print(f"[gpu_service_pi05] checkpoint={checkpoint} index={agent.infer_frame_idx} "
          f"action_shape={action_B_T_D.shape}", flush=True)
    return StepRequestFromPolicy.encode_from_raw(
        action=action_B_T_D, max_cache_action=cache_steps
    ).model_dump(mode="json")


if __name__ == "__main__":
    device = _resolve_device()
    agent, _ = get_agent(str(device))
    print(f"[gpu_service_pi05] ready with max_cache_action={agent.config.n_action_steps}")
