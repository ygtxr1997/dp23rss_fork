"""FastAPI service for the multi-task TCL foveation DP policy.

The service keeps the legacy RoboKit evaluator protocol and resolves a training
run from ``LOG_TIME``.  The evaluator may still send the old TCP state layout;
task selection is therefore temporarily configured locally.
"""

from functools import lru_cache
import copy
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Optional

import dill
import hydra
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from diffusion_policy.common.tcl_image_util import prepare_tcl_rgb, resolve_tcl_image_geometry
from diffusion_policy.common.tcl_data_util import (
    DP_NORM, action_bounds, configure_dp_action_clipping, make_normalization,
    norm_methods, transform_values,
)

from robokit.connects.protocols import (
    StepRequestEvent,
    StepRequestFromEvaluator,
    StepRequestFromPolicy,
)


"""How to use me?

export LOG_TIME=2026-10-19-21-08-48
export CKPT_INDEX=-1  # -1 always means checkpoints/latest.ckpt
CUDA_VISIBLE_DEVICES=0 uvicorn service.gpu_service_foveation:gpu_app --port 6080
"""

gpu_app = FastAPI()
REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TASK_ID = 1
DEFAULT_MAX_CACHE_ACTION = 32
DEFAULT_GRIPPER_THRESHOLD = 0.5

# Hydra writes unresolved logging and derived-field interpolations into some
# saved configs.  Register the same lightweight resolvers before loading one.
OmegaConf.register_new_resolver(
    "now", lambda pattern: datetime.now().strftime(str(pattern)), replace=True
)
OmegaConf.register_new_resolver("eval", eval, replace=True)


def _resolve_run_dir() -> Path:
    """Find a run directory under ``outputs`` using the configured log time."""
    log_time = os.environ.get("LOG_TIME", "").strip()
    if not log_time:
        raise FileNotFoundError("Set LOG_TIME, for example 2026-10-15-21-44-36.")
    outputs_dir = Path(os.environ.get("OUTPUT_ROOT", REPO_DIR / "outputs")).expanduser()
    if not outputs_dir.is_absolute():
        outputs_dir = REPO_DIR / outputs_dir
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {outputs_dir}")

    relative_time = log_time if "/" in log_time else None
    for fmt in ("%Y-%m-%d-%H-%M-%S", "%Y.%m.%d-%H.%M.%S"):
        try:
            parsed = datetime.strptime(log_time, fmt)
        except ValueError:
            continue
        relative_time = f"{parsed:%Y-%m-%d}/{parsed:%H-%M-%S}"
        break

    # A normal run is outputs/<project>/<date>/<time>; include the direct
    # outputs/<date>/<time> form as well.  This is two levels below the
    # project directory and avoids scanning the rest of the repository.
    candidates = []
    for first in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        candidates.append(first)
        second_level = sorted(path for path in first.iterdir() if path.is_dir())
        candidates.extend(second_level)
        for second in second_level:
            candidates.extend(sorted(path for path in second.iterdir() if path.is_dir()))
    matches = []
    for candidate in candidates:
        relative = candidate.relative_to(outputs_dir).as_posix()
        if (
            candidate.name == log_time
            or relative == log_time
            or relative == relative_time
            or (relative_time is not None and relative.endswith("/" + relative_time))
        ):
            matches.append(candidate)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one run for LOG_TIME={log_time!r} under {outputs_dir}; "
            f"found {matches}. Searched at most two levels below each project directory."
        )
    return matches[0].resolve()


def _resolve_checkpoint(run_dir: Path) -> Path:
    """Select a checkpoint by sorted index, with ``-1`` reserved for latest."""
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
    checkpoint_index = int(os.environ.get("CKPT_INDEX", "-1"))
    latest_path = checkpoint_dir / "latest.ckpt"
    if checkpoint_index == -1:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Missing latest checkpoint: {latest_path}")
        return latest_path
    if checkpoint_index < 0:
        raise ValueError("CKPT_INDEX must be -1 or a non-negative sorted checkpoint index")
    checkpoints = sorted(path for path in checkpoint_dir.glob("*.ckpt") if path.name != "latest.ckpt")
    if checkpoint_index >= len(checkpoints):
        raise IndexError(
            f"CKPT_INDEX={checkpoint_index} is out of range for {len(checkpoints)} checkpoints."
        )
    return checkpoints[checkpoint_index]


def _resolve_asset_dir(run_dir: Path, config) -> Optional[Path]:
    """Resolve the prepared TCL asset directory recorded by the run config."""
    for key in ("asset_dir",):
        value = config.get(key)
        if value is not None:
            asset_dir = Path(str(value)).expanduser()
            return asset_dir if asset_dir.is_absolute() else (REPO_DIR / asset_dir).resolve()
    dataset_cfg = config.get("task", {}).get("dataset", {})
    value = dataset_cfg.get("asset_dir")
    if value is None:
        return None
    asset_dir = Path(str(value)).expanduser()
    return asset_dir if asset_dir.is_absolute() else (REPO_DIR / asset_dir).resolve()


def _load_normalization(run_dir: Path, asset_dir: Optional[Path], payload: dict, config) -> dict:
    """Load trained transforms and check run metadata/config without opening replay data."""
    encoded = payload.get("pickles", {}).get("dataset_metadata")
    checkpoint_metadata = dill.loads(encoded) if encoded is not None else None
    dataset_cfg = config.get('task', {}).get('dataset', {})
    defaults = dict(DP_NORM, force=dataset_cfg.get('norm_force_type', 'quantile'))
    norm = norm_methods(dataset_cfg.get('norm'), defaults)
    candidates = [('checkpoint metadata', checkpoint_metadata)]
    path = run_dir / 'dataset_metadata.json'
    if path.is_file():
        candidates.append((str(path), json.loads(path.read_text())))
    result, source = None, None
    for label, metadata in candidates:
        if metadata is None:
            continue
        saved = metadata.get('normalization')
        if saved is None:
            # Legacy metadata without any action bounds may fall back to the explicit asset.
            if norm['action'] == 'minmax' and action_bounds(metadata, required=False) is None:
                continue
            single = isinstance(metadata.get('sources'), dict)
            legacy_methods = norm_methods(None, defaults)
            if norm != legacy_methods:
                raise ValueError('Legacy DP checkpoints require their original normalization methods')
            dimensions = dict(state=6, action=7)
            obs_meta = dataset_cfg.get('shape_meta', {}).get('obs', {})
            if any(key in obs_meta for key in ('force', 'force_torque')):
                dimensions['force'] = 6
            candidate = make_normalization(metadata, norm, dimensions=dimensions, legacy_single=single)
        else:
            candidate = make_normalization(metadata, norm, dimensions={
                key: field['dim'] for key, field in saved['fields'].items()}, saved=saved)
        if result is not None and candidate != result:
            raise ValueError('TCL checkpoint/run normalization contracts differ')
        result, source = candidate, label
    if result is None:
        if asset_dir is None:
            raise ValueError('Missing DP normalization statistics and no explicit asset_dir in training config')
        path = asset_dir / 'dataset_metadata.json'
        metadata = json.loads(path.read_text())
        result = make_normalization(metadata, norm, saved=metadata.get('normalization'))
        source = str(path)
    print(f'[gpu_service_foveation] normalization={result["methods"]}; source={source}', flush=True)
    return result


def _is_global_encoder_key(key: str) -> bool:
    return key.startswith("foveation_net.nets.") and ".global_encoder." in key


@lru_cache(maxsize=1)
def get_runtime() -> tuple[Any, Path, Path, dict, dict]:
    """Load the policy, checkpoint, run directory, normalization and image geometry."""
    run_dir = _resolve_run_dir()
    config_path = run_dir / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    configure_dp_action_clipping(config)
    image_resize_wh_ratio = OmegaConf.select(config, "task.dataset.image_resize_wh_ratio")
    image_resize_hw = resolve_tcl_image_geometry(
        OmegaConf.select(config, "task.dataset.shape_meta"), image_resize_wh_ratio)
    checkpoint_path = _resolve_checkpoint(run_dir)
    payload = torch.load(checkpoint_path, map_location="cpu", pickle_module=dill)

    policy = hydra.utils.instantiate(config.policy)
    ema_state = copy.deepcopy(payload["state_dicts"].get("ema_model", payload["state_dicts"]["model"]))
    live_state = payload["state_dicts"].get("model", {})
    for key, value in live_state.items():
        if _is_global_encoder_key(key):
            ema_state.setdefault(key, value)
    policy.load_state_dict(ema_state, strict=True)
    device = torch.device(os.environ.get("PI05_DEVICE", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Foveation GPU service requires CUDA")
    policy = policy.to(device).eval()
    policy.infer_frame_idx = 0
    asset_dir = _resolve_asset_dir(run_dir, config)
    normalization = _load_normalization(run_dir, asset_dir, payload, config)
    print(
        f"[gpu_service_foveation] run={run_dir} checkpoint={checkpoint_path} "
        f"task_id={DEFAULT_TASK_ID} device={device}", flush=True,
    )
    output_hw = {key: (image_resize_hw[key][1],) * 2 if key in image_resize_hw else (256, 256)
                 for key in ("image", "gripper")}
    print(f"[gpu_service_foveation] image_resize_wh_ratio={image_resize_wh_ratio} "
          f"content_hw={image_resize_hw or 'source aspect ratio'} output_hw={output_hw}", flush=True)
    return policy, checkpoint_path, run_dir, normalization, image_resize_hw


def _letterbox_images(image_B_H_W_C: np.ndarray, content_hw=None) -> torch.Tensor:
    """Letterbox uint8 RGB images into normalized ``[B,1,3,size,size]`` tensors."""
    if image_B_H_W_C.ndim != 4 or image_B_H_W_C.shape[-1] != 3:
        raise ValueError(f"Expected RGB images [B,H,W,3], got {image_B_H_W_C.shape}")
    # Use the same single-frame operations as the training dataset.
    return torch.stack([prepare_tcl_rgb(image_H_W_C, content_hw)
                        for image_H_W_C in image_B_H_W_C]).unsqueeze(1)


def _local_task_id_one_hot(batch_size: int) -> torch.Tensor:
    """Create the temporary local task id condition for one observation step."""
    task_id_B_1_10 = torch.zeros((batch_size, 1, 10), dtype=torch.float32)
    task_id_B_1_10[:, 0, DEFAULT_TASK_ID] = 1.0
    return task_id_B_1_10


def _build_observation(step_data: dict[str, Any], image_resize_hw: dict) -> dict[str, torch.Tensor]:
    """Convert one legacy evaluator request to the DP policy observation format."""
    gt_video_B_VT_H_W_C = step_data["gt_video"]
    tcp_state_B_T_D = step_data["tcp_state"]
    print(f"[gpu_service_foveation] tcp_state shape={None if tcp_state_B_T_D is None else tcp_state_B_T_D.shape}", flush=True)
    if tcp_state_B_T_D is None or tcp_state_B_T_D.ndim != 3:
        raise ValueError("tcp_state must have shape [B,Ts,D]")
    batch_size, time_steps, state_dim = tcp_state_B_T_D.shape
    if state_dim < 6:
        raise ValueError(f"tcp_state must contain at least six TCP pose values, got D={state_dim}")
    if gt_video_B_VT_H_W_C.ndim != 5 or gt_video_B_VT_H_W_C.shape[-1] != 3:
        raise ValueError("gt_video must have shape [B,V*Ts,H,W,3]")
    if gt_video_B_VT_H_W_C.shape[0] != batch_size or gt_video_B_VT_H_W_C.shape[1] != 2 * time_steps:
        raise ValueError("gt_video must contain two Ts-length camera sequences")
    # NOTE: Legacy evaluator may append force or other state fields. Keep only
    # TCP pose here; the local task id is temporary until evaluator support lands.
    print("[TODO][Warning] Read an explicit remote task id when the evaluator protocol is updated.")
    tcp_pose_B_1_6 = tcp_state_B_T_D[:, -1:, :6].astype(np.float32)
    primary_B_1_C_S_S = _letterbox_images(
        gt_video_B_VT_H_W_C[:, time_steps - 1], image_resize_hw.get("image"))
    gripper_B_1_C_S_S = _letterbox_images(
        gt_video_B_VT_H_W_C[:, 2 * time_steps - 1], image_resize_hw.get("gripper"))
    return {
        "image": primary_B_1_C_S_S,
        "gripper": gripper_B_1_C_S_S,
        "joint_state": torch.from_numpy(tcp_pose_B_1_6),
        "task_id": _local_task_id_one_hot(batch_size),
    }


def _effective_cache_action(request: StepRequestFromEvaluator, horizon: int) -> int:
    """Return the configured cache length bounded by the model horizon."""
    requested = request.max_cache_action
    value = DEFAULT_MAX_CACHE_ACTION if requested is None else int(requested)
    if value <= 0:
        raise ValueError("max_cache_action must be positive")
    return min(value, horizon)


def _denormalize_action(action_B_T_D: torch.Tensor, normalization: dict) -> np.ndarray:
    """Invert saved scaling, clamp minmax actions, and binarize the gripper."""
    field = normalization['fields']['action']
    action_B_T_D = action_B_T_D.detach().cpu().float()
    if field['method'] == 'minmax':
        action_B_T_D = action_B_T_D.clamp(-1.0, 1.0)
    action_B_T_D = transform_values(action_B_T_D, field, inverse=True).numpy()
    threshold = float(os.environ.get("GRIPPER_THRESHOLD", DEFAULT_GRIPPER_THRESHOLD))
    action_B_T_D[..., 6] = (action_B_T_D[..., 6] >= threshold).astype(np.float32)
    return action_B_T_D.astype(np.float32)


@gpu_app.get("/")
def read_root():
    """Return a lightweight health response."""
    return {"message": "TCL foveation GPU service"}


@gpu_app.get("/init")
def model_init():
    """Load the policy and report its deployment settings."""
    policy, checkpoint_path, run_dir, _, _ = get_runtime()
    return {
        "message": "Initialized.",
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "max_cache_action": min(DEFAULT_MAX_CACHE_ACTION, int(policy.horizon)),
        "task_id": DEFAULT_TASK_ID,
    }


@gpu_app.get("/reset")
def model_reset():
    """Reset the policy temporal state."""
    policy, _, _, _, _ = get_runtime()
    policy.reset()
    return {"max_cache_action": min(DEFAULT_MAX_CACHE_ACTION, int(policy.horizon))}


@gpu_app.post("/step")
def model_step(step_request: StepRequestFromEvaluator):
    """Decode a request, run foveation DP inference, and return raw actions."""
    policy, checkpoint_path, _, normalization, image_resize_hw = get_runtime()
    if step_request.event == StepRequestEvent.ROLLOUT_END:
        policy.reset()
        return StepRequestFromPolicy.encode_from_raw(
            action=np.zeros((1, 1, 7), dtype=np.float32), max_cache_action=0
        ).model_dump(mode="json")
    try:
        step_data = step_request.decode_to_raw()
        obs = _build_observation(step_data, image_resize_hw)
        obs['joint_state'] = transform_values(obs['joint_state'], normalization['fields']['state'])
        device = next(policy.parameters()).device
        obs = {key: value.to(device) for key, value in obs.items()}
        with torch.no_grad():
            action_pred_B_T_D = policy.predict_action(obs)["action_pred"]
        if action_pred_B_T_D.ndim != 3 or action_pred_B_T_D.shape[-1] != 7:
            raise ValueError(f"Expected action [B,T,7], got {tuple(action_pred_B_T_D.shape)}")
        out_action_B_T_D = _denormalize_action(action_pred_B_T_D, normalization)
        cache_steps = _effective_cache_action(step_request, out_action_B_T_D.shape[1])
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    policy.infer_frame_idx += 1
    print(f"[gpu_service_foveation] checkpoint={checkpoint_path} "
          f"index={policy.infer_frame_idx} action_shape={out_action_B_T_D.shape}", flush=True)
    return StepRequestFromPolicy.encode_from_raw(
        action=out_action_B_T_D, max_cache_action=cache_steps
    ).model_dump(mode="json")


if __name__ == "__main__":
    get_runtime()
