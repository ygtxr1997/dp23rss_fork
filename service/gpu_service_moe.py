import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import dill
import hydra
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf

from diffusion_policy.dataset.tcl_dataset import TCLDatasetHDF5, TCLImageDataset
from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from robokit.debug_utils.time_profiler import global_time_profiler as gtp


"""How to use me?
conda activate robodiff
cd code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork
export LOG_TIME=2026.06.25-02.36.45
export CKPT_INDEX=-2
CUDA_VISIBLE_DEVICES=0 uvicorn service.gpu_service_moe:gpu_app --port 6070
"""

gpu_app = FastAPI()
max_cache_action = int(os.environ.get("MAX_CACHE_ACTION", "32"))

DEFAULT_OUTPUT_ROOT = "/home/geyuan/code/dp23rss_fork/data/outputs"
DEFAULT_RUN_NAME = "train_diffusion_transformer_hybrid_pusht_image"
TCL_LOAD_KEYS = [
    "rel_actions",
    "primary_rgb",
    "gripper_rgb",
    "robot_obs",
    "language_text",
    "force_torque",
]

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _str2bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _resolve_train_dir() -> Path:
    train_dir = os.environ.get("TRAIN_DIR", "").strip()
    if train_dir:
        return Path(train_dir).expanduser().resolve()

    log_time = os.environ.get("LOG_TIME", "").strip()
    if not log_time:
        raise FileNotFoundError("Set `TRAIN_DIR` or `LOG_TIME` for the TCL GPU service.")
    if "-" not in log_time:
        raise ValueError(f"`LOG_TIME` must be like YYYY.MM.DD-HH.MM.SS, got: {log_time}")

    date_part, time_part = log_time.split("-", 1)
    output_root = Path(os.environ.get("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).expanduser()
    return (output_root / date_part / f"{time_part}_{DEFAULT_RUN_NAME}").resolve()


def _resolve_config_path(train_dir: Path) -> Path:
    config_path = train_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing hydra config: {config_path}")
    return config_path


def _resolve_checkpoint_path(train_dir: Path) -> Path:
    ckpt_dir = train_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint dir: {ckpt_dir}")

    ckpt_paths = sorted(p for p in ckpt_dir.iterdir() if p.suffix == ".ckpt")
    if not ckpt_paths:
        raise FileNotFoundError(f"No .ckpt files found under: {ckpt_dir}")

    ckpt_index = int(os.environ.get("CKPT_INDEX", "-2"))
    if ckpt_index < -len(ckpt_paths) or ckpt_index >= len(ckpt_paths):
        raise IndexError(
            f"`CKPT_INDEX`={ckpt_index} out of range for {len(ckpt_paths)} checkpoints."
        )
    return ckpt_paths[ckpt_index]


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _extract_stats(payload: Dict[str, Any], source: Path) -> Dict[str, Dict[str, Any]]:
    stats = payload.get("stats", payload)
    if not isinstance(stats, dict):
        raise KeyError(f"Statistics payload from {source} does not contain a stats dict.")
    return stats


def _stats_to_numpy(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
    np_stats = {}
    for key, stat_dict in stats.items():
        if not isinstance(stat_dict, dict):
            continue
        np_stats[key] = {
            sub_key: np.asarray(sub_value, dtype=np.float32)
            for sub_key, sub_value in stat_dict.items()
        }
    return np_stats


def _missing_required_stats(stats: Dict[str, Dict[str, Any]]) -> list:
    missing = []
    for key in ("rel_actions", "force_torque"):
        if key not in stats:
            missing.append(key)

    if "rel_actions" in stats:
        for sub_key in ("min", "max", "mean"):
            if sub_key not in stats["rel_actions"]:
                missing.append(f"rel_actions.{sub_key}")

    if "force_torque" in stats:
        for sub_key in ("p01", "p99"):
            if sub_key not in stats["force_torque"]:
                missing.append(f"force_torque.{sub_key}")
    return missing


def _dataset_paths_from_config(hydra_config) -> Tuple[Path, Path]:
    dataset_cfg = hydra_config.task.dataset
    target = str(dataset_cfg.get("_target_", ""))
    supported_targets = (
        "diffusion_policy.dataset.tcl_dataset.TCLImageDataset",
        "diffusion_policy.dataset.tcl_dataset.TCLMasterSlaveDataset",
    )
    if target not in supported_targets:
        raise ValueError(
            f"Unsupported dataset target for TCL statistics fallback: {target}. "
            f"Expected one of {supported_targets}."
        )

    data_root = Path(str(_to_plain(dataset_cfg.data_root))).expanduser()
    h5_path_value = dataset_cfg.get("h5_path", None)
    if h5_path_value is None or str(h5_path_value).strip() == "":
        raise ValueError("TCL statistics fallback requires `task.dataset.h5_path`.")
    h5_path = Path(str(_to_plain(h5_path_value))).expanduser()
    return data_root, h5_path


def _build_statistics_from_dataset(hydra_config) -> Dict[str, Any]:
    data_root, h5_path = _dataset_paths_from_config(hydra_config)
    source_stats_path = data_root / "statistics.json"
    if not source_stats_path.exists():
        raise FileNotFoundError(f"Missing source TCL statistics: {source_stats_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing source TCL h5 file: {h5_path}")

    with source_stats_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    stats = _extract_stats(payload, source_stats_path)

    tcl_hdf5_dataset = TCLDatasetHDF5(
        str(data_root),
        str(h5_path),
        use_extracted=True,
        load_keys=TCL_LOAD_KEYS,
    )
    all_force_torques = tcl_hdf5_dataset.dsets["force_torque"]
    stats["force_torque"]["p01"] = np.quantile(all_force_torques, q=0.01, axis=0)
    stats["force_torque"]["p99"] = np.quantile(all_force_torques, q=0.99, axis=0)
    return payload


@lru_cache(maxsize=8)
def get_dataset_stats(train_dir_str: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], str]:
    train_dir = Path(train_dir_str)
    hydra_config = OmegaConf.load(str(_resolve_config_path(train_dir)))
    statistics_path = train_dir / "statistics.json"

    if statistics_path.exists():
        with statistics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        stats = _extract_stats(payload, statistics_path)
        missing = _missing_required_stats(stats)
        if not missing:
            return _stats_to_numpy(stats), str(statistics_path)
        print(
            f"[gpu_service_moe] Existing statistics missing {missing}; "
            "rebuilding TCL force quantile statistics from dataset config."
        )

    payload = _build_statistics_from_dataset(hydra_config)
    stats = _extract_stats(payload, statistics_path)
    missing = _missing_required_stats(stats)
    if missing:
        raise KeyError(f"TCL statistics missing required keys: {missing}")

    try:
        with statistics_path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=4)
        print(f"[gpu_service_moe] Dumped TCL statistics to: {statistics_path}")
    except OSError as exc:
        print(f"[gpu_service_moe] Warning: failed to cache statistics at {statistics_path}: {exc}")

    return _stats_to_numpy(stats), str(statistics_path)


def _is_moe_policy(hydra_config) -> bool:
    backbone_type = str(hydra_config.policy.get("backbone_type", "transformer")).lower()
    return backbone_type == "moe"


def _instantiate_policy_for_inference(hydra_config):
    if _is_moe_policy(hydra_config):
        hydra_config.policy.teacher_ckpts = []
        print("[gpu_service_moe] Cleared policy.teacher_ckpts for MoE checkpoint inference.")
    return hydra.utils.instantiate(hydra_config.policy)


def _load_checkpoint_state(model, ckpt_path: Path, use_ema: bool):
    with ckpt_path.open("rb") as f:
        payload = torch.load(f, map_location="cpu", pickle_module=dill)

    state_dicts = payload.get("state_dicts", None)
    if not isinstance(state_dicts, dict):
        raise KeyError(f"Checkpoint does not contain `state_dicts`: {ckpt_path}")

    branch = "ema_model" if use_ema else "model"
    if branch not in state_dicts:
        raise KeyError(f"Checkpoint missing state_dicts['{branch}']: {ckpt_path}")

    model.load_state_dict(state_dicts[branch], strict=True)
    return branch


@lru_cache(maxsize=4)
def get_agent(device: str, use_ema: bool = True):
    train_dir = _resolve_train_dir()
    hydra_config = OmegaConf.load(str(_resolve_config_path(train_dir)))
    model = _instantiate_policy_for_inference(hydra_config)
    print(f"[gpu_service_moe] Instantiated model type: {type(model)}")

    ckpt_path = _resolve_checkpoint_path(train_dir)
    state_branch = _load_checkpoint_state(model, ckpt_path, use_ema=use_ema)
    model = model.to(device).eval()
    model.infer_frame_idx = 0

    dataset_stats, stats_path = get_dataset_stats(str(train_dir))
    action_dim = int(hydra_config.shape_meta.action.shape[0])
    stats_action_dim = int(dataset_stats["rel_actions"]["min"].shape[0])
    if stats_action_dim != action_dim:
        raise ValueError(
            f"Action dim mismatch between config and statistics: "
            f"config={action_dim}, stats={stats_action_dim}, file={stats_path}"
        )

    print(
        f"[gpu_service_moe] Loaded ckpt={ckpt_path}, branch={state_branch}, "
        f"device={device}, stats={stats_path}"
    )
    return model, hydra_config, str(ckpt_path), str(train_dir)


class SharedMemoryPool:
    def __init__(self):
        self._buffers: Dict[str, np.ndarray] = {}
        self._shapes: Dict[str, Tuple[int, ...]] = {}

    def get_or_allocate(
            self,
            key_to_shape: Union[str, Dict[str, Tuple[int, ...]]],
            dtype=np.float32) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(key_to_shape, str):
            return self._buffers[key_to_shape]

        for key, shape in key_to_shape.items():
            if key not in self._buffers or self._shapes.get(key) != shape:
                self._buffers[key] = np.empty(shape, dtype=dtype)
                self._shapes[key] = shape
                print(f"[gpu_service_moe] Allocated buffer `{key}` with shape {shape}.")
        return {key: self._buffers[key] for key in key_to_shape.keys()}

    def clear(self):
        self._buffers.clear()
        self._shapes.clear()


@lru_cache()
def get_mem_pool():
    return SharedMemoryPool()


def _shape_meta_obs(hydra_config):
    return hydra_config.shape_meta.obs


def _obs_has(hydra_config, key: str) -> bool:
    return key in _shape_meta_obs(hydra_config)


def _get_image_shape(hydra_config) -> Tuple[int, int, int]:
    if "image_shape" in hydra_config:
        c, h, w = hydra_config.image_shape
    else:
        c, h, w = hydra_config.shape_meta.obs.image.shape
    return int(c), int(h), int(w)


def _decode_step_request(step_request: StepRequestFromEvaluator, hydra_config):
    request_max_cache = int(getattr(step_request, "max_cache_action", max_cache_action))
    request_num_views = int(getattr(step_request, "num_camera_views", 2))

    if hasattr(step_request, "decode_to_raw_buffer"):
        c, h, w = _get_image_shape(hydra_config)
        mem_buffer = get_mem_pool()
        mem_buffer.get_or_allocate({
            "gt_video": (1, request_num_views * request_max_cache, h, w, c),
        }, dtype=np.uint8)
        with gtp("decode", group="process_request"):
            return step_request.decode_to_raw_buffer(
                out_video_buffer=mem_buffer.get_or_allocate("gt_video")
            )

    with gtp("decode", group="process_request"):
        return step_request.decode_to_raw()


def _split_video_views(gt_video: np.ndarray, tcp_state: np.ndarray) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if gt_video.ndim != 5:
        raise ValueError(f"`gt_video` must be [B,V*Ts,H,W,3], got shape={gt_video.shape}.")
    if tcp_state.ndim != 3:
        raise ValueError(f"`tcp_state` must be [B,Ts,D], got shape={tcp_state.shape}.")

    batch_size, time_steps, _ = tcp_state.shape
    if gt_video.shape[0] != batch_size:
        raise ValueError(f"`gt_video` B={gt_video.shape[0]} does not match tcp_state B={batch_size}.")
    if gt_video.shape[1] % time_steps != 0:
        raise ValueError(
            f"`gt_video` frame dimension must be a multiple of Ts={time_steps}, "
            f"got {gt_video.shape[1]}."
        )

    num_views = gt_video.shape[1] // time_steps
    if num_views < 1:
        raise ValueError("`gt_video` must contain at least one camera view.")

    video = torch.from_numpy(gt_video).float() / 127.5 - 1.0
    primary = video[:, :time_steps].permute(0, 1, 4, 2, 3).contiguous()
    gripper = None
    if num_views >= 2:
        gripper = video[:, time_steps:2 * time_steps].permute(0, 1, 4, 2, 3).contiguous()
    return primary, gripper


def _build_obs_dict(step_data: Dict[str, Any], hydra_config, dataset_stats, device: str) -> Dict[str, torch.Tensor]:
    gt_video = step_data["gt_video"]
    tcp_state = step_data["tcp_state"]
    if tcp_state is None:
        raise ValueError("`tcp_state` is required by the TCL GPU service.")
    if tcp_state.ndim != 3:
        raise ValueError(f"`tcp_state` must be [B,Ts,D], got shape={tcp_state.shape}.")

    batch_size, time_steps, state_dim = tcp_state.shape
    n_obs_steps = int(hydra_config.n_obs_steps)
    if time_steps < n_obs_steps:
        raise ValueError(
            f"`tcp_state` time length must be >= n_obs_steps={n_obs_steps}, got {time_steps}."
        )
    if state_dim < 6:
        raise ValueError(f"`tcp_state` must contain at least 6 TCP pose dims, got D={state_dim}.")

    primary_video, gripper_video = _split_video_views(gt_video, tcp_state)
    obs_dict = {
        "image": primary_video[:, -n_obs_steps:].to(device),
        "joint_state": torch.from_numpy(
            tcp_state[:, -n_obs_steps:, :6].astype(np.float32)
        ).to(device),
    }

    if _obs_has(hydra_config, "gripper"):
        if gripper_video is None:
            raise ValueError("Config expects `gripper` images, but request has fewer than 2 camera views.")
        obs_dict["gripper"] = gripper_video[:, -n_obs_steps:].to(device)

    if _obs_has(hydra_config, "force"):
        if state_dim < 12:
            raise ValueError(
                f"Config expects `force`, but `tcp_state` has D={state_dim}; expected pose+force dims."
            )
        force = tcp_state[:, -n_obs_steps:, -6:].astype(np.float32)
        force = TCLImageDataset.norm_state_or_force(
            force,
            norm_type="quantile",
            meta_data=dataset_stats["force_torque"],
        ).astype(np.float32)
        if bool(hydra_config.task.dataset.get("zero_force", False)):
            force = force * 0.0
        obs_dict["force"] = torch.from_numpy(force).to(device)

    if _obs_has(hydra_config, "success"):
        obs_dict["success"] = torch.ones(
            (batch_size, n_obs_steps, 1),
            dtype=torch.float32,
            device=device,
        )

    return obs_dict


def _denormalize_action(action_pred: torch.Tensor, dataset_stats) -> np.ndarray:
    if action_pred.ndim != 3:
        raise ValueError(f"`action_pred` must be [B,T,D], got shape={tuple(action_pred.shape)}.")

    action_min = dataset_stats["rel_actions"]["min"]
    action_max = dataset_stats["rel_actions"]["max"]
    action_mean = dataset_stats["rel_actions"]["mean"]
    if action_pred.shape[-1] != action_min.shape[0]:
        raise ValueError(
            f"Action dim mismatch: prediction D={action_pred.shape[-1]}, "
            f"statistics D={action_min.shape[0]}."
        )

    action = (action_pred.detach() * 0.5 + 0.5).cpu()
    action = action.clamp(0.0, 1.0).numpy().astype(np.float32)
    action = action * (action_max[None, None, :] - action_min[None, None, :]) + action_min[None, None, :]

    if action.shape[-1] > 6:
        threshold = float(action_mean[-1])
        action[..., 6:] = (action[..., 6:] >= threshold).astype(np.float32)
    return action


def _get_effective_max_cache_action(step_request: Optional[StepRequestFromEvaluator] = None) -> int:
    if step_request is None:
        return max_cache_action
    return int(getattr(step_request, "max_cache_action", max_cache_action))


@gpu_app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@gpu_app.get("/init")
def model_init():
    device = "cuda"
    use_ema = _str2bool(os.environ.get("USE_EMA"), True)
    _, hydra_config, ckpt_path, train_dir = get_agent(device=device, use_ema=use_ema)
    return {
        "message": "Initialized.",
        "max_cache_action": _get_effective_max_cache_action(),
        "ckpt_path": ckpt_path,
        "train_dir": train_dir,
        "use_ema": use_ema,
        "backbone_type": str(hydra_config.policy.get("backbone_type", "transformer")),
        "n_obs_steps": int(hydra_config.n_obs_steps),
    }


@gpu_app.get("/reset")
def model_reset():
    device = "cuda"
    use_ema = _str2bool(os.environ.get("USE_EMA"), True)
    agent, _, _, _ = get_agent(device=device, use_ema=use_ema)
    if hasattr(agent, "reset"):
        agent.reset()
    return {"max_cache_action": _get_effective_max_cache_action()}


@gpu_app.post("/step")
def model_step(step_request: StepRequestFromEvaluator):
    device = "cuda"
    use_ema = _str2bool(os.environ.get("USE_EMA"), True)
    agent, hydra_config, ckpt_path, train_dir = get_agent(device=device, use_ema=use_ema)
    dataset_stats, _ = get_dataset_stats(train_dir)
    print("[gpu_service_moe] Using cached ckpt from:", ckpt_path, "Model type:", type(agent))

    try:
        step_data = _decode_step_request(step_request, hydra_config)
        with gtp("cpu_build_obs", group="process_request"):
            obs_dict = _build_obs_dict(step_data, hydra_config, dataset_stats, device=device)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with gtp("predict_action", group="model_infer"):
        with torch.no_grad():
            result = agent.predict_action(obs_dict)
            action_pred = result["action_pred"]

    try:
        out_action = _denormalize_action(action_pred, dataset_stats)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    agent.infer_frame_idx += 1
    gtp.step()
    print(f"[gpu_service_moe] index={agent.infer_frame_idx}, output action shape: {out_action.shape}")

    with gtp("encode", group="process_request"):
        request_to_evaluator = StepRequestFromPolicy.encode_from_raw(
            action=out_action,
            max_cache_action=_get_effective_max_cache_action(step_request),
        )

    if agent.infer_frame_idx % 15 == 0:
        gtp.report()
    return request_to_evaluator.model_dump(mode="json")


if __name__ == "__main__":
    get_agent("cuda", use_ema=_str2bool(os.environ.get("USE_EMA"), True))
    print(f"[gpu_service_moe] ready with max_cache_action={max_cache_action}")
