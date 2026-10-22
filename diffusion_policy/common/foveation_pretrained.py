"""Small, inspectable artifacts for transferring a DP routing branch to Pi0.5."""

import copy
from pathlib import Path

import torch

from diffusion_policy.common.libero_task_util import LIBERO_TASK_NAMES
from diffusion_policy.common.tcl_data_util import DP_NORM, make_normalization


def _state_shapes(state_dict):
    return {name: tuple(value.shape) for name, value in state_dict.items()
            if torch.is_tensor(value)}


def make_foveation_artifact(policy, state_mean, state_std, *, task_mapping=None,
                            split_metadata=None):
    """Collect the two DP routing branches and their normalized condition contract."""
    if getattr(policy, "foveation_net", None) is None:
        raise ValueError("A DP policy with foveation_net is required")
    foveation = policy.foveation_net
    condition = getattr(policy, "router_condition_projection", None)
    if condition is None:
        raise ValueError("The DP policy must define router_condition_projection")
    # Pixel-space patch-bank buffers belong to the target geometry, not the router.
    branches = {
        key: {name: value.detach().cpu() for name, value in net.state_dict().items()
              if name.startswith(("global_encoder.", "router."))}
        for key, net in foveation.nets.items()
    }
    net = foveation.nets[foveation.view_keys[0]]
    router_config = {name: getattr(net, name) for name in (
        "router_type", "global_encoder_type", "global_feature_dim",
        "global_encoder_resize", "global_level", "router_hidden_dim", "router_cond_dim")}
    if any(any(getattr(branch, name) != value for name, value in router_config.items())
           or branch.router_temperature != net.router_temperature
           for branch in foveation.nets.values()):
        raise ValueError("Both camera branches must share routing configuration and temperature")
    artifact = {
        "camera_order": list(foveation.view_keys),
        "camera_orientation": {"image": policy.render_image_orientation, "gripper": "raw"},
        "router_config": router_config,
        "node_order": "level-major row-major quadtree",
        "source_geometry": {"input_hw": list(foveation.nets[foveation.view_keys[0]].input_hw),
                            "patch_size": int(foveation.nets[foveation.view_keys[0]].patch_size),
                            "num_levels": int(foveation.num_levels)},
        "target_geometry": {"image_size": 224, "patch_size": 14, "num_levels": 3,
                            "global_encoder_resize": 64},
        "router_temperature": float(foveation.nets[foveation.view_keys[0]].router_temperature),
        "task_mapping": (dict(task_mapping) if task_mapping is not None else
                          {name: index for index, name in enumerate(LIBERO_TASK_NAMES)}),
        "state_mean": torch.as_tensor(state_mean, dtype=torch.float32).flatten().cpu(),
        "state_std": torch.as_tensor(state_std, dtype=torch.float32).flatten().cpu(),
        "state_eps": 1.0e-6,
        "condition_projection": {name: value.detach().cpu() for name, value in condition.state_dict().items()},
        "condition_shapes": _state_shapes(condition.state_dict()),
        "branches": branches,
        "branch_shapes": {key: _state_shapes(value) for key, value in branches.items()},
        "split_metadata": split_metadata or {},
    }
    if split_metadata and 'normalization' in split_metadata:
        artifact['state_normalization'] = copy.deepcopy(split_metadata['normalization']['fields']['state'])
    if artifact["state_mean"].shape != (6,) or artifact["state_std"].shape != (6,):
        raise ValueError("DP state mean/std must contain the six joint-state values")
    _validate_artifact(artifact)
    return artifact


def save_foveation_artifact(path, artifact):
    """Save a CPU-only routing artifact without optimizer or policy state."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_foveation_artifact(path):
    """Load and validate required routing geometry and tensor shapes."""
    artifact = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    _validate_artifact(artifact)
    return artifact


def _validate_artifact(artifact):
    if 'state_normalization' in artifact:
        field = artifact['state_normalization']
        if field['dim'] != 6:
            raise ValueError('TCL routing state contract must contain six TCP values')
        methods = dict(DP_NORM, state=field['method'])
        make_normalization({'statistics': {'state': field['statistics']}}, methods,
                           dimensions=dict(state=6),
                           saved=dict(version=1, methods=methods, fields=dict(state=field)))
    required = ("camera_order", "node_order", "source_geometry", "target_geometry",
                "router_temperature", "task_mapping", "state_mean", "state_std",
                "condition_projection", "branches", "state_eps",
                "camera_orientation", "router_config")
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ValueError(f"Foveation artifact misses required fields: {missing}")
    if artifact["camera_order"] != ["image", "gripper"]:
        raise ValueError("Foveation artifact camera order must be image, gripper")
    if tuple(artifact["state_mean"].shape) != (6,) or tuple(artifact["state_std"].shape) != (6,):
        raise ValueError("Foveation artifact state statistics must have shape [6]")
    if artifact["target_geometry"] != {"image_size": 224, "patch_size": 14,
                                       "num_levels": 3, "global_encoder_resize": 64}:
        raise ValueError("Foveation artifact target geometry does not match Pi0.5")
    if artifact["source_geometry"] != {"input_hw": [256, 256], "patch_size": 16, "num_levels": 3}:
        raise ValueError("DP source geometry must be 256/patch16/3 levels")
    orientation = artifact["camera_orientation"]
    if orientation.get("image") not in ("rot180", "raw") or orientation.get("gripper") != "raw":
        raise ValueError("Routing transfer requires a raw or rot180 static image and raw wrist image")
    if artifact["node_order"] != "level-major row-major quadtree":
        raise ValueError("Routing node order differs")
    mapping = artifact["task_mapping"]
    if not isinstance(mapping, dict) or len(mapping) > 10:
        raise ValueError("Routing artifact task mapping must contain at most ten entries")
    task_ids = list(mapping.values())
    if any(type(task_id) is not int or not 0 <= task_id < 10 for task_id in task_ids):
        raise ValueError("Routing artifact task IDs must be unique integers in [0, 10)")
    if len(set(task_ids)) != len(task_ids) or any(not str(name) for name in mapping):
        raise ValueError("Routing artifact task mapping contains duplicate or empty entries")
    config = artifact["router_config"]
    if (config["router_type"] != "node_aligned_fpn" or config["global_encoder_type"] != "resnet10"
            or config["global_encoder_resize"] != 64 or config["global_level"] != 0):
        raise ValueError("Routing transfer requires ResNet10 FPN at global level 0 with resize64")
    for key in ("image", "gripper"):
        if key not in artifact["branches"]:
            raise ValueError(f"Foveation artifact misses {key} branch")
    return artifact

