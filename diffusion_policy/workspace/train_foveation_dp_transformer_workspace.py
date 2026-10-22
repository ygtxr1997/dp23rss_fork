"""Training workspace for the foveation diffusion policy."""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import json
import inspect
import math
import os
import pathlib
import random
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import Dict, Optional

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.checkpoint_selection import select_pretrained_checkpoint
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.foveation_video_util import make_foveation_vis_log
from diffusion_policy.common.foveation_training_util import (
    cosine_schedule_value, projected_dual_ascent,
)
from diffusion_policy.common.libero_task_util import LIBERO_TASK_NAMES
from diffusion_policy.common.tcl_data_util import configure_dp_action_clipping
from diffusion_policy.common.foveation_pretrained import make_foveation_artifact, save_foveation_artifact
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.policy.foveation_dp_transformer_policy import (
    FoveationDPTransformerPolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainFoveationDPTransformerWorkspace(BaseWorkspace):
    """Train and validate a foveation diffusion policy."""

    include_keys = (
        "global_step", "epoch", "next_epoch", "downstream_finetune_started", "budget_lambda",
        "dataset_metadata"
    )
    _ROUTING_PREFIXES = ("foveation_net.", "router_condition_projection.")

    @staticmethod
    def _validate_pretrained_weights_only(cfg):
        if not cfg.get("pretrained_weights_only", False):
            return
        if not cfg.get("pretrained_ckpt"):
            raise ValueError("pretrained_weights_only requires pretrained_ckpt")
        for key in ("training.resume", "training.router_only",
                    "reuse_pretrained_dataset_config", "data_seed_from_ckpt"):
            if OmegaConf.select(cfg, key, default=False):
                raise ValueError(f"pretrained_weights_only requires {key}=false")

    @classmethod
    def prepare_config(cls, cfg: OmegaConf):
        """Select pretrained weights and inherit data before resolving configuration."""
        cls._validate_pretrained_weights_only(cfg)
        mode = cfg.get("pretrained_ckpt_mode")
        reuse_dataset = cfg.get("reuse_pretrained_dataset_config", False)
        pretrained_ckpt = cfg.get("pretrained_ckpt")
        if not pretrained_ckpt:
            if mode is not None or reuse_dataset:
                raise ValueError(
                    "pretrained_ckpt_mode and reuse_pretrained_dataset_config require pretrained_ckpt"
                )
            return configure_dp_action_clipping(cfg)

        # Hydra may change cwd; source paths remain relative to the launch directory.
        source = hydra.utils.to_absolute_path(str(pathlib.Path(pretrained_ckpt).expanduser()))
        checkpoint_path = select_pretrained_checkpoint(source, mode)
        cfg.pretrained_ckpt = str(checkpoint_path)
        if reuse_dataset:
            source_path = checkpoint_path.parent.parent / ".hydra" / "config.yaml"
            source_cfg = OmegaConf.load(source_path)
            source_dataset = OmegaConf.select(source_cfg, "task.dataset")
            if source_dataset is None:
                raise ValueError(f"{source_path} lacks task.dataset")
            # Resolve in the source context, then replace before evaluation interpolations resolve.
            cfg.task.dataset = OmegaConf.create(
                OmegaConf.to_container(source_dataset, resolve=True)
            )
            print(f"[Dataset config] Inherited task.dataset from {source_path}")
        return configure_dp_action_clipping(cfg)

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self._validate_pretrained_weights_only(cfg)
        super().__init__(cfg, output_dir=output_dir)
        # Keep direct Workspace construction consistent with Hydra entrypoints.
        configure_dp_action_clipping(cfg)
        ddp_timeout = int(self._cfg_get(cfg, "training.ddp_timeout_seconds", 7200))
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=ddp_timeout))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[process_kwargs, ddp_kwargs],
            gradient_accumulation_steps=int(self._cfg_get(cfg, "training.gradient_accumulate_every", 1)),
        )

        reserve_gb = float(self._cfg_get(cfg, "training.reserve_gpu_memory_gb", 0))
        if not math.isfinite(reserve_gb) or reserve_gb < 0:
            raise ValueError("training.reserve_gpu_memory_gb must be finite and non-negative")
        if reserve_gb > 0:
            if self.accelerator.device.type != "cuda":
                raise ValueError("GPU memory reservation requires a CUDA device")
            # Each rank holds its own byte buffer; not a model buffer or checkpoint state.
            reserve_bytes = int(reserve_gb * 1024 ** 3)
            self._reserved_gpu_memory_N = torch.empty(
                reserve_bytes, dtype=torch.uint8, device=self.accelerator.device,
            )
            print(f"[GPU reservation] rank={self.accelerator.process_index} "
                  f"device={self.accelerator.device} "
                  f"holding {reserve_bytes / 1024 ** 3:.3f} GiB ({reserve_bytes} bytes)", flush=True)

        seed = int(self._cfg_get(cfg, "training.seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self._is_libero = str(cfg.task.dataset._target_).endswith(".LiberoFTDataset")
        self.vis_rot_image = bool(self._cfg_get(cfg, "vis_rot_image", False))

        self.model: FoveationDPTransformerPolicy = hydra.utils.instantiate(cfg.policy)
        if self.model.temporal_adaptive and (
            not bool(self._cfg_get(cfg, "training.router_only", False))
            or not self.model.obs_as_cond
        ):
            raise ValueError("Temporal routing requires paired router-only Stage B with obs_as_cond=true.")
        self.dataset_metadata = None
        pretrained_ckpt = self._cfg_get(cfg, "pretrained_ckpt", None)
        if pretrained_ckpt:
            self._load_pretrained_weights(pretrained_ckpt)

        self.downstream_finetune_start_epoch = self._cfg_get(
            cfg, "training.downstream_finetune_start_epoch", None
        )
        self.downstream_finetune_started = False
        self.next_epoch = 0
        start = self.downstream_finetune_start_epoch
        if start is not None:
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or not 0 <= start < cfg.training.num_epochs
            ):
                raise ValueError("downstream_finetune_start_epoch must be an integer in [0, num_epochs).")
            if not self._cfg_get(cfg, "training.router_only", False) or self.model.router_loss_type != "paired":
                raise ValueError("Downstream finetuning requires paired router-only Stage B.")
            if self._cfg_get(cfg, "training.target_num_patches_start", None) is not None:
                if start < int(self._cfg_get(cfg, "training.target_num_patches_decay_epochs", 0)):
                    raise ValueError("Downstream finetuning must start after the budget decay finishes.")

        if bool(self._cfg_get(cfg, "training.router_only", False)):
            foveation_net = self.model.foveation_net
            if (
                foveation_net is None
                or any(net.global_encoder is None or net.fix_split_mode is not None
                       or net.freeze_global_encoder or net.freeze_router
                       for net in foveation_net.nets.values())
                or bool(self._cfg_get(cfg, "policy.en_freeze_obs_encoder", False))
            ):
                raise ValueError(
                    "router_only requires a learned router with freeze_global_encoder, "
                    "freeze_router and en_freeze_obs_encoder all false."
                )
            if start is None:
                self.model.freeze_action_path()
            else:
                # Register both phases before optimizer creation and DDP wrapping.
                self.model.set_stage_b_phase(False)

        if self.model.router_loss_type == "paired":
            foveation_net = self.model.foveation_net
            if (
                not bool(self._cfg_get(cfg, "training.router_only", False))
                or foveation_net is None
                or any(net.random_topology_prob != 0.0 for net in foveation_net.nets.values())
            ):
                raise ValueError(
                    "Paired routing requires router_only=true and "
                    "random_topology_prob=0."
                )

        encoders = self._global_encoders(self.model)
        self._shared_encoder_keys = {
            key for key, encoder in encoders.items()
            if not any(param.requires_grad for param in encoder.parameters())
        }
        self._shared_encoder_prefixes = tuple(
            f"foveation_net.nets.{key}.global_encoder." for key in self.model.rgb_keys
            if key in self._shared_encoder_keys
        )
        self.ema_model: Optional[FoveationDPTransformerPolicy] = None
        if bool(self._cfg_get(cfg, "training.use_ema", False)):
            # Only frozen encoders are shared; trainable encoders need an EMA copy.
            memo = {id(encoders[key]): encoders[key] for key in self._shared_encoder_keys}
            self.ema_model = copy.deepcopy(self.model, memo)
            self._rebind_shared_encoder()
        self._assert_encoder_state()

        if bool(self._cfg_get(cfg, "policy.en_freeze_obs_encoder", False)):
            self.model.freeze_obs_encoder()
        self.model.print_training_status()
        optimizer_cfg = self._cfg_get(cfg, "optimizer", {})
        if not isinstance(optimizer_cfg, dict):
            optimizer_cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)
        self.optimizer = self.model.get_optimizer(**optimizer_cfg)
        self._assert_encoder_optimizer()
        self.global_step = 0
        self.epoch = 0
        lambda_init = float(self._cfg_get(cfg, "training.budget_lambda_init", 0.0))
        self.budget_lambda = {key: lambda_init for key in self.model.rgb_keys}
        self.budget_lambda_lr = float(self._cfg_get(cfg, "training.budget_lambda_lr", 0.01))
        if lambda_init < 0.0 or self.budget_lambda_lr < 0.0:
            raise ValueError("budget_lambda_init and budget_lambda_lr must be non-negative.")
        self._sync_budget_lambda()

    def _sync_budget_lambda(self) -> None:
        for policy in (self.model, self.ema_model):
            if policy is not None and self._unwrap(policy).foveation_net is not None:
                for key, net in self._unwrap(policy).foveation_net.nets.items():
                    net.budget_lambda = self.budget_lambda[key]

    def _update_budget_lambda(self, budget_stats_V_2: torch.Tensor) -> Dict[str, float]:
        """Update each camera's dual price once from the optimizer step's global counts."""
        metrics = {}
        for view, (key, net) in enumerate(self._unwrap(self.model).foveation_net.nets.items()):
            expected_mean = (budget_stats_V_2[view, 0] / budget_stats_V_2[view, 1]).item()
            gap = expected_mean - net.resolved_target_num_patches
            metrics[f"train/views/{key}/budget_lambda"] = self.budget_lambda[key]
            metrics[f"train/views/{key}/expected_num_patches_mean"] = expected_mean
            metrics[f"train/views/{key}/budget_gap"] = gap
            self.budget_lambda[key] = projected_dual_ascent(
                self.budget_lambda[key], expected_mean, net.resolved_target_num_patches,
                self.budget_lambda_lr, net.max_num_patches - net.min_num_patches,
            )
        self._sync_budget_lambda()
        return metrics

    def _reduce_sum(self, value_X: torch.Tensor) -> torch.Tensor:
        # Bypass old Accelerate's single-process vector reduction; distributed
        # reduction preserves the tensor shape and batches all camera statistics.
        if self.accelerator.num_processes == 1:
            return value_X
        return self.accelerator.reduce(value_X, reduction="sum")

    def _globalize_elu_budget(self, policy, losses) -> None:
        if "_budget_normalized_mean" not in losses:
            return
        nets = self._unwrap(policy).foveation_net.nets
        local_mean_V = losses["_budget_normalized_mean"]
        global_sum_V = self._reduce_sum(losses["_budget_expected_sum"].double())
        global_count_V = self._reduce_sum(losses["_budget_observation_count"].double())
        budget_losses = []
        for view, (key, net) in enumerate(nets.items()):
            budget_range = net.max_num_patches - net.min_num_patches
            global_mean_scalar = (
                (global_sum_V[view] / global_count_V[view] - net.min_num_patches) / budget_range
            ).to(local_mean_V.dtype)
            # Forward uses each camera's global mean; batch weighting/DDP supply
            # the observation weighting of its local differentiable contribution.
            mean_proxy_scalar = global_mean_scalar + (local_mean_V[view] - local_mean_V[view].detach())
            normalized_target = (net.resolved_target_num_patches - net.min_num_patches) / budget_range
            budget_loss_scalar = net.upper_bound_budget_loss(
                mean_proxy_scalar, normalized_target, net.budget_loss_type,
                net.budget_base_cost_weight, net.budget_lambda, net.budget_elu_tau,
            ) * net.budget_loss_weight
            budget_losses.append(budget_loss_scalar)
            losses[f"views/{key}/budget_loss"] = budget_loss_scalar
        # Accumulation averages per-forward ELU objectives; lambda uses the whole step.
        budget_loss_V = torch.stack(budget_losses)
        losses["_view_metric_sums"]["budget_loss"] = (
            budget_loss_V.detach().double() * losses["_view_metric_counts"]["budget_loss"]
        )
        budget_loss_scalar = budget_loss_V.mean()
        losses["total_loss"] = losses["total_loss"] - losses["budget_loss"] + budget_loss_scalar
        losses["budget_loss"] = budget_loss_scalar

    def _view_metrics(self, policy, losses) -> Dict[str, float]:
        metrics = {}
        keys = self._unwrap(policy).rgb_keys
        names = list(losses.get("_view_metric_sums", {}))
        if names:
            sums_M_V = torch.stack([losses["_view_metric_sums"][name].detach().double() for name in names])
            counts_M_V = torch.stack([losses["_view_metric_counts"][name].detach().double() for name in names])
            # One collective for all per-camera metric sums/counts.
            stats_2_M_V = self._reduce_sum(torch.stack([sums_M_V, counts_M_V])).cpu()
            for index, name in enumerate(names):
                for view, key in enumerate(keys):
                    if stats_2_M_V[1, index, view].item() > 0:
                        metrics[f"views/{key}/{name}"] = (
                            stats_2_M_V[0, index, view] / stats_2_M_V[1, index, view]
                        ).item()
        for field, name in (("_view_sample_count", "sample_count"),
                            ("_view_valid_count", "valid_pair_count"),
                            ("_view_matched_count", "matched_pair_count")):
            if field in losses:
                count_V = self._reduce_sum(losses[field].detach().double()).cpu()
                metrics.update({f"views/{key}/{name}": count_V[view].item() for view, key in enumerate(keys)})
        net_group = self._unwrap(policy).foveation_net
        if net_group is not None:
            for key, net in net_group.nets.items():
                metrics[f"views/{key}/budget_lambda"] = net.budget_lambda
                expected = metrics.get(f"views/{key}/expected_num_patches_mean")
                if expected is not None and net.resolved_target_num_patches is not None:
                    metrics[f"views/{key}/budget_gap"] = expected - net.resolved_target_num_patches
        return metrics

    def _load_pretrained_weights(self, pretrained_ckpt: str) -> None:
        """Load EMA weights, completing the shared encoder from the live state."""
        payload = torch.load(pretrained_ckpt, map_location="cpu", pickle_module=dill)
        self._check_view_order(payload["cfg"])
        if (str(self.cfg.task.dataset._target_).endswith((".LiberoFTDataset", ".TCLFoveationDataset"))
                and self._cfg_get(self.cfg, "training.router_only", False)):
            self._check_pretrained_config(payload["cfg"])
        weights_only = self.cfg.get("pretrained_weights_only", False)
        if not weights_only and "dataset_metadata" in payload["pickles"]:
            self.dataset_metadata = dill.loads(payload["pickles"]["dataset_metadata"])
        state = payload["state_dicts"]["ema_model"].copy()
        live_state = payload["state_dicts"]["model"]
        # Stage A stores its shared, frozen encoder only in the live model.
        # All other weights (including the untrained router) retain their EMA source.
        for name in live_state:
            if self._is_global_state_key(name):
                state.setdefault(name, live_state[name])
        self.model.load_state_dict(state, strict=True)
        if self.accelerator.is_main_process:
            print(f"[FoveationWorkspace] Loaded pretrained EMA weights: {pretrained_ckpt}")
            if weights_only:
                print("[FoveationWorkspace] Weights-only initialization; "
                      "using current dataset configuration and statistics.")

    def _check_view_order(self, source_cfg) -> None:
        obs_meta = self._cfg_get(source_cfg, "policy.shape_meta.obs", {})
        source_keys = [key for key, meta in obs_meta.items() if meta.get("type", "low_dim") == "rgb"]
        if source_keys != self._unwrap(self.model).rgb_keys:
            raise ValueError(f"Checkpoint RGB key order differs: {source_keys} vs {self._unwrap(self.model).rgb_keys}")

    def _check_pretrained_config(self, stage_a):
        # Compare effective sampling settings for legacy quantile checkpoints too.
        stage_a = configure_dp_action_clipping(copy.deepcopy(stage_a))
        # Constructor defaults make omitted legacy options comparable without
        # constructing a second network or downloading pretrained encoders.
        stage_b = self.cfg
        differences = []

        def canonical(value):
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            if isinstance(value, (list, tuple)):
                return [canonical(item) for item in value]
            if isinstance(value, dict):
                return {key: canonical(item) for key, item in value.items()}
            return value

        def compare(path, default=None):
            a = canonical(self._cfg_get(stage_a, path, default))
            b = canonical(self._cfg_get(stage_b, path, default))
            if a != b:
                differences.append(f"  {path}: A={a!r}, B={b!r}")

        allowed = {
            "router_loss_type", "paired_loss_weight", "paired_node_sampling",
            "temporal_adaptive", "temporal_loss_weight", "router_temperature",
            "freeze_global_encoder", "freeze_router", "en_freeze_obs_encoder",
            "fix_split_mode", "random_topology_prob", "torch_home",
            "target_num_patches", "budget_loss_weight", "budget_probability_type",
            "budget_loss_type", "budget_base_cost_weight", "budget_elu_tau",
            # Stage A weights replace initialization when loaded with strict=True.
            "visual_encoder_init",
        }
        for path in ("policy", "policy.foveation_net", "policy.noise_scheduler"):
            compare(path + "._target_")
            config_a = self._cfg_get(stage_a, path, {})
            config_b = self._cfg_get(stage_b, path, {})
            component = hydra.utils.get_class(config_b["_target_"])
            params = inspect.signature(component.__init__).parameters
            keys = set(config_a) | set(config_b) | set(params)
            for key in sorted(keys - allowed - {"self", "args", "kwargs", "foveation_net", "noise_scheduler"}):
                if key.startswith("_"):
                    continue
                param = params.get(key)
                default = None if param is None or param.default is inspect.Parameter.empty else param.default
                # None means all scheduler training timesteps in the policy API.
                if key == "num_inference_steps":
                    a = self._cfg_get(stage_a, path + "." + key,
                                      self._cfg_get(stage_a, "policy.noise_scheduler.num_train_timesteps", 1000))
                    b = self._cfg_get(stage_b, path + "." + key,
                                      self._cfg_get(stage_b, "policy.noise_scheduler.num_train_timesteps", 1000))
                    if a != b:
                        differences.append(f"  {path}.{key}: A={a!r}, B={b!r}")
                else:
                    compare(path + "." + key, default)
        for key, default in {
            "_target_": None, "dataset_subname": None, "hdf5_fns": None,
            "shape_meta": None, "horizon": None, "pad_before": 0, "pad_after": 0,
            "zero_force": False, "norm_force_type": "quantile",
            "load_future_obs": False, "max_len": None,
        }.items():
            compare("task.dataset." + key, default)
        if str(stage_b.task.dataset._target_).endswith(".TCLFoveationDataset"):
            # Dataset metadata validates canonical paths and ordered observation keys.
            for key, default in {"use_h5": False, "default_success_label": None}.items():
                compare("task.dataset." + key, default)
        if differences:
            raise ValueError("Stage A/B configuration mismatch:\n" + "\n".join(differences)
                             + "\nStage B requires matching Stage A architecture, observations and diffusion settings.")
        mode = self._cfg_get(stage_a, "policy.foveation_net.fix_split_mode", None)
        if self.accelerator.is_main_process:
            print(f"[Stage B] Stage A fix_split_mode={mode!r}")
            if isinstance(canonical(mode), list):
                print("[Stage B] Stage A used weighted level/random sampling.")
            elif mode is not None and not str(mode).startswith("random:"):
                print("[Stage B] Fixed-tree checkpoint: no joint multi-resolution training.")

    @staticmethod
    def _cfg_get(cfg, path: str, default=None):
        value = cfg
        for part in path.split("."):
            if value is None:
                return default
            if isinstance(value, dict):
                if part not in value:
                    return default
                value = value[part]
            else:
                value = getattr(value, part, default)
        return default if value is None else value

    @staticmethod
    def _unwrap(policy):
        return policy.module if hasattr(policy, "module") else policy

    @classmethod
    def _global_encoders(cls, policy):
        group = cls._unwrap(policy).foveation_net
        return {} if group is None else {
            key: net.global_encoder for key, net in group.nets.items() if net.global_encoder is not None
        }

    def _rebind_shared_encoder(self) -> None:
        if self.ema_model is None:
            return
        live = self._unwrap(self.model)
        ema = self._unwrap(self.ema_model)
        for key in self._shared_encoder_keys:
            ema.foveation_net.nets[key].global_encoder = live.foveation_net.nets[key].global_encoder

    def _assert_encoder_state(self) -> None:
        live_encoders = self._global_encoders(self.model)
        ema_encoders = self._global_encoders(self.ema_model) if self.ema_model is not None else {}
        for key, encoder in live_encoders.items():
            shared = key in self._shared_encoder_keys
            if self.ema_model is not None and (ema_encoders[key] is encoder) != shared:
                raise RuntimeError(f"{key}: only frozen encoders may share live/EMA weights.")
            if shared and (encoder.training or any(p.requires_grad for p in encoder.parameters())):
                raise RuntimeError(f"{key}: shared encoder must remain frozen and in eval mode.")

    def _assert_encoder_optimizer(self) -> None:
        optimizer_ids = {id(param) for group in self.optimizer.param_groups for param in group["params"]}
        for key, encoder in self._global_encoders(self.model).items():
            encoder_ids = {id(param) for param in encoder.parameters()}
            trainable_ids = {id(param) for param in encoder.parameters() if param.requires_grad}
            if (encoder_ids & optimizer_ids) != trainable_ids:
                raise RuntimeError(f"{key}: optimizer must contain exactly the trainable encoder parameters.")

    @staticmethod
    def _is_global_state_key(key: str) -> bool:
        return key.startswith("foveation_net.nets.") and ".global_encoder." in key

    def _state_dict_of(self, key, value, use_thread=True):
        state = super()._state_dict_of(key, value, use_thread=use_thread)
        if (
            self._shared_encoder_keys
            and key == "ema_model"
            and isinstance(state, dict)
        ):
            # The live state already carries the sole shared encoder copy.
            state = {
                name: tensor
                for name, tensor in state.items()
                if not name.startswith(self._shared_encoder_prefixes)
            }
        return state

    def _load_state_dict_into(self, key, obj, state, **kwargs):
        if (
            self._shared_encoder_keys
            and key == "ema_model"
            and isinstance(state, dict)
        ):
            incompatible = obj.load_state_dict(state, strict=False)
            missing = [
                name
                for name in incompatible.missing_keys
                if not name.startswith(self._shared_encoder_prefixes)
            ]
            if incompatible.unexpected_keys or missing:
                raise RuntimeError(
                    "EMA checkpoint has incompatible non-shared keys: "
                    f"missing={missing[:5]}, unexpected={incompatible.unexpected_keys[:5]}"
                )
            self._rebind_shared_encoder()
            return
        super()._load_state_dict_into(key, obj, state, **kwargs)

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        self._check_view_order(payload["cfg"])
        # Missing legacy input orientation means raw, regardless of old display flags.
        saved_orientation = self._cfg_get(payload["cfg"], "policy.render_image_orientation", "raw")
        if saved_orientation != self._unwrap(self.model).render_image_orientation:
            raise ValueError("Resume requires matching policy.render_image_orientation")
        restore_optimizer = "optimizer" in payload["state_dicts"] and "optimizer" not in (exclude_keys or ())
        if restore_optimizer:
            old_start = self._cfg_get(payload["cfg"], "training.downstream_finetune_start_epoch", None)
            if (old_start is None) != (self.downstream_finetune_start_epoch is None):
                raise ValueError(
                    "Cannot resume across router-only and staged optimizer layouts. "
                    "Use pretrained_ckpt with resume=false for a new finetuning run; "
                    "this does not restore the old optimizer or LR schedule."
                )
        super().load_payload(
            payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        self.epoch = max(self.epoch, self.next_epoch)
        self._sync_budget_lambda()
        if self.downstream_finetune_start_epoch is not None:
            self._unwrap(self.model).set_stage_b_phase(self.downstream_finetune_started)
            if self.ema_model is not None:
                self.ema_model.set_stage_b_phase(self.downstream_finetune_started)
                self.ema_model.eval()
            if restore_optimizer and self.downstream_finetune_started:
                self._clear_router_optimizer_state()
        self._rebind_shared_encoder()
        self._assert_encoder_state()

    def _clear_router_optimizer_state(self) -> None:
        """Discard inactive routing moments after the permanent downstream switch."""
        live = self._unwrap(self.model)
        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        # Keep parameter groups intact for DDP, the scheduler and checkpoint loading.
        for name in ("foveation_net", "router_condition_projection"):
            module = getattr(live, name, None)
            if module is None:
                continue
            for param in module.parameters():
                optimizer.state.pop(param, None)
                param.grad = None

    @torch.no_grad()
    def _set_stage_b_phase(self, ema, base_ema_exclude_prefixes) -> None:
        if self.downstream_finetune_start_epoch is None:
            return
        live = self._unwrap(self.model)
        entering = (
            not self.downstream_finetune_started
            and self.epoch >= self.downstream_finetune_start_epoch
        )
        if entering:
            self.accelerator.wait_for_everyone()
            self.optimizer.zero_grad(set_to_none=True)
            source_policy = self.ema_model if self.ema_model is not None else live
            for name in ("foveation_net", "router_condition_projection"):
                source_module = getattr(source_policy, name)
                if source_module is None:
                    continue
                state = source_module.state_dict()
                # Evaluation uses rank 0 EMA; BN buffers may differ across ranks.
                if self.accelerator.num_processes > 1:
                    for value_X in state.values():
                        torch.distributed.broadcast(value_X, src=0)
                getattr(live, name).load_state_dict(state)
            self.downstream_finetune_started = True
            self._clear_router_optimizer_state()
            if self.accelerator.is_main_process:
                source = "EMA" if self.ema_model is not None else "live"
                print(
                    f"[Stage B] epoch={self.epoch}: freeze {source} routing; finetune downstream; "
                    f"LRs={[group['lr'] for group in self.optimizer.param_groups]}"
                )
        live.set_stage_b_phase(self.downstream_finetune_started)
        if self.ema_model is not None:
            self.ema_model.set_stage_b_phase(self.downstream_finetune_started)
            self.ema_model.eval()
        if ema is not None:
            # Frozen branches must not drift through EMA arithmetic either.
            inactive = self._ROUTING_PREFIXES if self.downstream_finetune_started else ("obs_encoders.", "model.")
            ema.exclude_prefixes = tuple(dict.fromkeys((*base_ema_exclude_prefixes, *inactive)))

    def _temperature_at(self, step: int) -> Optional[float]:
        start = self._cfg_get(self.cfg, "training.router_temperature_start", None)
        end = self._cfg_get(self.cfg, "training.router_temperature_end", None)
        total = self._cfg_get(self.cfg, "training.router_temperature_steps", None)
        if total is None:
            total = self._cfg_get(
                self.cfg, "training.router_temperature_anneal_steps", None
            )
        schedule = str(
            self._cfg_get(self.cfg, "training.router_temperature_schedule", "linear")
        )
        if start is None:
            start = self._cfg_get(self.cfg, "training.router_temperature", None)
        if start is None:
            start = self._cfg_get(self.cfg, "policy.router_temperature", None)
        if start is None:
            return None
        start = float(start)
        end = start if end is None else float(end)
        total = max(1, int(total if total is not None else 1))
        progress = min(1.0, max(0.0, float(step) / total))
        if schedule == "cosine":
            progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        elif schedule == "constant":
            progress = 0.0
        elif schedule != "linear":
            raise ValueError(
                "router_temperature_schedule must be 'constant', 'linear' or 'cosine'."
            )
        return start + (end - start) * progress

    def _set_temperature(self) -> None:
        temperature = self._temperature_at(self.global_step)
        if temperature is None:
            return
        live = self._unwrap(self.model)
        live.set_router_temperature(temperature)
        if self.ema_model is not None:
            self.ema_model.set_router_temperature(temperature)

    def _set_target_num_patches(self) -> Optional[float]:
        live = self._unwrap(self.model)
        foveation_net = getattr(live, "foveation_net", None)
        if foveation_net is None:
            return None

        start = self._cfg_get(
            self.cfg,
            "training.target_num_patches_start",
            None,
        )
        if start is None:
            return foveation_net.resolved_target_num_patches

        decay_epochs = int(
            self._cfg_get(
                self.cfg,
                "training.target_num_patches_decay_epochs",
                0,
            )
        )
        if decay_epochs <= 0:
            raise ValueError("target_num_patches_decay_epochs must be positive.")

        final = self._cfg_get(
            self.cfg,
            "policy.foveation_net.target_num_patches",
            None,
        )
        start_patches = foveation_net.resolve_target_num_patches(start)
        final_patches = foveation_net.resolve_target_num_patches(final)
        if start_patches is None or final_patches is None:
            raise ValueError("Scheduled target_num_patches requires both endpoints.")

        target_patches = cosine_schedule_value(
            start_patches, final_patches, self.epoch, decay_epochs)

        # Attributes are not EMA parameters, so keep both policy copies in sync.
        live.set_target_num_patches(target_patches)
        if self.ema_model is not None:
            self.ema_model.set_target_num_patches(target_patches)
        return target_patches

    @staticmethod
    def _move_to_device(value, device):
        if isinstance(value, dict):
            return {
                key: TrainFoveationDPTransformerWorkspace._move_to_device(val, device)
                for key, val in value.items()
            }
        if torch.is_tensor(value):
            return value.to(device)
        return value

    @staticmethod
    def _scalar_losses(losses: Dict[str, object]) -> Dict[str, float]:
        return {
            key: float(value.detach().item())
            for key, value in losses.items()
            if torch.is_tensor(value) and value.ndim == 0 and not key.startswith("_")
        }

    @classmethod
    def _router_parameters(cls, policy):
        live = cls._unwrap(policy)
        modules = [] if live.foveation_net is None else [net.router for net in live.foveation_net.nets.values()]
        modules.append(live.router_condition_projection)
        return tuple(param for module in modules if module is not None
                     for param in module.parameters() if param.requires_grad)

    @classmethod
    def _router_grad_norm(cls, policy) -> float:
        router_params = cls._router_parameters(policy)
        squared_norm_scalar = None
        for param in router_params:
            if param.grad is None:
                continue
            param_squared_norm_scalar = param.grad.detach().float().square().sum()
            squared_norm_scalar = (
                param_squared_norm_scalar
                if squared_norm_scalar is None
                else squared_norm_scalar + param_squared_norm_scalar
            )
        if squared_norm_scalar is None:
            return 0.0
        return float(squared_norm_scalar.sqrt().item())

    @classmethod
    def _router_loss_grad_stats(cls, policy, losses, accelerator) -> Dict[str, float]:
        router_params = cls._router_parameters(policy)
        if not router_params:
            return {
                "grad/router_diffusion_grad_norm": 0.0,
                "grad/router_budget_grad_norm": 0.0,
            }

        def loss_grad_vector_P(loss_scalar: Optional[torch.Tensor]) -> torch.Tensor:
            # Paired diffusion losses are detached evaluation metrics, not objectives.
            if loss_scalar is None or not loss_scalar.requires_grad:
                return torch.zeros(
                    sum(param.numel() for param in router_params),
                    dtype=torch.float32,
                    device=router_params[0].device,
                )
            grads = torch.autograd.grad(
                loss_scalar,
                router_params,
                retain_graph=True,
                allow_unused=True,
            )
            # autograd.grad leaves .grad untouched; reduce the detached vector
            # to match the mean gradient that DDP applies to the optimizer.
            grad_P = torch.cat(
                [
                    (
                        torch.zeros(
                            param.numel(),
                            dtype=torch.float32,
                            device=param.device,
                        )
                        if grad is None
                        else grad.detach().float().reshape(-1)
                    )
                    for param, grad in zip(router_params, grads)
                ]
            )
            return accelerator.reduce(grad_P, reduction="mean")

        diffusion_grad_P = loss_grad_vector_P(losses.get("diffusion_loss"))
        budget_grad_P = loss_grad_vector_P(losses.get("budget_loss"))
        diffusion_grad_norm = float(diffusion_grad_P.norm().item())
        budget_grad_norm = float(budget_grad_P.norm().item())
        stats = {
            "grad/router_diffusion_grad_norm": diffusion_grad_norm,
            "grad/router_budget_grad_norm": budget_grad_norm,
        }
        if "paired_loss" in losses:
            paired_grad_P = loss_grad_vector_P(losses["paired_loss"])
            stats["grad/router_paired_grad_norm"] = float(paired_grad_P.norm().item())
        if diffusion_grad_norm > 0.0 and budget_grad_norm > 0.0:
            cosine_scalar = torch.dot(diffusion_grad_P, budget_grad_P) / (
                diffusion_grad_norm * budget_grad_norm
            )
            stats["grad/router_diffusion_budget_cosine"] = float(
                cosine_scalar.clamp(-1.0, 1.0).item()
            )
        return stats

    @staticmethod
    def _topology_diversity(
        foveation_outputs: Dict[str, Dict[str, torch.Tensor]],
        num_bank_patches: int,
    ) -> float:
        distance_sum_scalar = None
        num_distances = 0
        for foveation_out in foveation_outputs.values():
            indices_B_T_N = foveation_out["patch_bank_indices_B_T_N"]
            valid_B_T_N = foveation_out["token_valid_mask_B_T_N"]
            B, T, _ = indices_B_T_N.shape
            if B < 2:
                continue

            # Reconstruct leaf sets so topology comparison ignores token padding.
            leaf_counts_B_T_K = torch.zeros(
                (B, T, num_bank_patches),
                dtype=torch.long,
                device=indices_B_T_N.device,
            )
            leaf_counts_B_T_K.scatter_add_(
                dim=2,
                index=indices_B_T_N.clamp_min(0),
                src=valid_B_T_N.to(dtype=torch.long),
            )
            leaf_mask_B_T_K = leaf_counts_B_T_K > 0
            leaf_float_B_T_K = leaf_mask_B_T_K.float()
            intersection_T_B_B = torch.einsum(
                "btk,ctk->tbc",
                leaf_float_B_T_K,
                leaf_float_B_T_K,
            )
            leaf_count_B_T = leaf_float_B_T_K.sum(dim=-1)
            leaf_count_T_B = leaf_count_B_T.transpose(0, 1)
            union_T_B_B = (
                leaf_count_T_B.unsqueeze(2)
                + leaf_count_T_B.unsqueeze(1)
                - intersection_T_B_B
            )
            jaccard_distance_T_B_B = 1.0 - (
                intersection_T_B_B / union_T_B_B.clamp_min(1.0)
            )
            pair_mask_B_B = torch.triu(
                torch.ones((B, B), dtype=torch.bool, device=indices_B_T_N.device),
                diagonal=1,
            )
            distances_T_P = jaccard_distance_T_B_B[:, pair_mask_B_B]
            stream_distance_sum_scalar = distances_T_P.sum()
            distance_sum_scalar = (
                stream_distance_sum_scalar
                if distance_sum_scalar is None
                else distance_sum_scalar + stream_distance_sum_scalar
            )
            num_distances += distances_T_P.numel()

        if distance_sum_scalar is None or num_distances == 0:
            return 0.0
        return float((distance_sum_scalar / num_distances).item())

    def _task_sample_diagnostics(self, policy, dataset) -> Dict[str, float]:
        device = self.accelerator.device
        samples_per_task = int(self._cfg_get(self.cfg, "training.topology_samples_per_task", 64))
        diagnostic_seed = int(self._cfg_get(self.cfg, "training.seed", 0))
        sample_resample = bool(self._cfg_get(self.cfg, "training.sample_resample", False))
        # Change input windows/augmentation while keeping model diagnostic randomness fixed.
        sampling_seed = diagnostic_seed + self.epoch if sample_resample else diagnostic_seed
        is_libero = str(self.cfg.task.dataset._target_).endswith(".LiberoFTDataset")
        resolution_enabled = bool(self._cfg_get(self.cfg, "training.sample_resolution_advantage", False))
        task_names = ({value: name for name, value in dataset.task_mapping.items()}
                      if getattr(dataset, "_multi_source", False) else {})
        metrics = {}
        task_distances = []
        task_losses = []
        rng_devices = [device] if device.type == "cuda" else []
        for task_id, batch in dataset.iter_task_diagnostic_batches(samples_per_task, sampling_seed):
            task_name = (LIBERO_TASK_NAMES[task_id] if is_libero
                         and 0 <= task_id < len(LIBERO_TASK_NAMES) else task_names.get(task_id))
            batch_size = batch["task_id"].shape[0]
            if batch_size == 0 or (batch_size < 2 and not resolution_enabled):
                continue
            batch = self._move_to_device(batch, device)
            # Diagnostic sampling must not advance the training or diffusion RNG.
            with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
                torch.random.default_generator.manual_seed(diagnostic_seed + task_id)
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(diagnostic_seed + task_id)
                if resolution_enabled:
                    diagnostics = policy.compute_resolution_diagnostics(batch, seed=diagnostic_seed + task_id)
                    outputs = diagnostics["foveation_outputs"]
                    loss_L = diagnostics["loss_B_L"].mean(dim=0)
                    task_losses.append(loss_L)
                    if task_name is not None:
                        advantage_L = loss_L[:-1] - loss_L[1:]
                        metrics.update({
                            f"sample/resolution_advantage_{level}_to_{level + 1}_{task_name}": value
                            for level, value in enumerate(advantage_L.tolist())
                        })
                    del diagnostics
                else:
                    obs = {
                        key: value_B_T_D[:, :policy.n_obs_steps]
                        for key, value_B_T_D in batch["obs"].items()
                    }
                    outputs = policy.get_foveation_outputs(obs)
                    del obs
                if outputs and batch_size >= 2:
                    distance = self._topology_diversity(outputs, policy.foveation_net.num_bank_patches)
                    task_distances.append(distance)
                    if task_name is not None:
                        metrics[f"sample/topology_div_{task_name}"] = distance
            del outputs, batch
        if task_distances:
            metrics["sample/topology_div"] = float(np.mean(task_distances))
        else:
            print("[topology_div] No task has at least two diagnostic windows; skipping metric")
        if task_losses:
            # Equal task weights, regardless of how many windows each task provides.
            loss_L = torch.stack(task_losses).mean(dim=0)
            advantage_L = loss_L[:-1] - loss_L[1:]
            metrics.update({
                f"sample/resolution_advantage_{level}_to_{level + 1}": advantage
                for level, advantage in enumerate(advantage_L.tolist())
            })
        return metrics

    def _build_logger(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.accelerator.is_main_process:
            return None, nullcontext(None)
        try:
            import wandb

            run_dir = pathlib.Path(self.output_dir)
            logging_cfg = self._cfg_get(self.cfg, "logging", {})
            if not isinstance(logging_cfg, dict):
                logging_cfg = OmegaConf.to_container(logging_cfg, resolve=True)
            logging_cfg["name"] = f"{run_dir.parent.name}-{run_dir.name}"
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                **logging_cfg,
            )
        except Exception as exc:
            print(f"[FoveationWorkspace] wandb disabled: {exc}")
            wandb_run = None
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        return wandb_run, JsonLogger(log_path)

    def _uses_distributed_rollout(self):
        return (
            bool(self._cfg_get(self.cfg, "training.distributed_rollout", False))
            and str(self._cfg_get(self.cfg, "task.env_runner._target_", "")).endswith(".LiberoSocketRunner")
        )

    @contextmanager
    def _rollout_sync_group(self, enabled):
        # Long evaluator queues wait on CPU/Gloo, never an outstanding NCCL call.
        if not enabled or self.accelerator.num_processes == 1:
            yield None
            return
        if not torch.distributed.is_gloo_available():
            raise RuntimeError("Distributed rollout requires a PyTorch build with Gloo support")
        timeout = int(self._cfg_get(self.cfg, "training.ddp_timeout_seconds", 7200))
        if self._cfg_get(self.cfg, "task.env_runner.eval_timeout", None) is not None:
            runner_cfg = self.cfg.task.env_runner
            workers = self.accelerator.num_processes if self._uses_distributed_rollout() else 1
            queue_size = math.ceil(len(runner_cfg.hdf5_fns) / workers)
            timeout += queue_size * (
                runner_cfg.startup_timeout + runner_cfg.eval_timeout + 3 * runner_cfg.shutdown_timeout)
        group = torch.distributed.new_group(backend="gloo", timeout=timedelta(seconds=timeout))
        try:
            if self.accelerator.is_main_process:
                print(f"[Rollout] CPU/Gloo synchronization timeout={timeout}s")
            yield group
        finally:
            torch.distributed.destroy_process_group(group)

    def _gather_rollout_objects(self, value, group):
        if self.accelerator.num_processes == 1:
            return [value]
        gathered = [None] * self.accelerator.num_processes
        torch.distributed.all_gather_object(gathered, value, group=group)
        return gathered

    def _distributed_libero_rollout(self, env_runner, runner_cfg, group):
        accelerator = self.accelerator
        rank, world_size = accelerator.process_index, accelerator.num_processes
        local_error = None
        try:
            if rank != accelerator.local_process_index:
                raise ValueError("Distributed LIBERO rollout currently supports single-node DDP only")
            if self.ema_model is None:
                raise ValueError("Distributed LIBERO rollout requires training.use_ema=true")
            if env_runner is None:
                env_runner = hydra.utils.instantiate(runner_cfg, output_dir=self.output_dir)
            env_runner.configure_distributed(rank, accelerator.local_process_index, world_size)
            env_runner.set_context(self.dataset_metadata, self.epoch, self.global_step)
            if accelerator.is_main_process:
                env_runner.prepare_run()
        except BaseException as error:
            local_error = f"rank {rank}: {type(error).__name__}: {error}"
        readiness = self._gather_rollout_objects(
            dict(error=local_error, output_dir=os.path.realpath(self.output_dir)), group)
        errors = [item["error"] for item in readiness if item["error"] is not None]
        if errors:
            raise RuntimeError("LIBERO rollout setup failed: " + "; ".join(errors))
        if len({item["output_dir"] for item in readiness}) != 1:
            raise ValueError("Distributed LIBERO rollout requires a shared output directory")

        policy = self.ema_model
        # EMA buffers (e.g. BatchNorm) can differ despite synchronized DDP gradients.
        # Finish all model collectives before ranks start unequal evaluator queues.
        if world_size > 1:
            with torch.no_grad():
                for value_X in (*policy.parameters(), *policy.buffers()):
                    if value_X.numel() == 0:
                        continue
                    synced_X = value_X.detach().contiguous()
                    torch.distributed.broadcast(
                        synced_X, src=0, group=None if synced_X.is_cuda else group)
                    value_X.copy_(synced_X)
            if policy.device.type == "cuda":
                torch.cuda.synchronize(policy.device)
        task_indices = list(range(rank, len(env_runner.tasks), world_size))
        print(f"[Rollout] rank={rank}, policy_device={policy.device}, tasks={task_indices}")
        payloads, local_error = [], None
        try:
            payloads = env_runner.run_tasks(policy, task_indices)
        except BaseException as error:
            local_error = f"rank {rank}: {type(error).__name__}: {error}"
        gathered = self._gather_rollout_objects(dict(error=local_error, tasks=payloads), group)
        errors = [item["error"] for item in gathered if item["error"] is not None]
        if errors:
            raise RuntimeError("LIBERO rollout failed: " + "; ".join(errors))

        log, status = {}, [None]
        if accelerator.is_main_process:
            try:
                log = env_runner.summarize_results([
                    task for item in gathered for task in item["tasks"]])
            except BaseException as error:
                status[0] = f"{type(error).__name__}: {error}"
        # No rank enters validation/training while rank zero assembles media/summary.
        if world_size > 1:
            torch.distributed.broadcast_object_list(status, src=0, group=group)
        if status[0] is not None:
            raise RuntimeError("LIBERO rollout aggregation failed: " + status[0])
        return env_runner, log

    def _export_foveation_artifact(self):
        path = self._cfg_get(self.cfg, "foveation_export_path", None)
        if path is None:
            return
        if self.ema_model is None:
            raise ValueError("DP routing export requires training.use_ema=true")
        is_tcl = str(self.cfg.task.dataset._target_).endswith(".TCLFoveationDataset")
        if is_tcl:
            # Keep legacy identity fields; new TCL consumers use the saved state_normalization contract.
            state_mean = np.zeros(6, dtype=np.float32)
            state_std = np.ones(6, dtype=np.float32)
            task_mapping = self.dataset_metadata.get("task_mapping", {})
        else:
            # LiberoFTDataset normalizes states before the policy's identity normalizer.
            statistics = self.dataset_metadata["statistics"]["obs.ee_states"]
            state_mean = statistics["mean"][:6]
            state_std = statistics["std"][:6]
            task_mapping = None
        artifact = make_foveation_artifact(
            self.ema_model, state_mean, state_std,
            task_mapping=task_mapping,
            split_metadata=self.dataset_metadata,
        )
        path = pathlib.Path(path).expanduser()
        if not path.is_absolute():
            path = pathlib.Path(self.output_dir) / path
        save_foveation_artifact(path, artifact)

    def _validation(self, policy, val_dataloader, max_steps: Optional[int]):
        sums: Dict[str, torch.Tensor] = {}
        view_sums, view_counts = {}, {}
        count = torch.tensor(0.0, device=self.accelerator.device)
        policy.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if max_steps is not None and batch_idx >= int(max_steps):
                    break
                losses = policy.compute_loss(batch)
                self._globalize_elu_budget(policy, losses)
                for name, sum_V in losses.get("_view_metric_sums", {}).items():
                    view_sums[name] = view_sums.get(name, 0) + sum_V.detach().double()
                    view_counts[name] = view_counts.get(name, 0) + losses["_view_metric_counts"][name].detach().double()
                for key, value in losses.items():
                    if key.startswith("_") or not torch.is_tensor(value) or value.ndim != 0:
                        continue
                    sums.setdefault(key, torch.zeros_like(value))
                    sums[key] += value.detach()
                count += 1.0
        packed = list(sums.values()) + [count]
        if not packed:
            return {}
        # Older Accelerate reduces a vector to one scalar on a single process;
        # reduce scalars independently so the metric order remains intact.
        reduced = torch.stack(
            [self.accelerator.reduce(value, reduction="sum") for value in packed]
        )
        denominator = reduced[-1].clamp_min(1.0)
        keys = list(sums.keys())
        metrics = {f"val/{key}": (reduced[idx] / denominator).item() for idx, key in enumerate(keys)}
        view_metrics = self._view_metrics(policy, {"_view_metric_sums": view_sums, "_view_metric_counts": view_counts})
        metrics.update({f"val/{key}": value for key, value in view_metrics.items()})
        return metrics

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        accelerator = self.accelerator
        device = accelerator.device
        resume_payload = None
        weights_only = cfg.get("pretrained_weights_only", False)

        if bool(self._cfg_get(cfg, "training.resume", False)):
            path = self.get_checkpoint_path()
            if path.is_file():
                # The scheduler is created below after the loader length is known.
                resume_payload = self.load_checkpoint(
                    path=path,
                    exclude_keys=("lr_scheduler",),
                )

        dataset_kwargs = {}
        if str(cfg.task.dataset._target_).endswith((".LiberoFTDataset", ".TCLFoveationDataset")):
            requires_metadata = (resume_payload is not None or
                                 (self._cfg_get(cfg, "pretrained_ckpt", None) and not weights_only))
            if requires_metadata and self.dataset_metadata is None:
                raise ValueError("Checkpoint lacks dataset_metadata; cannot recover splits and statistics")
            if (str(cfg.task.dataset._target_).endswith(".LiberoFTDataset")
                    and self._cfg_get(cfg, "task.dataset.split_mode", "legacy") == "two_by_two"
                    and self._cfg_get(cfg, "task.dataset.split_stage", "a") == "b"
                    and resume_payload is None):
                metadata = copy.deepcopy(self.dataset_metadata)
                if (metadata is None or metadata.get("split_mode") != "two_by_two"
                        or metadata.get("split_stage") != "a"):
                    raise ValueError("New two_by_two Stage B requires a two_by_two Stage A checkpoint")
                # Switch only episode membership; the frozen policy keeps Stage A scaling.
                metadata["split_stage"] = "b"
                metadata["train"] = metadata["partitions"]["b_train"]
                metadata["val"] = metadata["partitions"]["a_val"]
                self.dataset_metadata = metadata
            data_seed_from_ckpt = bool(self._cfg_get(cfg, "data_seed_from_ckpt", False))
            if data_seed_from_ckpt:
                if self.dataset_metadata is None or "seed" not in self.dataset_metadata:
                    raise ValueError("data_seed_from_ckpt requires a checkpoint with dataset_metadata.seed")
                # Resume metadata takes precedence over the Stage A initialization checkpoint.
                # Persist the effective data seed without changing training randomness.
                restored_keys = ["seed"]
                if str(cfg.task.dataset._target_).endswith(".TCLFoveationDataset"):
                    restored_keys += [key for key in ("val_ratio", "max_train_episodes")
                                      if key in self.dataset_metadata]
                for key in restored_keys:
                    cfg.task.dataset[key] = self.dataset_metadata[key]
                    self.cfg.task.dataset[key] = cfg.task.dataset[key]
            if accelerator.is_main_process:
                seed_source = ("resume checkpoint" if resume_payload is not None else "pretrained checkpoint")
                if not data_seed_from_ckpt:
                    seed_source = "config"
                print(f"[Dataset] training.seed={cfg.training.seed}, "
                      f"dataset.seed={cfg.task.dataset.seed}, data seed source={seed_source}")
            dataset_kwargs["dataset_metadata"] = self.dataset_metadata
        if accelerator.is_main_process and (
            self._cfg_get(cfg, "reuse_pretrained_dataset_config", False)
            or self._cfg_get(cfg, "pretrained_ckpt_mode", None) is not None
            or self._cfg_get(cfg, "data_seed_from_ckpt", False)
            or weights_only
            or self._cfg_get(cfg, "task.dataset.norm.action", None) == "quantile"
        ):
            # Hydra saved before preparation; persist effective data and sampling settings.
            config_path = pathlib.Path(self.output_dir) / ".hydra" / "config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(self.cfg, config_path, resolve=True)
        asset_dir = self._cfg_get(cfg, 'task.dataset.asset_dir', None)
        if asset_dir is not None:
            resolved_asset = hydra.utils.to_absolute_path(str(pathlib.Path(asset_dir).expanduser()))
            cfg.task.dataset.asset_dir = self.cfg.task.dataset.asset_dir = resolved_asset
            if 'asset_dir' in cfg:
                cfg.asset_dir = self.cfg.asset_dir = resolved_asset
            if accelerator.is_main_process:
                config_path = pathlib.Path(self.output_dir) / '.hydra' / 'config.yaml'
                config_path.parent.mkdir(parents=True, exist_ok=True)
                OmegaConf.save(self.cfg, config_path, resolve=True)
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset, **dataset_kwargs)
        if hasattr(dataset, "get_dataset_metadata"):
            self.dataset_metadata = dataset.get_dataset_metadata()
            contract = self.dataset_metadata.get("contract")
            if contract is None:
                raise ValueError(
                    "TCL dataset metadata is missing the required contract. "
                    "The dataset code and metadata are out of sync for the DP training path; "
                    "for a Pi0.5 asset, rerun scripts/pi05/prepare_tcl_data.sh to regenerate "
                    "dataset_metadata.json."
                )
            if contract.get("n_obs_steps") != self.model.n_obs_steps:
                raise ValueError("Dataset contract n_obs_steps must match policy.n_obs_steps")
            if accelerator.is_main_process:
                metadata_path = pathlib.Path(self.output_dir) / "dataset_metadata.json"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(json.dumps(self.dataset_metadata, indent=2))
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError("The foveation workspace requires a BaseImageDataset.")
        if self.model.temporal_adaptive:
            from diffusion_policy.dataset.libero_dataset import LiberoFTDataset
            from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset, MergedPushTImageDataset

            is_tcl = str(cfg.task.dataset._target_).endswith(".TCLFoveationDataset")
            if not is_tcl and not isinstance(dataset, (PushTImageDataset, MergedPushTImageDataset, LiberoFTDataset)):
                raise ValueError("Temporal routing requires a PushT, LIBERO or TCL foveation dataset.")
            if dataset.n_obs_steps != self.model.n_obs_steps:
                raise ValueError("Temporal pairing requires dataset and policy n_obs_steps to match.")
        loader_cfg = self._cfg_get(cfg, "dataloader", {})
        val_loader_cfg = self._cfg_get(cfg, "val_dataloader", {})
        if not isinstance(loader_cfg, dict):
            loader_cfg = OmegaConf.to_container(loader_cfg, resolve=True)
        if not isinstance(val_loader_cfg, dict):
            val_loader_cfg = OmegaConf.to_container(val_loader_cfg, resolve=True)
        train_loader = DataLoader(dataset, **loader_cfg)
        val_dataset = dataset.get_validation_dataset()
        val_loader = DataLoader(
            val_dataset,
            drop_last=False,
            **val_loader_cfg,
        )
        if self.model.temporal_adaptive:
            # Initialize before workers start, including resumes at the phase boundary.
            downstream = self.downstream_finetune_started or (
                self.downstream_finetune_start_epoch is not None
                and self.epoch >= self.downstream_finetune_start_epoch
            )
            dataset.set_temporal_pairing(not downstream)
        # Weights-only finetuning adopts new scaling; frozen action paths keep checkpoint scaling.
        if resume_payload is None and (weights_only or not self._cfg_get(cfg, "pretrained_ckpt", None)):
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)
            if self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)

        num_epochs = int(self._cfg_get(cfg, "training.num_epochs", 1))
        grad_accum = int(self._cfg_get(cfg, "training.gradient_accumulate_every", 1))
        self.lr_scheduler = get_scheduler(
            self._cfg_get(cfg, "training.lr_scheduler", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=int(self._cfg_get(cfg, "training.lr_warmup_steps", 0)),
            num_training_steps=max(1, (len(train_loader) * num_epochs) // grad_accum),
            last_epoch=self.global_step - 1,
        )
        if resume_payload is not None:
            scheduler_state = resume_payload["state_dicts"].get("lr_scheduler")
            if scheduler_state is not None:
                self.lr_scheduler.load_state_dict(scheduler_state)
        self.model, self.optimizer, train_loader, val_loader, self.lr_scheduler = accelerator.prepare(
            self.model,
            self.optimizer,
            train_loader,
            val_loader,
            self.lr_scheduler,
        )
        self._rebind_shared_encoder()
        self._assert_encoder_state()

        ema = None
        base_ema_exclude_prefixes = ()
        if self.ema_model is not None:
            self.ema_model.to(device)
            ema_cfg = self._cfg_get(cfg, "ema", None)
            if ema_cfg is None:
                ema = EMAModel(model=self.ema_model)
            else:
                ema = hydra.utils.instantiate(ema_cfg, model=self.ema_model)
            # Shared frozen encoders already track live state through object identity;
            # trainable encoders own an independent EMA copy and must be updated.
            if self._shared_encoder_keys and hasattr(ema, "exclude_prefixes"):
                ema.exclude_prefixes = tuple(getattr(ema, "exclude_prefixes", ())) + self._shared_encoder_prefixes
            ema.optimization_step = int(self.global_step)
            base_ema_exclude_prefixes = tuple(ema.exclude_prefixes)

        wandb_run, log_context = self._build_logger()
        env_runner_cfg = self._cfg_get(cfg, "task.env_runner", None)
        env_runner = None
        # Instantiate lazily inside the synchronized rollout block so evaluator
        # configuration/startup failures are reported to every training rank.
        topk_manager = None
        if accelerator.is_main_process and self._cfg_get(cfg, "checkpoint.topk", None) is not None:
            topk_cfg = self._cfg_get(cfg, "checkpoint.topk", {})
            if not isinstance(topk_cfg, dict):
                topk_cfg = OmegaConf.to_container(topk_cfg, resolve=True)
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"),
                **topk_cfg,
            )
        max_train_steps = self._cfg_get(cfg, "training.max_train_steps", None)
        max_val_steps = self._cfg_get(cfg, "training.max_val_steps", None)
        router_grad_norm_every = self._cfg_get(
            cfg,
            "training.router_grad_norm_every",
            -1,
        )
        router_grad_norm_every = (
            -1 if router_grad_norm_every is None else int(router_grad_norm_every)
        )
        if router_grad_norm_every == 0 or router_grad_norm_every < -1:
            raise ValueError("router_grad_norm_every must be positive, null, or -1.")
        train_sampling_batch = None

        rollout_enabled = env_runner_cfg is not None and int(self._cfg_get(cfg, "training.rollout_every", -1)) != -1
        with self._rollout_sync_group(rollout_enabled) as rollout_group, log_context as json_logger:
            for _ in range(self.epoch, num_epochs):
                self._set_stage_b_phase(ema, base_ema_exclude_prefixes)
                if self._unwrap(self.model).temporal_adaptive:
                    dataset.set_temporal_pairing(not self.downstream_finetune_started)
                target_num_patches = self._set_target_num_patches()
                self.model.train()
                if bool(self._cfg_get(cfg, "policy.en_freeze_obs_encoder", False)):
                    self._unwrap(self.model).freeze_obs_encoder()
                epoch_losses = []
                step_log = {"epoch": self.epoch, "global_step": self.global_step}
                wandb_vis_log = {}
                budget_stats_V_2 = torch.zeros((len(self._unwrap(self.model).rgb_keys), 2), device=device, dtype=torch.float64)
                sample_count_scalar = torch.zeros((), device=device, dtype=torch.float64)
                # Each epoch starts a fresh window, including after max_train_steps truncation.
                accelerator.step = 0
                for batch_idx, batch in enumerate(
                    tqdm.tqdm(
                        train_loader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        disable=not accelerator.is_main_process,
                    )
                ):
                    if train_sampling_batch is None:
                        # Rollout/visualization retains only the original single window.
                        train_sampling_batch = {"obs": batch["obs"], "action": batch["action"]}
                    router_grad_stats = None
                    budget_metrics = {}
                    with accelerator.accumulate(self.model):
                        self._set_temperature()
                        losses = self.model(batch)
                        self._globalize_elu_budget(self.model, losses)
                        raw_loss = losses["total_loss"]
                        has_budget_stats = "_budget_expected_sum" in losses
                        backward_loss_scalar = raw_loss
                        if has_budget_stats:
                            budget_stats_V_2[:, 0] += losses["_budget_expected_sum"]
                            budget_stats_V_2[:, 1] += losses["_budget_observation_count"]
                            sample_count_scalar += batch["action"].shape[0]
                            # Each dataset sample contributes one window or one temporal
                            # pair, regardless of V. Keep the summed AMP loss near its mean.
                            batch_weight = batch["action"].shape[0] / loader_cfg["batch_size"]
                            backward_loss_scalar = raw_loss * batch_weight
                        if (
                            accelerator.sync_gradients
                            and not self.downstream_finetune_started
                            and router_grad_norm_every > 0
                            and self.global_step % router_grad_norm_every == 0
                        ):
                            router_grad_stats = self._router_loss_grad_stats(
                                self.model,
                                losses,
                                accelerator,
                            )
                        accelerator.backward(backward_loss_scalar)
                        if accelerator.sync_gradients:
                            if router_grad_stats is not None or has_budget_stats:
                                accelerator.unscale_gradients(self.optimizer)
                            if has_budget_stats:
                                global_budget_stats_V_2 = self._reduce_sum(budget_stats_V_2)
                                global_sample_count_scalar = accelerator.reduce(sample_count_scalar, reduction="sum")
                                # B counts dataset samples (pairs for temporal), not cameras.
                                # Undo DDP/accumulation averaging, including a short final step.
                                grad_scale_scalar = (
                                    grad_accum * accelerator.num_processes * loader_cfg["batch_size"]
                                    / global_sample_count_scalar
                                )
                                for param in self.model.parameters():
                                    if param.grad is not None:
                                        param.grad.mul_(grad_scale_scalar.to(param.grad.dtype))
                            if router_grad_stats is not None:
                                router_grad_stats["grad/router_grad_norm"] = (
                                    self._router_grad_norm(self.model)
                                )
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            if not accelerator.optimizer_step_was_skipped:
                                if ema is not None:
                                    ema.step(accelerator.unwrap_model(self.model))
                                if has_budget_stats:
                                    budget_metrics = self._update_budget_lambda(global_budget_stats_V_2)
                            budget_stats_V_2.zero_()
                            sample_count_scalar.zero_()
                    loss_value = float(raw_loss.detach().item())
                    epoch_losses.append(loss_value)
                    train_metrics = self._scalar_losses(losses)
                    train_metrics.update(self._view_metrics(self.model, losses))
                    if "temporal_loss" in losses:
                        # Log this forward's unweighted global pair mean, including invalid
                        # pairs in the denominator. Reduce scalars for older Accelerate.
                        pair_count_scalar = losses["temporal_loss"].new_tensor(
                            batch["action"].shape[0], dtype=torch.float64
                        )
                        temporal_sum_scalar = accelerator.reduce(
                            losses["temporal_loss"].detach().double() * pair_count_scalar,
                            reduction="sum",
                        )
                        global_pair_count_scalar = accelerator.reduce(pair_count_scalar, reduction="sum")
                        train_metrics["temporal_loss"] = (temporal_sum_scalar / global_pair_count_scalar).item()
                        # Match rate excludes placeholder windows. Sum counts across ranks
                        # before dividing, since each rank can have a different valid count.
                        global_matched_count_scalar = accelerator.reduce(
                            losses["_temporal_matched_pair_count"].double(), reduction="sum"
                        )
                        global_valid_count_scalar = accelerator.reduce(
                            losses["_temporal_valid_pair_count"].double(), reduction="sum"
                        )
                        train_metrics["temporal_match_fraction"] = (
                            global_matched_count_scalar / global_valid_count_scalar.clamp_min(1.0)
                        ).item()
                    step_log = {
                        "train/lr": self.lr_scheduler.get_last_lr()[0],
                        "train/downstream_finetune_active": int(self.downstream_finetune_started),
                        "epoch": self.epoch,
                        "global_step": self.global_step,
                        **{
                            f"train/{key}": value
                            for key, value in train_metrics.items()
                        },
                    }
                    if target_num_patches is not None:
                        step_log["train/target_num_patches"] = target_num_patches
                    if router_grad_stats is not None:
                        step_log.update(router_grad_stats)
                    step_log.update(budget_metrics)
                    if accelerator.is_main_process and wandb_run is not None:
                        wandb_run.log(step_log, step=self.global_step)
                    if accelerator.is_main_process and json_logger is not None:
                        json_logger.log(step_log)
                    self.global_step += 1
                    if max_train_steps is not None and batch_idx + 1 >= int(max_train_steps):
                        break

                # A truncated unsynchronized window must not leak into the next epoch.
                # Clear the model directly: AcceleratedOptimizer skips zero_grad in no_sync.
                self.model.zero_grad(set_to_none=True)
                accelerator.wait_for_everyone()
                step_log = {
                    "train/total_loss": (
                        float(np.mean(epoch_losses)) if epoch_losses else 0.0
                    ),
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                }
                rollout_every = int(
                    self._cfg_get(cfg, "training.rollout_every", -1)
                )
                rollout_due = (
                    env_runner_cfg is not None
                    and rollout_every != -1
                    and self.epoch % rollout_every == 0
                )
                if rollout_due:
                    if device.type == "cuda":
                        # Drain this rank's training work before entering the CPU wait.
                        torch.cuda.synchronize(device)
                    if self._uses_distributed_rollout():
                        env_runner, rollout_log = self._distributed_libero_rollout(
                            env_runner, env_runner_cfg, rollout_group)
                        step_log.update(rollout_log)
                    else:
                        rollout_error = None
                        rollout_status = [None]
                        if accelerator.is_main_process:
                            try:
                                if env_runner is None:
                                    runner_kwargs = {}
                                    if str(env_runner_cfg._target_).endswith(".TCLFoveationVideoRunner"):
                                        runner_kwargs["dataset"] = val_dataset
                                        runner_kwargs["log_wandb"] = (wandb_run is not None
                                            and self._cfg_get(cfg, "logging.mode", "online") != "disabled")
                                    env_runner = hydra.utils.instantiate(
                                        env_runner_cfg, output_dir=self.output_dir, **runner_kwargs)
                                if hasattr(env_runner, "set_context"):
                                    if (str(env_runner_cfg._target_).endswith(".LiberoSocketRunner")
                                            and self.ema_model is None):
                                        raise ValueError("LIBERO training evaluation requires training.use_ema=true")
                                    env_runner.set_context(self.dataset_metadata, self.epoch, self.global_step)
                                policy_rollout = self.ema_model if self.ema_model is not None else self._unwrap(self.model)
                                policy_rollout.eval()
                                step_log.update(env_runner.run(policy_rollout))
                            except BaseException as error:
                                rollout_error = error
                                rollout_status[0] = f"{type(error).__name__}: {error}"
                        # Gloo keeps this long wait off peer GPUs until rank-0 cleanup finishes.
                        if accelerator.num_processes > 1:
                            torch.distributed.broadcast_object_list(rollout_status, src=0, group=rollout_group)
                        if rollout_status[0] is not None:
                            raise RuntimeError(f"Rollout failed on rank 0: {rollout_status[0]}") from rollout_error
                if self.epoch % int(self._cfg_get(cfg, "training.val_every", 1)) == 0:
                    self._set_temperature()
                    policy_val = self.ema_model if self.ema_model is not None else self._unwrap(self.model)
                    step_log.update(self._validation(policy_val, val_loader, max_val_steps))

                if (
                    accelerator.is_main_process
                    and train_sampling_batch is not None
                    and self.epoch % int(self._cfg_get(cfg, "training.sample_every", 1)) == 0
                ):
                    policy_sample = self.ema_model if self.ema_model is not None else self._unwrap(self.model)
                    policy_sample.eval()
                    sample_batch = self._move_to_device(train_sampling_batch["obs"], device)
                    with torch.no_grad():
                        result = policy_sample.predict_action(
                            sample_batch,
                            return_foveation_outputs=True,
                        )
                    foveation_outputs = result.get("_foveation_outputs", {})
                    if foveation_outputs and policy_sample.foveation_net is not None:
                        if hasattr(dataset, "iter_task_diagnostic_batches"):
                            step_log.update(self._task_sample_diagnostics(policy_sample, dataset))
                        else:
                            if bool(self._cfg_get(cfg, "training.sample_resolution_advantage", False)):
                                raise ValueError("Resolution diagnostics require iter_task_diagnostic_batches().")
                            step_log["sample/topology_div"] = self._topology_diversity(
                                foveation_outputs, policy_sample.foveation_net.num_bank_patches)
                    if wandb_run is not None:
                        wandb_vis_log = make_foveation_vis_log(
                            policy_sample,
                            foveation_outputs,
                            vis_rot_image=(policy_sample.render_image_orientation == "raw"
                                           if self._is_libero else self.vis_rot_image),
                        )
                    target_action = train_sampling_batch["action"].to(device)
                    metric_action = result["action_pred"]
                    if policy_sample.pred_action_steps_only:
                        start = policy_sample.n_obs_steps - 1
                        target_action = target_action[:, start : start + policy_sample.n_action_steps]
                        metric_action = result["action"]
                    step_log["sample/train_action_mse_error"] = F.mse_loss(
                        metric_action, target_action
                    ).item()

                self.next_epoch = self.epoch + 1
                if self.epoch % int(self._cfg_get(cfg, "training.checkpoint_every", 1)) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and bool(
                        self._cfg_get(cfg, "checkpoint.save_last_ckpt", True)
                    ):
                        self.save_checkpoint()
                        self._export_foveation_artifact()
                    if (
                        accelerator.is_main_process
                        and accelerator.num_processes == 1
                        and bool(self._cfg_get(cfg, "checkpoint.save_last_snapshot", False))
                    ):
                        self.save_snapshot()
                    if accelerator.is_main_process and topk_manager is not None:
                        metric_dict = {key.replace("/", "_"): value for key, value in step_log.items()}
                        monitor_key = self._cfg_get(cfg, "checkpoint.topk.monitor_key", None)
                        if monitor_key in metric_dict:
                            topk_path = topk_manager.get_ckpt_path(metric_dict)
                            if topk_path is not None:
                                self.save_checkpoint(path=topk_path)

                self._unwrap(self.model).train()
                if bool(self._cfg_get(cfg, "policy.en_freeze_obs_encoder", False)):
                    self._unwrap(self.model).freeze_obs_encoder()
                self._assert_encoder_state()
                if self.ema_model is not None:
                    self.ema_model.eval()
                if accelerator.is_main_process and wandb_run is not None:
                    wandb_run.log(
                        {**step_log, **wandb_vis_log},
                        step=self.global_step,
                    )
                if accelerator.is_main_process and json_logger is not None:
                    json_logger.log(step_log)
                self.epoch += 1
                accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            self._export_foveation_artifact()
        accelerator.wait_for_everyone()


# Aliases make Hydra target names less brittle while retaining one implementation.
TrainFoveationWorkspace = TrainFoveationDPTransformerWorkspace
TrainFoveationDpTransformerWorkspace = TrainFoveationDPTransformerWorkspace
TrainDiffusionTransformerFoveationWorkspace = TrainFoveationDPTransformerWorkspace


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    cfg = TrainFoveationDPTransformerWorkspace.prepare_config(cfg)
    OmegaConf.resolve(cfg)
    TrainFoveationDPTransformerWorkspace(cfg).run()


if __name__ == "__main__":
    main()
