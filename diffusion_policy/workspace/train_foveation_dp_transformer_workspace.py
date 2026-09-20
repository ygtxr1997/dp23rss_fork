"""Training workspace for the foveation diffusion policy."""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import math
import os
import pathlib
import random
from contextlib import nullcontext
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
from diffusion_policy.common.json_logger import JsonLogger
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
        "global_step", "epoch", "next_epoch", "downstream_finetune_started", "budget_lambda"
    )
    _GLOBAL_MARKER = "foveation_net.global_encoder."
    _ROUTING_PREFIXES = ("foveation_net.", "router_condition_projection.")

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg, output_dir=output_dir)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[process_kwargs, ddp_kwargs],
            gradient_accumulation_steps=int(self._cfg_get(cfg, "training.gradient_accumulate_every", 1)),
        )

        seed = int(self._cfg_get(cfg, "training.seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.vis_rot_image = bool(self._cfg_get(cfg, "vis_rot_image", False))

        self.model: FoveationDPTransformerPolicy = hydra.utils.instantiate(cfg.policy)
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
                or foveation_net.global_encoder is None
                or foveation_net.fix_split_mode is not None
                or foveation_net.freeze_global_encoder
                or foveation_net.freeze_router
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
                or foveation_net.random_topology_prob != 0.0
            ):
                raise ValueError(
                    "Paired routing requires router_only=true and "
                    "random_topology_prob=0."
                )

        encoder = self._global_encoder(self.model)
        self._share_global_encoder = encoder is not None and not any(
            param.requires_grad for param in encoder.parameters()
        )
        self.ema_model: Optional[FoveationDPTransformerPolicy] = None
        if bool(self._cfg_get(cfg, "training.use_ema", False)):
            # Only frozen encoders are shared; trainable encoders need an EMA copy.
            memo = {id(encoder): encoder} if self._share_global_encoder else {}
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
        self.budget_lambda = float(self._cfg_get(cfg, "training.budget_lambda_init", 0.0))
        self.budget_lambda_lr = float(self._cfg_get(cfg, "training.budget_lambda_lr", 0.01))
        if self.budget_lambda < 0.0 or self.budget_lambda_lr < 0.0:
            raise ValueError("budget_lambda_init and budget_lambda_lr must be non-negative.")
        self._sync_budget_lambda()

    def _sync_budget_lambda(self) -> None:
        # A plain attribute avoids changing old model state dicts or averaging dual state.
        for policy in (self.model, self.ema_model):
            if policy is not None and self._unwrap(policy).foveation_net is not None:
                self._unwrap(policy).foveation_net.budget_lambda = self.budget_lambda

    def _update_budget_lambda(self, budget_stats_2: torch.Tensor) -> Dict[str, float]:
        """Update the shared price from global [expected-count sum, observation count]."""
        net = self._unwrap(self.model).foveation_net
        expected_mean = (budget_stats_2[0] / budget_stats_2[1]).item()
        metrics = {
            "train/budget_lambda": self.budget_lambda,
            "train/expected_num_patches_mean": expected_mean,
        }
        normalized_gap = (expected_mean - net.resolved_target_num_patches) / (
            net.max_num_patches - net.min_num_patches
        )
        # LR is in relative-price units; budget_loss_weight scales the router cost only.
        self.budget_lambda = max(0.0, self.budget_lambda + self.budget_lambda_lr * normalized_gap)
        self._sync_budget_lambda()
        return metrics

    def _globalize_elu_budget(self, policy, losses) -> None:
        """Replace local ELU cost with the current forward's global-mean cost."""
        if "_budget_normalized_mean" not in losses:
            return
        net = self._unwrap(policy).foveation_net
        local_mean_scalar = losses["_budget_normalized_mean"]
        # Reduce detached sums/counts, not nonlinear per-rank losses. Keep collectives
        # in the workspace: rollout and video routing may run on the main rank only.
        global_sum_scalar = self.accelerator.reduce(
            losses["_budget_expected_sum"].double(), reduction="sum"
        )
        global_count_scalar = self.accelerator.reduce(
            losses["_budget_observation_count"].double(), reduction="sum"
        )
        budget_range = net.max_num_patches - net.min_num_patches
        global_mean_scalar = (
            (global_sum_scalar / global_count_scalar - net.min_num_patches) / budget_range
        ).to(local_mean_scalar.dtype)
        # Forward uses the global mean; backward retains the local mean gradient.
        # Existing batch weighting/DDP normalization supplies observation weights.
        mean_proxy_scalar = global_mean_scalar + (local_mean_scalar - local_mean_scalar.detach())
        normalized_target = (net.resolved_target_num_patches - net.min_num_patches) / budget_range
        budget_loss_scalar = net.upper_bound_budget_loss(
            mean_proxy_scalar, normalized_target, net.budget_loss_type,
            net.budget_base_cost_weight, net.budget_lambda, net.budget_elu_tau,
        ) * net.budget_loss_weight
        # With accumulation > 1, this averages per-forward nonlinear objectives;
        # it is not ELU of the entire optimizer window's mean. Lambda still uses
        # that window's mean. Current Stage B uses accumulation=1.
        losses["total_loss"] = losses["total_loss"] - losses["budget_loss"] + budget_loss_scalar
        losses["budget_loss"] = budget_loss_scalar

    def _load_pretrained_weights(self, pretrained_ckpt: str) -> None:
        """Load EMA weights, completing the shared encoder from the live state."""
        payload = torch.load(pretrained_ckpt, map_location="cpu", pickle_module=dill)
        state = payload["state_dicts"]["ema_model"].copy()
        live_state = payload["state_dicts"]["model"]
        # Stage A stores its shared, frozen encoder only in the live model.
        # All other weights (including the untrained router) retain their EMA source.
        for name in live_state:
            if name.startswith(self._GLOBAL_MARKER):
                state.setdefault(name, live_state[name])
        self.model.load_state_dict(state, strict=True)
        if self.accelerator.is_main_process:
            print(f"[FoveationWorkspace] Loaded pretrained EMA weights: {pretrained_ckpt}")

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
    def _global_encoder(cls, policy):
        policy = cls._unwrap(policy)
        foveation_net = getattr(policy, "foveation_net", None)
        return (
            None
            if foveation_net is None
            else getattr(foveation_net, "global_encoder", None)
        )

    def _rebind_shared_encoder(self) -> None:
        if not self._share_global_encoder or self.ema_model is None:
            return
        live = self._unwrap(self.model)
        ema = self._unwrap(self.ema_model)
        live_net = getattr(live, "foveation_net", None)
        ema_net = getattr(ema, "foveation_net", None)
        if live_net is None or ema_net is None:
            return
        ema_net.global_encoder = live_net.global_encoder

    def _assert_encoder_state(self) -> None:
        live_encoder = self._global_encoder(self.model)
        if live_encoder is None:
            return
        if self.ema_model is not None:
            ema_encoder = self._global_encoder(self.ema_model)
            if self._share_global_encoder and ema_encoder is not live_encoder:
                raise RuntimeError("Frozen global encoders must be shared with EMA.")
            if not self._share_global_encoder and ema_encoder is live_encoder:
                raise RuntimeError("Trainable global encoders need a separate EMA copy.")
        if self._share_global_encoder:
            if any(param.requires_grad for param in live_encoder.parameters()):
                raise RuntimeError("A shared global encoder must remain frozen.")
            if live_encoder.training:
                raise RuntimeError("A frozen global encoder must remain in eval mode.")

    def _assert_encoder_optimizer(self) -> None:
        encoder = self._global_encoder(self.model)
        if encoder is None:
            return
        encoder_ids = {id(param) for param in encoder.parameters()}
        trainable_encoder_ids = {
            id(param) for param in encoder.parameters() if param.requires_grad
        }
        optimizer_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group["params"]
        }
        if (encoder_ids & optimizer_ids) != trainable_encoder_ids:
            raise RuntimeError(
                "Optimizer must contain exactly the trainable global encoder parameters."
            )

    @classmethod
    def _is_global_state_key(cls, key: str) -> bool:
        return cls._GLOBAL_MARKER in key or key.startswith("global_encoder.")

    def _state_dict_of(self, key, value, use_thread=True):
        state = super()._state_dict_of(key, value, use_thread=use_thread)
        if (
            self._share_global_encoder
            and key == "ema_model"
            and isinstance(state, dict)
        ):
            # The live state already carries the sole shared encoder copy.
            state = {
                name: tensor
                for name, tensor in state.items()
                if not self._is_global_state_key(name)
            }
        return state

    def _load_state_dict_into(self, key, obj, state, **kwargs):
        if (
            self._share_global_encoder
            and key == "ema_model"
            and isinstance(state, dict)
        ):
            incompatible = obj.load_state_dict(state, strict=False)
            missing = [
                name
                for name in incompatible.missing_keys
                if not self._is_global_state_key(name)
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
        if "optimizer" in payload["state_dicts"] and "optimizer" not in (exclude_keys or ()):
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
        self._rebind_shared_encoder()
        self._assert_encoder_state()

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
            inactive = self._ROUTING_PREFIXES if self.downstream_finetune_started else ("obs_encoder.", "model.")
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

        progress = min(1.0, max(0.0, float(self.epoch) / decay_epochs))
        cosine_progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        target_patches = start_patches + (
            final_patches - start_patches
        ) * cosine_progress

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
    def _router_grad_norm(cls, policy) -> float:
        live = cls._unwrap(policy)
        foveation_net = getattr(live, "foveation_net", None)
        router = (
            None
            if foveation_net is None
            else getattr(foveation_net, "router", None)
        )
        if router is None:
            return 0.0

        squared_norm_scalar = None
        for param in router.parameters():
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
        live = cls._unwrap(policy)
        foveation_net = getattr(live, "foveation_net", None)
        router = (
            None
            if foveation_net is None
            else getattr(foveation_net, "router", None)
        )
        router_params = (
            ()
            if router is None
            else tuple(param for param in router.parameters() if param.requires_grad)
        )
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

    @staticmethod
    def _make_foveation_vis_log(
        policy,
        foveation_outputs,
        max_batch: int = 8,
        vis_rot_image: bool = False,
    ):
        """Build wandb images for the first few samples of every RGB stream."""
        import wandb

        vis_log = {}
        for rgb_key, foveation_out in foveation_outputs.items():
            vis_B_T_H_W_C = policy.foveation_net.vis_ar_tokens(foveation_out)
            if vis_rot_image and rgb_key == "image":
                vis_B_T_H_W_C = np.flip(vis_B_T_H_W_C, axis=(2, 3)).copy()
            num_patches_B_T = foveation_out["num_patches_B_T"].detach().cpu()
            images = []
            for batch_idx in range(min(max_batch, vis_B_T_H_W_C.shape[0])):
                for time_idx in range(vis_B_T_H_W_C.shape[1]):
                    caption = (
                        f"b={batch_idx}, t={time_idx}, "
                        f"num_patches={int(num_patches_B_T[batch_idx, time_idx])}"
                    )
                    images.append(
                        wandb.Image(
                            vis_B_T_H_W_C[batch_idx, time_idx],
                            caption=caption,
                        )
                    )
            if images:
                vis_log[f"vis/train_foveation/{rgb_key}"] = images
        return vis_log

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

    def _validation(self, policy, val_dataloader, max_steps: Optional[int]):
        sums: Dict[str, torch.Tensor] = {}
        count = torch.tensor(0.0, device=self.accelerator.device)
        policy.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if max_steps is not None and batch_idx >= int(max_steps):
                    break
                losses = policy.compute_loss(batch)
                self._globalize_elu_budget(policy, losses)
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
        return {
            f"val/{key}": (reduced[idx] / denominator).item()
            for idx, key in enumerate(keys)
        }

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        accelerator = self.accelerator
        device = accelerator.device
        resume_payload = None

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError("The foveation workspace requires a BaseImageDataset.")
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
        if bool(self._cfg_get(cfg, "training.resume", False)):
            path = self.get_checkpoint_path()
            if path.is_file():
                # The scheduler is created below after the loader length is known.
                resume_payload = self.load_checkpoint(
                    path=path,
                    exclude_keys=("lr_scheduler",),
                )
        # A frozen action path must keep the normalization used by its checkpoint.
        if resume_payload is None and not self._cfg_get(cfg, "pretrained_ckpt", None):
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
            if self._share_global_encoder and hasattr(ema, "exclude_prefixes"):
                ema.exclude_prefixes = tuple(getattr(ema, "exclude_prefixes", ())) + (
                    self._GLOBAL_MARKER,
                )
            ema.optimization_step = int(self.global_step)
            base_ema_exclude_prefixes = tuple(ema.exclude_prefixes)

        wandb_run, log_context = self._build_logger()
        env_runner_cfg = self._cfg_get(cfg, "task.env_runner", None)
        env_runner = None
        if accelerator.is_main_process and env_runner_cfg is not None:
            # Simulation rollout is a main-process-only side effect.
            env_runner = hydra.utils.instantiate(
                env_runner_cfg,
                output_dir=self.output_dir,
            )
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

        with log_context as json_logger:
            for _ in range(self.epoch, num_epochs):
                self._set_stage_b_phase(ema, base_ema_exclude_prefixes)
                target_num_patches = self._set_target_num_patches()
                self.model.train()
                if bool(self._cfg_get(cfg, "policy.en_freeze_obs_encoder", False)):
                    self._unwrap(self.model).freeze_obs_encoder()
                epoch_losses = []
                step_log = {"epoch": self.epoch, "global_step": self.global_step}
                wandb_vis_log = {}
                budget_stats_2 = torch.zeros(2, device=device, dtype=torch.float64)
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
                        train_sampling_batch = batch
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
                            observation_count_scalar = losses["_budget_observation_count"]
                            budget_stats_2[0] += losses["_budget_expected_sum"]
                            budget_stats_2[1] += observation_count_scalar
                            # Fixed T/views: weighting by B also weights by observation count.
                            # Use the nominal batch size to avoid a large summed loss under AMP.
                            batch_weight = batch["action"].shape[0] / loader_cfg["batch_size"]
                            backward_loss_scalar = raw_loss * batch_weight
                            reference_count_scalar = observation_count_scalar / batch_weight
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
                                # Reduce scalars separately for older Accelerate versions.
                                global_budget_stats_2 = torch.stack([
                                    accelerator.reduce(value_scalar, reduction="sum")
                                    for value_scalar in budget_stats_2
                                ])
                                # Undo DDP and Accelerate averaging, then normalize by the
                                # actual global count (also handles a short final window).
                                grad_scale_scalar = (
                                    grad_accum * accelerator.num_processes * reference_count_scalar
                                    / global_budget_stats_2[1]
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
                                    budget_metrics = self._update_budget_lambda(global_budget_stats_2)
                            budget_stats_2.zero_()
                    loss_value = float(raw_loss.detach().item())
                    epoch_losses.append(loss_value)
                    train_metrics = self._scalar_losses(losses)
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
                    if accelerator.is_main_process:
                        policy_rollout = (
                            self.ema_model
                            if self.ema_model is not None
                            else self._unwrap(self.model)
                        )
                        policy_rollout.eval()
                        step_log.update(env_runner.run(policy_rollout))
                    # Keep other ranks out of validation collectives during rollout.
                    accelerator.wait_for_everyone()
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
                        step_log["sample/topology_div"] = self._topology_diversity(
                            foveation_outputs,
                            policy_sample.foveation_net.num_bank_patches,
                        )
                    if wandb_run is not None:
                        wandb_vis_log = self._make_foveation_vis_log(
                            policy_sample,
                            foveation_outputs,
                            vis_rot_image=self.vis_rot_image,
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
    TrainFoveationDPTransformerWorkspace(cfg).run()


if __name__ == "__main__":
    main()
