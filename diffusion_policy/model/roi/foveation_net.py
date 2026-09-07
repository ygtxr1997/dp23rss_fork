from contextlib import nullcontext
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.roi.jepa_helper import JEPAHelper
from diffusion_policy.model.roi.resnet_helper import ResNetHelper
from diffusion_policy.model.roi.vit_helper import (
    NodeAlignedViTRouter,
    QuadtreeViTEncoder,
)


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with per-frame AdaLN modulation."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        mlp_hidden_dim = int(round(hidden_dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )

        # AdaLN-Zero keeps a newly initialized router close to an identity stack.
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    @staticmethod
    def _modulate(
        x_BT_N_D: torch.Tensor,
        shift_BT_D: torch.Tensor,
        scale_BT_D: torch.Tensor,
    ) -> torch.Tensor:
        return (
            x_BT_N_D * (1.0 + scale_BT_D.unsqueeze(1))
            + shift_BT_D.unsqueeze(1)
        )

    def forward(
        self,
        x_BT_N_D: torch.Tensor,
        cond_BT_D: torch.Tensor,
    ) -> torch.Tensor:
        modulation_BT_6D = self.ada_ln(cond_BT_D)
        (
            shift_attn_BT_D,
            scale_attn_BT_D,
            gate_attn_BT_D,
            shift_mlp_BT_D,
            scale_mlp_BT_D,
            gate_mlp_BT_D,
        ) = modulation_BT_6D.chunk(6, dim=-1)

        attn_input_BT_N_D = self._modulate(
            self.norm1(x_BT_N_D),
            shift_attn_BT_D,
            scale_attn_BT_D,
        )
        attn_output_BT_N_D = self.self_attn(
            attn_input_BT_N_D,
            attn_input_BT_N_D,
            attn_input_BT_N_D,
            need_weights=False,
        )[0]
        x_BT_N_D = x_BT_N_D + gate_attn_BT_D.unsqueeze(1) * attn_output_BT_N_D

        mlp_input_BT_N_D = self._modulate(
            self.norm2(x_BT_N_D),
            shift_mlp_BT_D,
            scale_mlp_BT_D,
        )
        mlp_output_BT_N_D = self.mlp(mlp_input_BT_N_D)
        return x_BT_N_D + gate_mlp_BT_D.unsqueeze(1) * mlp_output_BT_N_D


class AdaptiveResolutionRouter(nn.Module):
    """Predict all fixed-order quadtree split logits from global image tokens."""

    def __init__(
        self,
        global_feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        cond_dim: int,
        num_internal_nodes: int,
    ) -> None:
        super().__init__()
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.num_internal_nodes = int(num_internal_nodes)

        self.input_projection = nn.Linear(self.global_feature_dim, self.hidden_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_internal_nodes),
        )

    def forward(
        self,
        global_features_B_T_N_D: torch.Tensor,
        router_condition_B_T_D: torch.Tensor,
    ) -> torch.Tensor:
        if global_features_B_T_N_D.ndim != 4:
            raise ValueError("Expected global features with shape (B, T, N, D).")
        B, T, N, D = global_features_B_T_N_D.shape
        if D != self.global_feature_dim:
            raise ValueError(
                f"Expected global feature dimension {self.global_feature_dim}, got {D}."
            )
        if N <= 0:
            raise ValueError("The global encoder must return at least one visual token.")
        if router_condition_B_T_D.shape != (B, T, self.hidden_dim):
            raise ValueError(
                "Expected projected router condition with shape "
                f"({B}, {T}, {self.hidden_dim})."
            )

        global_features_BT_N_D = global_features_B_T_N_D.reshape(B * T, N, D)
        router_condition_BT_D = router_condition_B_T_D.reshape(
            B * T,
            self.hidden_dim,
        )
        router_tokens_BT_N_D = self.input_projection(global_features_BT_N_D)
        for block in self.blocks:
            router_tokens_BT_N_D = block(
                router_tokens_BT_N_D,
                router_condition_BT_D,
            )

        # The Transformer keeps the global token count N unchanged. Each output
        # channel below has a fixed quadtree-node meaning and produces one logit.
        pooled_router_features_BT_D = self.final_norm(
            router_tokens_BT_N_D
        ).mean(dim=1)
        split_logits_BT_M = self.output_head(pooled_router_features_BT_D)
        return split_logits_BT_M.reshape(B, T, self.num_internal_nodes)


class NodeAlignedFPNRouter(nn.Module):
    """Predict node-aligned split logits with a spatial FPN and per-level FiLM."""

    def __init__(
        self,
        level_grid_hw: Tuple[Tuple[int, int], ...],
        level_in_channels: Dict[int, int],
        deepest_in_channels: int,
        hidden_dim: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        if hidden_dim % 8 != 0:
            raise ValueError("hidden_dim must be divisible by 8 for GroupNorm.")
        if not level_grid_hw:
            raise ValueError("NodeAlignedFPNRouter requires an internal node level.")

        self.level_grid_hw = tuple(
            tuple(int(size) for size in grid_hw) for grid_hw in level_grid_hw
        )
        self.level_in_channels = {
            int(level): int(channels)
            for level, channels in level_in_channels.items()
        }
        self.deepest_in_channels = int(deepest_in_channels)
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        if any(
            level < 0 or level >= len(self.level_grid_hw)
            for level in self.level_in_channels
        ):
            raise ValueError("Aligned ResNet node levels are out of range.")

        # Only stages with exact node-grid alignment receive lateral projections.
        self.lateral_projections = nn.ModuleDict(
            {
                str(level): nn.Conv2d(channels, self.hidden_dim, kernel_size=1)
                for level, channels in self.level_in_channels.items()
            }
        )
        # Keep the deepest ResNet map spatial instead of pooling it into one
        # frame-level bias; it is the seed of the top-down FPN path.
        self.deepest_projection = nn.Conv2d(
            self.deepest_in_channels,
            self.hidden_dim,
            kernel_size=1,
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.fusion_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=3,
                    padding=1,
                )
                for _ in self.level_grid_hw
            ]
        )
        self.fusion_norms = nn.ModuleList(
            [nn.GroupNorm(8, self.hidden_dim) for _ in self.level_grid_hw]
        )
        self.film_projections = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
                for _ in self.level_grid_hw
            ]
        )
        self.activation = nn.GELU()
        for film_projection in self.film_projections:
            # FiLM starts as identity, so enabling a new condition does not
            # perturb the spatial FPN before the modulation heads learn.
            nn.init.zeros_(film_projection.weight)
            nn.init.zeros_(film_projection.bias)
        # Every node level shares one spatial classifier instead of node-specific rows.
        self.output_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        level_features_B_T_C_H_W: Dict[int, torch.Tensor],
        deepest_features_B_T_C_H_W: torch.Tensor,
        cond_B_T_D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return all internal levels in level-major, row-major order."""
        if deepest_features_B_T_C_H_W.ndim != 5:
            raise ValueError("Expected ResNet features with shape (B, T, C, H, W).")
        B, T, deepest_channels = deepest_features_B_T_C_H_W.shape[:3]
        if deepest_channels != self.deepest_in_channels:
            raise ValueError(
                f"Expected {self.deepest_in_channels} deepest feature channels, "
                f"got {deepest_channels}."
            )

        lateral_features_BT_D_H_W: Dict[int, torch.Tensor] = {}
        for level, in_channels in self.level_in_channels.items():
            expected_shape = (B, T, in_channels, *self.level_grid_hw[level])
            features_B_T_C_H_W = level_features_B_T_C_H_W[level]
            if features_B_T_C_H_W.shape != expected_shape:
                raise ValueError(
                    f"Expected node level {level} features with shape "
                    f"{expected_shape}, got {tuple(features_B_T_C_H_W.shape)}."
                )
            features_BT_C_H_W = features_B_T_C_H_W.reshape(
                B * T,
                in_channels,
                *self.level_grid_hw[level],
            )
            lateral_features_BT_D_H_W[level] = self.lateral_projections[
                str(level)
            ](features_BT_C_H_W)

        deepest_features_BT_C_H_W = deepest_features_B_T_C_H_W.reshape(
            B * T,
            self.deepest_in_channels,
            *deepest_features_B_T_C_H_W.shape[-2:],
        )
        fused_features_BT_D_H_W = F.interpolate(
            self.deepest_projection(deepest_features_BT_C_H_W),
            size=self.level_grid_hw[0],
            mode="nearest",
        )

        condition_BT_D = None
        if cond_B_T_D is not None:
            if cond_B_T_D.shape != (B, T, self.cond_dim):
                raise ValueError(
                    f"Expected cond_B_T_D with shape {(B, T, self.cond_dim)}."
                )
            cond_BT_D = cond_B_T_D.to(
                device=fused_features_BT_D_H_W.device,
                dtype=fused_features_BT_D_H_W.dtype,
            ).reshape(B * T, self.cond_dim)
            condition_BT_D = self.condition_projection(cond_BT_D)

        logits_B_T_M = []
        for level in range(len(self.level_grid_hw)):
            if level > 0:
                fused_features_BT_D_H_W = F.interpolate(
                    fused_features_BT_D_H_W,
                    size=self.level_grid_hw[level],
                    mode="nearest",
                )
            if level in lateral_features_BT_D_H_W:
                fused_features_BT_D_H_W = (
                    fused_features_BT_D_H_W
                    + lateral_features_BT_D_H_W[level]
                )
            fused_features_BT_D_H_W = self.fusion_norms[level](
                self.fusion_convs[level](fused_features_BT_D_H_W)
            )
            if condition_BT_D is not None:
                # Modulate normalized channels at every node level; broadcasting
                # is limited to FiLM's per-channel scale and shift.
                modulation_BT_2D = self.film_projections[level](condition_BT_D)
                gamma_BT_D, beta_BT_D = modulation_BT_2D.chunk(2, dim=-1)
                fused_features_BT_D_H_W = (
                    fused_features_BT_D_H_W
                    * (1.0 + gamma_BT_D[:, :, None, None])
                    + beta_BT_D[:, :, None, None]
                )
            fused_features_BT_D_H_W = self.activation(
                fused_features_BT_D_H_W
            )
            logits_B_T_M.append(
                self.output_head(fused_features_BT_D_H_W).reshape(B, T, -1)
            )
        return torch.cat(logits_B_T_M, dim=-1)


class FoveationNet(nn.Module):
    """Build a mixed-resolution patch sequence with quadtree routing.

    Input images use the (B, C, T, H, W) convention. Pyramid levels are ordered
    from lowest to highest resolution, and all flattened nodes use level-major,
    row-major ordering. ``fix_split_mode`` bypasses the global encoder and router
    with a coarsest, finest, per-sample random, or persistent random topology.
    Random mode may provide multiple split ratios; each sample independently
    selects one ratio and shares the resulting topology across its frames.
    During training, ``random_topology_prob`` can replace each frame's hard tree
    with a random legal tree that preserves its router-selected patch count.
    Router action value ``split_action_value`` determines whether high or low
    logits mean split.
    """

    RESNET_STAGE_CHANNELS: Dict[str, int] = {
        "layer1": 64,
        "layer2": 128,
        "layer3": 256,
        "layer4": 512,
    }

    def __init__(
        self,
        in_channels: int = 3,
        input_hw: Tuple[int, int] = (1024, 1024),
        num_levels: int = 3,
        global_level: int = 0,
        global_encoder_type: str = "vjepa2_1_vit_giant_384",
        torch_home: Optional[str] = None,
        global_feature_dim: int = 1408,
        router_layers: int = 4,
        router_hidden_dim: int = 512,
        router_heads: int = 8,
        router_mlp_ratio: float = 4.0,
        router_dropout: float = 0.0,
        router_cond_dim: int = 512,
        patch_size: int = 16,
        freeze_global_encoder: bool = True,
        freeze_router: bool = False,
        router_temperature: float = 1.0,
        split_action_value: int = 1,
        target_num_patches: Optional[Union[int, float]] = None,
        budget_loss_weight: float = 0.0,
        budget_probability_type: str = "sampled_relaxed",
        fix_split_mode: Optional[str] = None,
        router_type: str = "pooled_global",
        random_topology_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        self.num_levels = int(num_levels)
        self.global_level = int(global_level)
        self.global_encoder_type = str(global_encoder_type)
        self.torch_home = torch_home
        self.global_feature_dim = int(global_feature_dim)
        self.router_layers = int(router_layers)
        self.router_hidden_dim = int(router_hidden_dim)
        self.router_heads = int(router_heads)
        self.router_mlp_ratio = float(router_mlp_ratio)
        self.router_dropout = float(router_dropout)
        self.router_cond_dim = int(router_cond_dim)
        self.patch_size = int(patch_size)
        self.freeze_global_encoder = bool(freeze_global_encoder)
        self.freeze_router = bool(freeze_router)
        self.router_temperature = float(router_temperature)
        self.split_action_value = int(split_action_value)
        self.target_num_patches = target_num_patches
        self.budget_loss_weight = float(budget_loss_weight)
        self.budget_probability_type = str(budget_probability_type)
        self.fix_split_mode = fix_split_mode
        (
            self.random_split_ratios,
            self.fixed_split_ratio,
        ) = self._parse_split_ratios(fix_split_mode)
        self.random_topology_prob = float(random_topology_prob)
        self.router_type = str(router_type)

        self._validate_geometry(
            router_layers=self.router_layers,
            router_heads=self.router_heads,
            router_mlp_ratio=self.router_mlp_ratio,
            router_dropout=self.router_dropout,
        )

        input_h, input_w = self.input_hw
        self.level_hw = tuple(
            (
                input_h // (2 ** (self.num_levels - 1 - level)),
                input_w // (2 ** (self.num_levels - 1 - level)),
            )
            for level in range(self.num_levels)
        )
        self.level_grid_hw = tuple(
            (level_h // self.patch_size, level_w // self.patch_size)
            for level_h, level_w in self.level_hw
        )
        self.level_num_patches = tuple(
            grid_h * grid_w for grid_h, grid_w in self.level_grid_hw
        )
        self.bank_level_offsets = self._prefix_offsets(self.level_num_patches)
        self.internal_level_offsets = self._prefix_offsets(
            self.level_num_patches[:-1]
        )
        self.num_bank_patches = sum(self.level_num_patches)
        self.num_internal_nodes = sum(self.level_num_patches[:-1])
        self.min_num_patches = self.level_num_patches[0]
        self.max_num_patches = self.level_num_patches[-1]
        self.resolved_target_num_patches = self.resolve_target_num_patches(
            self.target_num_patches
        )
        self.random_num_splits = (
            tuple(
                self._num_splits_from_ratio(split_ratio)
                for split_ratio in self.random_split_ratios
            )
            if self.random_split_ratios is not None
            else None
        )

        self._register_patch_metadata()
        fixed_split_hard_1_1_M = None
        if self.fixed_split_ratio is not None and self.num_internal_nodes > 0:
            target_num_splits = self._num_splits_from_ratio(
                self.fixed_split_ratio
            )
            num_splits_1_1 = torch.full(
                (1, 1),
                target_num_splits,
                dtype=torch.long,
            )
            # Preserve the global RNG stream so this ablation does not change
            # downstream policy initialization for the same training seed.
            with torch.random.fork_rng(devices=[]):
                fixed_split_hard_1_1_M = self._sample_random_split_mask(
                    num_splits_1_1
                )
        # Persistent model state keeps the exact topology across resume and eval.
        self.register_buffer(
            "fixed_split_hard_1_1_M",
            fixed_split_hard_1_1_M,
        )

        self.global_encoder = (
            self._load_global_encoder(self.global_encoder_type)
            if self.num_internal_nodes > 0
            else None
        )
        if self.global_encoder is not None and (
            self.freeze_global_encoder or self.fix_split_mode is not None
        ):
            self.global_encoder.requires_grad_(False)
            self.global_encoder.eval()

        if self.router_type == "pooled_global":
            self.router = AdaptiveResolutionRouter(
                global_feature_dim=self.global_feature_dim,
                hidden_dim=self.router_hidden_dim,
                num_layers=self.router_layers,
                num_heads=self.router_heads,
                mlp_ratio=self.router_mlp_ratio,
                dropout=self.router_dropout,
                cond_dim=self.router_cond_dim,
                num_internal_nodes=self.num_internal_nodes,
            )
        elif self.router_type == "node_aligned_fpn":
            stage_to_node_level = self._resnet_stage_to_node_level()
            self.router = NodeAlignedFPNRouter(
                level_grid_hw=self.level_grid_hw[:-1],
                level_in_channels={
                    node_level: self.RESNET_STAGE_CHANNELS[stage]
                    for stage, node_level in stage_to_node_level.items()
                },
                deepest_in_channels=self.RESNET_STAGE_CHANNELS["layer4"],
                hidden_dim=self.router_hidden_dim,
                cond_dim=self.router_cond_dim,
            )
        else:
            self.router = NodeAlignedViTRouter(
                level_grid_hw=self.level_grid_hw[:-1],
                node_dim=self.global_feature_dim,
                hidden_dim=self.router_hidden_dim,
                cond_dim=self.router_cond_dim,
            )
        if self.freeze_router or self.fix_split_mode is not None:
            self.router.requires_grad_(False)

    @staticmethod
    def _prefix_offsets(counts: Tuple[int, ...]) -> Tuple[int, ...]:
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        return tuple(offsets)

    @staticmethod
    def _parse_split_ratios(
        fix_split_mode: Optional[str],
    ) -> Tuple[Optional[Tuple[float, ...]], Optional[float]]:
        if fix_split_mode in (None, "coarsest", "finest"):
            return None, None
        if not isinstance(fix_split_mode, str):
            raise ValueError(
                "fix_split_mode must be None, 'coarsest', 'finest', or "
                "'random:<ratio>[,<ratio>...]' or 'fixed:<ratio>'."
            )
        if fix_split_mode.startswith("random:"):
            split_mode = "random"
        elif fix_split_mode.startswith("fixed:"):
            split_mode = "fixed"
        else:
            raise ValueError(
                "fix_split_mode must be None, 'coarsest', 'finest', or "
                "'random:<ratio>[,<ratio>...]' or 'fixed:<ratio>'."
            )

        ratio_text = fix_split_mode.removeprefix(f"{split_mode}:")
        if split_mode == "random":
            ratio_items = ratio_text.split(",")
            if not ratio_items or any(not item.strip() for item in ratio_items):
                raise ValueError(
                    "Random fix_split_mode requires a non-empty comma-separated "
                    "ratio list."
                )
            try:
                split_ratios = tuple(float(item) for item in ratio_items)
            except ValueError:
                raise ValueError(
                    "Every ratio in fix_split_mode='random:<ratio>[,<ratio>...]' "
                    "must be a number."
                ) from None
            if any(
                not np.isfinite(split_ratio)
                or not 0.0 <= split_ratio <= 1.0
                for split_ratio in split_ratios
            ):
                raise ValueError("Every split ratio must be in [0, 1].")
            return split_ratios, None

        try:
            split_ratio = float(ratio_text)
        except ValueError:
            raise ValueError(
                "fix_split_mode='fixed:<ratio>' requires one numeric ratio."
            ) from None
        if not np.isfinite(split_ratio) or not 0.0 <= split_ratio <= 1.0:
            raise ValueError("The split ratio must be in [0, 1].")
        return None, split_ratio

    def _num_splits_from_ratio(self, split_ratio: float) -> int:
        return int(float(split_ratio) * self.num_internal_nodes + 0.5)

    def _resnet_stage_to_node_level(self) -> Dict[str, int]:
        stage_to_node_level = {}
        # layer4 is always the spatial top-down seed; earlier stages provide
        # only node-aligned lateral skips and must not duplicate that feature.
        for layer_index in range(1, 4):
            # A stride-2 ResNet step corresponds to one coarser quadtree level.
            node_level = self.global_level + 3 - layer_index
            if 0 <= node_level <= self.num_levels - 2:
                stage_to_node_level[f"layer{layer_index}"] = node_level
        return stage_to_node_level

    def _validate_geometry(
        self,
        router_layers: int,
        router_heads: int,
        router_mlp_ratio: float,
        router_dropout: float,
    ) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if len(self.input_hw) != 2 or min(self.input_hw) <= 0:
            raise ValueError("input_hw must contain two positive integers.")
        if self.num_levels <= 0:
            raise ValueError("num_levels must be positive.")
        if not (0 <= self.global_level < self.num_levels):
            raise ValueError("global_level must be in [0, num_levels).")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        quadtree_stride = self.patch_size * (2 ** (self.num_levels - 1))
        if any(size % quadtree_stride != 0 for size in self.input_hw):
            raise ValueError(
                "Each input dimension must be divisible by "
                "patch_size * 2**(num_levels - 1) to form an exact quadtree."
            )
        if self.global_feature_dim <= 0 or self.router_hidden_dim <= 0:
            raise ValueError("Feature dimensions must be positive.")
        if self.router_cond_dim <= 0:
            raise ValueError("router_cond_dim must be positive.")
        if int(router_layers) <= 0:
            raise ValueError("router_layers must be positive.")
        if int(router_heads) <= 0:
            raise ValueError("router_heads must be positive.")
        if (
            self.router_type != "node_aligned_vit"
            and self.router_hidden_dim % int(router_heads) != 0
        ):
            raise ValueError("router_hidden_dim must be divisible by router_heads.")
        if (
            self.router_type == "node_aligned_vit"
            and self.global_feature_dim % int(router_heads) != 0
        ):
            raise ValueError("global_feature_dim must be divisible by router_heads.")
        if float(router_mlp_ratio) <= 0.0:
            raise ValueError("router_mlp_ratio must be positive.")
        if not (0.0 <= float(router_dropout) < 1.0):
            raise ValueError("router_dropout must be in [0, 1).")
        if self.router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive.")
        if self.split_action_value not in (0, 1):
            raise ValueError("split_action_value must be 0 or 1.")
        if not 0.0 <= self.random_topology_prob <= 1.0:
            raise ValueError("random_topology_prob must be in [0, 1].")
        if self.random_topology_prob > 0.0 and self.fix_split_mode is not None:
            if self.fixed_split_ratio is not None:
                warnings.warn(
                    "random_topology_prob is ignored when fix_split_mode uses "
                    "'fixed:<ratio>'.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    "random_topology_prob requires fix_split_mode=None."
                )
        if self.budget_loss_weight < 0.0:
            raise ValueError("budget_loss_weight must be non-negative.")
        if self.budget_probability_type not in (
            "sampled_relaxed",
            "clean_expected",
        ):
            raise ValueError(
                "budget_probability_type must be 'sampled_relaxed' or "
                "'clean_expected'."
            )
        if self.router_type not in (
            "pooled_global",
            "node_aligned_fpn",
            "node_aligned_vit",
        ):
            raise ValueError(
                "router_type must be 'pooled_global', 'node_aligned_fpn', or "
                "'node_aligned_vit'."
            )
        if self.router_type == "node_aligned_fpn":
            if self.num_levels < 2 or self.patch_size != 16:
                raise ValueError(
                    "node_aligned_fpn requires at least two levels and patch_size=16."
                )
            if self.global_encoder_type != "resnet10":
                raise ValueError("node_aligned_fpn currently requires resnet10.")
        if self.router_type == "node_aligned_vit":
            if self.num_levels < 2:
                raise ValueError("node_aligned_vit requires at least two levels.")
            if self.global_encoder_type != "quadtree_vit_tiny":
                raise ValueError(
                    "node_aligned_vit currently requires quadtree_vit_tiny."
                )

            global_scale = 2 ** (self.num_levels - 1 - self.global_level)
            global_h = self.input_hw[0] // global_scale
            global_w = self.input_hw[1] // global_scale
            fine_internal_grid_h = self.input_hw[0] // (2 * self.patch_size)
            fine_internal_grid_w = self.input_hw[1] // (2 * self.patch_size)
            if (
                global_h < fine_internal_grid_h
                or global_w < fine_internal_grid_w
                or global_h % fine_internal_grid_h != 0
                or global_w % fine_internal_grid_w != 0
            ):
                raise ValueError(
                    "The global-level input must split exactly into the finest "
                    "internal node grid."
                )
        elif self.global_encoder_type == "quadtree_vit_tiny":
            raise ValueError(
                "quadtree_vit_tiny currently requires node_aligned_vit."
            )

    def resolve_target_num_patches(
        self,
        target_num_patches: Optional[Union[int, float]],
    ) -> Optional[float]:
        """Convert a ratio or absolute leaf budget to an absolute count."""
        if target_num_patches is None:
            return None
        if isinstance(target_num_patches, bool):
            raise TypeError("target_num_patches must be an int, float, or None.")

        if isinstance(target_num_patches, float) and 0.0 <= target_num_patches <= 1.0:
            target = self.min_num_patches + target_num_patches * (
                self.max_num_patches - self.min_num_patches
            )
        else:
            target = float(target_num_patches)
        if not self.min_num_patches <= target <= self.max_num_patches:
            raise ValueError(
                "Absolute target_num_patches must lie between the minimum and "
                "maximum possible leaf counts."
            )
        return target

    def _register_patch_metadata(self) -> None:
        levels: List[torch.Tensor] = []
        boxes: List[torch.Tensor] = []
        centers: List[torch.Tensor] = []
        sizes: List[torch.Tensor] = []
        input_h, input_w = self.input_hw

        for level, (grid_h, grid_w) in enumerate(self.level_grid_hw):
            scale = 2 ** (self.num_levels - 1 - level)
            rows_H, cols_W = torch.meshgrid(
                torch.arange(grid_h, dtype=torch.float32),
                torch.arange(grid_w, dtype=torch.float32),
                indexing="ij",
            )
            patch_h = float(self.patch_size * scale)
            patch_w = float(self.patch_size * scale)
            x0_H_W = cols_W * patch_w
            y0_H_W = rows_H * patch_h
            x1_H_W = x0_H_W + patch_w
            y1_H_W = y0_H_W + patch_h
            boxes_H_W_4 = torch.stack(
                [x0_H_W, y0_H_W, x1_H_W, y1_H_W],
                dim=-1,
            )
            centers_H_W_2 = torch.stack(
                [
                    (x0_H_W + x1_H_W) / (2.0 * input_w),
                    (y0_H_W + y1_H_W) / (2.0 * input_h),
                ],
                dim=-1,
            )
            sizes_H_W_2 = torch.empty_like(centers_H_W_2)
            sizes_H_W_2[..., 0] = patch_h / input_h
            sizes_H_W_2[..., 1] = patch_w / input_w

            num_level_patches = grid_h * grid_w
            levels.append(torch.full((num_level_patches,), level, dtype=torch.long))
            boxes.append(boxes_H_W_4.reshape(num_level_patches, 4))
            centers.append(centers_H_W_2.reshape(num_level_patches, 2))
            sizes.append(sizes_H_W_2.reshape(num_level_patches, 2))

        self.register_buffer("bank_levels_K", torch.cat(levels, dim=0))
        self.register_buffer("bank_boxes_xyxy_K_4", torch.cat(boxes, dim=0))
        self.register_buffer("bank_centers_xy_norm_K_2", torch.cat(centers, dim=0))
        self.register_buffer("bank_sizes_hw_norm_K_2", torch.cat(sizes, dim=0))

        internal_children_M_4 = torch.full(
            (self.num_internal_nodes, 4),
            -1,
            dtype=torch.long,
        )
        for level in range(self.num_levels - 2):
            grid_h, grid_w = self.level_grid_hw[level]
            child_grid_w = self.level_grid_hw[level + 1][1]
            rows_H, cols_W = torch.meshgrid(
                torch.arange(grid_h, dtype=torch.long),
                torch.arange(grid_w, dtype=torch.long),
                indexing="ij",
            )
            child_rows_H_W_4 = torch.stack(
                [2 * rows_H, 2 * rows_H, 2 * rows_H + 1, 2 * rows_H + 1],
                dim=-1,
            )
            child_cols_H_W_4 = torch.stack(
                [2 * cols_W, 2 * cols_W + 1, 2 * cols_W, 2 * cols_W + 1],
                dim=-1,
            )
            child_start = self.internal_level_offsets[level + 1]
            children_H_W_4 = (
                child_start
                + child_rows_H_W_4 * child_grid_w
                + child_cols_H_W_4
            )
            start = self.internal_level_offsets[level]
            end = self.internal_level_offsets[level + 1]
            internal_children_M_4[start:end] = children_H_W_4.reshape(-1, 4)
        # This mapping is derived from geometry, so checkpoints need not store it.
        self.register_buffer(
            "internal_children_M_4",
            internal_children_M_4,
            persistent=False,
        )

    def _load_global_encoder(self, global_encoder_type: str) -> Optional[nn.Module]:
        if global_encoder_type == "quadtree_vit_tiny":
            return QuadtreeViTEncoder(
                in_channels=self.in_channels,
                input_hw=self.level_hw[self.global_level],
                level_grid_hw=self.level_grid_hw[:-1],
                token_dim=self.global_feature_dim,
                num_layers=self.router_layers,
                num_heads=self.router_heads,
                mlp_ratio=self.router_mlp_ratio,
                dropout=self.router_dropout,
            )
        if global_encoder_type in JEPAHelper.MODEL_NAMES:
            return JEPAHelper(
                model_name=global_encoder_type,
                torch_home=self.torch_home,
            )
        if global_encoder_type in ResNetHelper.MODEL_DIMS:
            return ResNetHelper(
                model_name=global_encoder_type,
                out_dim=self.global_feature_dim,
                project_output=self.router_type == "pooled_global",
            )
        supported = (
            tuple(JEPAHelper.MODEL_NAMES)
            + tuple(ResNetHelper.MODEL_DIMS)
            + ("quadtree_vit_tiny",)
        )
        raise ValueError(
            f"Unsupported global_encoder_type {global_encoder_type!r}; {supported}"
        )

    def train(self, mode: bool = True) -> "FoveationNet":
        super().train(mode)
        if self.global_encoder is not None and (
            self.freeze_global_encoder or self.fix_split_mode is not None
        ):
            self.global_encoder.eval()
        return self

    def set_router_temperature(self, router_temp: float) -> None:
        if router_temp <= 0.0:
            raise ValueError("router_temp must be positive.")
        self.router_temperature = float(router_temp)

    def set_target_num_patches(
        self,
        target_num_patches: Optional[Union[int, float]],
    ) -> Optional[float]:
        """Update the differentiable budget target and return its leaf count."""
        resolved_target = self.resolve_target_num_patches(target_num_patches)
        self.target_num_patches = target_num_patches
        self.resolved_target_num_patches = resolved_target
        return resolved_target

    def _build_image_pyramid(
        self,
        x_high_B_C_T_H_W: torch.Tensor,
    ) -> List[torch.Tensor]:
        B, C, T, H, W = x_high_B_C_T_H_W.shape
        x_high_BT_C_H_W = (
            x_high_B_C_T_H_W.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(B * T, C, H, W)
        )
        pyramid_high_to_low: List[torch.Tensor] = [x_high_BT_C_H_W]
        x_level_BT_C_H_W = x_high_BT_C_H_W
        for _ in range(self.num_levels - 1):
            x_level_BT_C_H_W = F.interpolate(
                x_level_BT_C_H_W,
                scale_factor=0.5,
                mode="area",
            )
            pyramid_high_to_low.append(x_level_BT_C_H_W)

        pyramid_low_to_high: List[torch.Tensor] = []
        for x_level_BT_C_H_W in reversed(pyramid_high_to_low):
            level_h, level_w = x_level_BT_C_H_W.shape[-2:]
            x_level_B_C_T_H_W = (
                x_level_BT_C_H_W.reshape(B, T, C, level_h, level_w)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
            )
            pyramid_low_to_high.append(x_level_B_C_T_H_W)
        return pyramid_low_to_high

    def _patchify_pyramid(
        self,
        image_pyramid: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        patch_levels: List[torch.Tensor] = []
        P = self.patch_size
        for level, x_level_B_C_T_H_W in enumerate(image_pyramid):
            B, C, T, level_h, level_w = x_level_B_C_T_H_W.shape
            grid_h, grid_w = self.level_grid_hw[level]
            x_level_B_T_C_H_W = x_level_B_C_T_H_W.permute(0, 2, 1, 3, 4)
            patches_B_T_H_W_C_P_P = (
                x_level_B_T_C_H_W.reshape(B, T, C, grid_h, P, grid_w, P)
                .permute(0, 1, 3, 5, 2, 4, 6)
                .contiguous()
            )
            patches_B_T_K_C_P_P = patches_B_T_H_W_C_P_P.reshape(
                B,
                T,
                grid_h * grid_w,
                C,
                P,
                P,
            )
            patch_levels.append(patches_B_T_K_C_P_P)
        patch_bank_B_T_K_C_P_P = torch.cat(patch_levels, dim=2)
        return patch_levels, patch_bank_B_T_K_C_P_P

    def _encode_global_images(
        self,
        global_images_B_C_T_H_W: torch.Tensor,
    ) -> torch.Tensor:
        encoder_context = (
            torch.no_grad() if self.freeze_global_encoder else nullcontext()
        )
        with encoder_context:
            global_features = self.global_encoder(
                global_images_B_C_T_H_W
            )
        if not isinstance(global_features, torch.Tensor):
            raise TypeError("The global encoder must return a tensor.")

        B, _, T, H, W = global_images_B_C_T_H_W.shape

        if global_features.ndim == 3:
            if H % 16 != 0 or W % 16 != 0:
                raise ValueError("VJEPA input height and width must be divisible by 16.")
            spatial_tokens = (H // 16) * (W // 16)
            B_encoder, total_tokens, feature_dim = global_features.shape
            if B_encoder != B:
                raise ValueError("Global encoder features must preserve batch size.")
            if total_tokens % spatial_tokens != 0:
                raise ValueError(
                    "VJEPA token count is incompatible with the input spatial grid."
                )
            T_out = total_tokens // spatial_tokens
            global_features_B_TN_D = global_features.reshape(
                B,
                T_out,
                spatial_tokens,
                feature_dim,
            )
        elif global_features.ndim == 4:
            B_encoder, T_out, N_encoder, feature_dim = global_features.shape
            if B_encoder != B:
                raise ValueError("Global encoder features must preserve batch size.")
            if N_encoder <= 0:
                raise ValueError("Global encoder must return at least one token.")
            global_features_B_TN_D = global_features
        else:
            raise ValueError(
                "Global encoder features must have shape (B, L, D) or (B, T, N, D)."
            )

        if T_out != T:
            if T_out <= 0 or T % T_out != 0:
                raise ValueError(
                    "Global encoder output time tokens must divide the input frame count."
                )
            global_features_B_T_N_D = global_features_B_TN_D.repeat_interleave(
                T // T_out,
                dim=1,
            )
        else:
            global_features_B_T_N_D = global_features_B_TN_D

        if global_features_B_T_N_D.shape[:2] != (B, T):
            raise ValueError("Global encoder features must preserve the input B and T.")
        if global_features_B_T_N_D.shape[-1] != self.global_feature_dim:
            raise ValueError(
                f"Expected global feature dimension {self.global_feature_dim}."
            )
        return global_features_B_T_N_D

    def _encode_global_stages(
        self,
        global_images_B_C_T_H_W: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoder_context = (
            torch.no_grad() if self.freeze_global_encoder else nullcontext()
        )
        with encoder_context:
            stage_features = self.global_encoder.forward_stages(
                global_images_B_C_T_H_W
            )
        return stage_features

    def _encode_quadtree_features(
        self,
        global_images_B_C_T_H_W: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        encoder_context = (
            torch.no_grad() if self.freeze_global_encoder else nullcontext()
        )
        with encoder_context:
            level_features_B_T_D_H_W = self.global_encoder(
                global_images_B_C_T_H_W
            )
        if not isinstance(level_features_B_T_D_H_W, dict):
            raise TypeError("The quadtree ViT encoder must return level features.")
        return level_features_B_T_D_H_W

    def _build_router_condition(
        self,
        cond_B_T_D: Optional[torch.Tensor],
        reference_B_T_N_D: torch.Tensor,
    ) -> torch.Tensor:
        B, T = reference_B_T_N_D.shape[:2]
        if cond_B_T_D is None:
            return reference_B_T_N_D.new_zeros((B, T, self.router_hidden_dim))
        if cond_B_T_D.shape != (B, T, self.router_cond_dim):
            raise ValueError(
                "Expected cond_B_T_D with shape "
                f"({B}, {T}, {self.router_cond_dim})."
            )
        cond_B_T_D = cond_B_T_D.to(
            device=reference_B_T_N_D.device,
            dtype=reference_B_T_N_D.dtype,
        )
        return self.router.condition_projection(cond_B_T_D)

    def _sample_random_split_mask(
        self,
        num_splits_B_T: torch.Tensor,
    ) -> torch.Tensor:
        B, T = num_splits_B_T.shape
        num_splits_BT = num_splits_B_T.reshape(B * T)
        split_hard_BT_M = torch.zeros(
            (B * T, self.num_internal_nodes),
            dtype=torch.bool,
            device=num_splits_B_T.device,
        )
        eligible_BT_M = torch.zeros_like(split_hard_BT_M)
        eligible_BT_M[:, : self.internal_level_offsets[1]] = True
        tree_indices_BT = torch.arange(B * T, device=num_splits_B_T.device)
        max_num_splits = int(num_splits_BT.max().detach().cpu())

        # Each tree grows only through its reachable frontier; per-frame split
        # counts may differ because topology dropout preserves router compute.
        for split_index in range(max_num_splits):
            active_BT = split_index < num_splits_BT
            random_scores_BT_M = torch.rand(
                (B * T, self.num_internal_nodes),
                device=num_splits_B_T.device,
            ).masked_fill(~eligible_BT_M, -1.0)
            selected_indices_BT = random_scores_BT_M.argmax(dim=-1)
            active_tree_indices_A = tree_indices_BT[active_BT]
            selected_indices_A = selected_indices_BT[active_BT]
            split_hard_BT_M[active_tree_indices_A, selected_indices_A] = True
            eligible_BT_M[active_tree_indices_A, selected_indices_A] = False

            children_A_4 = self.internal_children_M_4[selected_indices_A]
            valid_children_A_4 = children_A_4 >= 0
            child_tree_indices_A_4 = active_tree_indices_A[:, None].expand_as(
                children_A_4
            )
            eligible_BT_M[
                child_tree_indices_A_4[valid_children_A_4],
                children_A_4[valid_children_A_4],
            ] = True

        return split_hard_BT_M.reshape(B, T, self.num_internal_nodes)

    def _sample_split_decisions(
        self,
        split_logits_B_T_M: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            eps = torch.finfo(split_logits_B_T_M.dtype).eps
            uniform_noise_B_T_M = torch.rand_like(split_logits_B_T_M).clamp(
                min=eps,
                max=1.0 - eps,
            )
            logistic_noise_B_T_M = (
                uniform_noise_B_T_M.log()
                - torch.log1p(-uniform_noise_B_T_M)
            )
            action_soft_B_T_M = torch.sigmoid(
                (split_logits_B_T_M + logistic_noise_B_T_M)
                / self.router_temperature
            )
            action_hard_B_T_M = action_soft_B_T_M > 0.5
        else:
            action_soft_B_T_M = torch.sigmoid(
                split_logits_B_T_M / self.router_temperature
            )
            action_hard_B_T_M = split_logits_B_T_M > 0.0

        # Leaf construction always consumes canonical split probabilities and
        # decisions; only the router action value associated with split changes.
        if self.split_action_value == 1:
            split_soft_B_T_M = action_soft_B_T_M
            split_hard_B_T_M = action_hard_B_T_M
        else:
            split_soft_B_T_M = 1.0 - action_soft_B_T_M
            split_hard_B_T_M = ~action_hard_B_T_M
        return split_soft_B_T_M, split_hard_B_T_M

    @staticmethod
    def _replicate_to_children(x_B_T_H_W: torch.Tensor) -> torch.Tensor:
        return x_B_T_H_W.repeat_interleave(2, dim=-2).repeat_interleave(
            2,
            dim=-1,
        )

    def _build_soft_and_hard_leaf_masks(
        self,
        split_soft_B_T_M: torch.Tensor,
        split_hard_B_T_M: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = split_soft_B_T_M.shape[:2]
        root_h, root_w = self.level_grid_hw[0]
        soft_reach_B_T_H_W = split_soft_B_T_M.new_ones((B, T, root_h, root_w))
        hard_reach_B_T_H_W = torch.ones(
            (B, T, root_h, root_w),
            dtype=torch.bool,
            device=split_hard_B_T_M.device,
        )
        soft_leaf_levels: List[torch.Tensor] = []
        hard_leaf_levels: List[torch.Tensor] = []

        for level in range(self.num_levels - 1):
            grid_h, grid_w = self.level_grid_hw[level]
            start = self.internal_level_offsets[level]
            end = self.internal_level_offsets[level + 1]
            split_soft_B_T_H_W = split_soft_B_T_M[..., start:end].reshape(
                B,
                T,
                grid_h,
                grid_w,
            )
            split_hard_B_T_H_W = split_hard_B_T_M[..., start:end].reshape(
                B,
                T,
                grid_h,
                grid_w,
            )

            soft_leaf_B_T_H_W = soft_reach_B_T_H_W * (
                1.0 - split_soft_B_T_H_W
            )
            hard_leaf_B_T_H_W = hard_reach_B_T_H_W & (
                ~split_hard_B_T_H_W
            )
            soft_leaf_levels.append(soft_leaf_B_T_H_W.flatten(start_dim=2))
            hard_leaf_levels.append(hard_leaf_B_T_H_W.flatten(start_dim=2))

            soft_reach_B_T_H_W = self._replicate_to_children(
                soft_reach_B_T_H_W * split_soft_B_T_H_W
            )
            hard_reach_B_T_H_W = self._replicate_to_children(
                hard_reach_B_T_H_W & split_hard_B_T_H_W
            )

        # The highest-resolution level is leaf-only and needs no split logit.
        soft_leaf_levels.append(soft_reach_B_T_H_W.flatten(start_dim=2))
        hard_leaf_levels.append(hard_reach_B_T_H_W.flatten(start_dim=2))
        soft_leaf_probs_B_T_K = torch.cat(soft_leaf_levels, dim=-1)
        hard_leaf_mask_B_T_K = torch.cat(hard_leaf_levels, dim=-1)
        st_leaf_mask_B_T_K = (
            hard_leaf_mask_B_T_K.to(dtype=soft_leaf_probs_B_T_K.dtype)
            - soft_leaf_probs_B_T_K.detach()
            + soft_leaf_probs_B_T_K
        )
        return soft_leaf_probs_B_T_K, hard_leaf_mask_B_T_K, st_leaf_mask_B_T_K

    def _pack_selected_patches(
        self,
        patch_bank_B_T_K_C_P_P: torch.Tensor,
        hard_leaf_mask_B_T_K: torch.Tensor,
        st_leaf_mask_B_T_K: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, K, C, P, _ = patch_bank_B_T_K_C_P_P.shape
        num_patches_B_T = hard_leaf_mask_B_T_K.sum(dim=-1)
        N = int(num_patches_B_T.max().detach().cpu())

        bank_indices_1_1_K = torch.arange(
            K,
            device=hard_leaf_mask_B_T_K.device,
            dtype=torch.long,
        ).view(1, 1, K)
        invalid_indices_B_T_K = torch.full_like(
            bank_indices_1_1_K.expand(B, T, K),
            K,
        )
        ordered_indices_B_T_K = torch.where(
            hard_leaf_mask_B_T_K,
            bank_indices_1_1_K.expand(B, T, K),
            invalid_indices_B_T_K,
        )
        selected_indices_B_T_N = ordered_indices_B_T_K.sort(dim=-1).values[..., :N]
        token_positions_1_1_N = torch.arange(
            N,
            device=hard_leaf_mask_B_T_K.device,
            dtype=num_patches_B_T.dtype,
        ).view(1, 1, N)
        token_valid_mask_B_T_N = token_positions_1_1_N < num_patches_B_T.unsqueeze(-1)
        patch_bank_indices_B_T_N = selected_indices_B_T_N.masked_fill(
            ~token_valid_mask_B_T_N,
            -1,
        )
        safe_indices_B_T_N = selected_indices_B_T_N.clamp(max=K - 1)

        patch_indices_B_T_N_C_P_P = safe_indices_B_T_N[..., None, None, None].expand(
            B,
            T,
            N,
            C,
            P,
            P,
        )
        selected_patches_B_T_N_C_P_P = torch.gather(
            patch_bank_B_T_K_C_P_P,
            dim=2,
            index=patch_indices_B_T_N_C_P_P,
        )
        st_leaf_weights_B_T_N = torch.gather(
            st_leaf_mask_B_T_K,
            dim=2,
            index=safe_indices_B_T_N,
        )
        valid_weights_B_T_N = token_valid_mask_B_T_N.to(
            dtype=st_leaf_weights_B_T_N.dtype
        )
        st_leaf_weights_B_T_N = st_leaf_weights_B_T_N * valid_weights_B_T_N
        # Keep RGB content un-gated so the policy can place the ST multiplier
        # after its Conv stem; padding still stays numerically empty.
        valid_patches_B_T_N_1_1_1 = token_valid_mask_B_T_N.to(
            dtype=selected_patches_B_T_N_C_P_P.dtype
        )[..., None, None, None]
        patches_B_T_N_C_P_P = selected_patches_B_T_N_C_P_P * (
            valid_patches_B_T_N_1_1_1
        )
        return {
            "patches_B_T_N_C_P_P": patches_B_T_N_C_P_P,
            "token_valid_mask_B_T_N": token_valid_mask_B_T_N,
            "num_patches_B_T": num_patches_B_T,
            "st_leaf_weights_B_T_N": st_leaf_weights_B_T_N,
            # Global patch-bank index: levels low-to-high, then row-major.
            # Padding is explicit instead of aliasing the final bank entry.
            "patch_bank_indices_B_T_N": patch_bank_indices_B_T_N,
        }

    def _build_patch_metadata(
        self,
        selected_indices_B_T_N: torch.Tensor,
        token_valid_mask_B_T_N: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        levels_B_T_N = self.bank_levels_K[selected_indices_B_T_N]
        boxes_xyxy_B_T_N_4 = self.bank_boxes_xyxy_K_4[selected_indices_B_T_N]
        centers_xy_norm_B_T_N_2 = self.bank_centers_xy_norm_K_2[
            selected_indices_B_T_N
        ]
        sizes_hw_norm_B_T_N_2 = self.bank_sizes_hw_norm_K_2[selected_indices_B_T_N]

        levels_B_T_N = levels_B_T_N.masked_fill(~token_valid_mask_B_T_N, -1)
        valid_float_B_T_N_1 = token_valid_mask_B_T_N.unsqueeze(-1).to(
            dtype=boxes_xyxy_B_T_N_4.dtype
        )
        return {
            "levels_B_T_N": levels_B_T_N,
            "boxes_xyxy_B_T_N_4": boxes_xyxy_B_T_N_4 * valid_float_B_T_N_1,
            "centers_xy_norm_B_T_N_2": (
                centers_xy_norm_B_T_N_2 * valid_float_B_T_N_1
            ),
            "sizes_hw_norm_B_T_N_2": (
                sizes_hw_norm_B_T_N_2 * valid_float_B_T_N_1
            ),
        }

    @staticmethod
    def budget_loss(
        soft_leaf_probs_B_T_K: torch.Tensor,
        target_num_patches: Union[int, float],
        min_num_patches: Union[int, float],
        max_num_patches: Union[int, float],
    ) -> torch.Tensor:
        """Penalize leaf-count error after normalizing by its legal range."""
        expected_num_patches_B_T = soft_leaf_probs_B_T_K.sum(dim=-1)
        min_patches = float(min_num_patches)
        max_patches = float(max_num_patches)
        budget_range = max_patches - min_patches
        if budget_range < 0.0:
            raise ValueError("max_num_patches must be at least min_num_patches.")
        if budget_range == 0.0:
            return expected_num_patches_B_T.sum() * 0.0

        normalized_expected_B_T = (
            expected_num_patches_B_T - min_patches
        ) / budget_range
        normalized_target = expected_num_patches_B_T.new_tensor(
            (float(target_num_patches) - min_patches) / budget_range
        )
        return F.mse_loss(
            normalized_expected_B_T,
            normalized_target.expand_as(normalized_expected_B_T),
        )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        cond_B_T_D: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ## 1. Validate and normalize the high-resolution input.
        if x_B_C_T_H_W.ndim != 5:
            raise ValueError("Expected x_B_C_T_H_W with shape (B, C, T, H, W).")
        B, C, T, H, W = x_B_C_T_H_W.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}.")
        if (H, W) != self.input_hw:
            raise ValueError(f"Expected input_hw={self.input_hw}, got {(H, W)}.")
        if x_B_C_T_H_W.dtype == torch.uint8:
            x_float_B_C_T_H_W = x_B_C_T_H_W.float().div(127.5).sub(1.0)
        elif torch.is_floating_point(x_B_C_T_H_W):
            x_float_B_C_T_H_W = x_B_C_T_H_W
        else:
            raise TypeError("Images must be uint8 or floating point tensors.")

        ## 2. Build and patchify the native multi-resolution pyramid.
        image_pyramid = self._build_image_pyramid(x_float_B_C_T_H_W)
        _, patch_bank_B_T_K_C_P_P = self._patchify_pyramid(image_pyramid)

        ## 3. Predict every internal-node split logit in fixed order.
        router_soft_leaf_probs_B_T_K = None
        router_split_hard_B_T_M = None
        random_topology_mask_B_T = torch.zeros(
            (B, T),
            dtype=torch.bool,
            device=x_B_C_T_H_W.device,
        )
        if self.num_internal_nodes == 0:
            split_logits_B_T_M = x_float_B_C_T_H_W.new_empty((B, T, 0))
            split_soft_B_T_M = split_logits_B_T_M
            split_hard_B_T_M = torch.empty(
                (B, T, 0),
                dtype=torch.bool,
                device=x_B_C_T_H_W.device,
            )
        elif self.fix_split_mode is not None:
            # Fixed baselines bypass the global encoder/router while preserving
            # shape expected by diagnostics and downstream policy code.
            split_shape = (B, T, self.num_internal_nodes)
            if self.fixed_split_ratio is not None:
                split_hard_B_T_M = self.fixed_split_hard_1_1_M.expand(
                    split_shape
                )
            elif self.random_split_ratios is None:
                should_split = self.fix_split_mode == "finest"
                split_hard_B_T_M = torch.full(
                    split_shape,
                    should_split,
                    dtype=torch.bool,
                    device=x_B_C_T_H_W.device,
                )
            else:
                if len(self.random_num_splits) == 1:
                    # Preserve the existing single-ratio RNG stream.
                    num_splits_B_1 = torch.full(
                        (B, 1),
                        self.random_num_splits[0],
                        dtype=torch.long,
                        device=x_B_C_T_H_W.device,
                    )
                else:
                    split_counts_L = torch.as_tensor(
                        self.random_num_splits,
                        dtype=torch.long,
                        device=x_B_C_T_H_W.device,
                    )
                    ratio_indices_B = torch.randint(
                        len(self.random_num_splits),
                        (B,),
                        device=x_B_C_T_H_W.device,
                    )
                    num_splits_B_1 = split_counts_L[ratio_indices_B, None]
                # A sample uses one ratio and one tree for all observation frames.
                split_hard_B_T_M = self._sample_random_split_mask(
                    num_splits_B_1
                ).expand(B, T, self.num_internal_nodes)
            split_soft_B_T_M = split_hard_B_T_M.to(
                dtype=x_float_B_C_T_H_W.dtype
            )
            action_hard_B_T_M = (
                split_hard_B_T_M
                if self.split_action_value == 1
                else ~split_hard_B_T_M
            )
            split_logits_B_T_M = action_hard_B_T_M.to(
                dtype=x_float_B_C_T_H_W.dtype
            ).mul(2.0).sub(1.0)
        else:
            global_images_B_C_T_H_W = image_pyramid[self.global_level]
            if self.router_type == "pooled_global":
                global_features_B_T_N_D = self._encode_global_images(
                    global_images_B_C_T_H_W
                )
                router_condition_B_T_D = self._build_router_condition(
                    cond_B_T_D,
                    global_features_B_T_N_D,
                )
                split_logits_B_T_M = self.router(
                    global_features_B_T_N_D,
                    router_condition_B_T_D,
                )
            elif self.router_type == "node_aligned_fpn":
                stage_features = self._encode_global_stages(
                    global_images_B_C_T_H_W
                )
                stage_to_node_level = self._resnet_stage_to_node_level()
                level_features_B_T_C_H_W = {
                    node_level: stage_features[stage]
                    for stage, node_level in stage_to_node_level.items()
                }
                split_logits_B_T_M = self.router(
                    level_features_B_T_C_H_W,
                    stage_features["layer4"],
                    cond_B_T_D,
                )
            else:
                level_features_B_T_D_H_W = self._encode_quadtree_features(
                    global_images_B_C_T_H_W
                )
                split_logits_B_T_M = self.router(
                    level_features_B_T_D_H_W,
                    cond_B_T_D,
                )
            split_soft_B_T_M, split_hard_B_T_M = self._sample_split_decisions(
                split_logits_B_T_M
            )
            router_split_hard_B_T_M = split_hard_B_T_M

            if self.training and self.random_topology_prob > 0.0:
                (
                    router_soft_leaf_probs_B_T_K,
                    router_hard_leaf_mask_B_T_K,
                    _,
                ) = self._build_soft_and_hard_leaf_masks(
                    split_soft_B_T_M,
                    split_hard_B_T_M,
                )
                router_num_patches_B_T = router_hard_leaf_mask_B_T_K.sum(dim=-1)
                num_splits_B_T = (
                    router_num_patches_B_T - self.min_num_patches
                ) // 3
                random_split_hard_B_T_M = self._sample_random_split_mask(
                    num_splits_B_T
                )
                random_topology_mask_B_T = torch.rand(
                    (B, T),
                    device=x_B_C_T_H_W.device,
                ) < self.random_topology_prob
                random_topology_mask_B_T_1 = random_topology_mask_B_T.unsqueeze(-1)
                split_hard_B_T_M = torch.where(
                    random_topology_mask_B_T_1,
                    random_split_hard_B_T_M,
                    split_hard_B_T_M,
                )
                # A random tree is not a router decision. Using its hard mask as
                # the soft value makes the downstream ST gate constant on those frames.
                split_soft_B_T_M = torch.where(
                    random_topology_mask_B_T_1,
                    random_split_hard_B_T_M.to(dtype=split_soft_B_T_M.dtype),
                    split_soft_B_T_M,
                )

        ## 4. Recursively construct q, H, and the straight-through leaf mask m.
        (
            soft_leaf_probs_B_T_K,
            hard_leaf_mask_B_T_K,
            st_leaf_mask_B_T_K,
        ) = self._build_soft_and_hard_leaf_masks(
            split_soft_B_T_M,
            split_hard_B_T_M,
        )

        ## 5. Hard-gather legal leaves and pad each frame to the batch Nmax.
        packed = self._pack_selected_patches(
            patch_bank_B_T_K_C_P_P,
            hard_leaf_mask_B_T_K,
            st_leaf_mask_B_T_K,
        )
        metadata = self._build_patch_metadata(
            packed["patch_bank_indices_B_T_N"].clamp_min(0),
            packed["token_valid_mask_B_T_N"],
        )

        ## 6. Compute the differentiable expected token-budget penalty.
        if (
            self.fix_split_mode is not None
            or self.resolved_target_num_patches is None
            or self.budget_loss_weight == 0.0
        ):
            budget_loss = soft_leaf_probs_B_T_K.sum() * 0.0
        else:
            budget_leaf_probs_B_T_K = soft_leaf_probs_B_T_K
            if self.budget_probability_type == "clean_expected":
                clean_action_probs_B_T_M = torch.sigmoid(split_logits_B_T_M)
                clean_split_probs_B_T_M = (
                    clean_action_probs_B_T_M
                    if self.split_action_value == 1
                    else 1.0 - clean_action_probs_B_T_M
                )
                budget_split_hard_B_T_M = (
                    split_hard_B_T_M
                    if router_split_hard_B_T_M is None
                    else router_split_hard_B_T_M
                )
                budget_leaf_probs_B_T_K, _, _ = (
                    self._build_soft_and_hard_leaf_masks(
                        clean_split_probs_B_T_M,
                        budget_split_hard_B_T_M,
                    )
                )
            elif router_soft_leaf_probs_B_T_K is not None:
                # sampled_relaxed budget remains tied to the router sample,
                # not to a random topology used only by the action policy.
                budget_leaf_probs_B_T_K = router_soft_leaf_probs_B_T_K
            budget_loss = self.budget_loss(
                budget_leaf_probs_B_T_K,
                self.resolved_target_num_patches,
                self.min_num_patches,
                self.max_num_patches,
            ) * self.budget_loss_weight

        out = dict(packed)
        out.update(metadata)
        out["split_logits_B_T_M"] = split_logits_B_T_M
        out["random_topology_mask_B_T"] = random_topology_mask_B_T
        out["budget_loss"] = budget_loss
        return out

    def vis_ar_tokens(
        self,
        foveation_out: Dict[str, torch.Tensor],
        alpha: float = 0.1,
        border_width: int = 1,
    ) -> np.ndarray:
        """Render selected native-resolution patches with level colors and borders."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if border_width < 0:
            raise ValueError("border_width must be non-negative.")

        patches_B_T_N_C_P_P = foveation_out["patches_B_T_N_C_P_P"].detach().cpu()
        token_valid_mask_B_T_N = foveation_out[
            "token_valid_mask_B_T_N"
        ].detach().cpu()
        levels_B_T_N = foveation_out["levels_B_T_N"].detach().cpu()
        boxes_xyxy_B_T_N_4 = foveation_out["boxes_xyxy_B_T_N_4"].detach().cpu()
        B, T, N, C, _, _ = patches_B_T_N_C_P_P.shape
        if C != 3:
            raise ValueError("vis_ar_tokens currently expects RGB patches.")

        palette_L_3 = np.asarray(
            [
                [0, 114, 178],
                [230, 159, 0],
                [0, 158, 115],
                [204, 121, 167],
                [213, 94, 0],
                [86, 180, 233],
            ],
            dtype=np.float32,
        ) / 255.0
        input_h, input_w = self.input_hw
        vis_B_T_H_W_C = np.zeros((B, T, input_h, input_w, 3), dtype=np.float32)

        patches_B_T_N_H_W_C = patches_B_T_N_C_P_P.permute(0, 1, 2, 4, 5, 3)
        patches_B_T_N_H_W_C = patches_B_T_N_H_W_C.float().add(1.0).mul(0.5)
        patches_B_T_N_H_W_C = patches_B_T_N_H_W_C.clamp(0.0, 1.0).numpy()
        valid_B_T_N = token_valid_mask_B_T_N.numpy()
        levels_B_T_N_np = levels_B_T_N.numpy()
        boxes_B_T_N_4 = boxes_xyxy_B_T_N_4.round().to(torch.long).numpy()

        for batch_idx in range(B):
            for time_idx in range(T):
                for token_idx in range(N):
                    if not valid_B_T_N[batch_idx, time_idx, token_idx]:
                        continue
                    level = int(levels_B_T_N_np[batch_idx, time_idx, token_idx])
                    x0, y0, x1, y1 = boxes_B_T_N_4[
                        batch_idx,
                        time_idx,
                        token_idx,
                    ]
                    scale = 2 ** (self.num_levels - 1 - level)
                    patch_H_W_C = patches_B_T_N_H_W_C[
                        batch_idx,
                        time_idx,
                        token_idx,
                    ]
                    resized_patch_H_W_C = np.repeat(
                        np.repeat(patch_H_W_C, scale, axis=0),
                        scale,
                        axis=1,
                    )
                    color_3 = palette_L_3[level % len(palette_L_3)]
                    tinted_patch_H_W_C = (
                        (1.0 - alpha) * resized_patch_H_W_C
                        + alpha * color_3.reshape(1, 1, 3)
                    )
                    vis_B_T_H_W_C[batch_idx, time_idx, y0:y1, x0:x1] = (
                        tinted_patch_H_W_C
                    )

                # Draw borders after all patches have been reconstructed so a
                # neighboring patch cannot overwrite an already-drawn edge.
                for token_idx in range(N):
                    if (
                        border_width == 0
                        or not valid_B_T_N[batch_idx, time_idx, token_idx]
                    ):
                        continue
                    x0, y0, x1, y1 = boxes_B_T_N_4[
                        batch_idx,
                        time_idx,
                        token_idx,
                    ]
                    line_width = min(border_width, x1 - x0, y1 - y0)
                    frame_H_W_C = vis_B_T_H_W_C[batch_idx, time_idx]
                    frame_H_W_C[y0:y0 + line_width, x0:x1] = 1.0
                    frame_H_W_C[y0:y1, x0:x0 + line_width] = 1.0
                    if x1 == input_w:
                        frame_H_W_C[y0:y1, x1 - line_width:x1] = 1.0
                    if y1 == input_h:
                        frame_H_W_C[y1 - line_width:y1, x0:x1] = 1.0

        return np.rint(vis_B_T_H_W_C * 255.0).clip(0, 255).astype(np.uint8)
