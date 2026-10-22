"""Diffusion policy consuming the mixed-resolution output of FoveationNet."""

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import DictConfig, OmegaConf

from diffusion_policy.model.common.normalizer import (
    EmptyNormalizer,
    LinearNormalizer,
)
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)
from diffusion_policy.model.roi.foveation_net import FoveationNet
from diffusion_policy.model.roi.multi_view_foveation_net import MultiViewFoveationNet
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class FoveationObsEncoder(nn.Module):
    """Turn padded native-pixel leaves into one fixed-size latent per frame."""

    def __init__(
        self,
        in_channels: int,
        input_hw: Tuple[int, int],
        patch_size: int,
        num_levels: int,
        position_encoding_types: Sequence[str] = ("apt_interpolated",),
        pooling_type: str = "mean",
        patch_embedding_type: str = "linear",
        st_gate_location: str = "patch_input",
        patch_level_adapter_channels: Optional[int] = None,
        visual_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        visual_encoder_init: str = "copy",
    ) -> None:
        """Configure patch encoding and optional independent Transformer initialization."""
        super().__init__()
        if visual_encoder_init not in {"copy", "trunc_normal"}:
            raise ValueError("visual_encoder_init must be 'copy' or 'trunc_normal'.")
        self.visual_encoder_init = visual_encoder_init
        self.in_channels = int(in_channels)
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        self.patch_size = int(patch_size)
        self.num_levels = int(num_levels)
        if isinstance(position_encoding_types, str):
            raise TypeError("position_encoding_types must be a sequence of strings.")
        self.position_encoding_types = tuple(
            str(encoding_type) for encoding_type in position_encoding_types
        )
        self.pooling_type = str(pooling_type)
        self.patch_embedding_type = str(patch_embedding_type)
        self.st_gate_location = str(st_gate_location)
        self.patch_level_adapter_channels = (
            None
            if patch_level_adapter_channels is None
            else int(patch_level_adapter_channels)
        )
        self.visual_dim = int(visual_dim)
        if self.in_channels <= 0 or self.patch_size <= 0 or self.num_levels <= 0:
            raise ValueError("in_channels, patch_size and num_levels must be positive.")
        if len(self.input_hw) != 2 or min(self.input_hw) <= 0:
            raise ValueError("input_hw must contain two positive integers.")
        supported_position_encodings = {"apt_interpolated", "level_embedding"}
        unknown_position_encodings = (
            set(self.position_encoding_types) - supported_position_encodings
        )
        if unknown_position_encodings:
            raise ValueError(
                "Unsupported position encodings: "
                f"{sorted(unknown_position_encodings)}."
            )
        if len(set(self.position_encoding_types)) != len(self.position_encoding_types):
            raise ValueError("position_encoding_types cannot contain duplicates.")
        if self.pooling_type not in {"mean", "cls_token"}:
            raise ValueError("pooling_type must be 'mean' or 'cls_token'.")
        if self.patch_embedding_type not in {"linear", "conv"}:
            raise ValueError("patch_embedding_type must be 'linear' or 'conv'.")
        if self.st_gate_location not in {"patch_input", "content_embedding"}:
            raise ValueError(
                "st_gate_location must be 'patch_input' or 'content_embedding'."
            )
        if (
            self.patch_level_adapter_channels is not None
            and self.patch_level_adapter_channels <= 0
        ):
            raise ValueError("patch_level_adapter_channels must be positive or None.")
        if (
            self.patch_level_adapter_channels is not None
            and self.patch_embedding_type != "conv"
        ):
            raise ValueError("Patch-level Conv adapters require patch_embedding_type='conv'.")
        if self.visual_dim <= 0 or int(num_layers) <= 0 or int(num_heads) <= 0:
            raise ValueError("visual_dim, num_layers and num_heads must be positive.")
        if self.visual_dim % int(num_heads) != 0:
            raise ValueError("visual_dim must be divisible by num_heads.")
        if float(mlp_ratio) <= 0.0 or not (0.0 <= float(dropout) < 1.0):
            raise ValueError("mlp_ratio must be positive and dropout must be in [0, 1).")

        input_h, input_w = self.input_hw
        quadtree_stride = self.patch_size * (2 ** (self.num_levels - 1))
        if input_h % quadtree_stride != 0 or input_w % quadtree_stride != 0:
            raise ValueError(
                "Each input dimension must be divisible by "
                "patch_size * 2**(num_levels - 1)."
            )
        self.level_grid_hw = tuple(
            (
                input_h // (self.patch_size * 2 ** (self.num_levels - 1 - level)),
                input_w // (self.patch_size * 2 ** (self.num_levels - 1 - level)),
            )
            for level in range(self.num_levels)
        )
        self.num_bank_patches = sum(
            grid_h * grid_w for grid_h, grid_w in self.level_grid_hw
        )

        patch_dim = self.in_channels * self.patch_size * self.patch_size
        if self.patch_embedding_type == "linear":
            # Keep this module and its state-dict keys compatible with old checkpoints.
            self.patch_projection = nn.Linear(patch_dim, self.visual_dim)
        else:
            conv_patch_hw = (self.patch_size + 3) // 4
            # Leaf-local Conv stem adds locality without computing dense image features.
            self.patch_projection = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(
                    32,
                    48,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, 48),
                nn.GELU(),
                nn.Flatten(),
                nn.Linear(48 * conv_patch_hw * conv_patch_hw, self.visual_dim),
            )
        self.patch_level_adapters = None
        if self.patch_level_adapter_channels is not None:
            # The shared stem learns common local features; these small residual
            # branches specialize its (BTN, 48, P/4, P/4) output by quadtree level.
            self.patch_level_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            48,
                            self.patch_level_adapter_channels,
                            kernel_size=1,
                            bias=False,
                        ),
                        nn.GELU(),
                        nn.Conv2d(
                            self.patch_level_adapter_channels,
                            48,
                            kernel_size=1,
                            bias=False,
                        ),
                    )
                    for _ in range(self.num_levels)
                ]
            )
            for adapter in self.patch_level_adapters:
                # Start from the shared-Conv baseline while allowing each level
                # to learn its own residual correction.
                nn.init.zeros_(adapter[-1].weight)
        self.fine_pos_embed_1_D_Hf_Wf = None
        if "apt_interpolated" in self.position_encoding_types:
            fine_grid_h, fine_grid_w = self.level_grid_hw[-1]
            self.fine_pos_embed_1_D_Hf_Wf = nn.Parameter(
                torch.empty(1, self.visual_dim, fine_grid_h, fine_grid_w)
            )
            nn.init.trunc_normal_(self.fine_pos_embed_1_D_Hf_Wf, std=0.02)
        self.level_embed_L_D = None
        if "level_embedding" in self.position_encoding_types:
            self.level_embed_L_D = nn.Parameter(
                torch.empty(self.num_levels, self.visual_dim)
            )
            nn.init.trunc_normal_(self.level_embed_L_D, std=0.02)
        self.cls_token_1_1_D = None
        if self.pooling_type == "cls_token":
            self.cls_token_1_1_D = nn.Parameter(
                torch.empty(1, 1, self.visual_dim)
            )
            nn.init.trunc_normal_(self.cls_token_1_1_D, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.visual_dim,
            nhead=int(num_heads),
            dim_feedforward=int(round(self.visual_dim * mlp_ratio)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_layers),
        )
        self.final_norm = nn.LayerNorm(self.visual_dim)
        self.output_dim = self.visual_dim
        if self.visual_encoder_init == "trunc_normal":
            # Construction is on CPU; preserve the RNG stream used by the action model.
            with torch.random.fork_rng(devices=[]):
                self._init_transformer_weights()

    def _init_transformer_weights(self) -> None:
        """Independently initialize cloned layers without changing patch embeddings."""
        for layer in self.token_encoder.layers:
            # Q/K/V share a (3D, D) parameter, which an nn.Linear traversal misses.
            nn.init.trunc_normal_(layer.self_attn.in_proj_weight, std=0.02)
            nn.init.zeros_(layer.self_attn.in_proj_bias)
            for projection in (layer.self_attn.out_proj, layer.linear1, layer.linear2):
                nn.init.trunc_normal_(projection.weight, std=0.02)
                nn.init.zeros_(projection.bias)
            for norm in (layer.norm1, layer.norm2):
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)

    def _build_multiscale_position_bank(self) -> torch.Tensor:
        """Interpolate one learned fine-grid map to every quadtree level."""
        fine_grid_hw = self.level_grid_hw[-1]
        level_positions = []
        for grid_hw in self.level_grid_hw:
            if grid_hw == fine_grid_hw:
                pos_1_D_H_W = self.fine_pos_embed_1_D_Hf_Wf
            else:
                pos_1_D_H_W = F.interpolate(
                    self.fine_pos_embed_1_D_Hf_Wf,
                    size=grid_hw,
                    mode="bicubic",
                    align_corners=False,
                    antialias=False,
                )
            grid_h, grid_w = grid_hw
            # NCHW -> row-major (row, col, channel), matching FoveationNet.
            level_positions.append(
                pos_1_D_H_W.permute(0, 2, 3, 1).reshape(
                    grid_h * grid_w,
                    self.visual_dim,
                )
            )
        position_bank_K_D = torch.cat(level_positions, dim=0)
        if position_bank_K_D.shape != (self.num_bank_patches, self.visual_dim):
            raise RuntimeError(
                "APT position bank geometry mismatch: "
                f"input_hw={self.input_hw}, patch_size={self.patch_size}, "
                f"num_levels={self.num_levels}, grids={self.level_grid_hw}."
            )
        return position_bank_K_D

    def _gather_position_embeddings(
        self,
        position_bank_K_D: torch.Tensor,
        patch_bank_indices_B_T_N: torch.Tensor,
        token_valid_mask_B_T_N: torch.Tensor,
    ) -> torch.Tensor:
        if patch_bank_indices_B_T_N.shape != token_valid_mask_B_T_N.shape:
            raise ValueError("Patch bank indices and valid mask must have equal shapes.")
        if patch_bank_indices_B_T_N.dtype != torch.long:
            raise TypeError("patch_bank_indices_B_T_N must have dtype torch.long.")
        safe_indices_B_T_N = torch.where(
            token_valid_mask_B_T_N,
            patch_bank_indices_B_T_N,
            torch.zeros_like(patch_bank_indices_B_T_N),
        )
        # F.embedding keeps gradients to the position bank and rejects an
        # out-of-range index at a valid token instead of silently clamping it.
        position_B_T_N_D = F.embedding(
            safe_indices_B_T_N,
            position_bank_K_D,
        )
        return position_B_T_N_D * token_valid_mask_B_T_N.unsqueeze(-1).to(
            position_B_T_N_D.dtype
        )

    def _gather_level_embeddings(
        self,
        levels_B_T_N: torch.Tensor,
        token_valid_mask_B_T_N: torch.Tensor,
    ) -> torch.Tensor:
        if levels_B_T_N.shape != token_valid_mask_B_T_N.shape:
            raise ValueError("Levels and valid mask must have equal shapes.")
        if levels_B_T_N.dtype != torch.long:
            raise TypeError("levels_B_T_N must have dtype torch.long.")
        safe_levels_B_T_N = torch.where(
            token_valid_mask_B_T_N,
            levels_B_T_N,
            torch.zeros_like(levels_B_T_N),
        )
        # Valid out-of-range levels remain untouched so F.embedding reports
        # inconsistent FoveationNet metadata instead of silently clamping it.
        level_B_T_N_D = F.embedding(
            safe_levels_B_T_N,
            self.level_embed_L_D,
        )
        return level_B_T_N_D * token_valid_mask_B_T_N.unsqueeze(-1).to(
            level_B_T_N_D.dtype
        )

    def forward(self, foveation_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        patches_B_T_N_C_P_P = foveation_out["patches_B_T_N_C_P_P"]
        token_valid_mask_B_T_N = foveation_out["token_valid_mask_B_T_N"].bool()
        st_leaf_weights_B_T_N = foveation_out["st_leaf_weights_B_T_N"]
        patch_bank_indices_B_T_N = None
        if "apt_interpolated" in self.position_encoding_types:
            patch_bank_indices_B_T_N = foveation_out[
                "patch_bank_indices_B_T_N"
            ]
        levels_B_T_N = None
        if (
            "level_embedding" in self.position_encoding_types
            or self.patch_level_adapters is not None
        ):
            levels_B_T_N = foveation_out["levels_B_T_N"]

        if patches_B_T_N_C_P_P.ndim != 6:
            raise ValueError("Foveation patches must have shape (B, T, N, C, P, P).")
        B, T, N, C, P_h, P_w = patches_B_T_N_C_P_P.shape
        if N <= 0:
            raise ValueError("Foveation output must contain at least one padded token.")
        if (C, P_h, P_w) != (self.in_channels, self.patch_size, self.patch_size):
            raise ValueError(
                "Foveation patch shape does not match the observation encoder: "
                f"got {(C, P_h, P_w)}, expected "
                f"{(self.in_channels, self.patch_size, self.patch_size)}."
            )
        if token_valid_mask_B_T_N.shape != (B, T, N):
            raise ValueError("token_valid_mask has an incompatible shape.")
        if st_leaf_weights_B_T_N.shape != (B, T, N):
            raise ValueError("st_leaf_weights has an incompatible shape.")
        if (
            patch_bank_indices_B_T_N is not None
            and patch_bank_indices_B_T_N.shape != (B, T, N)
        ):
            raise ValueError("patch_bank_indices has an incompatible shape.")
        if levels_B_T_N is not None and levels_B_T_N.shape != (B, T, N):
            raise ValueError("levels_B_T_N has an incompatible shape.")
        if not torch.is_floating_point(patches_B_T_N_C_P_P):
            patches_B_T_N_C_P_P = patches_B_T_N_C_P_P.float()
        projection_param = next(self.patch_projection.parameters())
        patches_B_T_N_C_P_P = patches_B_T_N_C_P_P.to(
            device=projection_param.device,
            dtype=projection_param.dtype,
        )
        token_valid_mask_B_T_N = token_valid_mask_B_T_N.to(
            device=patches_B_T_N_C_P_P.device
        )
        st_leaf_weights_B_T_N = st_leaf_weights_B_T_N.to(
            device=patches_B_T_N_C_P_P.device,
            dtype=patches_B_T_N_C_P_P.dtype,
        )
        if patch_bank_indices_B_T_N is not None:
            patch_bank_indices_B_T_N = patch_bank_indices_B_T_N.to(
                device=patches_B_T_N_C_P_P.device
            )
        if levels_B_T_N is not None:
            levels_B_T_N = levels_B_T_N.to(device=patches_B_T_N_C_P_P.device)

        # Keep the dynamic N dimension local to this encoder. Padding is never
        # included as an attention key or in the final pooling.
        if self.st_gate_location == "patch_input":
            patches_B_T_N_C_P_P = patches_B_T_N_C_P_P * (
                st_leaf_weights_B_T_N[..., None, None, None]
            )
        if self.patch_embedding_type == "linear":
            patches_BTN_Din = patches_B_T_N_C_P_P.reshape(B * T * N, -1)
        else:
            patches_BTN_Din = patches_B_T_N_C_P_P.reshape(
                B * T * N,
                C,
                P_h,
                P_w,
            )
        if self.patch_level_adapters is None:
            pixel_BTN_D = self.patch_projection(patches_BTN_Din)
        else:
            shared_BTN_C_H_W = self.patch_projection[:-2](patches_BTN_Din)
            adapter_update_BTN_C_H_W = torch.zeros_like(shared_BTN_C_H_W)
            levels_BTN = levels_B_T_N.reshape(B * T * N)
            valid_BTN = token_valid_mask_B_T_N.reshape(B * T * N)
            for level, adapter in enumerate(self.patch_level_adapters):
                level_indices_M = torch.nonzero(
                    valid_BTN & levels_BTN.eq(level),
                    as_tuple=False,
                ).flatten()
                if level_indices_M.numel() == 0:
                    continue
                level_features_M_C_H_W = shared_BTN_C_H_W.index_select(
                    0, level_indices_M
                )
                adapter_update_BTN_C_H_W = adapter_update_BTN_C_H_W.index_add(
                    0,
                    level_indices_M,
                    adapter(level_features_M_C_H_W),
                )
            adapted_BTN_C_H_W = shared_BTN_C_H_W + adapter_update_BTN_C_H_W
            pixel_BTN_D = self.patch_projection[-1](adapted_BTN_C_H_W.flatten(1))
        pixel_BT_N_D = pixel_BTN_D.reshape(B * T, N, self.visual_dim)
        if self.st_gate_location == "content_embedding":
            # Gate content before adding spatial/level embeddings so token
            # LayerNorm cannot remove one scalar shared by the whole token.
            st_leaf_weights_BT_N_1 = st_leaf_weights_B_T_N.reshape(
                B * T, N, 1
            ).to(dtype=pixel_BT_N_D.dtype)
            pixel_BT_N_D = pixel_BT_N_D * st_leaf_weights_BT_N_1
        tokens_BT_N_D = pixel_BT_N_D
        if "apt_interpolated" in self.position_encoding_types:
            position_bank_K_D = self._build_multiscale_position_bank()
            position_BT_N_D = self._gather_position_embeddings(
                position_bank_K_D,
                patch_bank_indices_B_T_N,
                token_valid_mask_B_T_N,
            ).to(dtype=pixel_BT_N_D.dtype).reshape(
                B * T, N, self.visual_dim
            )
            tokens_BT_N_D = tokens_BT_N_D + position_BT_N_D
        if "level_embedding" in self.position_encoding_types:
            level_BT_N_D = self._gather_level_embeddings(
                levels_B_T_N,
                token_valid_mask_B_T_N,
            ).to(dtype=pixel_BT_N_D.dtype).reshape(
                B * T, N, self.visual_dim
            )
            tokens_BT_N_D = tokens_BT_N_D + level_BT_N_D

        valid_BT_N = token_valid_mask_B_T_N.reshape(B * T, N)
        attention_padding_BT_N = ~valid_BT_N
        if self.pooling_type == "cls_token":
            # CLS is non-spatial, so it has no APT position embedding. It is
            # always a valid token and attends only to valid patch tokens.
            cls_token_BT_1_D = self.cls_token_1_1_D.expand(B * T, -1, -1)
            tokens_BT_S_D = torch.cat([cls_token_BT_1_D, tokens_BT_N_D], dim=1)
            attention_padding_BT_S = torch.cat(
                [
                    torch.zeros(
                        B * T,
                        1,
                        dtype=torch.bool,
                        device=attention_padding_BT_N.device,
                    ),
                    attention_padding_BT_N,
                ],
                dim=1,
            )
        else:
            all_padding_BT = attention_padding_BT_N.all(dim=1)
            if all_padding_BT.any():
                # FoveationNet normally guarantees at least one leaf; this keeps
                # malformed/custom outputs from producing all-masked NaNs.
                attention_padding_BT_N[all_padding_BT, 0] = False
            tokens_BT_S_D = tokens_BT_N_D
            attention_padding_BT_S = attention_padding_BT_N
        tokens_BT_S_D = self.token_encoder(
            tokens_BT_S_D,
            src_key_padding_mask=attention_padding_BT_S,
        )
        tokens_BT_S_D = self.final_norm(tokens_BT_S_D)
        if self.pooling_type == "cls_token":
            return tokens_BT_S_D[:, 0].reshape(B, T, self.visual_dim)

        valid_float_BT_N_1 = valid_BT_N.unsqueeze(-1).to(tokens_BT_S_D.dtype)
        pooled_BT_D = (tokens_BT_S_D * valid_float_BT_N_1).sum(dim=1)
        pooled_BT_D = pooled_BT_D / valid_float_BT_N_1.sum(dim=1).clamp_min(1.0)
        return pooled_BT_D.reshape(B, T, self.visual_dim)

    def output_shape(self) -> Tuple[int]:
        return (self.visual_dim,)


class FoveationDPTransformerPolicy(BaseImagePolicy):
    """Diffusion policy with a mixed-resolution visual observation path."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        foveation_net: Optional[Union[nn.Module, dict]] = None,
        visual_dim: int = 256,
        visual_encoder_layers: int = 2,
        visual_encoder_heads: int = 8,
        visual_encoder_mlp_ratio: float = 4.0,
        visual_encoder_dropout: float = 0.0,
        position_encoding_types: Sequence[str] = ("apt_interpolated",),
        pooling_type: str = "mean",
        patch_embedding_type: str = "linear",
        st_gate_location: str = "patch_input",
        patch_level_adapter_channels: Optional[int] = None,
        n_layer: int = 8,
        n_cond_layers: int = 0,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        time_as_cond: bool = True,
        obs_as_cond: bool = True,
        pred_action_steps_only: bool = False,
        router_temperature: Optional[float] = None,
        router_condition_key: Optional[Union[str, Sequence[str]]] = None,
        router_loss_type: str = "st",
        paired_loss_weight: float = 1.0,
        temporal_adaptive: bool = False,
        temporal_loss_weight: float = 1.0,
        paired_node_sampling: str = "reachable",
        mask_policy_state: bool = False,
        visual_encoder_init: str = "copy",
        render_image_orientation: str = "raw",
        **kwargs,
    ) -> None:
        """Configure routing losses and optional low-dimensional action-input masking."""
        super().__init__()
        if render_image_orientation not in ("raw", "rot180"):
            raise ValueError("render_image_orientation must be 'raw' or 'rot180'")
        # Plain configuration only: preserve legacy checkpoint state-dict keys.
        self._render_image_orientation = render_image_orientation
        if router_loss_type not in {"st", "paired"}:
            raise ValueError("router_loss_type must be 'st' or 'paired'.")
        if paired_loss_weight <= 0.0:
            raise ValueError("paired_loss_weight must be positive.")
        self.router_loss_type = router_loss_type
        self.paired_loss_weight = float(paired_loss_weight)
        if paired_node_sampling != "reachable":
            raise ValueError(
                "paired_node_sampling only accepts 'reachable'; 'single_level' was removed. "
                "Override old checkpoint configs with paired_node_sampling=reachable."
            )
        if not temporal_loss_weight >= 0.0:
            raise ValueError("temporal_loss_weight must be non-negative.")
        if temporal_adaptive and router_loss_type != "paired":
            raise ValueError("temporal_adaptive requires router_loss_type=paired.")
        self.temporal_adaptive = bool(temporal_adaptive)
        self.temporal_loss_weight = float(temporal_loss_weight)
        self.paired_node_sampling = paired_node_sampling
        if "position_encoding_type" in kwargs:
            raise ValueError(
                "position_encoding_type was renamed to position_encoding_types; "
                "pass a list such as [apt_interpolated, level_embedding]."
            )
        if foveation_net is None and "foveation" in kwargs:
            foveation_net = kwargs.pop("foveation")
        shape_meta = (
            OmegaConf.to_container(shape_meta, resolve=True)
            if isinstance(shape_meta, DictConfig)
            else copy.deepcopy(shape_meta)
        )
        obs_shape_meta = shape_meta["obs"]
        action_shape = tuple(shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise ValueError("The action shape must be one-dimensional.")
        self.obs_shape_meta = obs_shape_meta
        self.action_dim = int(action_shape[0])
        self.rgb_keys: List[str] = []
        self.low_dim_keys: List[str] = []
        self.low_dim_dims: List[int] = []
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            shape = tuple(attr["shape"])
            if obs_type == "rgb":
                if len(shape) != 3:
                    raise ValueError(f"RGB observation {key!r} must have shape (C,H,W).")
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.low_dim_keys.append(key)
                self.low_dim_dims.append(math.prod(shape))
            else:
                raise ValueError(f"Unsupported observation type {obs_type!r} for {key!r}.")

        self.router_condition_key = router_condition_key
        self.router_condition_keys = (
            [] if router_condition_key is None else
            [router_condition_key] if isinstance(router_condition_key, str) else list(router_condition_key)
        )
        router_condition_input_dim = None
        if self.router_condition_keys:
            if not self.rgb_keys:
                raise ValueError("router_condition_key requires at least one RGB observation.")
            router_condition_input_dim = 0
            for key in self.router_condition_keys:
                if key not in obs_shape_meta:
                    raise ValueError(f"Router condition {key!r} is missing from shape_meta.obs.")
                condition_meta = obs_shape_meta[key]
                if condition_meta.get("type", "low_dim") != "low_dim":
                    raise ValueError("router_condition_key must refer to low-dim observations.")
                router_condition_input_dim += math.prod(condition_meta["shape"])

        routing_template = None
        self.obs_encoders = nn.ModuleDict()
        if self.rgb_keys:
            rgb_shape = tuple(obs_shape_meta[self.rgb_keys[0]]["shape"])
            for key in self.rgb_keys[1:]:
                if tuple(obs_shape_meta[key]["shape"]) != rgb_shape:
                    raise ValueError(
                        "All RGB observations must share C/H/W and foveation geometry; "
                        f"{self.rgb_keys[0]!r} and {key!r} differ."
                    )
            routing_template = self._instantiate_foveation_net(
                foveation_net,
                in_channels=rgb_shape[0],
                input_hw=rgb_shape[1:],
            )
            global_encoder = getattr(routing_template, "global_encoder", None)
            encoder_is_frozen = bool(
                getattr(routing_template, "freeze_global_encoder", True)
                or getattr(routing_template, "fix_split_mode", None) is not None
            )
            encoder_type = str(
                getattr(routing_template, "global_encoder_type", "")
            )
            if encoder_type.startswith("vjepa") and not encoder_is_frozen:
                raise ValueError("V-JEPA global encoders must remain frozen.")
            if global_encoder is not None and encoder_is_frozen:
                if any(param.requires_grad for param in global_encoder.parameters()):
                    raise ValueError("The configured global encoder must be frozen.")
                global_encoder.eval()
            for key in self.rgb_keys:
                encoder = FoveationObsEncoder(
                    in_channels=rgb_shape[0],
                    input_hw=rgb_shape[1:],
                    patch_size=int(routing_template.patch_size),
                    num_levels=int(routing_template.num_levels),
                    position_encoding_types=position_encoding_types,
                    pooling_type=pooling_type,
                    patch_embedding_type=patch_embedding_type,
                    st_gate_location=st_gate_location,
                    patch_level_adapter_channels=patch_level_adapter_channels,
                    visual_dim=visual_dim,
                    num_layers=visual_encoder_layers,
                    num_heads=visual_encoder_heads,
                    mlp_ratio=visual_encoder_mlp_ratio,
                    dropout=visual_encoder_dropout,
                    visual_encoder_init=visual_encoder_init,
                )
                foveation_grids = tuple(
                    tuple(grid_hw) for grid_hw in routing_template.level_grid_hw
                )
                if encoder.level_grid_hw != foveation_grids:
                    raise ValueError(
                        "Observation encoder and FoveationNet patch grids differ: "
                        f"encoder={encoder.level_grid_hw}, "
                        f"foveation={foveation_grids}."
                    )
                if encoder.num_bank_patches != int(
                    routing_template.num_bank_patches
                ):
                    raise ValueError(
                        "Observation encoder and FoveationNet bank sizes differ."
                    )
                self.obs_encoders[key] = encoder

        self.router_condition_projection: Optional[nn.Linear] = None
        if router_condition_input_dim is not None:
            self.router_condition_projection = nn.Linear(
                router_condition_input_dim,
                int(routing_template.router_cond_dim),
            )
            if (
                getattr(routing_template, "freeze_router", False)
                or getattr(routing_template, "fix_split_mode", None) is not None
            ):
                self.router_condition_projection.requires_grad_(False)

        self.foveation_net = (
            MultiViewFoveationNet(self.rgb_keys, routing_template)
            if routing_template is not None else None
        )

        self.low_dim_dim = sum(self.low_dim_dims)
        self.obs_feature_dim = (
            len(self.rgb_keys) * int(visual_dim) + self.low_dim_dim
        )
        if self.obs_feature_dim <= 0:
            raise ValueError("At least one RGB or low-dimensional observation is required.")

        self.horizon = int(horizon)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        if min(self.horizon, self.n_action_steps, self.n_obs_steps) <= 0:
            raise ValueError("horizon, n_action_steps and n_obs_steps must be positive.")
        if self.n_obs_steps > self.horizon or self.n_action_steps > self.horizon:
            raise ValueError("n_obs_steps and n_action_steps cannot exceed horizon.")
        self.obs_as_cond = bool(obs_as_cond)
        self.mask_policy_state = bool(mask_policy_state)
        self.pred_action_steps_only = bool(pred_action_steps_only)
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(noise_scheduler.config.num_train_timesteps)
        )
        input_dim = self.action_dim if self.obs_as_cond else self.action_dim + self.obs_feature_dim
        cond_dim = self.obs_feature_dim if self.obs_as_cond else 0
        self.model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=input_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_cond_layers=n_cond_layers,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=self.obs_as_cond,
            **kwargs,
        )
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if self.obs_as_cond else self.obs_feature_dim,
            max_n_obs_steps=self.n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self._ensure_obs_normalizers()
        self.router_temperature = router_temperature
        self.extra_kwargs = kwargs
        self.infer_frame_idx = 0
        self.cache_action = None
        self._obs_encoder_frozen = False
        self._action_path_frozen = False
        self._downstream_finetune = False
        self._staged_training = False

        if router_temperature is not None:
            self.set_router_temperature(router_temperature)

    @staticmethod
    def _instantiate_foveation_net(
        foveation_net: Optional[Union[nn.Module, dict]],
        in_channels: int,
        input_hw: Tuple[int, int],
    ) -> FoveationNet:
        if foveation_net is None:
            return FoveationNet(in_channels=in_channels, input_hw=input_hw)
        if isinstance(foveation_net, (dict, DictConfig)):
            foveation_net = hydra.utils.instantiate(foveation_net)
        if not isinstance(foveation_net, nn.Module):
            raise TypeError("foveation_net must be an nn.Module or Hydra config.")
        if int(getattr(foveation_net, "in_channels", in_channels)) != in_channels:
            raise ValueError("FoveationNet channel count does not match shape_meta.")
        if tuple(getattr(foveation_net, "input_hw", input_hw)) != tuple(input_hw):
            raise ValueError("FoveationNet input_hw does not match shape_meta.")
        return foveation_net

    def _ensure_obs_normalizers(self) -> None:
        for key in list(self.obs_shape_meta.keys()) + ["action"]:
            if key not in self.normalizer.params_dict:
                self.normalizer[key] = EmptyNormalizer.create_identity()

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())
        self._ensure_obs_normalizers()

    def set_router_temperature(self, router_temp: float) -> None:
        if float(router_temp) <= 0.0:
            raise ValueError("router_temp must be positive.")
        if self.foveation_net is not None:
            self.foveation_net.set_router_temperature(float(router_temp))
        self.router_temperature = float(router_temp)

    def set_target_num_patches(
        self,
        target_num_patches: Optional[Union[int, float]],
    ) -> Optional[float]:
        """Update the foveation budget target and return its leaf count."""
        if self.foveation_net is None:
            return None
        return self.foveation_net.set_target_num_patches(target_num_patches)

    @property
    def render_image_orientation(self):
        """Return the static-camera input orientation used by preprocessing."""
        return self._render_image_orientation

    def _prepare_rgb(self, key: str, image_B_T_C_H_W: torch.Tensor) -> torch.Tensor:
        # Rotate spatial axes before either dtype path, without mutating observations.
        if key == "image" and self.render_image_orientation == "rot180":
            image_B_T_C_H_W = image_B_T_C_H_W.flip((-2, -1))
        if image_B_T_C_H_W.dtype == torch.uint8:
            return image_B_T_C_H_W
        # Dataset normalizers provide the floating [-1, 1] convention expected
        # by FoveationNet; avoid a per-batch device synchronization here.
        image_B_T_C_H_W = self.normalizer[key].normalize(image_B_T_C_H_W)
        if not torch.is_floating_point(image_B_T_C_H_W):
            image_B_T_C_H_W = image_B_T_C_H_W.float()
        elif image_B_T_C_H_W.dtype != torch.float32:
            image_B_T_C_H_W = image_B_T_C_H_W.float()
        return image_B_T_C_H_W

    @torch.no_grad()
    def get_foveation_outputs(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return per-RGB routing outputs using predict_action's input convention."""
        # Skip visual token encoding and action diffusion for offline diagnostics.
        _, _, aux = self._route_observations(obs_dict, return_foveation_outputs=True)
        return aux.get("foveation_outputs", {})

    def _encode_observations(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_foveation_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        routing_by_view, normalized_low_dim_by_key, aux = self._route_observations(
            obs_dict, return_foveation_outputs=return_foveation_outputs
        )
        features_B_T_D = self._encode_routed_observations(
            routing_by_view, normalized_low_dim_by_key
        )
        return features_B_T_D, aux

    def _route_observations(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_foveation_outputs: bool = False,
        return_routing_state: bool = False,
    ):
        # Keep routing registered with DDP, but build no routing graph in finetuning.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self._downstream_finetune):
            return self._route_observations_impl(
                obs_dict, return_foveation_outputs, return_routing_state
            )

    def _route_observations_impl(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_foveation_outputs: bool = False,
        return_routing_state: bool = False,
    ):
        if not obs_dict:
            raise ValueError("obs_dict cannot be empty.")
        normalized_low_dim_by_key = {
            key: self.normalizer[key].normalize(obs_dict[key]) for key in self.low_dim_keys
        }
        router_cond_B_T_D = None
        if self.router_condition_projection is not None:
            conditions = []
            for key in self.router_condition_keys:
                value_B_T_X = normalized_low_dim_by_key[key]
                conditions.append(value_B_T_X.reshape(*value_B_T_X.shape[:2], -1))
            # One shared state projection; each camera retains its own fusion layers.
            router_cond_B_T_D = self.router_condition_projection(torch.cat(conditions, dim=-1))
        routing_by_view = {}
        aux = {"budget_loss": None, "num_patches_mean": None, "random_topology_fraction": None}
        if self.foveation_net is not None:
            images_by_view = {
                key: self._prepare_rgb(key, obs_dict[key]) for key in self.rgb_keys
            }
            shapes = {image_B_T_C_H_W.shape[:2] for image_B_T_C_H_W in images_by_view.values()}
            if len(shapes) != 1:
                raise ValueError("All RGB observations must share B and T.")
            routing_by_view, aux = self.foveation_net(
                images_by_view, router_cond_B_T_D, return_routing_state=return_routing_state,
            )
            if return_foveation_outputs:
                aux["foveation_outputs"] = {
                    key: {
                        name: value_B_T_X.detach()
                        for name, value_B_T_X in out.items()
                        if torch.is_tensor(value_B_T_X) and value_B_T_X.ndim >= 2
                        and name not in {"patch_bank_B_T_K_C_P_P", "split_hard_B_T_M"}
                    }
                    for key, out in routing_by_view.items()
                }
        return routing_by_view, normalized_low_dim_by_key, aux

    def _encode_routed_observations(self, routing_by_view, normalized_low_dim_by_key):
        features_by_view = {
            key: self.obs_encoders[key](routing_by_view[key]) for key in self.rgb_keys
        }
        return self._fuse_observation_features(features_by_view, normalized_low_dim_by_key)

    def _fuse_observation_features(self, features_by_view, normalized_low_dim_by_key):
        features = [features_by_view[key] for key in self.rgb_keys]
        for key in self.low_dim_keys:
            value_B_T_X = normalized_low_dim_by_key[key]
            value_B_T_D = value_B_T_X.reshape(*value_B_T_X.shape[:2], -1)
            if self.mask_policy_state and key != "task_id":
                value_B_T_D = torch.zeros_like(value_B_T_D)
            features.append(value_B_T_D)
        reference_B_T_D = features[0]
        return torch.cat([
            value_B_T_D.to(device=reference_B_T_D.device, dtype=reference_B_T_D.dtype)
            for value_B_T_D in features
        ], dim=-1)

    def _encode_obs_batch(
        self,
        obs_dict: Dict[str, torch.Tensor],
        time_steps: Optional[int] = None,
        return_foveation_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        sliced = obs_dict
        if time_steps is not None:
            sliced = {key: value[:, :time_steps] for key, value in obs_dict.items()}
        return self._encode_observations(sliced, return_foveation_outputs)

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        generator=None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"use_da", "domain_adapt"}
        }
        trajectory = torch.randn(
            condition_data.shape,
            device=condition_data.device,
            dtype=condition_data.dtype,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(trajectory, timestep, cond)
            trajectory = self.noise_scheduler.step(
                model_output,
                timestep,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_foveation_outputs: bool = False,
    ) -> Dict[str, object]:
        if "past_action" in obs_dict:
            raise AssertionError("past_action is not supported.")
        obs_features_B_To_D, aux = self._encode_obs_batch(
            obs_dict,
            self.n_obs_steps,
            return_foveation_outputs,
        )
        B = obs_features_B_To_D.shape[0]
        To = self.n_obs_steps
        if self.obs_as_cond:
            cond = obs_features_B_To_D
            # TransformerForDiffusion owns fixed-horizon attention masks, so keep
            # the sampling length fixed and select action steps after denoising.
            sample_shape = (B, self.horizon, self.action_dim)
            condition_data = torch.zeros(sample_shape, device=self.device, dtype=self.dtype)
            condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        else:
            cond = None
            condition_data = torch.zeros(
                (B, self.horizon, self.action_dim + self.obs_feature_dim),
                device=self.device,
                dtype=self.dtype,
            )
            condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
            condition_data[:, :To, self.action_dim:] = obs_features_B_To_D
            condition_mask[:, :To, self.action_dim:] = True
        nsample = self.conditional_sample(condition_data, condition_mask, cond=cond)
        naction_pred = nsample[..., : self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = To - 1
        action = action_pred[:, start : start + self.n_action_steps]
        self.cache_action = action_pred
        result = {"action": action, "action_pred": action_pred}
        if return_foveation_outputs:
            result["_foveation_outputs"] = aux.get("foveation_outputs", {})
        return result

    def reset(self) -> None:
        self.infer_frame_idx = 0
        self.cache_action = None

    def get_optimizer(
        self,
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        visual_encoder_lr_multiplier: float = 1.0,
    ) -> torch.optim.Optimizer:
        if visual_encoder_lr_multiplier < 0.0:
            raise ValueError("visual_encoder_lr_multiplier must be non-negative.")

        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        for group in optim_groups:
            group["params"] = [param for param in group["params"] if param.requires_grad]
        optim_groups = [group for group in optim_groups if group["params"]]
        known_ids = {id(param) for group in optim_groups for param in group["params"]}
        no_decay_ids = set()
        for encoder in self.obs_encoders.values():
            for name in ("fine_pos_embed_1_D_Hf_Wf", "level_embed_L_D", "cls_token_1_1_D"):
                param = getattr(encoder, name, None)
                if param is not None:
                    no_decay_ids.add(id(param))

        visual_decay_params = []
        visual_no_decay_params = []
        for param in self.obs_encoders.parameters():
            if param.requires_grad and id(param) not in known_ids:
                target = (
                    visual_no_decay_params
                    if id(param) in no_decay_ids
                    else visual_decay_params
                )
                target.append(param)
                known_ids.add(id(param))

        # Only FoveationObsEncoder uses the visual LR; routing stays at the base LR.
        visual_learning_rate = learning_rate * visual_encoder_lr_multiplier
        if visual_decay_params:
            optim_groups.append(
                {
                    "params": visual_decay_params,
                    "weight_decay": obs_encoder_weight_decay,
                    "lr": visual_learning_rate,
                }
            )
        if visual_no_decay_params:
            optim_groups.append(
                {
                    "params": visual_no_decay_params,
                    "weight_decay": 0.0,
                    "lr": visual_learning_rate,
                }
            )

        routing_params = []
        for module in (self.foveation_net, self.router_condition_projection):
            if module is None:
                continue
            for param in module.parameters():
                if param.requires_grad and id(param) not in known_ids:
                    routing_params.append(param)
                    known_ids.add(id(param))
        if routing_params:
            optim_groups.append(
                {
                    "params": routing_params,
                    "weight_decay": obs_encoder_weight_decay,
                }
            )
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def set_stage_b_phase(self, downstream_finetune: bool) -> None:
        """Select the active Stage B branch without changing DDP parameter registration."""
        self._staged_training = True
        self._downstream_finetune = downstream_finetune
        self._action_path_frozen = not downstream_finetune
        self.train(self.training)

    def freeze_action_path(self) -> None:
        """Freeze visual/action weights while retaining input gradients for routing."""
        self._action_path_frozen = True
        for module in (self.obs_encoders, self.model):
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def freeze_obs_encoder(self) -> None:
        self._obs_encoder_frozen = True
        for module in (
            self.obs_encoders,
            self.foveation_net,
            self.router_condition_projection,
        ):
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._action_path_frozen:
            # eval disables action-path dropout without breaking the ST gradient.
            for module in (self.obs_encoders, self.model):
                if module is not None:
                    module.eval()
        if self._downstream_finetune:
            for module in (self.foveation_net, self.router_condition_projection):
                if module is not None:
                    module.eval()
        if self._obs_encoder_frozen:
            for module in (
                self.obs_encoders,
                self.foveation_net,
                self.router_condition_projection,
            ):
                if module is not None:
                    module.eval()
        return self

    def print_parameter_statistics(
        self,
    ) -> Dict[str, Dict[str, Union[int, float, str]]]:
        """Print and return parameter counts for the policy's key modules."""
        named_modules = []
        if self.foveation_net is not None:
            for key, net in self.foveation_net.nets.items():
                if net.global_encoder is not None:
                    named_modules.append((f"foveation.{key}.global_encoder", net.global_encoder))
                if net.router is not None:
                    named_modules.append((f"foveation.{key}.router", net.router))
        if self.router_condition_projection is not None:
            named_modules.append(("foveation.shared_condition_projection", self.router_condition_projection))
        named_modules.extend((f"visual.{key}", encoder) for key, encoder in self.obs_encoders.items())
        named_modules.append(("action.diffusion_transformer", self.model))

        policy_total = sum(param.numel() for param in self.parameters())
        statistics: Dict[str, Dict[str, Union[int, float, str]]] = {}

        def add_statistics(name: str, module: nn.Module) -> None:
            total = sum(param.numel() for param in module.parameters())
            trainable = sum(
                param.numel() for param in module.parameters() if param.requires_grad
            )
            if total == 0:
                status = "no parameters"
            elif trainable == 0:
                status = "frozen"
            elif trainable == total:
                status = "trainable"
            else:
                status = "partially trainable"
            if self._staged_training and module is not self:
                routing = name.startswith("foveation.")
                active = routing != self._downstream_finetune
                status += "; active" if active else "; no-grad/eval"
            statistics[name] = {
                "total": total,
                "trainable": trainable,
                "frozen": total - trainable,
                "percent_of_policy": (
                    100.0 * total / policy_total if policy_total > 0 else 0.0
                ),
                "status": status,
            }

        for name, module in named_modules:
            add_statistics(name, module)
        add_statistics("policy.total", self)

        def format_count(count: int) -> str:
            if count >= 1_000_000_000:
                return f"{count / 1_000_000_000:.3f}B"
            if count >= 1_000_000:
                return f"{count / 1_000_000:.3f}M"
            if count >= 1_000:
                return f"{count / 1_000:.1f}K"
            return str(count)

        name_width = max(len(name) for name in statistics)
        print("\n[FoveationDPTransformerPolicy] Parameter statistics")
        print(
            f"  {'module':<{name_width}}  {'total':>10}  {'trainable':>10}  "
            f"{'frozen':>10}  {'share':>7}  status"
        )
        print(
            f"  {'-' * name_width}  {'-' * 10}  {'-' * 10}  "
            f"{'-' * 10}  {'-' * 7}  {'-' * 19}"
        )
        for name, module_statistics in statistics.items():
            print(
                f"  {name:<{name_width}}  "
                f"{format_count(int(module_statistics['total'])):>10}  "
                f"{format_count(int(module_statistics['trainable'])):>10}  "
                f"{format_count(int(module_statistics['frozen'])):>10}  "
                f"{float(module_statistics['percent_of_policy']):>6.1f}%  "
                f"{module_statistics['status']}"
            )
        return statistics

    def print_training_status(self) -> None:
        self.print_parameter_statistics()
        if self._staged_training:
            active = "visual encoder + action policy" if self._downstream_finetune else "routing"
            print(f"[Stage B] Registered parameters remain grad-enabled; active updates: {active}")
        print(
            f"[FoveationDPTransformerPolicy] obs_dim={self.obs_feature_dim}, "
            f"mask_policy_state={self.mask_policy_state}"
        )
        if self.foveation_net is not None:
            for key, net in self.foveation_net.nets.items():
                if net.global_encoder is not None:
                    status = "trainable" if any(p.requires_grad for p in net.global_encoder.parameters()) else "frozen"
                    print(f"[FoveationDPTransformerPolicy] {key}: {net.global_encoder_type} ({status})")

    def _diffusion_loss_per_sample(
        self,
        obs_features_B_T_D: torch.Tensor,
        nactions_B_H_D: torch.Tensor,
        *,
        noise_B_H_D: Optional[torch.Tensor] = None,
        timesteps_B: Optional[torch.Tensor] = None,
        condition_mask_B_H_D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Keep the existing masked MSE reduction while allowing paired random inputs."""
        B = nactions_B_H_D.shape[0]
        if self.obs_as_cond:
            cond_B_T_D = obs_features_B_T_D
            trajectory_B_H_D = nactions_B_H_D
        else:
            cond_B_T_D = None
            trajectory_B_H_D = torch.cat(
                [nactions_B_H_D, obs_features_B_T_D], dim=-1
            )
        if condition_mask_B_H_D is None:
            condition_mask_B_H_D = (
                torch.zeros_like(trajectory_B_H_D, dtype=torch.bool)
                if self.pred_action_steps_only
                else self.mask_generator(trajectory_B_H_D.shape)
            )
        if noise_B_H_D is None:
            noise_B_H_D = torch.randn_like(trajectory_B_H_D)
        if timesteps_B is None:
            timesteps_B = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=trajectory_B_H_D.device,
            ).long()
        noisy_trajectory_B_H_D = self.noise_scheduler.add_noise(
            trajectory_B_H_D, noise_B_H_D, timesteps_B
        )
        loss_mask_B_H_D = ~condition_mask_B_H_D
        if self.pred_action_steps_only:
            loss_mask_B_H_D = torch.zeros_like(trajectory_B_H_D, dtype=torch.bool)
            start = self.n_obs_steps - 1
            loss_mask_B_H_D[:, start : start + self.n_action_steps, : self.action_dim] = True
        noisy_trajectory_B_H_D[condition_mask_B_H_D] = trajectory_B_H_D[
            condition_mask_B_H_D
        ]
        pred_B_H_D = self.model(noisy_trajectory_B_H_D, timesteps_B, cond_B_T_D)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type not in {"epsilon", "sample"}:
            raise ValueError(f"Unsupported prediction type {prediction_type!r}.")
        target_B_H_D = noise_B_H_D if prediction_type == "epsilon" else trajectory_B_H_D
        loss_B_H_D = F.mse_loss(pred_B_H_D, target_B_H_D, reduction="none")
        loss_B_H_D = loss_B_H_D * loss_mask_B_H_D.to(loss_B_H_D.dtype)
        return loss_B_H_D.reshape(B, -1).mean(dim=1)

    @torch.no_grad()
    def compute_resolution_diagnostics(
        self, batch: Dict[str, object], *, seed: int
    ) -> Dict[str, object]:
        """Return paired per-level action losses and the original routing outputs."""
        if not self.obs_as_cond:
            raise ValueError("Resolution diagnostics require obs_as_cond=true.")
        if self.training:
            raise ValueError("Resolution diagnostics require eval mode.")
        if self.foveation_net is None or not self.rgb_keys:
            raise ValueError("Resolution diagnostics require RGB foveation inputs.")
        obs_dict = {
            key: value_B_T_D[:, :self.n_obs_steps]
            for key, value_B_T_D in batch["obs"].items()
        }
        nactions_B_H_D = self.normalizer["action"].normalize(batch["action"])
        if nactions_B_H_D.shape[1] != self.horizon:
            raise ValueError(
                f"Expected action horizon {self.horizon}, got {nactions_B_H_D.shape[1]}."
            )
        device = nactions_B_H_D.device
        rng_devices = [device] if device.type == "cuda" else []
        # Preserve training RNG, including the randomness used by the original topology.
        with torch.random.fork_rng(devices=rng_devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            routing_by_view, normalized_low_dim_by_key, aux = self._route_observations(
                obs_dict, return_foveation_outputs=True, return_routing_state=True
            )
            # A separate generator decouples diffusion inputs from random-tree draws.
            generator = torch.Generator(device=device).manual_seed(seed)
            noise_B_H_D = torch.randn(
                nactions_B_H_D.shape, device=device, dtype=nactions_B_H_D.dtype,
                generator=generator,
            )
            timesteps_B = torch.randint(
                self.noise_scheduler.config.num_train_timesteps,
                (nactions_B_H_D.shape[0],), device=device, generator=generator,
            )
            condition_mask_B_H_D = (
                torch.zeros_like(nactions_B_H_D, dtype=torch.bool)
                if self.pred_action_steps_only
                else self.mask_generator(nactions_B_H_D.shape, seed=seed)
            )
            level_losses = []
            for level in range(self.foveation_net.num_levels):
                level_by_view = {}
                for key, net in self.foveation_net.nets.items():
                    out = routing_by_view[key]
                    split_hard_B_T_M = torch.zeros_like(out["split_hard_B_T_M"])
                    split_hard_B_T_M[..., :net.internal_level_offsets[level]] = True
                    level_by_view[key] = net.pack_topology(
                        out["patch_bank_B_T_K_C_P_P"], split_hard_B_T_M,
                    )
                obs_features_B_T_D = self._encode_routed_observations(
                    level_by_view, normalized_low_dim_by_key,
                )
                level_losses.append(self._diffusion_loss_per_sample(
                    obs_features_B_T_D, nactions_B_H_D,
                    noise_B_H_D=noise_B_H_D, timesteps_B=timesteps_B,
                    condition_mask_B_H_D=condition_mask_B_H_D,
                ))
            loss_B_L = torch.stack(level_losses, dim=-1)
        return {"loss_B_L": loss_B_L, "foveation_outputs": aux["foveation_outputs"]}

    def _compute_paired_loss(
        self,
        obs_dict: Dict[str, torch.Tensor],
        nactions_B_H_D: torch.Tensor,
        return_foveation_outputs: bool,
        temporal_pair_valid_P: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self._action_path_frozen:
            raise ValueError("Paired loss requires freeze_action_path() before training.")
        if self.obs_as_cond:
            obs_dict = {key: value[:, :self.n_obs_steps] for key, value in obs_dict.items()}
        routing_by_view, normalized_low_dim_by_key, aux = self._route_observations(
            obs_dict, return_foveation_outputs=return_foveation_outputs, return_routing_state=True,
        )
        intervention = self.foveation_net.sample_paired_interventions(routing_by_view, temporal_pair_valid_P)
        selected_logits_B = intervention["selected_logits_B"].float()
        selected_view_B = intervention["selected_view_B"]
        base_split_B = intervention["base_split_B"]
        B = nactions_B_H_D.shape[0]  # Temporal packs 2P windows as [all A, all B].
        device = nactions_B_H_D.device
        actual_delta_B = torch.zeros(B, dtype=torch.long, device=device)
        with torch.no_grad():
            base_features_by_view = {
                key: self.obs_encoders[key](routing_by_view[key]) for key in self.rgb_keys
            }
            stop_features_by_view = {key: value_B_T_D.clone() for key, value_B_T_D in base_features_by_view.items()}
            split_features_by_view = {key: value_B_T_D.clone() for key, value_B_T_D in base_features_by_view.items()}
            for key, selected in intervention["by_view"].items():
                batch_R = selected["batch_indices_R"]
                frame_R = selected["frame_indices_R"]
                out = routing_by_view[key]
                net = self.foveation_net.nets[key]
                base_is_split_R = base_split_B[batch_R]
                # Only the opposite arm needs encoding. The selected subtree's
                # descendants are unchanged, so one arm equals the base tree.
                changed_split_R_M = torch.where(
                    base_is_split_R[:, None], selected["stop_R_M"], selected["split_R_M"],
                )
                bank_R_1_K_C_P_P = out["patch_bank_B_T_K_C_P_P"][batch_R, frame_R].unsqueeze(1)
                changed_out = net.pack_topology(bank_R_1_K_C_P_P, changed_split_R_M[:, None])
                changed_features_R_D = self.obs_encoders[key](changed_out)[:, 0]
                stop_features_by_view[key][batch_R[base_is_split_R], frame_R[base_is_split_R]] = (
                    changed_features_R_D[base_is_split_R]
                )
                split_features_by_view[key][batch_R[~base_is_split_R], frame_R[~base_is_split_R]] = (
                    changed_features_R_D[~base_is_split_R]
                )
                base_count_R = out["num_patches_B_T"][batch_R, frame_R]
                changed_count_R = changed_out["num_patches_B_T"][:, 0]
                actual_delta_B[batch_R] = torch.where(
                    base_is_split_R, base_count_R - changed_count_R, changed_count_R - base_count_R,
                )
            stop_features_B_T_D = self._fuse_observation_features(stop_features_by_view, normalized_low_dim_by_key)
            split_features_B_T_D = self._fuse_observation_features(split_features_by_view, normalized_low_dim_by_key)
            trajectory_B_H_D = nactions_B_H_D
            if not self.obs_as_cond:
                trajectory_B_H_D = torch.cat([nactions_B_H_D, stop_features_B_T_D], dim=-1)
            condition_mask_B_H_D = (
                torch.zeros_like(trajectory_B_H_D, dtype=torch.bool)
                if self.pred_action_steps_only else self.mask_generator(trajectory_B_H_D.shape)
            )
            noise_B_H_D = torch.randn_like(trajectory_B_H_D)
            random_batch_size = B if temporal_pair_valid_P is None else temporal_pair_valid_P.shape[0]
            timesteps_B = torch.randint(
                self.noise_scheduler.config.num_train_timesteps, (random_batch_size,), device=device,
            )
            if temporal_pair_valid_P is not None:
                timesteps_B = timesteps_B.repeat(2)
            # Arm-major layout: each window reuses its own noise/timestep/mask.
            loss_2B = self._diffusion_loss_per_sample(
                torch.cat([stop_features_B_T_D, split_features_B_T_D]),
                nactions_B_H_D.repeat(2, 1, 1),
                noise_B_H_D=noise_B_H_D.repeat(2, 1, 1),
                timesteps_B=timesteps_B.repeat(2),
                condition_mask_B_H_D=condition_mask_B_H_D.repeat(2, 1, 1),
            )
            stop_loss_B, split_loss_B = loss_2B.chunk(2)

        advantage_B = stop_loss_B - split_loss_B
        spatial_terms_B = -selected_logits_B.sigmoid() * advantage_B.detach()
        paired_loss = spatial_terms_B.mean()
        diffusion_loss = torch.where(base_split_B, split_loss_B, stop_loss_B).mean()
        budget_loss = aux["budget_loss"]
        losses = {
            "total_loss": self.paired_loss_weight * paired_loss + budget_loss,
            "diffusion_loss": diffusion_loss,
            "budget_loss": budget_loss,
            "num_patches_mean": aux["num_patches_mean"],
            "random_topology_fraction": aux["random_topology_fraction"],
            "paired_loss": paired_loss,
            "paired_stop_diffusion_loss": stop_loss_B.mean(),
            "paired_split_diffusion_loss": split_loss_B.mean(),
            "paired_advantage_mean": advantage_B.mean(),
            "paired_advantage_abs_mean": advantage_B.abs().mean(),
            "paired_advantage_positive_fraction": (advantage_B > 0).float().mean(),
        }
        metric_sums = dict(aux["_view_metric_sums"])
        metric_counts = dict(aux["_view_metric_counts"])

        def add_view_metric(name, values_B, views_B, weights_B=None):
            values_B = values_B.detach().float()
            if weights_B is None:
                weights_B = torch.ones_like(values_B)
            metric_sums[name] = values_B.new_zeros(len(self.rgb_keys)).scatter_add_(0, views_B, values_B)
            metric_counts[name] = values_B.new_zeros(len(self.rgb_keys)).scatter_add_(0, views_B, weights_B.float())

        for name, values_B in (
            ("paired_loss", spatial_terms_B), ("paired_advantage_mean", advantage_B),
            ("paired_advantage_abs_mean", advantage_B.abs()),
            ("paired_advantage_positive_fraction", (advantage_B > 0).float()),
            ("paired_stop_diffusion_loss", stop_loss_B), ("paired_split_diffusion_loss", split_loss_B),
        ):
            add_view_metric(name, values_B, selected_view_B)
        if temporal_pair_valid_P is not None:
            P = temporal_pair_valid_P.shape[0]
            matched_P = temporal_pair_valid_P & (actual_delta_B[:P] == actual_delta_B[P:])
            gain_P = advantage_B[:P] - advantage_B[P:]
            prefer_a_P = (selected_logits_B[:P] - selected_logits_B[P:]).sigmoid()
            temporal_terms_P = -prefer_a_P * gain_P.detach() * matched_P.float()
            losses["temporal_loss"] = temporal_terms_P.mean()
            losses["_temporal_matched_pair_count"] = matched_P.sum()
            losses["_temporal_valid_pair_count"] = temporal_pair_valid_P.sum()
            losses["total_loss"] = losses["total_loss"] + self.temporal_loss_weight * losses["temporal_loss"]
            add_view_metric("temporal_loss", temporal_terms_P, selected_view_B[:P])
            add_view_metric("temporal_gain", gain_P, selected_view_B[:P])
            add_view_metric("temporal_match_fraction", matched_P, selected_view_B[:P], temporal_pair_valid_P)
            losses["_view_valid_count"] = metric_counts["temporal_match_fraction"]
            losses["_view_matched_count"] = metric_sums["temporal_match_fraction"]
        losses["_view_sample_count"] = torch.bincount(
            selected_view_B[:random_batch_size], minlength=len(self.rgb_keys),
        )
        if return_foveation_outputs:
            losses["_foveation_outputs"] = aux["foveation_outputs"]
        losses.update({key: value for key, value in aux.items() if key.startswith(("_budget_", "views/"))})
        losses["_view_metric_sums"] = metric_sums
        losses["_view_metric_counts"] = metric_counts
        return losses

    def compute_loss(
        self,
        batch: Dict[str, object],
        return_foveation_outputs: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute paired routing, ST diffusion, or downstream-only finetuning loss."""
        if kwargs.pop("return_roi_params", False):
            return_foveation_outputs = True
        obs_dict = batch["obs"]
        actions_B_H_D = batch["action"]
        paired_training = self.training and self.router_loss_type == "paired" and not self._downstream_finetune
        temporal_pair_valid_P = None
        if paired_training and self.temporal_adaptive:
            if "temporal_pair" not in batch or "temporal_pair_valid" not in batch:
                raise ValueError("Temporal router training requires dataset.set_temporal_pairing(True).")
            pair_batch = batch["temporal_pair"]
            obs_dict = {
                key: torch.cat([value_B_T_D, pair_batch["obs"][key]], dim=0)
                for key, value_B_T_D in obs_dict.items()
            }
            actions_B_H_D = torch.cat([actions_B_H_D, pair_batch["action"]], dim=0)
            temporal_pair_valid_P = batch["temporal_pair_valid"]
        nactions_B_H_D = self.normalizer["action"].normalize(actions_B_H_D)
        if nactions_B_H_D.shape[1] != self.horizon:
            raise ValueError(
                f"Expected action horizon {self.horizon}, got {nactions_B_H_D.shape[1]}."
            )
        if paired_training:
            return self._compute_paired_loss(
                obs_dict, nactions_B_H_D, return_foveation_outputs, temporal_pair_valid_P
            )

        obs_features_B_T_D, aux = self._encode_obs_batch(
            obs_dict,
            self.n_obs_steps if self.obs_as_cond else None,
            return_foveation_outputs,
        )
        diffusion_loss = self._diffusion_loss_per_sample(
            obs_features_B_T_D, nactions_B_H_D
        ).mean()

        budget_loss = aux["budget_loss"]
        if budget_loss is None:
            budget_loss = diffusion_loss * 0.0
        num_patches_mean = aux["num_patches_mean"]
        if num_patches_mean is None:
            num_patches_mean = diffusion_loss.detach() * 0.0
        random_topology_fraction = aux["random_topology_fraction"]
        if random_topology_fraction is None:
            random_topology_fraction = diffusion_loss.detach() * 0.0
        total_loss = diffusion_loss if self._downstream_finetune else diffusion_loss + budget_loss
        losses = {
            "diffusion_loss": diffusion_loss,
            "budget_loss": budget_loss,
            "num_patches_mean": num_patches_mean,
            "random_topology_fraction": random_topology_fraction,
            "total_loss": total_loss,
        }
        if return_foveation_outputs:
            losses["_foveation_outputs"] = aux.get("foveation_outputs", {})
        losses.update({key: value for key, value in aux.items() if key.startswith(("views/", "_view_"))})
        if not self._downstream_finetune:
            losses.update({key: value for key, value in aux.items() if key.startswith("_budget_")})
        return losses

    def forward(self, batch, return_foveation_outputs: bool = False, **kwargs):
        return self.compute_loss(
            batch,
            return_foveation_outputs=return_foveation_outputs,
            **kwargs,
        )


# Keep likely config/class spellings compatible with the existing policy names.
FoveationTransformerHybridImagePolicy = FoveationDPTransformerPolicy
DiffusionTransformerFoveationPolicy = FoveationDPTransformerPolicy
FoveationPolicy = FoveationDPTransformerPolicy

__all__ = [
    "FoveationObsEncoder",
    "FoveationDPTransformerPolicy",
    "FoveationTransformerHybridImagePolicy",
    "DiffusionTransformerFoveationPolicy",
    "FoveationPolicy",
]
