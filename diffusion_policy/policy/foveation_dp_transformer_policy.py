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
    ) -> None:
        super().__init__()
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
        router_condition_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
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
        router_condition_input_dim = None
        if self.router_condition_key is not None:
            if self.router_condition_key not in obs_shape_meta:
                raise ValueError(
                    f"Router condition {self.router_condition_key!r} is missing from shape_meta.obs."
                )
            condition_meta = obs_shape_meta[self.router_condition_key]
            if condition_meta.get("type", "low_dim") != "low_dim":
                raise ValueError("router_condition_key must refer to a low-dim observation.")
            if not self.rgb_keys:
                raise ValueError("router_condition_key requires at least one RGB observation.")
            router_condition_input_dim = math.prod(condition_meta["shape"])

        self.foveation_net: Optional[FoveationNet] = None
        self.obs_encoder: Optional[FoveationObsEncoder] = None
        if self.rgb_keys:
            rgb_shape = tuple(obs_shape_meta[self.rgb_keys[0]]["shape"])
            for key in self.rgb_keys[1:]:
                if tuple(obs_shape_meta[key]["shape"]) != rgb_shape:
                    raise ValueError(
                        "All RGB observations must share C/H/W and foveation geometry; "
                        f"{self.rgb_keys[0]!r} and {key!r} differ."
                    )
            self.foveation_net = self._instantiate_foveation_net(
                foveation_net,
                in_channels=rgb_shape[0],
                input_hw=rgb_shape[1:],
            )
            global_encoder = getattr(self.foveation_net, "global_encoder", None)
            encoder_is_frozen = bool(
                getattr(self.foveation_net, "freeze_global_encoder", True)
                or getattr(self.foveation_net, "fix_split_mode", None) is not None
            )
            encoder_type = str(
                getattr(self.foveation_net, "global_encoder_type", "")
            )
            if encoder_type.startswith("vjepa") and not encoder_is_frozen:
                raise ValueError("V-JEPA global encoders must remain frozen.")
            if global_encoder is not None and encoder_is_frozen:
                if any(param.requires_grad for param in global_encoder.parameters()):
                    raise ValueError("The configured global encoder must be frozen.")
                global_encoder.eval()
            self.obs_encoder = FoveationObsEncoder(
                in_channels=rgb_shape[0],
                input_hw=rgb_shape[1:],
                patch_size=int(self.foveation_net.patch_size),
                num_levels=int(self.foveation_net.num_levels),
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
            )
            foveation_grids = tuple(
                tuple(grid_hw) for grid_hw in self.foveation_net.level_grid_hw
            )
            if self.obs_encoder.level_grid_hw != foveation_grids:
                raise ValueError(
                    "Observation encoder and FoveationNet patch grids differ: "
                    f"encoder={self.obs_encoder.level_grid_hw}, "
                    f"foveation={foveation_grids}."
                )
            if self.obs_encoder.num_bank_patches != int(
                self.foveation_net.num_bank_patches
            ):
                raise ValueError(
                    "Observation encoder and FoveationNet bank sizes differ."
                )

        self.router_condition_projection: Optional[nn.Linear] = None
        if router_condition_input_dim is not None:
            self.router_condition_projection = nn.Linear(
                router_condition_input_dim,
                int(self.foveation_net.router_cond_dim),
            )
            if (
                getattr(self.foveation_net, "freeze_router", False)
                or getattr(self.foveation_net, "fix_split_mode", None) is not None
            ):
                self.router_condition_projection.requires_grad_(False)

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

    def _prepare_rgb(self, key: str, image_B_T_C_H_W: torch.Tensor) -> torch.Tensor:
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

    def _encode_observations(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_foveation_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if not obs_dict:
            raise ValueError("obs_dict cannot be empty.")
        features: List[torch.Tensor] = []
        aux: Dict[str, object] = {
            "budget_loss": None,
            "num_patches_mean": None,
            "random_topology_fraction": None,
        }
        normalized_low_dim_by_key = {
            key: self.normalizer[key].normalize(obs_dict[key])
            for key in self.low_dim_keys
        }
        router_cond_B_T_D = None
        if self.router_condition_projection is not None:
            condition_B_T = normalized_low_dim_by_key[self.router_condition_key]
            condition_B_T_D = condition_B_T.reshape(
                condition_B_T.shape[0], condition_B_T.shape[1], -1
            )
            router_cond_B_T_D = self.router_condition_projection(condition_B_T_D)

        if self.rgb_keys:
            rgb_inputs = []
            batch_size = None
            time_size = None
            for key in self.rgb_keys:
                image_B_T_C_H_W = obs_dict[key]
                if image_B_T_C_H_W.ndim != 5:
                    raise ValueError(f"RGB observation {key!r} must be (B,T,C,H,W).")
                if batch_size is None:
                    batch_size, time_size = image_B_T_C_H_W.shape[:2]
                if image_B_T_C_H_W.shape[:2] != (batch_size, time_size):
                    raise ValueError("All RGB observations must share B and T.")
                rgb_inputs.append(self._prepare_rgb(key, image_B_T_C_H_W))

            merged_B_T_C_H_W = torch.cat(rgb_inputs, dim=0)
            merged_B_C_T_H_W = merged_B_T_C_H_W.permute(0, 2, 1, 3, 4).contiguous()
            merged_cond_B_T_D = None
            if router_cond_B_T_D is not None:
                # RGB streams are concatenated as full batch blocks, so repeat
                # the shared low-dim condition in the same order.
                merged_cond_B_T_D = router_cond_B_T_D.repeat(
                    len(self.rgb_keys), 1, 1
                )
            merged_out = self.foveation_net(
                merged_B_C_T_H_W,
                cond_B_T_D=merged_cond_B_T_D,
            )
            aux["budget_loss"] = merged_out["budget_loss"]
            aux["num_patches_mean"] = merged_out["num_patches_B_T"].float().mean()
            aux["random_topology_fraction"] = merged_out[
                "random_topology_mask_B_T"
            ].float().mean()
            route_outputs = []
            for route_idx, key in enumerate(self.rgb_keys):
                start = route_idx * batch_size
                end = start + batch_size
                route_out = {
                    name: value[start:end]
                    for name, value in merged_out.items()
                    if (
                        torch.is_tensor(value)
                        and value.ndim > 0
                        and value.shape[0] == merged_B_T_C_H_W.shape[0]
                    )
                }
                route_outputs.append(route_out)
                features.append(self.obs_encoder(route_out))
            if return_foveation_outputs:
                aux["foveation_outputs"] = {
                    key: {
                        name: value.detach()
                        for name, value in route_out.items()
                    }
                    for key, route_out in zip(self.rgb_keys, route_outputs)
                }

        for key in self.low_dim_keys:
            value_B_T = normalized_low_dim_by_key[key]
            features.append(value_B_T.reshape(value_B_T.shape[0], value_B_T.shape[1], -1))

        if not features:
            raise ValueError("No supported observations were provided.")
        reference = features[0]
        features = [value.to(device=reference.device, dtype=reference.dtype) for value in features]
        return torch.cat(features, dim=-1), aux

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
        if self.obs_encoder is not None:
            fine_pos_embed = getattr(
                self.obs_encoder,
                "fine_pos_embed_1_D_Hf_Wf",
                None,
            )
            if fine_pos_embed is not None:
                no_decay_ids.add(id(fine_pos_embed))
            level_embed_L_D = getattr(self.obs_encoder, "level_embed_L_D", None)
            if level_embed_L_D is not None:
                no_decay_ids.add(id(level_embed_L_D))
            cls_token_1_1_D = getattr(self.obs_encoder, "cls_token_1_1_D", None)
            if cls_token_1_1_D is not None:
                no_decay_ids.add(id(cls_token_1_1_D))

        visual_decay_params = []
        visual_no_decay_params = []
        if self.obs_encoder is not None:
            for param in self.obs_encoder.parameters():
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

    def freeze_action_path(self) -> None:
        """Freeze visual/action weights while retaining input gradients for routing."""
        self._action_path_frozen = True
        for module in (self.obs_encoder, self.model):
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def freeze_obs_encoder(self) -> None:
        self._obs_encoder_frozen = True
        for module in (
            self.obs_encoder,
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
            for module in (self.obs_encoder, self.model):
                if module is not None:
                    module.eval()
        if self._obs_encoder_frozen:
            for module in (
                self.obs_encoder,
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
            global_encoder = getattr(self.foveation_net, "global_encoder", None)
            if global_encoder is not None:
                encoder_type = getattr(
                    self.foveation_net,
                    "global_encoder_type",
                    type(global_encoder).__name__,
                )
                named_modules.append(
                    (f"foveation.global_encoder ({encoder_type})", global_encoder)
                )
            router = getattr(self.foveation_net, "router", None)
            if router is not None:
                named_modules.append(("foveation.router", router))
        if self.router_condition_projection is not None:
            named_modules.append(
                ("foveation.condition_projection", self.router_condition_projection)
            )
        if self.obs_encoder is not None:
            named_modules.append(("visual.obs_encoder", self.obs_encoder))
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
        print(f"[FoveationDPTransformerPolicy] obs_dim={self.obs_feature_dim}")
        if self.foveation_net is not None:
            encoder = getattr(self.foveation_net, "global_encoder", None)
            if encoder is not None:
                encoder_type = getattr(
                    self.foveation_net,
                    "global_encoder_type",
                    type(encoder).__name__,
                )
                status = (
                    "trainable"
                    if any(param.requires_grad for param in encoder.parameters())
                    else "frozen"
                )
                print(
                    f"[FoveationDPTransformerPolicy] global_encoder="
                    f"{encoder_type} ({status})"
                )

    def compute_loss(
        self,
        batch: Dict[str, object],
        return_foveation_outputs: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if kwargs.pop("return_roi_params", False):
            return_foveation_outputs = True
        obs_dict = batch["obs"]
        nactions = self.normalizer["action"].normalize(batch["action"])
        B, horizon = nactions.shape[:2]
        if horizon != self.horizon:
            raise ValueError(f"Expected action horizon {self.horizon}, got {horizon}.")

        if self.obs_as_cond:
            nobs_features_B_To_D, aux = self._encode_obs_batch(
                obs_dict,
                self.n_obs_steps,
                return_foveation_outputs,
            )
            cond = nobs_features_B_To_D
            trajectory = nactions
        else:
            nobs_features_B_T_D, aux = self._encode_obs_batch(
                obs_dict,
                None,
                return_foveation_outputs,
            )
            cond = None
            trajectory = torch.cat([nactions, nobs_features_B_T_D], dim=-1)

        condition_mask = (
            torch.zeros_like(trajectory, dtype=torch.bool)
            if self.pred_action_steps_only
            else self.mask_generator(trajectory.shape)
        )
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        if self.pred_action_steps_only:
            loss_mask = torch.zeros_like(trajectory, dtype=torch.bool)
            start = self.n_obs_steps - 1
            loss_mask[:, start : start + self.n_action_steps, : self.action_dim] = True
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        pred = self.model(noisy_trajectory, timesteps, cond)

        prediction_type = self.noise_scheduler.config.prediction_type
        target = noise if prediction_type == "epsilon" else trajectory
        if prediction_type not in {"epsilon", "sample"}:
            raise ValueError(f"Unsupported prediction type {prediction_type!r}.")
        diffusion_loss = F.mse_loss(pred, target, reduction="none")
        diffusion_loss = (diffusion_loss * loss_mask.to(diffusion_loss.dtype)).reshape(B, -1).mean()

        budget_loss = aux["budget_loss"]
        if budget_loss is None:
            budget_loss = diffusion_loss * 0.0
        num_patches_mean = aux["num_patches_mean"]
        if num_patches_mean is None:
            num_patches_mean = diffusion_loss.detach() * 0.0
        random_topology_fraction = aux["random_topology_fraction"]
        if random_topology_fraction is None:
            random_topology_fraction = diffusion_loss.detach() * 0.0
        total_loss = diffusion_loss + budget_loss
        losses = {
            "diffusion_loss": diffusion_loss,
            "budget_loss": budget_loss,
            "num_patches_mean": num_patches_mean,
            "random_topology_fraction": random_topology_fraction,
            "total_loss": total_loss,
        }
        if return_foveation_outputs:
            losses["_foveation_outputs"] = aux.get("foveation_outputs", {})
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
