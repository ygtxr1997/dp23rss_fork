# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SigLIP vision encoder for PyTorch, aligned with JAX models/siglip.py.

Implements the So400m/14 ViT variant with pool_type="none" used by Pi0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .utils import _str_to_dtype


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(
        self,
        dim: int,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        mlp_dim = mlp_dim or 4 * dim
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x_B_N_D: torch.Tensor) -> torch.Tensor:
        """Encode a batch of patch tokens."""
        x_B_N_D = self.fc1(x_B_N_D)
        x_B_N_D = F.gelu(x_B_N_D)
        x_B_N_D = self.dropout(x_B_N_D)
        x_B_N_D = self.fc2(x_B_N_D)
        return x_B_N_D


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = MlpBlock(dim, mlp_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x_B_N_D: torch.Tensor, token_valid_mask_B_N=None) -> torch.Tensor:
        """Encode tokens while excluding packed padding from attention keys/values."""
        # Cast only the LayerNorm input. Keep residual dtype (OpenPI mixed precision).
        norm_dtype = self.norm1.weight.dtype
        y_B_N_D = self.norm1(x_B_N_D.to(norm_dtype))
        y_B_N_D, _ = self.attn(y_B_N_D, y_B_N_D, y_B_N_D,
                              key_padding_mask=None if token_valid_mask_B_N is None else ~token_valid_mask_B_N)
        y_B_N_D = self.dropout1(y_B_N_D)
        x_B_N_D = x_B_N_D + y_B_N_D

        y_B_N_D = self.norm2(x_B_N_D.to(norm_dtype))
        y_B_N_D = self.mlp(y_B_N_D)
        y_B_N_D = self.dropout2(y_B_N_D)
        x_B_N_D = x_B_N_D + y_B_N_D
        return x_B_N_D


class Encoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Encoder1DBlock(
                    dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.gradient_checkpointing = use_gradient_checkpointing
        # Non-reentrant checkpointing supports the nested expert inputs.
        self.gradient_checkpointing_use_reentrant = False

    def forward(self, x_B_N_D: torch.Tensor, token_valid_mask_B_N=None) -> torch.Tensor:
        """Encode a batch using the same leaf validity in every block."""
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x_B_N_D = torch.utils.checkpoint.checkpoint(
                    layer, x_B_N_D, token_valid_mask_B_N,
                    use_reentrant=self.gradient_checkpointing_use_reentrant
                )
            else:
                x_B_N_D = layer(x_B_N_D, token_valid_mask_B_N)
        x_B_N_D = self.norm(x_B_N_D.to(self.norm.weight.dtype))
        return x_B_N_D


class _ResidualPatchAdapter(nn.Module):
    """Apply a zero-initialized residual transform to one quadtree level."""

    def __init__(self, dim: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x_V_D: torch.Tensor) -> torch.Tensor:
        """Compute in parameter dtype and preserve the residual input dtype."""
        # The patch stem produces fp32 tokens even when adapter weights are bf16.
        input_dtype = x_V_D.dtype
        x_V_D = x_V_D.to(self.down.weight.dtype)
        return self.up(F.gelu(self.down(x_V_D))).to(input_dtype)


class SigLIPViT(nn.Module):
    """Dense So400m/14 encoder with its pretrained 2048-dimensional projection."""

    def __init__(self, dtype: str = "bfloat16", gradient_checkpointing: bool = False,
                 level_embedding: bool = False, patch_level_adapter: bool = False,
                 patch_level_adapter_channels: int = 64,
                 apt_position_encoding: bool = False):
        super().__init__()
        self.dtype_mm = _str_to_dtype(dtype)
        self.stem = nn.Conv2d(3, 1152, kernel_size=14, stride=14, bias=True)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 256, 1152))
        self.encoder = Encoder(1152, 27, 16, 4304,
                               use_gradient_checkpointing=gradient_checkpointing)
        self.head = nn.Linear(1152, 2048)
        # Zero initialized so mixed-foveation starts from the pretrained dense embedding.
        self.level_embedding = nn.Parameter(torch.zeros(3, 1152)) if level_embedding else None
        self.patch_level_adapters = nn.ModuleList(
            [_ResidualPatchAdapter(1152, patch_level_adapter_channels) for _ in range(3)]
        ) if patch_level_adapter else None
        self.apt_position = nn.Parameter(torch.zeros(1, 16, 16, 1152)) \
            if apt_position_encoding else None
        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_patches(self, patches_B_N_C_P_P, bank_indices_B_N, levels_B_N, token_valid_mask_B_N):
        """Encode packed 14x14 leaves through one shared SigLIP transformer."""
        B, N = patches_B_N_C_P_P.shape[:2]
        # The masked third camera bypasses the encoder in Pi05Model. Real cameras
        # always have leaves; padding never passes through the patch stem.
        patches_V_C_P_P = patches_B_N_C_P_P[token_valid_mask_B_N].float()
        stem_weight_O_C_P_P = self.stem.weight.float()
        stem_bias_O = self.stem.bias.float()
        try:
            x_V_D = F.conv2d(patches_V_C_P_P, stem_weight_O_C_P_P,
                             stem_bias_O, stride=14).flatten(1)
        except RuntimeError as error:
            # Some CUDA/cuDNN versions fail on this batched 14x14 patch kernel
            # during rollout.  Retry with the native CUDA implementation; the
            # patch count and tensor shapes stay unchanged.
            if "CUDNN_STATUS_INTERNAL_ERROR" not in str(error):
                raise
            with torch.backends.cudnn.flags(enabled=False):
                x_V_D = F.conv2d(patches_V_C_P_P, stem_weight_O_C_P_P,
                                 stem_bias_O, stride=14).flatten(1)
        x_B_N_D = x_V_D.new_zeros(B, N, 1152)
        x_B_N_D[token_valid_mask_B_N] = x_V_D
        pos_levels = []
        for grid in (4, 8, 16):
            pos_1_D_H_W = F.interpolate(
                self.pos_embedding.float().reshape(1, 16, 16, -1).permute(0, 3, 1, 2),
                (grid, grid), mode='bicubic', align_corners=False)
            pos_levels.append(pos_1_D_H_W.permute(0, 2, 3, 1).reshape(grid * grid, -1))
        # Bank indices are global: coarse [0,16), medium [16,80), fine [80,336).
        pos_K_D = torch.cat(pos_levels)
        valid_B_N_1 = token_valid_mask_B_N[..., None]
        if self.patch_level_adapters is not None:
            # Apply each adapter only to valid tokens belonging to its level.
            for level, adapter in enumerate(self.patch_level_adapters):
                level_mask_B_N = token_valid_mask_B_N & (levels_B_N == level)
                if level_mask_B_N.any():
                    x_B_N_D[level_mask_B_N] += adapter(x_B_N_D[level_mask_B_N])
        x_B_N_D = x_B_N_D + pos_K_D[bank_indices_B_N.clamp_min(0)] * valid_B_N_1
        if self.apt_position is not None:
            apt_levels = []
            for grid in (4, 8, 16):
                apt_1_D_H_W = F.interpolate(
                    self.apt_position.float().permute(0, 3, 1, 2),
                    (grid, grid), mode='bicubic', align_corners=False)
                apt_levels.append(apt_1_D_H_W.permute(0, 2, 3, 1).reshape(grid * grid, -1))
            apt_K_D = torch.cat(apt_levels)
            x_B_N_D = x_B_N_D + apt_K_D[bank_indices_B_N.clamp_min(0)] * valid_B_N_1
        if self.level_embedding is not None:
            x_B_N_D = x_B_N_D + self.level_embedding[levels_B_N.clamp_min(0)] * valid_B_N_1
        x_B_N_D = self.encoder(x_B_N_D.to(self.dtype_mm), token_valid_mask_B_N)
        return self.head(x_B_N_D).masked_fill(~token_valid_mask_B_N[..., None], 0)

    def forward(self, image_B_H_W_C: torch.Tensor) -> torch.Tensor:
        """Map float32 RGB in [-1,1] to patch tokens at the requested grid size."""
        # Preserve the fp32 patch embedding/position island, including its gradients.
        x_B_D_H_W = F.conv2d(image_B_H_W_C.permute(0, 3, 1, 2).float(),
                            self.stem.weight.float(), self.stem.bias.float(), stride=14)
        x_B_N_D = x_B_D_H_W.flatten(2).transpose(1, 2)
        grid_h, grid_w = x_B_D_H_W.shape[-2:]
        # The patch stem floors non-aligned sizes (e.g. 64 -> 4 patches);
        # positional interpolation follows the actual patch grid.
        pos_B_N_D = self.pos_embedding.float().reshape(1, 16, 16, -1)
        pos_B_N_D = F.interpolate(pos_B_N_D.permute(0, 3, 1, 2), (grid_h, grid_w),
                                   mode='bicubic', align_corners=False)
        x_B_N_D = x_B_N_D + pos_B_N_D.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        x_B_N_D = self.encoder(x_B_N_D.to(self.dtype_mm))
        return self.head(x_B_N_D)
