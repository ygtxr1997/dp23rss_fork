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

"""Gemma backbone for PyTorch, aligned with JAX models/gemma.py.

Supports:
- Multiple experts sharing self-attention but with independent FFN
- RMSNorm (regular and adaptive/adaRMS)
- GQA (Grouped Query Attention) with RoPE
- KV caching for efficient inference
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .utils import _str_to_dtype, gelu_glu

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    """Dimensions of one Gemma expert."""

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


Variant = Literal["gemma_300m", "gemma_2b"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    raise ValueError(f"Unknown variant: {variant}")


class RMSNorm(nn.Module):
    """RMSNorm with optional adaptive mode (adaRMS)."""

    def __init__(
        self,
        dim: int,
        adaptive: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.adaptive = adaptive
        if not self.adaptive:
            self.scale = nn.Parameter(torch.zeros(dim))
        else:
            self.ada_modulation = nn.Linear(dim, dim * 3)
            nn.init.zeros_(self.ada_modulation.weight)
            nn.init.zeros_(self.ada_modulation.bias)

    def forward(
        self, x_B_T_D: torch.Tensor, cond_B_D: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: input tensor of shape (..., dim)
            cond: optional conditioning for adaptive RMSNorm, shape (..., dim)

        Returns:
            (normalized_output, gate) where gate is None for regular RMSNorm
        """
        dtype = x_B_T_D.dtype
        x_B_T_D_float = x_B_T_D.float()
        var_B_T_1 = torch.mean(x_B_T_D_float**2, dim=-1, keepdim=True)
        normed_B_T_D = x_B_T_D_float * torch.rsqrt(var_B_T_1 + 1e-6)

        if not self.adaptive:
            scale_B_1_D = 1.0 + self.scale.float()
            normed_B_T_D = normed_B_T_D * scale_B_1_D
            return normed_B_T_D.to(dtype), None

        # adaptive RMSNorm
        target_dtype = self.ada_modulation.weight.dtype
        modulation_B_D3 = self.ada_modulation(cond_B_D.to(target_dtype))  # (..., 3*dim)
        scale_B_1_D, shift_B_1_D, gate_B_1_D = torch.chunk(modulation_B_D3, 3, dim=-1)
        if x_B_T_D.dim() == 3:
            scale_B_1_D = scale_B_1_D.unsqueeze(-2)
            shift_B_1_D = shift_B_1_D.unsqueeze(-2)
            gate_B_1_D = gate_B_1_D.unsqueeze(-2)

        normed_B_T_D = normed_B_T_D * (1.0 + scale_B_1_D.float()) + shift_B_1_D.float()
        return normed_B_T_D.to(dtype), gate_B_1_D.to(dtype)


class Embedder(nn.Module):
    """Embedder module."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight)

    def forward(self, x_B_T_D: torch.Tensor) -> torch.Tensor:
        """Embed token IDs."""
        return self.encode(x_B_T_D)

    def encode(self, x_B_T_D: torch.Tensor) -> torch.Tensor:
        """Apply vocabulary embedding and Gemma scaling."""
        x_B_T_D = self.embedding(x_B_T_D)
        x_B_T_D = x_B_T_D * math.sqrt(self.embed_dim)
        return x_B_T_D

    def decode(self, x_B_T_D: torch.Tensor) -> torch.Tensor:
        """Project hidden states onto the shared vocabulary."""
        return F.linear(x_B_T_D, self.embedding.weight)


class Attention(nn.Module):
    """Multi-expert Grouped Query Attention with RoPE."""

    def __init__(
        self,
        configs: Sequence[Config],
        attention_backend: str = 'sdpa',
    ):
        super().__init__()
        if attention_backend not in ('sdpa', 'math'):
            raise ValueError("attention_backend must be 'sdpa' or 'math'")
        self.expert_configs = configs
        self.attention_backend = attention_backend
        self.num_heads = configs[0].num_heads
        self.num_kv_heads = configs[0].num_kv_heads
        self.head_dim = configs[0].head_dim
        self.num_query_groups = self.num_heads // self.num_kv_heads

        # Q projections per expert
        self.q_proj = nn.ModuleList()
        self.k_proj = nn.ModuleList()
        self.v_proj = nn.ModuleList()
        self.o_proj = nn.ModuleList()

        for config in configs:
            if config.num_kv_heads == config.num_heads:
                # Combined QKV projection
                self.q_proj.append(
                    nn.Linear(
                        config.width, 3 * config.num_heads * config.head_dim, bias=False
                    )
                )
                self.k_proj.append(None)  # handled by q_proj
                self.v_proj.append(None)
            else:
                self.q_proj.append(
                    nn.Linear(
                        config.width, config.num_heads * config.head_dim, bias=False
                    )
                )
                self.k_proj.append(
                    nn.Linear(
                        config.width, config.num_kv_heads * config.head_dim, bias=False
                    )
                )
                self.v_proj.append(
                    nn.Linear(
                        config.width, config.num_kv_heads * config.head_dim, bias=False
                    )
                )

            self.o_proj.append(
                nn.Linear(config.num_heads * config.head_dim, config.width, bias=False)
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for i, config in enumerate(self.expert_configs):
            # Q projection
            nn.init.normal_(self.q_proj[i].weight, std=1.0 / math.sqrt(config.width))
            if self.k_proj[i] is not None:
                nn.init.normal_(
                    self.k_proj[i].weight, std=1.0 / math.sqrt(config.width)
                )
                nn.init.normal_(
                    self.v_proj[i].weight, std=1.0 / math.sqrt(config.width)
                )
            # O projection
            nn.init.normal_(
                self.o_proj[i].weight,
                std=1.0 / math.sqrt(config.num_heads * config.head_dim),
            )

    def forward(
        self,
        xs: list[torch.Tensor | None],
        positions_B_T: torch.Tensor,
        attn_mask_B_1_T_S: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
        """Multi-expert attention forward.

        Concatenates all expert inputs along the sequence dimension (like JAX),
        computes unified self-attention, then splits outputs back per expert.

        Args:
            xs: list of inputs, one per expert (or None)
            positions: (B, T) positional indices covering full concatenated sequence
            attn_mask: (B, 1, T, S) attention mask
            kv_cache: optional (k, v) cache tuple

        Returns:
            (outputs, new_kv_cache)
        """
        dtype = next(x_B_T_D.dtype for x_B_T_D in xs if x_B_T_D is not None)

        q_parts, k_parts, v_parts = [], [], []
        for i, x_B_T_D in enumerate(xs):
            if x_B_T_D is None:
                q_parts.append(None)
                k_parts.append(None)
                v_parts.append(None)
                continue

            B, T, _ = x_B_T_D.shape
            if self.k_proj[i] is None:
                qkv_B_T_3_N_H = self.q_proj[i](x_B_T_D)
                qkv_B_T_3_N_H = qkv_B_T_3_N_H.reshape(B, T, 3, self.num_heads, self.head_dim)
                q_B_T_N_H, k_B_S_K_H, v_B_S_K_H = qkv_B_T_3_N_H.unbind(dim=2)
            else:
                q_B_T_N_H = self.q_proj[i](x_B_T_D).reshape(B, T, self.num_heads, self.head_dim)
                k_B_S_K_H = self.k_proj[i](x_B_T_D).reshape(B, T, self.num_kv_heads, self.head_dim)
                v_B_S_K_H = self.v_proj[i](x_B_T_D).reshape(B, T, self.num_kv_heads, self.head_dim)

            q_parts.append(q_B_T_N_H)
            k_parts.append(k_B_S_K_H)
            v_parts.append(v_B_S_K_H)

        # Concatenate along sequence dimension (dim=1) — matching JAX jnp.concatenate(axis=1)
        q_B_T_N_H = torch.cat([q_B_T_N_H for q_B_T_N_H in q_parts if q_B_T_N_H is not None], dim=1)
        k_B_S_K_H = torch.cat([k_B_S_K_H for k_B_S_K_H in k_parts if k_B_S_K_H is not None], dim=1)
        v_B_S_K_H = torch.cat([v_B_S_K_H for v_B_S_K_H in v_parts if v_B_S_K_H is not None], dim=1)

        # Apply RoPE
        q_B_T_N_H = _apply_rope(q_B_T_N_H, positions_B_T=positions_B_T)
        k_B_S_K_H = _apply_rope(k_B_S_K_H, positions_B_T=positions_B_T)

        # KV cache: concatenate along sequence dim
        if kv_cache is not None:
            cache_k_B_P_K_H, cache_v_B_P_K_H = kv_cache
            k_B_S_K_H = torch.cat([cache_k_B_P_K_H, k_B_S_K_H], dim=1)
            v_B_S_K_H = torch.cat([cache_v_B_P_K_H, v_B_S_K_H], dim=1)

        new_kv_cache = (k_B_S_K_H, v_B_S_K_H)

        if self.attention_backend == 'sdpa':
            # SDPA consumes [B, heads, T, H] and keeps the GQA contraction
            # inside the fused kernel. The boolean mask uses True=visible,
            # matching make_attn_mask and the previous torch.where path.
            q_B_N_T_H = q_B_T_N_H.transpose(1, 2)
            k_B_K_S_H = k_B_S_K_H.transpose(1, 2)
            v_B_K_S_H = v_B_S_K_H.transpose(1, 2)
            encoded_B_N_T_H = F.scaled_dot_product_attention(
                q_B_N_T_H,
                k_B_K_S_H,
                v_B_K_S_H,
                attn_mask=attn_mask_B_1_T_S,
                dropout_p=0.0,
                enable_gqa=True,
            )
            encoded_B_T_N_H = encoded_B_N_T_H.transpose(1, 2)
            outputs = []
            start = 0
            for i, x_B_T_D in enumerate(xs):
                if x_B_T_D is not None:
                    end = start + x_B_T_D.shape[1]
                    expert_out_B_T_D = encoded_B_T_N_H[:, start:end].reshape(
                        x_B_T_D.shape[0], x_B_T_D.shape[1], -1)
                    outputs.append(self.o_proj[i](expert_out_B_T_D))
                    start = end
                else:
                    outputs.append(None)
            return outputs, new_kv_cache

        # Apply mask: shape (B, 1, T, S) -> broadcast to (B, K, G, T, S)
        if attn_mask_B_1_T_S.dim() == 4:
            attn_mask_B_1_T_S = attn_mask_B_1_T_S[:, 0:1, :, :]  # (B, 1, T, S)

        # GQA einsum pattern matching JAX:
        q_B_T_N_H = q_B_T_N_H * (self.head_dim**-0.5)

        # q: (B, T, num_heads, H) -> rearrange to (B, T, K, G, H)
        # k: (B, S, num_kv_heads, H) -> stays (B, S, K, H)
        K = self.num_kv_heads
        G = self.num_heads // K

        q_B_T_K_G_H = q_B_T_N_H.reshape(q_B_T_N_H.shape[0], q_B_T_N_H.shape[1], K, G, self.head_dim)
        k_B_S_K_H_view = k_B_S_K_H.reshape(k_B_S_K_H.shape[0], k_B_S_K_H.shape[1], K, self.head_dim)
        v_B_S_K_H_view = v_B_S_K_H.reshape(v_B_S_K_H.shape[0], v_B_S_K_H.shape[1], K, self.head_dim)

        # einsum "BTKGH,BSKH->BKGTS"
        logits_B_K_G_T_S = torch.einsum("BTKGH,BSKH->BKGTS", q_B_T_K_G_H.float(), k_B_S_K_H_view.float())

        # Align mask to logits shape: logits is (B, K, G, T, S), mask is (B, 1, T, S)
        # We need mask to be (B, 1, 1, T, S) so it broadcasts to (B, K, G, T, S)
        big_neg = -2.3819763e38
        mask_B_K_G_T_S = attn_mask_B_1_T_S[:, :, None, :, :].expand_as(logits_B_K_G_T_S).bool()
        masked_logits_B_K_G_T_S = torch.where(
            mask_B_K_G_T_S,
            logits_B_K_G_T_S,
            torch.tensor(big_neg, dtype=logits_B_K_G_T_S.dtype, device=logits_B_K_G_T_S.device),
        )

        probs_B_K_G_T_S = F.softmax(masked_logits_B_K_G_T_S, dim=-1).to(dtype)

        # einsum "BKGTS,BSKH->BTKGH"
        encoded_B_T_N_H = torch.einsum("BKGTS,BSKH->BTKGH", probs_B_K_G_T_S, v_B_S_K_H_view.to(dtype))
        encoded_B_T_N_H = encoded_B_T_N_H.reshape(
            encoded_B_T_N_H.shape[0], encoded_B_T_N_H.shape[1], K * G, self.head_dim
        )
        # encoded: (B, T_total, num_heads, head_dim)

        # Split back to per-expert outputs
        outputs = []
        start = 0
        for i, x_B_T_D in enumerate(xs):
            if x_B_T_D is not None:
                end = start + x_B_T_D.shape[1]
                expert_out_B_T_D = encoded_B_T_N_H[:, start:end].reshape(B, x_B_T_D.shape[1], -1)
                expert_out_B_T_D = self.o_proj[i](expert_out_B_T_D)
                outputs.append(expert_out_B_T_D)
                start = end
            else:
                outputs.append(None)

        return outputs, new_kv_cache


class FeedForward(nn.Module):
    """Feed forward module."""

    def __init__(self, features: int, hidden_dim: int):
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.w_gating = nn.Parameter(torch.empty(2, features, hidden_dim))
        self.w_linear = nn.Parameter(torch.empty(hidden_dim, features))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w_gating[0], std=1.0 / math.sqrt(self.features))
        nn.init.normal_(self.w_gating[1], std=1.0 / math.sqrt(self.features))
        nn.init.normal_(self.w_linear, std=1.0 / math.sqrt(self.hidden_dim))

    def forward(self, x_B_T_D: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward projection."""
        dtype = x_B_T_D.dtype
        ff_gate_B_T_M = torch.matmul(x_B_T_D, self.w_gating[0].to(dtype))
        ff1_B_T_M = torch.matmul(x_B_T_D, self.w_gating[1].to(dtype))
        activations_B_T_M = gelu_glu(ff_gate_B_T_M, ff1_B_T_M)
        outputs = torch.matmul(activations_B_T_M, self.w_linear.to(dtype))
        return outputs


class Block(nn.Module):
    """Transformer block with multi-expert attention and FFN."""

    def __init__(
        self,
        configs: Sequence[Config],
        adarms: Sequence[bool] | bool = False,
        dropout: float = 0.0,
        attention_backend: str = 'sdpa',
    ):
        super().__init__()
        self.configs = configs
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if isinstance(adarms, bool):
            adarms = [adarms] * len(configs)
        self.adarms = list(adarms)

        self.attn = Attention(configs, attention_backend=attention_backend)
        self.pre_attention_norms = nn.ModuleList(
            [RMSNorm(c.width, adaptive=adarms[i]) for i, c in enumerate(configs)]
        )
        self.pre_ffw_norms = nn.ModuleList(
            [RMSNorm(c.width, adaptive=adarms[i]) for i, c in enumerate(configs)]
        )

        self.mlps = nn.ModuleList([FeedForward(c.width, c.mlp_dim) for c in configs])

    def forward(
        self,
        xs: list[torch.Tensor | None],
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        positions_B_T: torch.Tensor,
        attn_mask_B_1_T_S: torch.Tensor,
        adarms_cond: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            xs: per-expert inputs
            kv_cache: optional KV cache
            positions: (B, T) position indices
            attn_mask: attention mask
            adarms_cond: per-expert adaptive RMS norm conditioning (None = regular)

        Returns:
            (outputs, new_kv_cache)
        """
        # Pre-attention norm
        pre_attn = []
        gates = []
        for i, x_B_T_D in enumerate(xs):
            if x_B_T_D is not None:
                x_norm_B_T_D, gate_B_1_D = self.pre_attention_norms[i](x_B_T_D, adarms_cond[i])
                pre_attn.append(x_norm_B_T_D)
                gates.append(gate_B_1_D)
            else:
                pre_attn.append(None)
                gates.append(None)

        # Attention
        post_attn, kv_cache = self.attn(pre_attn, positions_B_T, attn_mask_B_1_T_S, kv_cache)
        post_attn = [self.dropout(p) if p is not None else None for p in post_attn]

        # Gated residual for attention
        xs = [
            _gated_residual(x_B_T_D, y_B_T_D, gate_B_1_D)
            for x_B_T_D, y_B_T_D, gate_B_1_D in zip(xs, post_attn, gates, strict=True)
        ]

        # Pre-FFN norm
        out = []
        gates = []
        for i, (x_B_T_D, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x_B_T_D is not None:
                x_norm_B_T_D, gate_B_1_D = self.pre_ffw_norms[i](x_B_T_D, adarms_cond[i])
                x_ffn_B_T_D = self.mlps[i](x_norm_B_T_D)
                out.append(x_ffn_B_T_D)
                gates.append(gate_B_1_D)
            else:
                out.append(None)
                gates.append(None)

        # Dropout and gated residual for FFN
        out = [self.dropout(o) if o is not None else None for o in out]
        xs = [
            _gated_residual(x_B_T_D, y_B_T_D, gate_B_1_D)
            for x_B_T_D, y_B_T_D, gate_B_1_D in zip(xs, out, gates, strict=True)
        ]

        return xs, kv_cache


class Module(nn.Module):
    """Gemma transformer with multi-expert support."""

    def __init__(
        self,
        configs: Sequence[Config],
        embed_dtype: str = "bfloat16",
        adarms: Sequence[bool] | bool = False,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = True,
        attention_backend: str = 'sdpa',
    ):
        super().__init__()
        self.configs = configs
        self.embed_dtype = _str_to_dtype(embed_dtype)
        if isinstance(adarms, bool):
            adarms = [adarms] * len(configs)
        self.adarms = list(adarms)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=configs[0].width,
        )

        self.layers = nn.ModuleList(
            [
                Block(
                    configs,
                    adarms=self.adarms,
                    dropout=dropout,
                    attention_backend=attention_backend,
                )
                for _ in range(configs[0].depth)
            ]
        )

        self.final_norms = nn.ModuleList(
            [RMSNorm(c.width, adaptive=self.adarms[i]) for i, c in enumerate(configs)]
        )

        self.gradient_checkpointing = use_gradient_checkpointing
        # Non-reentrant checkpointing supports the nested expert inputs.
        self.gradient_checkpointing_use_reentrant = False

    def embed(self, tokens_B_L: torch.Tensor) -> torch.Tensor:
        """Embed token indices."""
        return self.embedder(tokens_B_L).to(self.embed_dtype)

    def forward(
        self,
        embedded: Sequence[torch.Tensor | None],
        positions_B_T: torch.Tensor,
        mask_B_T_S: torch.Tensor,
        adarms_cond: Sequence[torch.Tensor | None] | None = None,
        *,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
        """Full transformer forward pass.

        Args:
            embedded: list of embedded tokens, one per expert (or None)
            positions: (B, T) position indices
            mask: (B, T, S) attention mask (bool)
            adarms_cond: per-expert adaptive conditioning (or None)
            kv_cache: optional KV cache for inference

        Returns:
            (outputs, new_kv_cache)
        """
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        # Expand mask: (B, T, S) -> (B, 1, T, S)
        mask_B_T_S = mask_B_T_S.unsqueeze(1)

        xs = list(embedded)

        # Cast all embedded tokens to embed_dtype, matching JAX:
        #   embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        xs = [e.to(self.embed_dtype) if e is not None else None for e in xs]

        # KV cache: per-layer cache for inference. During training, no cache is used.
        # In JAX's nn.scan, each layer gets its own element from the cache list.
        # We replicate this with per-layer caches.
        if kv_cache is None:
            layer_kv_caches = [None] * len(self.layers)
        else:
            layer_kv_caches = (
                list(kv_cache)
                if isinstance(kv_cache, list | tuple)
                else [kv_cache] * len(self.layers)
            )

        new_layer_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv = layer_kv_caches[i] if i < len(layer_kv_caches) else None
            if self.gradient_checkpointing and self.training:
                xs, new_kv = torch.utils.checkpoint.checkpoint(
                    layer,
                    xs,
                    layer_kv,
                    positions_B_T,
                    mask_B_T_S,
                    adarms_cond,
                    use_reentrant=self.gradient_checkpointing_use_reentrant,
                )
            else:
                xs, new_kv = layer(xs, layer_kv, positions_B_T, mask_B_T_S, adarms_cond)
            new_layer_kv_caches.append(new_kv)

        # Return the list of per-layer KV caches
        kv_cache = tuple(new_layer_kv_caches)

        # Final norm
        outputs = []
        for i in range(len(self.configs)):
            if xs[i] is not None:
                out, _ = self.final_norms[i](xs[i], adarms_cond[i])
                outputs.append(out)
            else:
                outputs.append(None)

        return outputs, kv_cache


def _apply_rope(
    x_B_T_D: torch.Tensor, *, positions_B_T: torch.Tensor, max_wavelength: float = 10000.0
) -> torch.Tensor:
    """Apply RoPE to input tensor."""
    head_dim = x_B_T_D.shape[-1]
    freq_exponents_H2 = (2.0 / head_dim) * torch.arange(
        head_dim // 2, dtype=torch.float32, device=x_B_T_D.device
    )
    timescale_H2 = max_wavelength**freq_exponents_H2
    radians_B_T_1_H2 = positions_B_T[..., None].float() / timescale_H2[None, None, :]
    radians_B_T_1_H2 = radians_B_T_1_H2[..., None, :]  # (B, T, 1, head_dim//2)

    sin_B_T_1_H2, cos_B_T_1_H2 = torch.sin(radians_B_T_1_H2), torch.cos(radians_B_T_1_H2)
    x1_B_T_N_H2, x2_B_T_N_H2 = torch.chunk(x_B_T_D, 2, dim=-1)
    res_B_T_N_H = torch.cat([x1_B_T_N_H2 * cos_B_T_1_H2 - x2_B_T_N_H2 * sin_B_T_1_H2, x2_B_T_N_H2 * cos_B_T_1_H2 + x1_B_T_N_H2 * sin_B_T_1_H2], dim=-1)
    return res_B_T_N_H.to(x_B_T_D.dtype)



def _gated_residual(x_B_T_D, y_B_T_D, gate_B_1_D):
    if x_B_T_D is None:
        return None
    if gate_B_1_D is None:
        return x_B_T_D + y_B_T_D
    return x_B_T_D + y_B_T_D * gate_B_1_D
