# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0; see licenses/RLinf-LICENSE.
"""Dense pi0.5 flow model extracted from RLinf, without task/framework wrappers."""
from dataclasses import dataclass, replace

import torch
from torch import nn
import torch.nn.functional as F

from . import gemma, siglip
from .config import Pi05Config

IMAGE_KEYS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')


@dataclass
class Observation:
    """Already transformed model input; images are B x 224 x 224 x 3 float32."""

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    foveation_outputs: dict[str, dict[str, torch.Tensor]] | None = None
    patches: dict[str, torch.Tensor] | None = None
    patch_bank_indices: dict[str, torch.Tensor] | None = None
    levels: dict[str, torch.Tensor] | None = None
    token_valid_masks: dict[str, torch.Tensor] | None = None


def make_attn_mask(valid_B_N, autoregressive_N):
    """Combine prefix validity with block-causal attention boundaries."""
    blocks_B_N = autoregressive_N.expand(valid_B_N.shape[0], -1).int().cumsum(1)
    return ((blocks_B_N[:, None, :] <= blocks_B_N[:, :, None])
            & valid_B_N[:, None, :] & valid_B_N[:, :, None])


def posemb_sincos(time_B, width, min_period=4e-3, max_period=4.0):
    """Encode flow time with the upstream sine/cosine periods."""
    fraction_D2 = torch.linspace(0, 1, width // 2, device=time_B.device, dtype=torch.float32)
    period_D2 = min_period * (max_period / min_period) ** fraction_D2
    phase_B_D2 = torch.einsum('i,j->ij', time_B.float(), 1 / period_D2 * 2 * torch.pi)
    return torch.cat((phase_B_D2.sin(), phase_B_D2.cos()), dim=-1).to(time_B.dtype)


class Pi05Model(nn.Module):
    """SigLIP, PaliGemma and adaRMS action expert with native flow matching."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        self.llm = gemma.Module(
            [gemma.get_config('gemma_2b'), gemma.get_config('gemma_300m')],
            embed_dtype=config.dtype, adarms=[False, True],
            use_gradient_checkpointing=config.gradient_checkpointing,
            attention_backend=config.attention_backend)
        self.img = siglip.SigLIPViT(config.dtype, config.gradient_checkpointing,
            level_embedding=config.visual_input_mode == 'mixed_foveation' and config.foveation_level_embedding,
            patch_level_adapter=config.visual_input_mode == 'mixed_foveation' and config.foveation_patch_level_adapter,
            patch_level_adapter_channels=config.foveation_patch_level_adapter_channels,
            apt_position_encoding=config.visual_input_mode == 'mixed_foveation' and config.foveation_apt_position_encoding)
        self.action_in_proj = nn.Linear(32, 1024)
        self.action_out_proj = nn.Linear(1024, 32)
        self.time_mlp_in = nn.Linear(1024, 1024)
        self.time_mlp_out = nn.Linear(1024, 1024)
        for layer in (self.action_in_proj, self.action_out_proj, self.time_mlp_in, self.time_mlp_out):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        self.set_precision(config.dtype)

    def set_precision(self, precision: str):
        """Preserve upstream fp32 islands and fp32 action/time projections."""
        self.config = replace(self.config, dtype=precision)
        dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[precision]
        self.llm.to(dtype=dtype)
        self.img.to(dtype=dtype)
        self.llm.embed_dtype = self.img.dtype_mm = dtype
        if dtype == torch.bfloat16:
            for name, param_D in self.named_parameters():
                if any(part in name for part in ('img.stem.', 'img.pos_embedding',
                                                 'pre_attention_norms', 'pre_ffw_norms', 'final_norms')):
                    param_D.data = param_D.data.float()
        for layer in (self.action_in_proj, self.action_out_proj, self.time_mlp_in, self.time_mlp_out):
            layer.float()

    def set_gradient_checkpointing(self, enabled: bool):
        """Enable non-reentrant recomputation without changing train/eval mode."""
        self.config = replace(self.config, gradient_checkpointing=enabled)
        self.llm.gradient_checkpointing = enabled
        self.img.encoder.gradient_checkpointing = enabled

    def embed_prefix(self, observation: Observation):
        """Encode cameras in checkpoint order, followed by language tokens."""
        tokens, masks = [], []
        if self.config.visual_input_mode == 'mixed_foveation' and observation.foveation_outputs is None and observation.patches is None:
            raise ValueError('mixed_foveation requires packed patch metadata')
        for name in IMAGE_KEYS:
            if self.config.visual_input_mode == 'mixed_foveation':
                routed = (observation.foveation_outputs or {})
                if name in routed:
                    patches = routed[name]['patches_B_N_C_P_P']
                    bank_indices = routed[name]['patch_bank_indices_B_N']
                    levels = routed[name]['levels_B_N']
                    valid = routed[name]['token_valid_mask_B_N']
                else:
                    patches = observation.patches[name]
                    bank_indices = observation.patch_bank_indices[name]
                    levels = observation.levels[name]
                    valid = observation.token_valid_masks[name]
                image_valid_B_N = observation.image_masks[name][:, None] & valid
                active_B = image_valid_B_N.any(-1)
                # Avoid all-masked softmax rows for the placeholder camera.
                image_B_N_D = torch.zeros(*image_valid_B_N.shape, 2048,
                    device=image_valid_B_N.device, dtype=self.img.dtype_mm)
                if active_B.any():
                    image_B_N_D[active_B] = self.img.forward_patches(
                        patches[active_B], bank_indices[active_B], levels[active_B], image_valid_B_N[active_B])
            else:
                image_B_N_D = self.img(observation.images[name])
                image_valid_B_N = observation.image_masks[name][:, None].expand(-1, image_B_N_D.shape[1])
            tokens.append(image_B_N_D)
            masks.append(image_valid_B_N)
        tokens.append(self.llm.embed(observation.tokenized_prompt))
        masks.append(observation.tokenized_prompt_mask)
        prefix_B_P_D = torch.cat(tokens, dim=1)
        valid_B_P = torch.cat(masks, dim=1)
        return prefix_B_P_D, valid_B_P, torch.zeros(valid_B_P.shape[1], dtype=torch.bool, device=valid_B_P.device)

    def embed_suffix(self, action_B_T_D, time_B):
        """Encode noisy actions and adaRMS time conditioning; no state token."""
        suffix_B_T_D = self.action_in_proj(action_B_T_D.float())
        time_B_D = posemb_sincos(time_B.float(), 1024)
        time_B_D = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_B_D))))
        valid_B_T = torch.ones(suffix_B_T_D.shape[:2], dtype=torch.bool, device=action_B_T_D.device)
        autoregressive_T = torch.zeros(valid_B_T.shape[1], dtype=torch.bool, device=action_B_T_D.device)
        autoregressive_T[0] = True
        return suffix_B_T_D, valid_B_T, autoregressive_T, time_B_D

    def compute_loss(self, observation, action_B_T_D, *, noise=None, time=None):
        """Return B x action_horizon x 32 native velocity MSE."""
        action_B_T_D = action_B_T_D.float()
        noise_B_T_D = torch.randn_like(action_B_T_D) if noise is None else noise.to(action_B_T_D)
        if time is None:
            # Match RLinf's CPU Beta sampling; restoring CPU RNG restores flow times.
            time_B = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0)).sample((action_B_T_D.shape[0],))
            time_B = time_B.to(action_B_T_D) * 0.999 + 0.001
        else:
            time_B = time.to(action_B_T_D)
        if noise_B_T_D.shape != action_B_T_D.shape or time_B.shape != action_B_T_D.shape[:1]:
            raise ValueError(
                f'Expected noise Bx{self.config.action_horizon}x32 and time B')
        noisy_B_T_D = time_B[:, None, None] * noise_B_T_D + (1 - time_B[:, None, None]) * action_B_T_D
        prefix_B_P_D, prefix_valid_B_P, prefix_ar_P = self.embed_prefix(observation)
        suffix_B_T_D, suffix_valid_B_T, suffix_ar_T, time_B_D = self.embed_suffix(noisy_B_T_D, time_B)
        valid_B_N = torch.cat((prefix_valid_B_P, suffix_valid_B_T), dim=1)
        ar_N = torch.cat((prefix_ar_P, suffix_ar_T))
        outputs, _ = self.llm(
            [prefix_B_P_D, suffix_B_T_D], positions_B_T=valid_B_N.int().cumsum(1) - 1,
            mask_B_T_S=make_attn_mask(valid_B_N, ar_N), adarms_cond=[None, time_B_D])
        velocity_B_T_D = self.action_out_proj(outputs[1].float())
        return (velocity_B_T_D - (noise_B_T_D - action_B_T_D)).square()

    def build_prefix_cache(self, observation):
        """Build one observation's prefix cache for all Euler steps."""
        prefix_B_P_D, valid_B_P, ar_P = self.embed_prefix(observation)
        outputs, cache = self.llm(
            [prefix_B_P_D, None], positions_B_T=valid_B_P.int().cumsum(1) - 1,
            mask_B_T_S=make_attn_mask(valid_B_P, ar_P))
        return outputs[0], valid_B_P, cache

    def run_suffix(self, action_B_T_D, time_B, prefix_valid_B_P, cache):
        """Predict velocity using an immutable prefix KV cache."""
        suffix_B_T_D, valid_B_T, ar_T, time_B_D = self.embed_suffix(action_B_T_D, time_B)
        mask_B_T_N = torch.cat((prefix_valid_B_P[:, None, :].expand(-1, valid_B_T.shape[1], -1),
                                make_attn_mask(valid_B_T, ar_T)), dim=-1)
        positions_B_T = prefix_valid_B_P.sum(-1)[:, None] + valid_B_T.int().cumsum(-1) - 1
        outputs, _ = self.llm([None, suffix_B_T_D], positions_B_T=positions_B_T,
                              mask_B_T_S=mask_B_T_N, adarms_cond=[None, time_B_D], kv_cache=cache)
        return self.action_out_proj(outputs[1].float())

    @torch.no_grad()
    def sample_actions(self, observation, *, noise=None, generator=None):
        """Generate a normalized action chunk with the ordinary Euler solver."""
        batch_size = observation.state.shape[0]
        device = observation.state.device
        shape = (batch_size, self.config.action_horizon, self.config.action_dim)
        action_B_T_D = (torch.randn(shape, device=device, generator=generator) if noise is None
                        else noise.to(device=device, dtype=torch.float32))
        if action_B_T_D.shape != shape:
            raise ValueError(f'Expected initial noise {shape}')
        _, valid_B_P, cache = self.build_prefix_cache(observation)
        dt = -1.0 / self.config.num_steps
        time_value = 1.0
        for _ in range(self.config.num_steps):
            time_B = torch.full((batch_size,), time_value, device=device, dtype=torch.float32)
            velocity_B_T_D = self.run_suffix(action_B_T_D, time_B, valid_B_P, cache)
            action_B_T_D = action_B_T_D + dt * velocity_B_T_D
            time_value += dt
        return action_B_T_D
