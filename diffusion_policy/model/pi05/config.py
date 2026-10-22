"""Explicit configuration for the dense LIBERO pi0.5 baseline."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Pi05Config:
    """Model shape and inference settings shared with saved policy assets."""

    dtype: str = 'bfloat16'
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    num_steps: int = 5
    n_action_steps: int = 5
    discrete_state_input: bool = False
    extra_delta_transform: bool = False
    gradient_checkpointing: bool = False
    attention_backend: str = 'sdpa'
    render_image_orientation: str = 'rot180'
    image_size: int = 224
    visual_input_mode: str = 'dense'
    foveation_patch_size: int = 14
    foveation_num_levels: int = 3
    foveation_split_mode: object = 'level:2'
    foveation_position_encoding: str = 'siglip_multiscale'
    foveation_level_embedding: bool = True
    foveation_level_adapter: bool = False
    foveation_patch_level_adapter: bool = False
    foveation_patch_level_adapter_channels: int = 64
    foveation_apt_position_encoding: bool = False
    foveation_random_seed: int = 42
    foveation_router_type: str = 'none'
    foveation_global_encoder_type: str = 'resnet10'
    foveation_global_feature_dim: int = 512
    foveation_global_encoder_resize: object = None
    foveation_torch_home: object = None
    foveation_freeze_global_encoder: bool = True
    foveation_router_layers: int = 4
    foveation_router_hidden_dim: int = 512
    foveation_router_heads: int = 8
    foveation_router_mlp_ratio: float = 4.0
    foveation_router_dropout: float = 0.0
    foveation_router_cond_dim: int = 512
    foveation_router_temperature: float = 1.0
    foveation_random_topology_prob: float = 0.0
    foveation_target_num_patches: object = None
    foveation_budget_loss_weight: float = 0.0
    foveation_budget_loss_type: str = 'per_observation_mse'
    foveation_budget_base_cost_weight: float = 0.01
    foveation_budget_probability_type: str = 'sampled_relaxed'
    foveation_budget_elu_tau: float = 0.025

    def __post_init__(self):
        # Hydra ListConfig must not leak into weights-only-loadable checkpoints.
        if not isinstance(self.foveation_split_mode, str) and self.foveation_split_mode is not None:
            object.__setattr__(self, 'foveation_split_mode', list(self.foveation_split_mode))
        # Keep the pre-existing short field as a read-compatible alias.
        if self.foveation_level_adapter and not self.foveation_patch_level_adapter:
            object.__setattr__(self, 'foveation_patch_level_adapter', True)
        if self.render_image_orientation not in ('raw', 'rot180'):
            raise ValueError("render_image_orientation must be 'raw' or 'rot180'")
        if self.dtype not in ('float32', 'bfloat16'):
            raise ValueError('pi0.5 supports float32 or selected-parameter bfloat16')
        if self.action_dim != 32 or not 1 <= self.action_horizon <= 50 or self.max_token_len != 200:
            raise ValueError('Pi05 requires action_dim=32, horizon in [1, 50], token_len=200')
        if self.discrete_state_input or self.extra_delta_transform:
            raise ValueError('The fixed LIBERO recipe disables state tokens and extra deltas')
        if self.attention_backend not in ('sdpa', 'math'):
            raise ValueError("attention_backend must be 'sdpa' or 'math'")
        if self.visual_input_mode not in ('dense', 'mixed_foveation'):
            raise ValueError('visual_input_mode must be dense or mixed_foveation')
        if self.foveation_patch_size != 14 or self.foveation_num_levels != 3:
            raise ValueError('Stage A uses three 14-pixel levels')
        if self.visual_input_mode == 'mixed_foveation':
            if self.image_size != 224:
                raise ValueError('mixed_foveation requires image_size=224')
            if self.foveation_position_encoding != 'siglip_multiscale':
                raise ValueError('mixed_foveation requires siglip_multiscale positions')
            if self.foveation_patch_level_adapter_channels <= 0:
                raise ValueError('foveation_patch_level_adapter_channels must be positive')
            if self.foveation_split_mode is not None and self.foveation_router_type != 'none':
                raise ValueError('Fixed mixed topology requires foveation_router_type=none')
        if self.visual_input_mode == 'mixed_foveation' and self.foveation_split_mode is None:
            if self.foveation_router_type not in ('pooled_global', 'node_aligned_fpn'):
                raise ValueError('Learned mixed topology requires a learned router')
        if self.foveation_router_type in ('pooled_global', 'node_aligned_fpn'):
            if self.foveation_global_feature_dim <= 0 or self.foveation_router_cond_dim <= 0:
                raise ValueError('Router feature and condition dimensions must be positive')
            if self.foveation_router_temperature <= 0:
                raise ValueError('Router temperature must be positive')
        if self.foveation_global_encoder_resize is not None:
            resize = int(self.foveation_global_encoder_resize)
            if resize <= 0 or resize % 16 != 0:
                raise ValueError('foveation_global_encoder_resize must be a positive multiple of 16')
        if self.foveation_router_type == 'node_aligned_fpn':
            if self.foveation_global_encoder_type != 'resnet10':
                raise ValueError('node_aligned_fpn requires resnet10')
            if self.foveation_global_encoder_resize != 64:
                raise ValueError('Pi05 node_aligned_fpn requires global_encoder_resize=64')
        jepa_dims = {
            'vjepa2_1_vit_base_384': 768,
            'vjepa2_1_vit_large_384': 1024,
            'vjepa2_1_vit_giant_384': 1408,
            'vjepa2_1_vit_gigantic_384': 1664,
        }
        if self.foveation_global_encoder_type in jepa_dims:
            expected_dim = jepa_dims[self.foveation_global_encoder_type]
            if self.foveation_global_feature_dim != expected_dim:
                raise ValueError(
                    f'{self.foveation_global_encoder_type} requires '
                    f'foveation_global_feature_dim={expected_dim}')
            if self.foveation_router_type == 'pooled_global' and self.foveation_global_encoder_resize is None:
                raise ValueError('JEPA pooled_global routing requires foveation_global_encoder_resize')
        if self.foveation_budget_loss_weight < 0 or self.foveation_budget_base_cost_weight < 0:
            raise ValueError('Budget weights must be non-negative')
        if not 0.0 <= self.foveation_random_topology_prob <= 1.0:
            raise ValueError('foveation_random_topology_prob must be in [0, 1]')
        if self.image_size < 14:
            raise ValueError('image_size must be at least one SigLIP patch (14 pixels)')
        if self.num_steps <= 0 or not 1 <= self.n_action_steps <= self.action_horizon:
            raise ValueError('Invalid flow or execution step count')
