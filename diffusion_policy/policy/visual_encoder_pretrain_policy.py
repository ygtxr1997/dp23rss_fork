from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import diffusion_policy.model.vision.crop_randomizer as dmvc

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
from robomimic.models.obs_nets import ObservationEncoder


class VisualEncoderPretrainPolicy(BaseImagePolicy):
    def __init__(
            self,
            shape_meta: dict,
            horizon: int,
            n_obs_steps: int,
            crop_shape=(76, 76),
            obs_encoder_group_norm: bool = False,
            eval_fixed_crop: bool = False,
            regressor_hidden_dim: int = 256,
            regressor_dropout: float = 0.0,
            agent_pos_loss_weight: float = 1.0,
            action_loss_weight: float = 1.0,
            image_only_input: bool = True,
            state_target_key: str = "agent_pos",
    ):
        super().__init__()
        assert horizon >= 1, f"`horizon` should be >= 1, got {horizon}"
        assert n_obs_steps >= 1, f"`n_obs_steps` should be >= 1, got {n_obs_steps}"
        assert n_obs_steps <= horizon, f"`n_obs_steps` should be <= `horizon`, got {n_obs_steps} > {horizon}"

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                obs_config['rgb'].append(key)
            elif obs_type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        config: DictConfig = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')

        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for _, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for _, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        base_policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )
        obs_encoder: ObservationEncoder = base_policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16),
                    num_channels=x.num_features)
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_feature_dim = obs_encoder.output_shape()[0]
        if state_target_key not in shape_meta["obs"]:
            raise ValueError(
                f"`state_target_key={state_target_key}` not in shape_meta['obs']. "
                f"Available keys: {list(shape_meta['obs'].keys())}"
            )
        state_dim = shape_meta["obs"][state_target_key]["shape"][0]
        max_future_action_steps = horizon - (n_obs_steps - 1)
        full_state_steps = horizon

        if regressor_hidden_dim > 0:
            state_head = nn.Sequential(
                nn.Linear(obs_feature_dim * n_obs_steps, regressor_hidden_dim),
                nn.GELU(),
                nn.Dropout(regressor_dropout),
                nn.Linear(regressor_hidden_dim, full_state_steps * state_dim),
            )
            action_head = nn.Sequential(
                nn.Linear(obs_feature_dim * n_obs_steps, regressor_hidden_dim),
                nn.GELU(),
                nn.Dropout(regressor_dropout),
                nn.Linear(regressor_hidden_dim, max_future_action_steps * action_dim),
            )
        else:
            state_head = nn.Linear(obs_feature_dim * n_obs_steps, full_state_steps * state_dim)
            action_head = nn.Linear(obs_feature_dim * n_obs_steps, max_future_action_steps * action_dim)

        self.obs_encoder = obs_encoder
        self.state_head = state_head
        self.action_head = action_head
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_future_action_steps = max_future_action_steps
        self.full_state_steps = full_state_steps
        self.agent_pos_loss_weight = float(agent_pos_loss_weight)
        self.action_loss_weight = float(action_loss_weight)
        self.image_only_input = bool(image_only_input)
        self.state_target_key = state_target_key

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            obs_encoder_weight_decay: float,
            regressor_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        optim_groups = [
            {
                "params": self.obs_encoder.parameters(),
                "weight_decay": obs_encoder_weight_decay
            },
            {
                "params": list(self.state_head.parameters()) + list(self.action_head.parameters()),
                "weight_decay": regressor_weight_decay
            }
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas
        )
        return optimizer

    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        norm_keys = set(self.normalizer.params_dict.keys())
        obs_in = {k: v for k, v in obs_dict.items() if k in norm_keys}
        if len(obs_in) == 0:
            raise ValueError(
                f"No observation key matches normalizer keys. obs keys={list(obs_dict.keys())}, "
                f"normalizer keys={list(norm_keys)}"
            )

        nobs = self.normalizer.normalize(obs_in)
        value = next(iter(nobs.values()))
        batch_size, total_steps = value.shape[:2]
        used_steps = min(total_steps, self.n_obs_steps)
        if used_steps != self.n_obs_steps:
            raise ValueError(
                f"Observed steps mismatch: expect >= {self.n_obs_steps}, got {total_steps}"
            )
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :used_steps, ...].reshape(-1, *x.shape[2:])
        )
        if self.image_only_input:
            this_nobs = {
                k: (v if k == "image" else torch.zeros_like(v))
                for k, v in this_nobs.items()
            }
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(batch_size, used_steps, -1)
        return nobs_features

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "obs" not in batch or "action" not in batch:
            raise ValueError("batch must contain `obs` and `action`.")
        if self.state_target_key not in batch["obs"]:
            raise ValueError(f"batch['obs'] must contain `{self.state_target_key}`.")

        obs_features = self.encode_obs(batch["obs"])  # [B, To, D]
        batch_size, used_steps, _ = obs_features.shape
        if used_steps != self.n_obs_steps:
            raise ValueError(
                f"Observed steps mismatch after encoding: expect {self.n_obs_steps}, got {used_steps}"
            )

        flat_features = obs_features.reshape(batch_size, -1)

        pred_state_flat = self.state_head(flat_features)
        pred_state = pred_state_flat.reshape(batch_size, self.full_state_steps, self.state_dim)
        target_state = batch["obs"][self.state_target_key][:, :self.full_state_steps, :]
        if target_state.shape != pred_state.shape:
            raise ValueError(
                f"State shape mismatch: pred {pred_state.shape} vs target {target_state.shape}."
            )

        action_start = used_steps - 1
        target_actions = batch["action"][:, action_start:, :]  # [B, T-To+1, D_action]
        pred_action_flat = self.action_head(flat_features)
        pred_actions_all = pred_action_flat.reshape(batch_size, self.max_future_action_steps, self.action_dim)
        pred_actions = pred_actions_all[:, :target_actions.shape[1], :]
        if pred_actions.shape != target_actions.shape:
            raise ValueError(
                f"Action shape mismatch: pred {pred_actions.shape} vs target {target_actions.shape}."
            )

        state_loss = F.mse_loss(pred_state, target_state)
        action_loss = F.mse_loss(pred_actions, target_actions)
        total_loss = self.agent_pos_loss_weight * state_loss + self.action_loss_weight * action_loss

        state_mae = F.l1_loss(pred_state, target_state)
        return {
            "total_loss": total_loss,
            "state_loss": state_loss.detach(),
            "action_loss": action_loss.detach(),
            "state_mae": state_mae.detach(),
            "action_mae": F.l1_loss(pred_actions, target_actions).detach(),
            # backward compatibility for existing workspace code
            "agent_pos_loss": state_loss.detach(),
            "agent_pos_mae": state_mae.detach(),
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compute_loss(batch)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("VisualEncoderPretrainPolicy does not support action prediction.")
