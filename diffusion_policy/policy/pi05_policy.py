"""Standalone raw-observation pi0.5 policy for the dense LIBERO baseline."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from diffusion_policy.model.pi05.config import Pi05Config
from diffusion_policy.model.pi05.model import Pi05Model
from diffusion_policy.model.pi05.preprocessing import Pi05Preprocessor
from diffusion_policy.model.pi05.checkpoint import load_model_weights, sha256_file
from diffusion_policy.model.roi.foveation_net import FoveationNet
from diffusion_policy.model.roi.multi_view_foveation_net import MultiViewFoveationNet
from diffusion_policy.model.pi05.model import Observation, IMAGE_KEYS
from diffusion_policy.common.foveation_pretrained import load_foveation_artifact
from diffusion_policy.common.tcl_data_util import transform_values


class Pi05Policy(nn.Module):
    """Own model-specific transforms and produce controller-space action chunks."""

    def __init__(self, model_path=None, tokenizer_path=None, norm_stats_path=None,
                 config=None, assets=None):
        super().__init__()
        if assets is None:
            if tokenizer_path is None or norm_stats_path is None:
                raise ValueError('Provide tokenizer and matching normalization assets explicitly')
            assets = dict(tokenizer_proto=Path(tokenizer_path).read_bytes(),
                          norm_stats=json.loads(Path(norm_stats_path).read_text()))
        model_config = Pi05Config(**dict(config or {}))
        self._assets = copy.deepcopy(assets)
        self._assets['tokenizer_sha256'] = hashlib.sha256(self._assets['tokenizer_proto']).hexdigest()
        self.preprocessor = Pi05Preprocessor(
            self._assets['tokenizer_proto'], self._assets['norm_stats'],
            model_config.max_token_len, model_config.image_size, model_config.visual_input_mode,
            model_config.foveation_split_mode, model_config.foveation_random_seed,
            render_image_orientation=model_config.render_image_orientation,
            normalization=self._assets.get('normalization'))
        self.model = Pi05Model(model_config)
        # Lightweight diagnostics populated by the shared routing path.
        self.last_visual_stats = None
        self.stage_b_phase = 'native'
        self.stage_b_init_report = None
        self.foveation_artifact = None
        if model_config.visual_input_mode == 'mixed_foveation':
            template = FoveationNet(
                input_hw=(model_config.image_size, model_config.image_size),
                num_levels=model_config.foveation_num_levels,
                patch_size=model_config.foveation_patch_size,
                fix_split_mode=model_config.foveation_split_mode,
                router_type=model_config.foveation_router_type,
                global_encoder_type=model_config.foveation_global_encoder_type,
                global_feature_dim=model_config.foveation_global_feature_dim,
                global_encoder_resize=model_config.foveation_global_encoder_resize,
                torch_home=model_config.foveation_torch_home,
                router_layers=model_config.foveation_router_layers,
                router_hidden_dim=model_config.foveation_router_hidden_dim,
                router_heads=model_config.foveation_router_heads,
                router_mlp_ratio=model_config.foveation_router_mlp_ratio,
                router_dropout=model_config.foveation_router_dropout,
                router_cond_dim=model_config.foveation_router_cond_dim,
                router_temperature=model_config.foveation_router_temperature,
                random_topology_prob=model_config.foveation_random_topology_prob,
                target_num_patches=model_config.foveation_target_num_patches,
                budget_loss_weight=model_config.foveation_budget_loss_weight,
                budget_loss_type=model_config.foveation_budget_loss_type,
                budget_base_cost_weight=model_config.foveation_budget_base_cost_weight,
                budget_probability_type=model_config.foveation_budget_probability_type,
                budget_elu_tau=model_config.foveation_budget_elu_tau,
                freeze_global_encoder=model_config.foveation_freeze_global_encoder,
            )
            self.foveation_net = MultiViewFoveationNet(('image', 'gripper'), template)
            if model_config.foveation_router_type != 'none':
                # DP-compatible condition: six TCP values plus a ten-slot task one-hot.
                self.router_condition_projection = nn.Linear(16, model_config.foveation_router_cond_dim)
                nn.init.zeros_(self.router_condition_projection.bias)
        if model_path is not None:
            path = load_model_weights(self.model, model_path)
            self._assets['base_weights_sha256'] = sha256_file(path)
            manifest = path.parent / 'manifest.json'
            if manifest.exists():
                self._assets['base_manifest'] = json.loads(manifest.read_text())

    @property
    def config(self):
        """Return the effective model configuration saved with the policy."""
        return self.model.config

    @property
    def render_image_orientation(self):
        """Return the static-camera input orientation used by preprocessing."""
        return self.config.render_image_orientation

    @property
    def device(self):
        """Return the model device; precision is deliberately module-specific."""
        return self.model.action_in_proj.weight.device

    def forward(self, batch, **kwargs):
        """Dispatch native SFT or the differentiable StageB paired objective."""
        paired = kwargs.pop('paired', self.stage_b_phase == 'router_only')
        if paired:
            return self.compute_paired_loss(batch, **kwargs)
        return self.compute_loss(batch, **kwargs)

    def compute_loss(self, batch, *, noise=None, time=None, reduction='mean', augmentation_generator=None):
        """Return mean native loss, or one loss per window with reduction='none'."""
        observation = self.preprocessor.prepare(batch['obs'], device=self.device, train=self.training,
                                                 augmentation_generator=augmentation_generator)
        observation.foveation_outputs = self._route_observation(
            observation, task_ids=batch.get('task_id'), raw_state=batch['obs'].get('joint_state'))
        actions_B_T_D = torch.as_tensor(batch['action'], device=self.device, dtype=torch.float32)
        expected_shape = (observation.state.shape[0], self.config.action_horizon, 7)
        if actions_B_T_D.shape != expected_shape:
            raise ValueError(f'action must be a full raw Bx{self.config.action_horizon}x7 window')
        actions_B_T_D = self.preprocessor.normalize(actions_B_T_D, 'actions')
        loss_B_T_D = self.model.compute_loss(observation, actions_B_T_D, noise=noise, time=time)
        # Ordinary SFT supervises the complete action horizon, including padded coordinates.
        loss_B = loss_B_T_D.mean(dim=(1, 2))
        if reduction == 'none':
            return loss_B
        if reduction != 'mean':
            raise ValueError("reduction must be 'mean' or 'none'")
        return loss_B.mean()

    def set_stage_b_phase(self, phase: str) -> None:
        """Switch between trainable router paired loss and downstream SFT."""
        if phase not in ('native', 'router_only', 'downstream'):
            raise ValueError("phase must be 'native', 'router_only', or 'downstream'")
        self.stage_b_phase = phase
        if not hasattr(self, 'foveation_net'):
            return
        router_modules = [self.foveation_net]
        if hasattr(self, 'router_condition_projection'):
            router_modules.append(self.router_condition_projection)
        for module in router_modules:
            module.requires_grad_(phase == 'router_only')
            if phase == 'router_only' and hasattr(module, 'nets'):
                for net in module.nets.values():
                    if net.freeze_global_encoder and net.global_encoder is not None:
                        net.global_encoder.requires_grad_(False)
        self.model.requires_grad_(phase != 'router_only')
        self._set_patch_adapter_grad(phase == 'native')
        if phase == 'router_only':
            self.model.eval()
            self.foveation_net.train()
            if hasattr(self, 'router_condition_projection'):
                self.router_condition_projection.train()
        elif phase == 'downstream':
            self.model.train()
            self.foveation_net.eval()
            if hasattr(self, 'router_condition_projection'):
                self.router_condition_projection.eval()
        else:
            self.train()

    def _set_patch_adapter_grad(self, enabled: bool) -> None:
        """Keep StageA SigLIP adapter/APT parameters fixed throughout StageB."""
        image_encoder = getattr(self.model, 'img', None)
        if image_encoder is None:
            return
        for name in ('patch_level_adapters', 'apt_position'):
            module = getattr(image_encoder, name, None)
            if module is not None:
                module.requires_grad_(enabled)
        level_embedding = getattr(image_encoder, 'level_embedding', None)
        if level_embedding is not None:
            level_embedding.requires_grad_(enabled)

    def load_foveation_artifact(self, path) -> None:
        """Load compatible DP routing weights and condition statistics."""
        artifact = load_foveation_artifact(path)
        if list(self.foveation_net.view_keys) != artifact['camera_order']:
            raise ValueError('Routing artifact camera order differs from the policy')
        if self.config.render_image_orientation != artifact['camera_orientation']['image']:
            raise ValueError('Routing artifact static-camera orientation differs')
        for key, net in self.foveation_net.nets.items():
            if tuple(net.input_hw) != (224, 224) or net.patch_size != 14 or net.num_levels != 3:
                raise ValueError('Routing artifact requires Pi05 224/patch14/3 levels')
            if any(getattr(net, name) != value for name, value in artifact['router_config'].items()):
                raise ValueError(f'Routing artifact configuration differs in {key}')
            source = artifact['branches'][key]
            expected = {name for name in net.state_dict()
                        if name.startswith(('global_encoder.', 'router.'))}
            if set(source) != expected:
                raise ValueError(f'Routing artifact parameter keys differ in {key}')
            # Load only transferable modules; retain Pi05's 224/patch14 geometry buffers.
            for prefix in ('global_encoder', 'router'):
                module_state = {name[len(prefix) + 1:]: value for name, value in source.items()
                                if name.startswith(prefix + '.')}
                getattr(net, prefix).load_state_dict(module_state, strict=True)
        self.router_condition_projection.load_state_dict(artifact['condition_projection'], strict=True)
        self.foveation_artifact = artifact
        self.set_router_temperature(float(artifact['router_temperature']))

    def train(self, mode=True):
        """Reapply StageB train/eval constraints after a caller changes mode."""
        super().train(mode)
        if not hasattr(self, 'foveation_net'):
            return self
        if self.stage_b_phase == 'router_only':
            self.model.eval()
            self.foveation_net.train(mode)
            if hasattr(self, 'router_condition_projection'):
                self.router_condition_projection.train(mode)
        elif self.stage_b_phase == 'downstream':
            self.model.train(mode)
            self.foveation_net.eval()
            if hasattr(self, 'router_condition_projection'):
                self.router_condition_projection.eval()
        return self

    def set_target_num_patches(self, target_num_patches):
        """Update the per-camera differentiable patch budget."""
        if hasattr(self, 'foveation_net'):
            return self.foveation_net.set_target_num_patches(target_num_patches)
        return None

    def set_router_temperature(self, router_temperature: float) -> None:
        """Set the sampling temperature on both camera routers."""
        if hasattr(self, 'foveation_net'):
            self.foveation_net.set_router_temperature(router_temperature)

    def compute_paired_loss(self, batch, *, noise=None, time=None,
                            paired_loss_weight=1.0, temporal_loss_weight=1.0):
        """Compute Spatial/Temporal router loss with shared flow inputs.

        The paired action losses are detached advantages; only selected router
        logits receive the policy-gradient style signal in the router-only phase.
        """
        if not hasattr(self, 'foveation_net') or self.stage_b_phase != 'router_only':
            raise ValueError('Paired loss requires a mixed-foveation router-only policy')
        temporal = batch.get('temporal_pair')
        temporal_valid = batch.get('temporal_pair_valid')
        if temporal is not None and temporal_valid is None:
            temporal_valid = temporal.get('temporal_pair_valid', temporal.get('valid'))
        temporal_valid = None if temporal_valid is None else temporal_valid.to(self.device).bool()
        if temporal_valid is not None:
            obs = {key: torch.cat((batch['obs'][key], temporal['obs'][key]), dim=0)
                   if torch.is_tensor(batch['obs'][key]) else batch['obs'][key] + temporal['obs'][key]
                   for key in batch['obs']}
            actions_raw = torch.cat((batch['action'], temporal['action']), dim=0)
            task_ids = torch.cat((batch['task_id'], temporal['task_id']), dim=0)
            raw_state = torch.cat((batch['obs']['joint_state'], temporal['obs']['joint_state']), dim=0)
        else:
            obs, actions_raw = batch['obs'], batch['action']
            task_ids = batch.get('task_id')
            raw_state = batch['obs'].get('joint_state')
        observation = self.preprocessor.prepare(obs, device=self.device, train=True)
        cond_B_T_D = self._router_condition(observation, task_ids, raw_state)
        images = {
            key: observation.images[source].permute(0, 3, 1, 2).unsqueeze(1)
            for key, source in zip(('image', 'gripper'), IMAGE_KEYS[:2])
        }
        routing, aux = self.foveation_net(images, cond_B_T_D=cond_B_T_D, return_routing_state=True)
        self.last_visual_stats = {
            'camera_tokens': {key: routing[key]['num_patches_B_T'].detach().cpu().long().tolist()
                              for key in routing},
            'expected_camera_tokens': {key: routing[key]['expected_num_patches_B_T'].detach().cpu().tolist()
                                       for key in routing},
            'random_topology_fraction': {key: routing[key]['random_topology_mask_B_T'].float().mean().item()
                                         for key in routing},
            'budget_loss': {key: routing[key]['budget_loss'].detach().item() for key in routing},
        }
        intervention = self.foveation_net.sample_paired_interventions(routing, temporal_valid)
        B = actions_raw.shape[0]
        actions_B_T_D = self.preprocessor.normalize(
            torch.as_tensor(actions_raw, device=self.device, dtype=torch.float32), 'actions')
        if noise is None:
            noise_B_T_D = torch.randn_like(actions_B_T_D)
        else:
            noise_B_T_D = noise.to(device=self.device, dtype=actions_B_T_D.dtype)
        if time is None:
            time_sample = torch.distributions.Beta(1.5, 1.0).sample(
                (temporal_valid.shape[0] if temporal_valid is not None else B,)
            ).to(self.device) * 0.999 + 0.001
            time_B = torch.cat((time_sample, time_sample), dim=0) \
                if temporal_valid is not None else time_sample
        else:
            time_B = time.to(self.device)
            if temporal_valid is not None:
                time_B = torch.cat((time_B[:temporal_valid.shape[0]],
                                    time_B[:temporal_valid.shape[0]]), dim=0)
        stop_losses = []
        split_losses = []
        actual_delta_B = torch.zeros(B, dtype=torch.long, device=self.device)
        for arm_name in ('stop', 'split'):
            arm_outputs = {}
            for key, out in routing.items():
                net = self.foveation_net.nets[key]
                frame_B = intervention['selected_frame_B']
                patch_bank = out['patch_bank_B_T_K_C_P_P'][torch.arange(B, device=self.device), frame_B]
                split_hard = out['split_hard_B_T_M'][torch.arange(B, device=self.device), frame_B].clone()
                record = intervention['by_view'].get(key)
                if record is not None:
                    local = torch.zeros(B, dtype=torch.bool, device=self.device)
                    local[record['batch_indices_R']] = True
                    chosen = record[f'{arm_name}_R_M']
                    split_hard[local] = chosen
                packed = net.pack_topology(patch_bank[:, None], split_hard[:, None])
                if temporal_valid is not None and arm_name == 'stop':
                    batch_indices_R = intervention['by_view'].get(key, {}).get('batch_indices_R')
                    if batch_indices_R is not None:
                        base_count_R = out['num_patches_B_T'][
                            batch_indices_R, frame_B[batch_indices_R]]
                        changed_count_R = packed['num_patches_B_T'][batch_indices_R, 0]
                        base_is_split_R = intervention['base_split_B'][batch_indices_R]
                        # Only the stop arm changes windows whose selected node
                        # was split in the base tree. The split arm handles the
                        # complementary windows in the next arm pass.
                        actual_delta_B[batch_indices_R] = torch.where(
                            base_is_split_R,
                            base_count_R - changed_count_R,
                            actual_delta_B[batch_indices_R],
                        )
                if temporal_valid is not None and arm_name == 'split':
                    batch_indices_R = intervention['by_view'].get(key, {}).get('batch_indices_R')
                    if batch_indices_R is not None:
                        base_count_R = out['num_patches_B_T'][
                            batch_indices_R, frame_B[batch_indices_R]]
                        changed_count_R = packed['num_patches_B_T'][batch_indices_R, 0]
                        base_is_split_R = intervention['base_split_B'][batch_indices_R]
                        actual_delta_B[batch_indices_R] = torch.where(
                            ~base_is_split_R,
                            changed_count_R - base_count_R,
                            actual_delta_B[batch_indices_R],
                        )
                # Keep the shared [B, T=1, ...] contract; the camera adapter
                # removes T exactly once below.
                arm_outputs[key] = packed
            observation.foveation_outputs = self._adapt_foveation_outputs(arm_outputs, B)
            loss_B = self.model.compute_loss(observation, actions_B_T_D,
                                             noise=noise_B_T_D, time=time_B).mean(dim=(1, 2))
            (stop_losses if arm_name == 'stop' else split_losses).append(loss_B)
        stop_loss_B, split_loss_B = stop_losses[0], split_losses[0]
        advantage_B = stop_loss_B - split_loss_B
        selected_logit_B = intervention['selected_logits_B']
        spatial_terms_B = -selected_logit_B.sigmoid() * advantage_B.detach()
        paired_loss = spatial_terms_B.mean()
        base_loss = torch.where(intervention['base_split_B'], split_loss_B, stop_loss_B).mean()
        temporal_loss = advantage_B.new_zeros(())
        temporal_match = advantage_B.new_zeros(())
        temporal_valid_count = advantage_B.new_zeros(())
        temporal_matched_count = advantage_B.new_zeros(())
        P = 0
        valid = torch.zeros(0, dtype=torch.bool, device=advantage_B.device)
        gain_P = advantage_B.new_zeros((0,))
        temporal_terms_P = advantage_B.new_zeros((0,))
        if temporal_valid is not None:
            P = temporal_valid.shape[0]
            valid = temporal_valid & (actual_delta_B[:P] == actual_delta_B[P:])
            gain_P = advantage_B[:P] - advantage_B[P:]
            prefer_a_P = (selected_logit_B[:P] - selected_logit_B[P:]).sigmoid()
            temporal_terms_P = -prefer_a_P * gain_P.detach() * valid.to(gain_P.dtype)
            temporal_loss = temporal_terms_P.mean()
            temporal_valid_count = temporal_valid.sum().to(gain_P.dtype)
            temporal_matched_count = valid.sum().to(gain_P.dtype)
            temporal_match = temporal_matched_count / temporal_valid_count.clamp_min(1.0)
        budget_loss = aux['budget_loss']
        view_metric_sums = dict(aux['_view_metric_sums'])
        view_metric_counts = dict(aux['_view_metric_counts'])

        def add_view_metric(name, values_B, views_B, weights_B=None):
            values_B = values_B.detach().float()
            if weights_B is None:
                weights_B = torch.ones_like(values_B)
            sums_V = values_B.new_zeros(len(self.foveation_net.view_keys))
            counts_V = values_B.new_zeros(len(self.foveation_net.view_keys))
            view_metric_sums[name] = sums_V.scatter_add_(0, views_B, values_B)
            view_metric_counts[name] = counts_V.scatter_add_(0, views_B, weights_B.float())

        selected_view_B = intervention['selected_view_B']
        for name, values_B in (
                ('paired_loss', spatial_terms_B),
                ('paired_advantage_mean', advantage_B),
                ('paired_advantage_abs_mean', advantage_B.abs()),
                ('paired_advantage_positive_fraction', (advantage_B > 0).float()),
                ('paired_stop_diffusion_loss', stop_loss_B),
                ('paired_split_diffusion_loss', split_loss_B)):
            add_view_metric(name, values_B, selected_view_B)
        total_loss = (float(paired_loss_weight) * paired_loss
                      + float(temporal_loss_weight) * temporal_loss
                      + budget_loss)
        if temporal_valid is not None:
            add_view_metric('temporal_loss', temporal_terms_P, selected_view_B[:P])
            add_view_metric('temporal_gain', gain_P, selected_view_B[:P])
            add_view_metric('temporal_match_fraction', valid, selected_view_B[:P], temporal_valid)
        temporal_valid_by_view_V = advantage_B.new_zeros(len(self.foveation_net.view_keys))
        temporal_matched_by_view_V = advantage_B.new_zeros(len(self.foveation_net.view_keys))
        if temporal_valid is not None:
            temporal_valid_by_view_V.scatter_add_(0, selected_view_B[:P], temporal_valid.float())
            temporal_matched_by_view_V.scatter_add_(0, selected_view_B[:P], valid.float())
        result = dict(loss=total_loss, total_loss=total_loss, diffusion_loss=base_loss.detach(),
                    paired_loss=paired_loss, paired_stop_diffusion_loss=stop_loss_B.mean().detach(),
                    paired_split_diffusion_loss=split_loss_B.mean().detach(),
                    paired_advantage_mean=advantage_B.mean().detach(), paired_advantage_abs_mean=advantage_B.abs().mean().detach(),
                    paired_advantage_positive_fraction=(advantage_B > 0).float().mean().detach(),
                    temporal_loss=temporal_loss, temporal_match_fraction=temporal_match.detach(),
                    temporal_valid_pair_count=temporal_valid_count,
                    temporal_matched_pair_count=temporal_matched_count,
                    budget_loss=budget_loss, budget_lambda=budget_loss.new_tensor(
                        self.foveation_net.nets['image'].budget_lambda),
                    expected_num_patches_mean=torch.stack([routing[key]['expected_num_patches_B_T'].mean()
                                                            for key in routing]).mean().detach(),
                    num_patches_mean=torch.stack([routing[key]['num_patches_B_T'].float().mean()
                                                  for key in routing]).mean().detach(),
                    target_num_patches=budget_loss.new_tensor(
                        self.foveation_net.resolved_target_num_patches or 0.0),
                    _view_metric_sums=view_metric_sums,
                    _view_metric_counts=view_metric_counts,
                    _temporal_view_valid_count=temporal_valid_by_view_V,
                    _temporal_view_matched_count=temporal_matched_by_view_V,
                    routing=routing)
        result.update({key: value for key, value in aux.items()
                      if key.startswith('_budget_')})
        return result

    @torch.no_grad()
    def predict_action(self, obs, *, noise=None, generator=None, return_foveation_outputs=False):
        """Return raw action predictions and the configured execution prefix."""
        was_training = self.training
        self.eval()
        try:
            observation = self.preprocessor.prepare(obs, device=self.device)
            observation.foveation_outputs = self._route_observation(
                observation, task_ids=obs.get('task_id'), raw_state=obs.get('joint_state'))
            actions_B_T_D = self.model.sample_actions(observation, noise=noise, generator=generator)
            actions_B_T_D = self.preprocessor.decode_actions(actions_B_T_D)
            result = {'action_pred': actions_B_T_D, 'action': actions_B_T_D[:, :self.config.n_action_steps]}
            if return_foveation_outputs:
                result['_foveation_outputs'] = self._public_foveation_outputs(observation.foveation_outputs)
            return result
        finally:
            self.train(was_training)

    def reset(self):
        """No cross-observation cache; leave the caller's random generator untouched."""

    def get_extra_state(self):
        """Embed tokenizer, statistics and configuration in ordinary state_dict saves."""
        artifact_contract = None
        if self.foveation_artifact is not None:
            artifact_contract = {key: copy.deepcopy(self.foveation_artifact[key])
                                 for key in ('state_mean', 'state_std', 'state_eps',
                                             'router_temperature', 'camera_order', 'target_geometry')}
        return {'version': 5, 'config': asdict(self.config), 'assets': copy.deepcopy(self._assets),
                'foveation_artifact': artifact_contract}

    def get_split_sampler_state(self):
        """Retained for callers of older workspace code; topology uses Torch RNG."""
        return None

    def set_split_sampler_state(self, state):
        """Compatibility no-op for pre-v3 checkpoints."""

    def set_extra_state(self, state):
        """Restore preprocessing without relying on the original asset paths."""
        saved_config = Pi05Config(**state['config'])
        if state['version'] not in (1, 2, 3, 4, 5) or asdict(saved_config) != asdict(self.config):
            raise ValueError('Policy configuration differs from the saved checkpoint')
        if state['version'] == 1 and saved_config.visual_input_mode == 'mixed_foveation':
            raise ValueError('Legacy mixed checkpoints lack the corrected topology/sampler state; restart Stage A from base')
        self._assets = copy.deepcopy(state['assets'])
        self.preprocessor = Pi05Preprocessor(
            self._assets['tokenizer_proto'], self._assets['norm_stats'],
            self.config.max_token_len, self.config.image_size, self.config.visual_input_mode,
            self.config.foveation_split_mode, self.config.foveation_random_seed,
            render_image_orientation=self.config.render_image_orientation,
            normalization=self._assets.get('normalization'))
        artifact_contract = state.get('foveation_artifact')
        if artifact_contract is not None:
            self.foveation_artifact = artifact_contract
            self.set_router_temperature(float(artifact_contract['router_temperature']))

    def _router_condition(self, observation, task_ids=None, raw_state=None):
        """Build [B,1,16] DP condition from raw six-state values and task IDs."""
        if raw_state is None:
            state_B_6 = observation.state[:, :6]
        else:
            state_B_6 = torch.as_tensor(raw_state, device=self.device, dtype=torch.float32)[:, :6]
        if self.foveation_artifact is not None and 'state_normalization' in self.foveation_artifact:
            state_B_6 = transform_values(state_B_6, self.foveation_artifact['state_normalization'])
        elif self.foveation_artifact is not None:
            mean_B_6 = self.foveation_artifact['state_mean'].to(self.device)
            std_B_6 = self.foveation_artifact['state_std'].to(self.device)
            state_B_6 = (state_B_6 - mean_B_6) / std_B_6.clamp_min(self.foveation_artifact['state_eps'])
        if task_ids is None:
            task_ids_B = torch.zeros(state_B_6.shape[0], dtype=torch.long, device=self.device)
        else:
            task_ids_tensor = torch.as_tensor(task_ids, device=self.device)
            if task_ids_tensor.ndim == 2 and task_ids_tensor.shape[-1] == 10:
                task_ids_B = task_ids_tensor.argmax(dim=-1).long()
            else:
                task_ids_B = task_ids_tensor.to(dtype=torch.long).reshape(-1)
        if task_ids_B.shape[0] != state_B_6.shape[0] or (task_ids_B < 0).any() or (task_ids_B >= 10).any():
            raise ValueError('task_id must be a B-vector with values in [0, 9]')
        task_onehot_B_10 = torch.zeros((state_B_6.shape[0], 10), device=self.device, dtype=state_B_6.dtype)
        task_onehot_B_10.scatter_(1, task_ids_B[:, None], 1.0)
        condition_B_16 = torch.cat((state_B_6, task_onehot_B_10), dim=-1)
        return self.router_condition_projection(condition_B_16[:, None].to(
            self.router_condition_projection.weight.dtype))

    def _route_observation(self, observation, *, return_routing_state=False, task_ids=None, raw_state=None):
        """Route the two real cameras and adapt outputs to Pi0.5 camera slots."""
        if not hasattr(self, 'foveation_net'):
            return None
        images = {
            key: observation.images[source].permute(0, 3, 1, 2).unsqueeze(1)
            for key, source in zip(('image', 'gripper'), IMAGE_KEYS[:2])
        }
        cond_B_T_D = None
        if hasattr(self, 'router_condition_projection'):
            cond_B_T_D = self._router_condition(observation, task_ids, raw_state)
        routed, _ = self.foveation_net(images, cond_B_T_D=cond_B_T_D,
                                       return_routing_state=return_routing_state)
        self.last_visual_stats = {
            'camera_tokens': {
                key: routed[key]['num_patches_B_T'].detach().cpu().long().tolist()
                for key in routed
            },
            'expected_camera_tokens': {
                key: routed[key]['expected_num_patches_B_T'].detach().cpu().tolist()
                for key in routed
            },
            'random_topology_fraction': {
                key: routed[key]['random_topology_mask_B_T'].float().mean().item()
                for key in routed
            },
            'budget_loss': {key: routed[key]['budget_loss'].detach().item() for key in routed},
        }
        return self._adapt_foveation_outputs(routed, observation.state.shape[0])

    def _adapt_foveation_outputs(self, routed, batch_size):
        """Map real camera trees to model slots and append a masked placeholder."""
        outputs = {}
        for key, out in routed.items():
            adapted = {}
            for name, value in out.items():
                if not torch.is_tensor(value):
                    continue
                if '_B_T' in name:
                    if value.ndim < 2 or value.shape[1] != 1:
                        raise ValueError('Pi05 camera adapter requires a single observation frame')
                    adapted[name.replace('_B_T', '_B', 1)] = value[:, 0]
                elif '_B' in name:
                    # Paired arms are already packed to [B, ...].
                    adapted[name] = value
            outputs[IMAGE_KEYS[0] if key == 'image' else IMAGE_KEYS[1]] = adapted
        # Placeholder camera has no valid prefix tokens and never encodes black pixels.
        outputs[IMAGE_KEYS[2]] = {
            'patches_B_N_C_P_P': torch.zeros(batch_size, 1, 3, 14, 14, device=self.device),
            'patch_bank_indices_B_N': torch.full((batch_size, 1), -1, dtype=torch.long, device=self.device),
            'levels_B_N': torch.zeros((batch_size, 1), dtype=torch.long, device=self.device),
            'token_valid_mask_B_N': torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device),
        }
        return outputs

    @staticmethod
    def _public_foveation_outputs(outputs):
        """Expose routing under the policy's external camera names."""
        public = {}
        for key, value in outputs.items():
            if key not in IMAGE_KEYS[:2]:
                continue
            public['image' if key == IMAGE_KEYS[0] else 'gripper'] = {
                name.replace('_B', '_B_T', 1): tensor_B_X[:, None]
                for name, tensor_B_X in value.items() if torch.is_tensor(tensor_B_X)
            }
        return public

    @torch.no_grad()
    def get_foveation_outputs(self, obs):
        """Prepare raw observations and return routing metadata without sampling actions."""
        observation = obs if isinstance(obs, Observation) else self.preprocessor.prepare(obs, device=self.device)
        return self._public_foveation_outputs(self._route_observation(
            observation, task_ids=None if isinstance(obs, Observation) else obs.get('task_id'),
            raw_state=None if isinstance(obs, Observation) else obs.get('joint_state')))

    @torch.no_grad()
    def compute_resolution_diagnostics(self, batch, *, seed, batch_size):
        """Return paired native flow losses and full-task leaf sets with bounded model batches."""
        if self.training or not hasattr(self, 'foveation_net'):
            raise ValueError('Resolution diagnostics require mixed_foveation in eval mode')
        if batch_size <= 0:
            raise ValueError('Diagnostic batch_size must be positive')
        actions_B_T_D = torch.as_tensor(batch['action'], device=self.device, dtype=torch.float32)
        expected_tail = (self.config.action_horizon, 7)
        if actions_B_T_D.ndim != 3 or tuple(actions_B_T_D.shape[1:]) != expected_tail or actions_B_T_D.shape[0] == 0:
            raise ValueError(
                f'Diagnostics require a nonempty raw Bx{self.config.action_horizon}x7 action window')
        actions_B_T_D = self.preprocessor.normalize(actions_B_T_D, 'actions')
        rng_devices = [self.device] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=rng_devices):
            # Draw task-wide flow inputs independently of routing and microbatch size.
            torch.random.default_generator.manual_seed(seed)
            time_B = torch.distributions.Beta(1.5, 1.0).sample((actions_B_T_D.shape[0],))
            time_B = time_B.to(actions_B_T_D) * 0.999 + 0.001
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noise_B_T_D = torch.randn(actions_B_T_D.shape, device=self.device,
                                      dtype=actions_B_T_D.dtype, generator=generator)
            torch.random.default_generator.manual_seed(seed)
            if self.device.type == 'cuda':
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
            losses, leaves = [], {key: [] for key in self.foveation_net.view_keys}
            for start in range(0, actions_B_T_D.shape[0], batch_size):
                stop = start + batch_size
                obs = {key: value[start:stop] for key, value in batch['obs'].items()}
                observation = self.preprocessor.prepare(obs, device=self.device)
                images = {
                    key: observation.images[source].permute(0, 3, 1, 2).unsqueeze(1)
                    for key, source in zip(('image', 'gripper'), IMAGE_KEYS[:2])
                }
                cond_B_T_D = self._router_condition(
                    observation, batch.get('task_id')[start:stop],
                    batch['obs'].get('joint_state')[start:stop]) if hasattr(self, 'router_condition_projection') else None
                routed, _ = self.foveation_net(
                    images, cond_B_T_D=cond_B_T_D, return_routing_state=True)
                for key, out in routed.items():
                    # Fixed-width CPU metadata retains cross-microbatch Jaccard pairs.
                    indices_B_T_N = out['patch_bank_indices_B_T_N'].detach().cpu()
                    valid_B_T_N = out['token_valid_mask_B_T_N'].detach().cpu()
                    pad = self.foveation_net.num_bank_patches - indices_B_T_N.shape[-1]
                    leaves[key].append((torch.nn.functional.pad(indices_B_T_N, (0, pad), value=-1),
                                        torch.nn.functional.pad(valid_B_T_N, (0, pad), value=False)))
                level_losses = []
                for level in range(self.foveation_net.num_levels):
                    level_by_view = {}
                    for key, net in self.foveation_net.nets.items():
                        out = routed[key]
                        split_B_T_M = torch.zeros_like(out['split_hard_B_T_M'])
                        split_B_T_M[..., :net.internal_level_offsets[level]] = True
                        level_by_view[key] = net.pack_topology(out['patch_bank_B_T_K_C_P_P'], split_B_T_M)
                    observation.foveation_outputs = self._adapt_foveation_outputs(
                        level_by_view, observation.state.shape[0])
                    loss_B_T_D = self.model.compute_loss(
                        observation, actions_B_T_D[start:stop],
                        noise=noise_B_T_D[start:stop], time=time_B[start:stop])
                    # Match SFT, including the padded action coordinates.
                    level_losses.append(loss_B_T_D.mean(dim=(1, 2)).cpu())
                losses.append(torch.stack(level_losses, dim=-1))
            outputs = {
                key: {'patch_bank_indices_B_T_N': torch.cat([item[0] for item in chunks]),
                      'token_valid_mask_B_T_N': torch.cat([item[1] for item in chunks])}
                for key, chunks in leaves.items()
            }
        return {'loss_B_L': torch.cat(losses), 'foveation_outputs': outputs}

    def save_policy(self, path):
        """Save an inference policy; optimizer/EMA training state belongs to the workspace."""
        torch.save(self.state_dict(), path)

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        """Restore a self-contained policy without reloading base assets."""
        state = torch.load(Path(path).expanduser(), map_location='cpu', weights_only=True, mmap=True)
        if '_extra_state' not in state:
            raise ValueError('Expected a self-contained policy .pt; full training checkpoints are not inference files')
        extra = state['_extra_state']
        policy = cls(config=extra['config'], assets=extra['assets'])
        policy.load_state_dict(state, strict=True)
        return policy.to(device=device).eval()
