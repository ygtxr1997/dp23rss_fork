from typing import Dict, Tuple, List, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import builtins, types, inspect
from omegaconf import DictConfig
import pathlib
import dill
import copy
import hydra

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
from robomimic.models.obs_nets import ObservationEncoder
from robomimic.models.base_nets import VisualCore
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            # da
            use_da=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
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

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder: ObservationEncoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
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

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
            is_da=use_da,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # For real-world inference
        self.infer_frame_idx = 0
        self.cached_action = None
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        ## remove useless args
        useless_keys = ['use_da', 'domain_adapt']
        passed_kwargs = {k: v for k, v in self.kwargs.items() if k not in useless_keys}
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **passed_kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


class DiffusionTransformerHybridImagePolicyHDFree(DiffusionTransformerHybridImagePolicy):
    def __init__(self,
                 domain_adapt: DictConfig,
                 **kwargs):
        super().__init__(**kwargs)
        self.src_ckpt = domain_adapt.src_ckpt
        self.load_from_src_ckpt(self.src_ckpt)  # will have 2 copies: one is for target; another is for source (fixed)

        # (Source) Create model copies for domain adaptation AFTER loading pretrained weights
        self.src_obs_encoder: ObservationEncoder = copy.deepcopy(self.obs_encoder)
        self.src_model: TransformerForDiffusion = copy.deepcopy(self.model)
        self.placeholder_param = torch.nn.Parameter(torch.ones([1]))

        # (Loss) For domain adaptation
        self.use_da_vis1: bool = domain_adapt.use_da_visual in ('both', 'static')
        self.use_da_vis2: bool = domain_adapt.use_da_visual in ('both', 'gripper')  # Not implemented!
        self.use_da_act: bool = domain_adapt.use_da_act
        self.act_loss_from: str = domain_adapt.act_loss_from
        self.act_layers: int = domain_adapt.act_layers
        self.act_weights: str = domain_adapt.act_weights

        # (Target) Register Adapter blocks for trainable modules
        from diffusion_policy.model.domain_adapt.wgan import WGAN_GP
        if 'adapter' in domain_adapt.act_weights:
            self.model.register_adapter()
        debug_diff_loss = domain_adapt.debug_diff_loss
        if debug_diff_loss:
            self.use_da_vis1 = self.use_da_vis2 = False
        if self.use_da_vis1:
            self.da_vis1_loss: WGAN_GP = copy.deepcopy(domain_adapt.visual_da).to(self.device)
        if self.use_da_vis2:
            self.da_vis2_loss: WGAN_GP = copy.deepcopy(domain_adapt.visual_da).to(self.device)
        if self.use_da_act:
            self.da_act_loss: WGAN_GP = copy.deepcopy(domain_adapt.action_da).to(self.device)
        self.cache_da_d_loss = 0.
        self.cache_wdist = 0.
        self.cache_da_g_loss = 0.
        self.shuffle_target_goal = domain_adapt.shuffle_target_goal
        self.cfg_drop_ratio = domain_adapt.cfg_drop_ratio
        self.reg_source_diff_loss = domain_adapt.reg_source_diff_loss
        self.use_dann_lambda = domain_adapt.use_dann_lambda
        self.debug_diff_loss = domain_adapt.debug_diff_loss

        self.optimizer_config = domain_adapt.optimizer_config

        # (Target) Using Monkey Patching trick to replace class method
        def forward_with_da(self: VisualCore, inputs):
            """
            Forward pass through visual core.
            inputs: (112, 3, 84, 84)
            """
            ndim = len(self.input_shape)
            assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
            x = self.nets[:-2](inputs)
            # print(x.shape, x.mean(), x.min(), x.max())
            x = self.nets[-2](x)
            # print(x.shape, x.mean(), x.min(), x.max())
            x = self.nets[-1](x)  # (112, 64)
            self.cache_vis_out = x
            # print("[DEBUG] forward_with_da:", inputs.shape, x.shape, x.mean(), x.min(), x.max())
            if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
                raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                    str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
                                 )
            return x

        self.src_obs_encoder.obs_nets['image'].forward = types.MethodType(
            forward_with_da, self.src_obs_encoder.obs_nets['image'])
        self.obs_encoder.obs_nets['image'].forward = types.MethodType(
            forward_with_da, self.obs_encoder.obs_nets['image'])

    def load_from_src_ckpt(self, src_ckpt, use_ema: bool = True):
        self.src_ckpt = src_ckpt
        path = pathlib.Path(src_ckpt)
        payload = torch.load(path.open('rb'), pickle_module=dill)

        def load_state_dict_partial(model, state_dict, skip_keys=[]):
            """
            加载状态字典时跳过某些键
            Args:
                model (nn.Module): 模型
                state_dict (dict): 要加载的状态字典
                skip_keys (list): 需要跳过的键列表
            """
            # 过滤掉需要跳过的键
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k not in skip_keys:
                    need_skip = False
                    for skip_key in skip_keys:  # skip_keys may contain short keywords
                        if skip_key in k:
                            need_skip = True
                    if not need_skip:
                        filtered_state_dict[k] = v

            # 加载过滤后的状态字典
            # 获取模型的 state_dict 键和 filtered_state_dict 键
            model_keys = set(model.state_dict().keys())
            state_dict_keys = set(filtered_state_dict.keys())

            # 找到 missing_keys 和 unexpected_keys
            missing_keys = list(model_keys - state_dict_keys)
            unexpected_keys = list(state_dict_keys - model_keys)

            for k in missing_keys:
                filtered_state_dict[k] = model.state_dict()[k]

            model.load_state_dict(filtered_state_dict, strict=True)

            # Load normalizer
            norm_keys = set({k: None for k in state_dict.keys() if 'normalizer.' in k}.keys())
            norm_state_dict = {k[len("normalizer."):]: v for k, v in state_dict.items() if 'normalizer.' in k}
            model.normalizer.load_state_dict(norm_state_dict)

            # 打印结果
            print("Skip keys:", skip_keys)
            print("Normalizer keys:", norm_keys)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", list(set(unexpected_keys) - norm_keys))

        # print(payload['state_dicts']['model'].keys())
        # print(self.normalizer.state_dict().keys())
        # exit()
        skip_keys = [
            '.ia3_',  # IA3 Adapter
        ]
        if use_ema:
            load_state_dict_partial(self, payload['state_dicts']['ema_model'], skip_keys)
        else:
            load_state_dict_partial(self, payload['state_dicts']['model'], skip_keys)
        print(f"[DiffusionTransformerHybridImagePolicyHDFree] Loaded src_ckpt (use_ema={use_ema}) "
              f"from: {self.src_ckpt}")

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data, condition_mask,
                           cond=None, generator=None,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            # cond: torch.Size([56, 2, 66]) timesteps: torch.Size([]) noisy_trajectory torch.Size([56, 10, 2])
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)  # (B,66)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        ## remove useless args
        useless_keys = ['use_da', 'domain_adapt']
        passed_kwargs = {k: v for k, v in self.kwargs.items() if k not in useless_keys}
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **passed_kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @staticmethod
    def set_requires_grad(model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad
        if requires_grad:
            model.train()
        else:
            model.eval()

    @staticmethod
    def freeze_params(model):
        def set_parameter_requires_grad(model, requires_grad):
            for name, child in model.named_children():
                for param in child.parameters():
                    param.requires_grad = requires_grad
        set_parameter_requires_grad(model, requires_grad=False)

    @staticmethod
    def trainable_params(model, sub_name: str = None):
        m = model
        if sub_name is not None:
            m = getattr(model, sub_name)
        return filter(lambda p: p.requires_grad, m.parameters())

    def calc_grad_and_param_norm(self, module: Union[nn.Module, List[nn.Module]],
                                 sqrt_out: bool = True,
                                 ):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        total_ratio_norm = 0.0
        if isinstance(module, list):
            for m in module:
                m_grad, m_param, m_ratio = self.calc_grad_and_param_norm(m, sqrt_out=False)  # recursive
                total_grad_norm += m_grad
                total_param_norm += m_param
                total_ratio_norm += m_ratio
        else:
            assert isinstance(module, nn.Module)
            for name, p in module.named_parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
                    total_ratio_norm += (p.grad.norm().item() / (1e-8 + p.data.norm().item())) ** 2
                total_param_norm += p.norm().item() ** 2
        if sqrt_out:
            total_grad_norm = total_grad_norm ** 0.5
            total_param_norm = total_param_norm ** 0.5
            total_ratio_norm = total_ratio_norm ** 0.5
        return total_grad_norm, total_param_norm, total_ratio_norm

    def get_optimizer(
            self,
            transformer_weight_decay: float,
            obs_encoder_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]
    ) -> List[torch.optim.Optimizer]:
        g_vis1_optim_groups = []
        d_vis1_optim_groups = []
        g_act_optim_groups = []
        d_act_optim_groups = []

        ''' Frozen modules '''
        self.set_requires_grad(self.src_obs_encoder, False)
        self.set_requires_grad(self.src_model, False)

        ''' Visual Encoder '''
        self.set_requires_grad(self.obs_encoder, False)
        if self.use_da_vis1:
            self.freeze_params(self.obs_encoder)
            self.set_requires_grad(self.obs_encoder.obs_nets['image'].nets[-1], True)
            g_vis1_optim_groups.extend([
                {"params": self.trainable_params(self.obs_encoder), "lr": self.optimizer_config.vis1_lr},
            ])
        else:
            g_vis1_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])  # placeholder

        ''' Transformer Encoder & Decoder '''
        self.set_requires_grad(self.model, False)
        if self.use_da_act:
            self.set_requires_grad(self.model, True)
            unfreeze_ca = "ca" in self.act_weights
            unfreeze_adapter = "adapter" in self.act_weights
            if not self.debug_diff_loss:  # when NOT debug diff loss, finetuning CA params of diffusion policy
                self.model.freeze_backbone(
                    unfreeze_params=self.act_weights,
                )
            else:
                # Debug diff loss
                self.model.inner_model.freeze_backbone(
                    unfreeze_params=self.act_weights
                )  # using the same setting with da_act
                pass  # finetuning all params
            g_act_optim_groups.extend([
                {"params": self.model.trainable_params(), "lr": self.optimizer_config.act_lr},
            ])
        else:
            g_act_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])  # placeholder

        ''' Adaptation Discriminator '''
        if self.use_da_vis1:
            self.set_requires_grad(self.da_vis1_loss, True)
            d_vis1_optim_groups.extend([
                {"params": self.da_vis1_loss.parameters(), "lr": self.optimizer_config.vis1_lr},
            ])
        else:
            d_vis1_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])

        if self.use_da_act:
            self.set_requires_grad(self.da_act_loss, True)
            d_act_optim_groups.extend([
                {"params": self.da_act_loss.parameters(),
                 "lr": self.optimizer_config.act_lr},
            ])
        else:
            d_act_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])

        ''' Optimizer '''
        g_vis1_optimizer = torch.optim.AdamW(g_vis1_optim_groups,
                                             weight_decay=obs_encoder_weight_decay,
                                             betas=betas)
        d_vis1_optimizer = torch.optim.AdamW(d_vis1_optim_groups,
                                             weight_decay=obs_encoder_weight_decay,
                                             betas=betas)

        g_act_optimizer = torch.optim.AdamW(g_act_optim_groups,
                                            weight_decay=transformer_weight_decay,
                                            betas=betas)
        d_act_optimizer = torch.optim.AdamW(d_act_optim_groups,
                                            weight_decay=transformer_weight_decay,
                                            betas=betas)

        param_obs_cnt = sum(p.numel() for p in self.trainable_params(self.obs_encoder))
        param_act_cnt = sum(p.numel() for p in self.trainable_params(self.model))
        param_dis_cnt = 0
        if self.use_da_vis1:
            param_dis_cnt += sum(p.numel() for p in self.trainable_params(self, "da_vis1_loss"))
        if self.use_da_act:
            param_dis_cnt += sum(p.numel() for p in self.trainable_params(self, "da_act_loss"))
        param_all_cnt = sum(p.numel() for p in self.trainable_params(self))
        print(f"[DiffusionTransformerHybridImagePolicyHDFree] Optimizer configured. "
              f"Obs trainable params: {param_obs_cnt}, "
              f"Act trainable params: {param_act_cnt}, "
              f"Discriminator trainable params: {param_dis_cnt / 1e6:.2f}M, "
              f"Total trainable params: {param_all_cnt / 1e6:.2f}M. "
              f"All params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M.")
        return (g_vis1_optimizer, d_vis1_optimizer,
                g_act_optimizer, d_act_optimizer,
                )

        ## Vanilla optimizer
        # optim_groups = self.model.get_optim_groups(
        #     weight_decay=transformer_weight_decay)
        # optim_groups.append({
        #     "params": self.obs_encoder.parameters(),
        #     "weight_decay": obs_encoder_weight_decay
        # })
        # optimizer = torch.optim.AdamW(
        #     optim_groups, lr=learning_rate, betas=betas
        # )
        # return optimizer

    def compute_loss(self, batch: dict, optimizers = None, lr_schedulers = None, batch_idx: int = None):
        """ Called by BaseWorkspace.run() """
        ''' (1) Vanilla batch'''
        if batch.get("src") is None:
            return super().compute_loss(batch)
        
        ''' (2) Source-Target batch'''
        (g_vis1_opt, d_vis1_opt, g_act_opt, d_act_opt) = optimizers
        (g_vis1_sch, d_vis1_sch, g_act_sch, d_act_sch) = lr_schedulers

        (total_loss, action_loss, cont_loss, id_loss, img_gen_loss,
         da_d1_loss, da_g1_loss, da_d2_loss, da_g2_loss, da_d_act_loss, da_g_act_loss,
         w_dist_1, gp_1, w_dist_2, gp_2, w_dist_act, gp_act) = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )
        losses = {
            'total_loss': total_loss,
            'action_loss': action_loss,
            'cont_loss': cont_loss,
            'img_gen_loss': img_gen_loss,
            'da_d1_loss': da_d1_loss,
            'da_g1_loss': da_g1_loss,
            'da_d2_loss': da_d2_loss,
            'da_g2_loss': da_g2_loss,
            'da_d_act_loss': da_d_act_loss,
            'da_g_act_loss': da_g_act_loss,
            'w_dist_1': w_dist_1,
            'gp_1': gp_1,
            'w_dist_2': w_dist_2,
            'gp_2': gp_2,
            'w_dist_act': w_dist_act,
            'gp_act': gp_act,
        }
        s_batch_len = 0
        t_batch_len = 0
        total_bs = 0

        source_act_0 = None
        common_noise = None
        common_sigmas = None
        common_sigma_emb = None
        max_bs = None
        use_zero_goal = np.random.uniform(0, 1) <= self.cfg_drop_ratio

        # normalize input
        assert 'valid_mask' not in batch
        s_batch, t_batch = batch["src"], batch["tgt"]

        s_nobs = self.normalizer.normalize(s_batch['obs'])
        s_nactions = self.normalizer['action'].normalize(s_batch['action'])
        batch_size = s_nactions.shape[0]
        horizon = s_nactions.shape[1]
        To = self.n_obs_steps

        t_nobs = self.normalizer.normalize(t_batch['obs'])
        t_nactions = self.normalizer['action'].normalize(t_batch['action'])

        # handle different ways of passing observation
        cond = None
        s_trajectory = s_nactions
        t_trajectory = t_nactions
        if self.obs_as_cond:  # go here
            # reshape B, T, ... to B*T
            s_this_nobs = dict_apply(s_nobs,
                                   lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            s_nobs_features = self.src_obs_encoder(s_this_nobs)
            t_this_nobs = dict_apply(t_nobs,
                                     lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            t_nobs_features = self.obs_encoder(t_this_nobs)
            s_vis_emb = self.src_obs_encoder.obs_nets['image'].cache_vis_out  # (B,64)
            t_vis_emb = self.obs_encoder.obs_nets['image'].cache_vis_out  # (B,64)
            # reshape back to B, T, Do
            s_cond = s_nobs_features.reshape(batch_size, To, -1)
            t_cond = t_nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                s_trajectory = s_nactions[:, start:end]
                t_trajectory = t_nactions[:, start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            s_condition_mask = torch.zeros_like(s_trajectory, dtype=torch.bool)
            t_condition_mask = torch.zeros_like(t_trajectory, dtype=torch.bool)
        else:
            s_condition_mask = self.mask_generator(s_trajectory.shape)
            t_condition_mask = self.mask_generator(t_trajectory.shape)

        # Sample noise that we'll add to the images
        # (1) Source
        s_noise = torch.randn(s_trajectory.shape, device=s_trajectory.device)
        bsz = s_trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=s_trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        s_noisy_trajectory = self.noise_scheduler.add_noise(
            s_trajectory, s_noise, timesteps)
        # (2) Target
        t_noise = torch.randn(t_trajectory.shape, device=t_trajectory.device)
        bsz = t_trajectory.shape[0]
        # DO NOT: Sample a random timestep for each image
        # timesteps = torch.randint(
        #     0, self.noise_scheduler.config.num_train_timesteps,
        #     (bsz,), device=t_trajectory.device
        # ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        t_noisy_trajectory = self.noise_scheduler.add_noise(
            s_trajectory, t_noise, timesteps)  # Should be the same

        # compute loss mask
        s_loss_mask = ~s_condition_mask

        # apply conditioning
        s_noisy_trajectory[s_condition_mask] = s_trajectory[s_condition_mask]

        # Predict the noise residual
        s_pred = self.src_model(s_noisy_trajectory, timesteps, s_cond)
        t_pred = self.model(t_noisy_trajectory, timesteps, t_cond)

        # print(s_pred.shape, t_pred.shape)
        noise_loss = torch.nn.MSELoss()
        # print(noise_loss(s_pred, t_pred).item())

        common_sigma_emb = self.src_model.cache_time_emb.squeeze(1)  # (B,1,256)
        s_mlp_list = self.src_model.decoder.cache_mlp_outputs
        t_mlp_list = self.model.decoder.cache_mlp_outputs

        if self.act_layers < 0:
            left, right = self.act_layers, None
        else:
            left, right = None, self.act_layers
        t_feat_for_da_act = []
        s_feat_for_da_act = []
        if 'mlp' in self.act_loss_from:
            # s_feat_for_da_act.extend(s_mlp_list[left:right])
            # t_feat_for_da_act.extend(t_mlp_list[left:right])
            s_feat_for_da_act.append(s_pred)
            t_feat_for_da_act.append(t_pred)

        t_feat_for_da_vis1 = t_vis_emb
        s_feat_for_da_vis1 = s_vis_emb
        # print(s_vis_emb.shape, t_vis_emb.shape)  # (B,64)
        # print(len(s_feat_for_da_act), len(t_feat_for_da_act))  # 6
        # print(s_feat_for_da_act[0].shape, t_feat_for_da_act[0].shape)  # (B,10,256)

        ''' 1. Update D '''
        if self.use_da_vis1:
            da_loss_dict = self.da_vis1_loss.forward(
                [t_feat_for_da_vis1.clone().detach()],  # avoid grad of G_target
                [s_feat_for_da_vis1.detach()],  # avoid grad of G_source
                is_discriminator_batch=True,
            )
            da_d_1_loss = da_loss_dict['loss']
            w_dist = da_loss_dict['w_dist']
            gp = da_loss_dict['gp']  # just for log
            losses['da_d1_loss'] += da_d_1_loss / 1
            losses['w_dist_1'] += w_dist
            losses['gp_1'] += gp

            d_vis1_opt.zero_grad()
            # self.manual_backward(losses['da_d1_loss'], retain_graph=False)  # no need to retrain graph
            losses['da_d1_loss'].backward(retain_graph=False)
            d_vis1_opt.step()
            d_vis1_sch.step()

        if self.use_da_act:
            da_act_loss_dict = self.da_act_loss.forward(
                [x.clone().detach() for x in t_feat_for_da_act],  # avoid grad of G_target
                [x.clone().detach() for x in s_feat_for_da_act],  # avoid grad of G_source
                is_discriminator_batch=True,
                sigmas=common_sigma_emb,
            )
            da_d_act_loss = da_act_loss_dict['loss']
            w_dist = da_act_loss_dict['w_dist']
            gp = da_act_loss_dict['gp']  # just for log
            losses['da_d_act_loss'] += da_d_act_loss / 1
            losses['w_dist_act'] += w_dist
            losses['gp_act'] += gp

            d_act_opt.zero_grad()
            # self.manual_backward(losses['da_d_act_loss'], retain_graph=False)  # no need to retrain graph
            losses['da_d_act_loss'].backward(retain_graph=True)
            d_act_opt.step()
            d_act_sch.step()

        ''' 2. Update G'''
        backward_loss = torch.tensor(0.0).to(self.device)
        if self.use_da_act:
            da_act_loss_dict = self.da_act_loss.forward(
                t_feat_for_da_act,  # update G_target
                s_feat_for_da_act,  # avoid grad of G_source
                is_discriminator_batch=False,
                sigmas=common_sigma_emb,
            )
            da_g_act_loss = da_act_loss_dict['loss']
            gp = da_act_loss_dict['gp']  # just for log
            losses['da_g_act_loss'] += da_g_act_loss / 1

            g_act_opt.zero_grad()
            retain_graph = self.use_da_vis1 or self.use_da_vis2  # Keep backward graph for later modules
            if not self.debug_diff_loss:
                act_back_loss = losses['da_g_act_loss'] + losses['action_loss']
                # self.manual_backward(act_back_loss, retain_graph=retain_graph)
                act_back_loss.backward(retain_graph=retain_graph)
            elif self.current_epoch >= 1 or batch_idx > 10:  # Only for debug
                # self.manual_backward(backward_loss)
                backward_loss.backward()
            g_act_opt.step()
            g_act_sch.step()

        if self.use_da_vis1:
            da_loss_dict = self.da_vis1_loss.forward(
                [t_feat_for_da_vis1],  # update G_target
                [s_feat_for_da_vis1],  # avoid grad of G_source
                is_discriminator_batch=False,
            )
            da_g1_loss = da_loss_dict['loss']
            gp = da_loss_dict['gp']  # just for log
            losses['da_g1_loss'] += da_g1_loss / 1

            backward_loss += losses['da_g1_loss']
            g_vis1_opt.zero_grad()

        losses['total_loss'] += backward_loss + losses['da_g_act_loss']
        if self.use_da_vis1:
            # self.manual_backward(backward_loss)  # backward vis1 and vis2 together
            backward_loss.backward()

        if self.use_da_vis1:
            g_vis1_opt.step()
        g_vis1_sch.step()

        # Get grad_norm
        vis1_grad_norm, vis1_total_norm, _ = self.calc_grad_and_param_norm(self.obs_encoder)
        act_grad_norm, act_total_norm, _ = self.calc_grad_and_param_norm(self.model)
        losses["train/vis1_grad_norm"] = vis1_grad_norm
        losses["train/vis1_total_norm"] = vis1_total_norm
        losses["train/act_grad_norm"] = act_grad_norm
        losses["train/act_total_norm"] = act_total_norm

        # if not self.automatic_optimization:
        #     self.on_before_zero_grad()
        # Log the metrics
        # self._log_training_metrics(losses, total_bs)

        # print(losses)
        # exit(0)

        return losses


        ## Compute MSE diffusion loss
        # pred_type = self.noise_scheduler.config.prediction_type
        # if pred_type == 'epsilon':
        #     target = s_noise
        # elif pred_type == 'sample':
        #     target = trajectory
        # else:
        #     raise ValueError(f"Unsupported prediction type {pred_type}")

        # loss = F.mse_loss(pred, target, reduction='none')
        # loss = loss * loss_mask.type(loss.dtype)
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean()
        # return loss
