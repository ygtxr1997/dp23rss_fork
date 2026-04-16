from typing import Dict, Tuple, List, Union, Optional
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
from accelerate import Accelerator, PartialState

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.moe_for_diffusion import MoEForDiffusion
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


# Copied from: video2world_force_dit.py
class ActionEncoder(nn.Module):
    def __init__(self, in_features: int, output_dim: int):
        super().__init__()
        self.layer = nn.Linear(in_features, output_dim)
    def forward(self, x):
        return self.layer(x)
    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.layer.in_features)
        torch.nn.init.trunc_normal_(self.layer.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.layer.bias)


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
            # force
            use_force: bool = False,
            # moe
            en_freeze_obs_encoder: bool = False,
            load_parts_from_ckpt: str = None,
            ffn_expand_factor: Union[float, List[float]] = 4.,
            backbone_type: str = "transformer",  # transformer | moe
            teacher_ckpts: List[str] = None,  # for moe from multiple teachers
            moe_topk: int = 2,
            router_noisy_std: float = 0.0,
            moe_aux_loss_weight: float = 0.0,
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
        print("[DEBUG] ObsUtils initialized with:", config.observation.modalities.obs)

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

        # NOTE: obs_encoder will automatically handle the obs in shape_meta, no need to create by ourselves
        self.force_embedder = None
        self.use_force = use_force
        '''
        Without Force:
        obs_feature_dim:134, cond_dim:134, input_dim:7, output_dim:7
        With Force:
        obs_feature_dim:140, cond_dim:140, input_dim:7, output_dim:7
        '''
        print(f"[DEBUG] obs_feature_dim:{obs_feature_dim}, cond_dim:{cond_dim}, input_dim:{input_dim}, output_dim:{output_dim} ")

        self.backbone_type = backbone_type
        self.obs_encoder = obs_encoder  # will be set weight in _bulild_backbone()
        model = self._build_backbone(
            # Shared params
            ffn_expand_factor=ffn_expand_factor,
            backbone_type=backbone_type,
            # Original `TransformerForDiffusion` params
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
            # MoE specific params
            teacher_ckpts=teacher_ckpts,
            moe_topk=moe_topk,
            router_noisy_std=router_noisy_std,
        )

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

        self.pretrained_ckpt = None

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # MoE related
        self.en_freeze_obs_encoder = en_freeze_obs_encoder
        self.load_parts_from_ckpt = load_parts_from_ckpt or "all"  # default: `all`
        self.teacher_ckpts = teacher_ckpts
        self.moe_topk = moe_topk
        self.router_noisy_std = router_noisy_std
        self.moe_aux_loss_weight = moe_aux_loss_weight

        # For real-world inference
        self.infer_frame_idx = 0
        self.cache_action = None

    def _build_backbone(self, backbone_type: str,
                        ffn_expand_factor: Union[float, List[float]],
                        teacher_ckpts: List[str],
                        use_ema: bool = True,
                        **model_kwargs
                        ):
        if backbone_type == "transformer":
            assert isinstance(ffn_expand_factor, (int, float)), "`ffn_expand_factor` should be float for transformer backbone"
            backbone_kwargs = dict(model_kwargs)
            backbone_kwargs["is_da"] = backbone_kwargs.get("is_da", False)
            return TransformerForDiffusion(
                ffn_expand_factor=ffn_expand_factor,
                **backbone_kwargs
            )

        if backbone_type == "moe":
            # Check `num_experts`
            teacher_ckpts = [] or teacher_ckpts  # avoid None

            if ffn_expand_factor is None and len(teacher_ckpts) == 0:
                raise ValueError("For MoE backbone, either `ffn_expand_factor` or `teacher_ckpts` must be provided.")
            if ffn_expand_factor is None and len(teacher_ckpts) > 0:
                raise ValueError("For MoE backbone, `ffn_expand_factor` must be provided when `teacher_ckpts` is given to determine the number of experts.")
            if ffn_expand_factor is not None and len(teacher_ckpts) > 0:
                ffn_expand_factor = [ffn_expand_factor] if isinstance(ffn_expand_factor, (int, float)) else ffn_expand_factor
                assert len(ffn_expand_factor) == len(teacher_ckpts), \
                    (f"The length of `ffn_expand_factor` should match `teacher_ckpts`. "
                     f"Got {len(ffn_expand_factor)} vs {len(teacher_ckpts)}.")
                # We use a strict rule to avoid complicated checking logic

            if len(teacher_ckpts) > 0:
                # A) 从多 teacher ckpt 合并
                teachers = []
                teacher_obs_encoders = []

                def _extract_component_state_dict(payload: dict, use_ema_flag: bool, component_name: str):
                    branch = "ema_model" if use_ema_flag else "model"
                    root = payload["state_dicts"][branch]
                    if not isinstance(root, dict):
                        raise ValueError(
                            f"Unexpected checkpoint structure at state_dicts['{branch}']: {type(root)}")

                    # case 1) nested dict under component key
                    if component_name in root and isinstance(root[component_name], dict):
                        nested = root[component_name]
                        prefix = f"{component_name}."
                        prefixed_nested = {k[len(prefix):]: v for k, v in nested.items() if k.startswith(prefix)}
                        return prefixed_nested if len(prefixed_nested) > 0 else nested

                    # case 2) policy-flat keys with component prefix
                    prefix = f"{component_name}."
                    prefixed = {k[len(prefix):]: v for k, v in root.items() if k.startswith(prefix)}
                    if len(prefixed) > 0:
                        return prefixed

                    # case 3) already component-only state dict (for `model`)
                    if component_name == "model":
                        return root
                    return None

                for t_idx, (ckpt, ffn_factor) in enumerate(zip(teacher_ckpts, ffn_expand_factor)):
                    # Merge action transformers
                    teacher_kwargs = dict(model_kwargs)
                    teacher_kwargs["is_da"] = False
                    teacher = TransformerForDiffusion(
                        ffn_expand_factor=ffn_factor,
                        **teacher_kwargs
                    )
                    payload = torch.load(pathlib.Path(ckpt).open("rb"), map_location="cpu", pickle_module=dill)
                    teacher_sd = _extract_component_state_dict(payload, use_ema_flag=use_ema, component_name="model")
                    teacher.load_state_dict(teacher_sd, strict=True)
                    teachers.append(teacher)

                    # Merge obs_encoders
                    assert self.obs_encoder is not None, "obs_encoder should be initialized before building MoE backbone, since we need to load teacher obs_encoder weights from ckpt"
                    obs_sd = _extract_component_state_dict(
                        payload, use_ema_flag=use_ema, component_name="obs_encoder")
                    if obs_sd is None:
                        raise ValueError(
                            f"Cannot find `obs_encoder` state dict in teacher checkpoint: {ckpt}")
                    teacher_obs_encoder = copy.deepcopy(self.obs_encoder)
                    teacher_obs_encoder.load_state_dict(obs_sd, strict=True)
                    teacher_obs_encoders.append(teacher_obs_encoder)

                return MoEForDiffusion.from_teacher_models(
                    teachers=teachers,
                    moe_topk=model_kwargs["moe_topk"],
                    router_noisy_std=model_kwargs["router_noisy_std"],
                    freeze_experts=False,  # NOTE: how to set this?
                    target_obs_encoder=self.obs_encoder,
                    teacher_obs_encoders=teacher_obs_encoders,
                    is_main_process=PartialState().is_main_process,
                )
            else:
                # B) 无 teacher 直接初始化 MoE
                return MoEForDiffusion(
                    num_experts=len(ffn_expand_factor),
                    **model_kwargs
                )
        raise ValueError(f"[DiffusionTransformerHybridImagePolicy] Unsupported backbone_type = {backbone_type}")

    def freeze_obs_encoder(self):
        for param in self.obs_encoder.parameters():
            param.requires_grad = False
        self.obs_encoder.eval()

    def print_training_status(self):
        if PartialState().is_main_process:
            self.count_moe_param_groups(self)

    def load_weight_from_ckpt(self, pretrained_ckpt, use_ema: bool = True):
        self.pretrained_ckpt = pretrained_ckpt
        path = pathlib.Path(pretrained_ckpt)
        payload = torch.load(path.open('rb'), pickle_module=dill)

        def load_state_dict_partial(model, state_dict, load_keys=None, skip_keys=None):
            """
            对 ckpt 的 keys，先按 load_keys 选择，再按 skip_keys 排除
            Args:
                model (nn.Module): 模型
                state_dict (dict): 要加载的状态字典
                load_keys (list): 只加载包含任一子串的键；None/[] 表示全量候选
                skip_keys (list): 需要跳过的键列表
            """
            # 过滤掉需要跳过的键
            load_keys = [] if load_keys is None else [str(x) for x in load_keys if str(x) != ""]
            skip_keys = [] if skip_keys is None else [str(x) for x in skip_keys if str(x) != ""]

            def _match_any(name: str, patterns: List[str]) -> bool:
                return any(p in name for p in patterns)

            # 1) include by load_keys
            selected_state_dict = {}
            for k, v in state_dict.items():
                if len(load_keys) == 0 or _match_any(k, load_keys):  # load_keys 为空表示全量候选
                    selected_state_dict[k] = v
            # 2) exclude by skip_keys
            filtered_state_dict = {}
            for k, v in selected_state_dict.items():
                if not _match_any(k, skip_keys):
                    filtered_state_dict[k] = v

            # 获取模型的 state_dict 键和 filtered_state_dict 键
            model_keys = set(model.state_dict().keys())
            state_dict_keys = set(filtered_state_dict.keys())

            # 找到 missing_keys 和 unexpected_keys
            missing_keys = list(model_keys - state_dict_keys)
            unexpected_keys = list(state_dict_keys - model_keys)
            loaded_keys = list(state_dict_keys & model_keys)

            for k in missing_keys:
                filtered_state_dict[k] = model.state_dict()[k]

            model.load_state_dict(filtered_state_dict, strict=True)

            # Load normalizer (do not consider `load_keys` and `skip_keys` for normalizer)
            norm_keys = set({k: None for k in state_dict.keys() if 'normalizer.' in k}.keys())
            norm_state_dict = {k[len("normalizer."):]: v for k, v in state_dict.items() if 'normalizer.' in k}
            model.normalizer.load_state_dict(norm_state_dict)

            if PartialState().is_main_process:
                print(f"Load keys (len={len(load_keys)}):", load_keys[:10], "..." if len(load_keys) > 10 else "")
                print(f"Skip keys (len={len(skip_keys)}):", skip_keys[:10], "..." if len(skip_keys) > 10 else "")
                print(f"Normalizer keys (len={len(list(norm_keys))}):", list(norm_keys)[:10], "..." if len(list(norm_keys)) > 10 else "")
                print(f"Missing keys (len={len(missing_keys)}):", missing_keys[:10], "..." if len(missing_keys) > 10 else "")
                print(f"Unexpected keys (len={len(list(set(unexpected_keys) - norm_keys))}):", list(set(unexpected_keys) - norm_keys))
                print(f"Loaded keys (len={len(loaded_keys)}):")

        # print(payload['state_dicts']['model'].keys())
        # print(self.normalizer.state_dict().keys())
        # print("[DEBUG] Keys in checkpoint model state_dict:", payload['state_dicts']['model'].keys())
        # self.count_moe_param_groups(self)
        # self.count_moe_param_groups(payload['state_dicts']['model'])

        load_keys: List[str] = []  # `None` or `[]` means all keys will be loaded (except those in `skip_keys`)
        skip_keys = [
            '.ia3_',  # IA3 Adapter
        ]

        # Skip some layers for MoE fine-tuning, see `policy.load_parts_from_ckpt` in config yaml
        if self.load_parts_from_ckpt != "all":
            load_keys = [str(part) for part in self.load_parts_from_ckpt.split(",")]

        if use_ema:
            load_state_dict_partial(self, payload['state_dicts']['ema_model'], load_keys, skip_keys)
        else:
            load_state_dict_partial(self, payload['state_dicts']['model'], load_keys, skip_keys)
        print(f"[DiffusionTransformerHybridImagePolicy] Loaded pretrained_ckpt (use_ema={use_ema}) "
              f"from: {self.pretrained_ckpt}, load_keys={load_keys}(empty means `all`), skip_keys={skip_keys}")

    @staticmethod
    def count_moe_param_groups(policy):
        """
        统计 DiffusionTransformerHybridImagePolicy 各模块参数量。
        """
        import re

        # 定义正则表达式规则与组名的映射
        rules = {
            r'^obs_encoder\..*': 'obs_encoder.*',
            r'^model\.cond_obs_emb\..*': 'model.cond_obs_emb.*',
            r'^model\.input_emb\..*': 'model.input_emb.*',
            r'^model\.(cond_)?pos_emb$': 'model.pos_emb / cond_pos_emb',
            r'^model\.encoder\..*': 'model.encoder.* (cond MLP)',
            r'^model\..*\.self_attn\..*': 'model.*.self_attn.*',
            r'^model\..*\.multihead_attn\..*': 'model.*.multihead_attn.*',
            r'^model\..*\.linear[12]\..*': 'model.*.linear1/2 (FFN)',
            r'^model\..*\.norm[123]\..*': 'model.*.norm1/2/3.*',
            r'^model\.ln_f\..*': 'model.ln_f.*',
            r'^model\.head\..*': 'model.head.*',
            r'^normalizer\..*': 'normalizer.*',
        }

        # 1. 改变初始化：包含 total 和 trainable
        groups = {name: {'total': 0, 'trainable': 0} for name in rules.values()}
        groups['buffers/dummy'] = {'total': 0, 'trainable': 0}
        groups['OTHER'] = {'total': 0, 'trainable': 0}

        other_keys = []

        # 2. 修改遍历逻辑：获取 requires_grad
        is_module = isinstance(policy, nn.Module)
        if is_module:
            named_params = list(policy.named_parameters()) + list(policy.named_buffers())
        else:
            named_params = policy.items()

        for name, param in named_params:
            n = param.numel()
            is_trainable = param.requires_grad if hasattr(param, 'requires_grad') and is_module else False

            matched = False
            for pattern, group_name in rules.items():
                if re.match(pattern, name):
                    groups[group_name]['total'] += n
                    groups[group_name]['trainable'] += (n if is_trainable else 0)
                    matched = True
                    break

            if not matched:
                groups['OTHER']['total'] += n
                groups['OTHER']['trainable'] += (n if is_trainable else 0)
                other_keys.append(name)

        # buffers（非 parameter，需单独遍历）
        if isinstance(policy, nn.Module):
            for name, buf in policy.named_buffers():
                n = buf.numel()
                is_trainable = buf.requires_grad if hasattr(buf, 'requires_grad') and is_module else False

                if '_dummy' in name or 'mask' in name:
                    groups['buffers/dummy']['total'] += n
                    groups['buffers/dummy']['trainable'] += (n if is_trainable else 0)
                # buffers 不计入 total params

        # 3. 计算总计（这里以 total 为例）
        total = sum(
            v['total'] for k, v in groups.items() if k not in ('buffers/dummy', 'OTHER')) + groups['OTHER']['total']
        total_trainable = sum(
            v['trainable'] for k, v in groups.items() if k not in ('buffers/dummy', 'OTHER')) + groups['OTHER'][
                              'trainable']

        # 计算 MoE 冻结和微调参数量
        moe_frozen = groups['obs_encoder.*']['total'] + groups['model.*.linear1/2 (FFN)']['total']
        moe_finetune = sum(groups[k]['total'] for k in [
            'model.*.self_attn.*', 'model.*.multihead_attn.*', 'model.*.norm1/2/3.*',
            'model.ln_f.*', 'model.encoder.* (cond MLP)', 'model.cond_obs_emb.*',
            'model.input_emb.*', 'model.pos_emb / cond_pos_emb', 'model.head.*'
        ])

        def fmt(n):
            if n >= 1e6:  return f"{n / 1e6:.3f} M"
            if n >= 1e3:  return f"{n / 1e3:.1f} K"
            return str(n)

        W = 46
        print(f"\n{'=' * (W + 22)}")
        print(f"  MoE 参数分组统计")
        print(f"{'=' * (W + 22)}")
        print(f"  {'组别':<{W}} {'参数量':>10}  {'占比':>6}")
        print(f"  {'-' * W}  {'-' * 10}  {'-' * 6}")

        categories = [
            ('── 视觉编码器（Frozen）', None),
            ('obs_encoder.*', '🔒 frozen'),
            ('── Transformer 条件侧', None),
            ('model.cond_obs_emb.*', '✏️  finetune'),
            ('model.encoder.* (cond MLP)', '✏️  finetune'),
            ('model.pos_emb / cond_pos_emb', '✏️  finetune'),
            ('model.input_emb.*', '✏️  finetune'),
            ('── Transformer Decoder', None),
            ('model.*.self_attn.*', '✏️  finetune'),
            ('model.*.multihead_attn.*', '✏️  finetune'),
            ('model.*.linear1/2 (FFN)', '🔒 frozen (MoE experts)'),
            ('model.*.norm1/2/3.*', '✏️  finetune'),
            ('model.ln_f.*', '✏️  finetune'),
            ('model.head.*', '✏️  finetune'),
            ('── 其他', None),
            ('normalizer.*', '—  not trained'),
            ('OTHER', '❓ check'),
        ]

        for item in categories:
            if item[1] is None:
                print(f"  {item[0]}")
                continue
            k, tag = item
            v_tot = groups[k]['total']
            v_train = groups[k]['trainable']
            pct = v_tot / total * 100 if total > 0 else 0
            train_pct = (v_train / v_tot * 100) if v_tot > 0 else 0
            print(f"    {k:<{W - 2}} {fmt(v_tot):>10}  {pct:>5.1f}% | Train: {fmt(v_train):>10} ({train_pct:>5.1f}%)   {tag}")

        print(f"  {'─' * W}  {'─' * 10}  {'─' * 6}")
        print(f"  {'TOTAL (parameters)':<{W}} {fmt(total):>10}  100.0%")
        print(f"  {'  ├─ MoE Frozen (obs_encoder + FFN experts)':<{W}} {fmt(moe_frozen):>10}  {moe_frozen / total * 100:>5.1f}%")
        print(f"  {'  └─ MoE Finetune':<{W}} {fmt(moe_finetune):>10}  {moe_finetune / total * 100:>5.1f}%")
        print(f"  {'└─ Trainable':<{W}} {fmt(total_trainable):>10}  {total_trainable / total * 100:>5.1f}%")
        if other_keys:
            print("OTHER keys:", other_keys)

        return groups

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
            if self.pred_action_steps_only:  # default:False
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
        # print('action_pred:', action_pred.shape, )

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]

        self.cache_action = action_pred
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def reset(self):
        self.infer_frame_idx = 0
        self.cache_action = None

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

    def compute_loss(self, batch) -> Dict[str, torch.Tensor]:
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
        if self.obs_as_cond:  # default: go here
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

        losses = {}
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        # optional MoE auxiliary loss
        if self.backbone_type == "moe" and self.moe_aux_loss_weight > 0:
            aux = getattr(self.model, "last_moe_aux_loss", None)
            if aux is not None:
                loss = loss + self.moe_aux_loss_weight * aux
                losses["moe_aux_loss"] = aux.detach().cpu().item()

        losses["total_loss"] = loss

        return losses

    def forward(self, batch):
        loss = self.compute_loss(batch)
        return loss


class DiffusionTransformerHybridImagePolicyHDFree(DiffusionTransformerHybridImagePolicy):
    def __init__(self,
                 domain_adapt: DictConfig,
                 **kwargs):
        super().__init__(**kwargs)
        self.src_ckpt = domain_adapt.src_ckpt
        self.load_from_src_ckpt(self.src_ckpt, use_ema=False)  # will have 2 copies: one is for target; another is for source (fixed)

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

        self.src_obs_encoder.obs_nets['gripper'].forward = types.MethodType(
            forward_with_da, self.src_obs_encoder.obs_nets['gripper'])
        self.obs_encoder.obs_nets['gripper'].forward = types.MethodType(
            forward_with_da, self.obs_encoder.obs_nets['gripper'])

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

        self.cache_action = action_pred

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
        g_vis2_optim_groups = []
        d_vis2_optim_groups = []
        g_act_optim_groups = []
        d_act_optim_groups = []

        ''' Frozen modules '''
        self.set_requires_grad(self.src_obs_encoder, False)
        self.set_requires_grad(self.src_model, False)

        ''' Visual Encoder '''
        self.set_requires_grad(self.obs_encoder, False)
        if self.use_da_vis1 or self.use_da_vis2:
            self.freeze_params(self.obs_encoder)
            if self.use_da_vis1:
                self.set_requires_grad(self.obs_encoder.obs_nets['image'].nets[-1], True)
                g_vis1_optim_groups.extend([
                    {"params": self.trainable_params(self.obs_encoder.obs_nets['image']), "lr": self.optimizer_config.vis1_lr},
                ])
            else:
                g_vis1_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])

            if self.use_da_vis2:
                self.set_requires_grad(self.obs_encoder.obs_nets['gripper'].nets[-1], True)
                g_vis2_optim_groups.extend([
                    {"params": self.trainable_params(self.obs_encoder.obs_nets['gripper']), "lr": self.optimizer_config.vis2_lr},
                ])
            else:
                g_vis2_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])

        else:
            g_vis1_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])  # placeholder
            g_vis2_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])

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

        if self.use_da_vis2:
            self.set_requires_grad(self.da_vis2_loss, True)
            d_vis2_optim_groups.extend([
                {"params": self.da_vis2_loss.parameters(), "lr": self.optimizer_config.vis2_lr},
            ])
        else:
            d_vis2_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])

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

        g_vis2_optimizer = torch.optim.AdamW(g_vis2_optim_groups,
                                             weight_decay=obs_encoder_weight_decay,
                                             betas=betas)
        d_vis2_optimizer = torch.optim.AdamW(d_vis2_optim_groups,
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
        if self.use_da_vis2:
            param_dis_cnt += sum(p.numel() for p in self.trainable_params(self, "da_vis2_loss"))
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
                g_vis2_optimizer, d_vis2_optimizer,
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

    def compute_loss(self, batch: dict, optimizers = None, lr_schedulers = None, batch_idx: int = None,
                     accelerator: Accelerator = None):
        """ Called by BaseWorkspace.run() """
        ''' (1) Vanilla batch'''
        if batch.get("src") is None:
            return super().compute_loss(batch)
        
        ''' (2) Source-Target batch'''
        (g_vis1_opt, d_vis1_opt, g_vis2_opt, d_vis2_opt, g_act_opt, d_act_opt) = optimizers
        (g_vis1_sch, d_vis1_sch, g_vis2_sch, d_vis2_sch, g_act_sch, d_act_sch) = lr_schedulers

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
        s_trajectory = s_nactions  # value will not be used
        t_trajectory = t_nactions
        if self.obs_as_cond:  # go here
            # reshape B, T, ... to B*T
            s_this_nobs = dict_apply(s_nobs,
                                   lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            s_nobs_features = self.src_obs_encoder(s_this_nobs)
            t_this_nobs = dict_apply(t_nobs,
                                     lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            t_nobs_features = self.obs_encoder(t_this_nobs)
            s_vis1_emb = self.src_obs_encoder.obs_nets['image'].cache_vis_out  # (B,64)
            t_vis1_emb = self.obs_encoder.obs_nets['image'].cache_vis_out  # (B,64)
            s_vis2_emb = self.src_obs_encoder.obs_nets['gripper'].cache_vis_out
            t_vis2_emb = self.obs_encoder.obs_nets['gripper'].cache_vis_out
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
            s_feat_for_da_act.extend(s_mlp_list[left:right])
            t_feat_for_da_act.extend(t_mlp_list[left:right])
            # s_feat_for_da_act.append(s_pred)
            # t_feat_for_da_act.append(t_pred)

        t_feat_for_da_vis1 = t_vis1_emb
        s_feat_for_da_vis1 = s_vis1_emb
        t_feat_for_da_vis2 = t_vis2_emb
        s_feat_for_da_vis2 = s_vis2_emb
        # print(s_vis_emb.shape, t_vis_emb.shape)  # (B,64)
        # print(len(s_feat_for_da_act), len(t_feat_for_da_act))  # 8, 8
        # print(s_feat_for_da_act[0].shape, t_feat_for_da_act[0].shape)  # (B,horizon,256)
        # exit()

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
            # losses['da_d1_loss'].backward(retain_graph=False)
            if accelerator is not None:
                accelerator.backward(losses['da_d1_loss'], retain_graph=False)
            else:
                losses['da_d1_loss'].backward(retain_graph=False)
            d_vis1_opt.step()
            d_vis1_sch.step()

        if self.use_da_vis2:
            da_loss_dict = self.da_vis2_loss.forward(
                [t_feat_for_da_vis2.clone().detach()],  # avoid grad of G_target
                [s_feat_for_da_vis2.detach()],  # avoid grad of G_source
                is_discriminator_batch=True,
            )
            da_d_2_loss = da_loss_dict['loss']
            w_dist = da_loss_dict['w_dist']
            gp = da_loss_dict['gp']  # just for log
            losses['da_d2_loss'] += da_d_2_loss / 1
            losses['w_dist_2'] += w_dist
            losses['gp_2'] += gp

            d_vis2_opt.zero_grad()
            # self.manual_backward(losses['da_d2_loss'], retain_graph=False)  # no need to retrain graph
            # losses['da_d2_loss'].backward(retain_graph=False)
            if accelerator is not None:
                accelerator.backward(losses['da_d2_loss'], retain_graph=False)
            else:
                losses['da_d2_loss'].backward(retain_graph=False)
            d_vis2_opt.step()
            d_vis2_sch.step()

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
            # losses['da_d_act_loss'].backward(retain_graph=True)
            if accelerator is not None:
                accelerator.backward(losses['da_d_act_loss'], retain_graph=True)
            else:
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
                # act_back_loss.backward(retain_graph=retain_graph)
                if accelerator is not None:
                    accelerator.backward(act_back_loss, retain_graph=retain_graph)
                else:
                    act_back_loss.backward(retain_graph=retain_graph)
            elif self.current_epoch >= 1 or batch_idx > 10:  # Only for debug
                # backward_loss.backward()
                if accelerator is not None:
                    accelerator.backward(backward_loss)
                else:
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

        if self.use_da_vis2:
            da_loss_dict = self.da_vis2_loss.forward(
                [t_feat_for_da_vis2],  # update G_target
                [s_feat_for_da_vis2],  # avoid grad of G_source
                is_discriminator_batch=False,
            )
            da_g2_loss = da_loss_dict['loss']
            gp = da_loss_dict['gp']  # just for log
            losses['da_g2_loss'] += da_g2_loss / 1

            backward_loss += losses['da_g2_loss']
            g_vis2_opt.zero_grad()

        losses['total_loss'] += backward_loss + losses['da_g_act_loss']
        if self.use_da_vis1 or self.use_da_vis2:
            # self.manual_backward(backward_loss)  # backward vis1 and vis2 together
            # backward_loss.backward()
            if accelerator is not None:
                accelerator.backward(backward_loss)
            else:
                backward_loss.backward()

        if self.use_da_vis1:
            g_vis1_opt.step()
        if self.use_da_vis2:
            g_vis2_opt.step()
        g_vis1_sch.step()
        g_vis2_sch.step()

        # Get grad_norm
        vis1_grad_norm, vis1_total_norm, _ = self.calc_grad_and_param_norm(self.obs_encoder.obs_nets['image'])
        vis2_grad_norm, vis2_total_norm, _ = self.calc_grad_and_param_norm(self.obs_encoder.obs_nets['gripper'])
        act_grad_norm, act_total_norm, _ = self.calc_grad_and_param_norm(self.model)
        losses["train/vis1_grad_norm"] = vis1_grad_norm
        losses["train/vis1_total_norm"] = vis1_total_norm
        losses["train/vis2_grad_norm"] = vis2_grad_norm
        losses["train/vis2_total_norm"] = vis2_total_norm
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
