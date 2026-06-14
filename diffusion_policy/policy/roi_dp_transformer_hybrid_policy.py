from typing import Dict, Tuple, List, Union, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import builtins, types, inspect
from omegaconf import DictConfig, OmegaConf
import pathlib
import dill
import copy
import hydra
from accelerate import Accelerator, PartialState

from diffusion_policy.model.common.normalizer import LinearNormalizer, EmptyNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.moe_for_diffusion import MoEForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.roi.roinet import RoiNet
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

from robokit.debug_utils.time_profiler import global_time_profiler as gtp


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
                 roi_net=None,
                 roi_area_loss_weight: float = 0.01,
                 roi_area_loss_weight_increase_ratio: Optional[float] = None,
                 roi_area_loss_weight_increase_per_steps: Optional[int] = None,
                 roi_area_loss_weight_max: float = 0.01,
                 roi_center_diversity_loss_weight: float = 0.0,
                 roi_center_diversity_margin: float = 0.25,
                 p_drop_full_image: float = 0.3,
                 concat_roi: bool = False,
                 use_image_roi_only: bool = False,
                 roi_coord_conv: bool = False,
                 roi_coord_conv_detach: bool = True,
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

        shape_meta = OmegaConf.to_container(shape_meta, resolve=True) \
            if isinstance(shape_meta, DictConfig) else copy.deepcopy(shape_meta)
        concat_roi = bool(concat_roi)
        use_image_roi_only = bool(use_image_roi_only)
        roi_coord_conv = bool(roi_coord_conv)
        roi_coord_conv_detach = bool(roi_coord_conv_detach)
        obs_shape_meta = shape_meta['obs']
        encoder_shape_meta = copy.deepcopy(shape_meta)
        if roi_coord_conv and concat_roi:
            raise ValueError("`roi_coord_conv=True` is not supported with `concat_roi=True`.")
        if 'image_roi' in obs_shape_meta:
            roi_shape = list(obs_shape_meta['image_roi']['shape'])
            if len(roi_shape) != 3:
                raise ValueError(
                    "`image_roi` shape must be [C, H, W]. "
                    f"Got image_roi={roi_shape}."
                )
            encoder_shape_meta['obs']['image_roi']['shape'] = [
                roi_shape[0] + 1,
                roi_shape[1],
                roi_shape[2],
            ]
        if roi_coord_conv:
            if 'image_roi' not in obs_shape_meta:
                raise ValueError("`roi_coord_conv=True` requires `image_roi` in shape_meta['obs'].")
            roi_shape = list(obs_shape_meta['image_roi']['shape'])
            encoder_shape_meta['obs']['image_roi']['shape'] = [
                roi_shape[0] + 3,
                roi_shape[1],
                roi_shape[2],
            ]
        if use_image_roi_only:
            if 'image' not in obs_shape_meta or 'image_roi' not in obs_shape_meta:
                raise ValueError(
                    "`use_image_roi_only=True` requires `image` and `image_roi` in shape_meta['obs']. "
                    "`image` is still used to generate ROI crops."
                )
            if 'image' in encoder_shape_meta['obs']:
                del encoder_shape_meta['obs']['image']
        elif concat_roi:
            if 'image' not in obs_shape_meta or 'image_roi' not in obs_shape_meta:
                raise ValueError("`concat_roi=True` requires `image` and `image_roi` in shape_meta['obs'].")

            image_shape = list(obs_shape_meta['image']['shape'])
            roi_shape = list(obs_shape_meta['image_roi']['shape'])
            if len(image_shape) != 3 or len(roi_shape) != 3:
                raise ValueError(
                    "`concat_roi=True` expects `image` and `image_roi` shapes to be [C, H, W]. "
                    f"Got image={image_shape}, image_roi={roi_shape}."
                )
            if image_shape[:2] != roi_shape[:2]:
                raise ValueError(
                    "`concat_roi=True` requires `image` and `image_roi` to have matching C and H. "
                    f"Got image={image_shape}, image_roi={roi_shape}."
                )

            encoder_shape_meta['obs']['image']['shape'] = [
                image_shape[0] + 1,
                image_shape[1],
                image_shape[2] + roi_shape[2],
            ]
            del encoder_shape_meta['obs']['image_roi']

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        self.obs_shape_meta = obs_shape_meta
        self.encoder_obs_shape_meta = encoder_shape_meta['obs']
        self.concat_roi = concat_roi
        self.use_image_roi_only = use_image_roi_only
        self.roi_coord_conv = roi_coord_conv
        self.roi_coord_conv_detach = roi_coord_conv_detach
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in self.encoder_obs_shape_meta.items():
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
        print(
            "[DEBUG] concat_roi:", self.concat_roi,
            "use_image_roi_only:", self.use_image_roi_only,
            "roi_coord_conv:", self.roi_coord_conv,
            "roi_coord_conv_detach:", self.roi_coord_conv_detach,
            "obs_key_shapes:", obs_key_shapes,
        )

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
                    num_groups=x.num_features // 16,
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
        print(
            f"[DEBUG] obs_feature_dim:{obs_feature_dim}, cond_dim:{cond_dim}, input_dim:{input_dim}, output_dim:{output_dim} ")

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
        self.roi_area_loss_weight = float(roi_area_loss_weight)
        self.roi_center_diversity_loss_weight = float(roi_center_diversity_loss_weight)
        self.roi_center_diversity_margin = float(roi_center_diversity_margin)
        if self.roi_center_diversity_loss_weight < 0.0:
            raise ValueError("`roi_center_diversity_loss_weight` must be non-negative.")
        if self.roi_center_diversity_margin < 0.0:
            raise ValueError("`roi_center_diversity_margin` must be non-negative.")
        self.roi_area_loss_weight_increase_ratio = (
            None if roi_area_loss_weight_increase_ratio is None
            else float(roi_area_loss_weight_increase_ratio)
        )
        self.roi_area_loss_weight_increase_per_steps = (
            None if roi_area_loss_weight_increase_per_steps is None
            else int(roi_area_loss_weight_increase_per_steps)
        )
        if (
                (self.roi_area_loss_weight_increase_ratio is None)
                != (self.roi_area_loss_weight_increase_per_steps is None)
        ):
            raise ValueError(
                "`roi_area_loss_weight_increase_ratio` and "
                "`roi_area_loss_weight_increase_per_steps` must be set together."
            )
        if (
                self.roi_area_loss_weight_increase_per_steps is not None
                and self.roi_area_loss_weight_increase_per_steps <= 0
        ):
            raise ValueError("`roi_area_loss_weight_increase_per_steps` must be > 0.")
        self.roi_area_loss_weight_max = roi_area_loss_weight_max
        self._roi_area_loss_step = 0
        self._effective_roi_area_loss_weight = min(
            self.roi_area_loss_weight,
            self.roi_area_loss_weight_max,
        )
        self.p_drop_full_image = float(p_drop_full_image)
        image_shape = list(self.obs_shape_meta['image']['shape'])
        roi_shape = list(self.obs_shape_meta['image_roi']['shape'])
        if roi_net is None:
            roi_net = RoiNet(in_channels=image_shape[0], roi_size=roi_shape[1:])
        elif isinstance(roi_net, (dict, DictConfig)):
            roi_net = hydra.utils.instantiate(roi_net)
        self.roi_net: RoiNet = roi_net
        self.image_obs_size = tuple(image_shape[1:])

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
            assert isinstance(ffn_expand_factor,
                              (int, float)), "`ffn_expand_factor` should be float for transformer backbone"
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
                raise ValueError(
                    "For MoE backbone, `ffn_expand_factor` must be provided when `teacher_ckpts` is given to determine the number of experts.")
            if ffn_expand_factor is not None and len(teacher_ckpts) > 0:
                ffn_expand_factor = [ffn_expand_factor] if isinstance(ffn_expand_factor,
                                                                      (int, float)) else ffn_expand_factor
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
            # 2) exclude normalizer, skipped keys, unexpected keys and shape-mismatched keys
            model_state_dict = model.state_dict()
            model_keys = set(model_state_dict.keys())
            filtered_state_dict = {}
            candidate_keys = set()
            shape_mismatch_keys = []
            for k, v in selected_state_dict.items():
                if _match_any(k, skip_keys) or 'normalizer.' in k:
                    continue
                candidate_keys.add(k)
                if k not in model_state_dict:
                    continue
                if hasattr(v, "shape") and tuple(v.shape) != tuple(model_state_dict[k].shape):
                    shape_mismatch_keys.append(k)
                    continue
                filtered_state_dict[k] = v

            # 获取模型的 state_dict 键和 filtered_state_dict 键
            state_dict_keys = set(filtered_state_dict.keys())

            # 找到 missing_keys 和 unexpected_keys
            missing_keys = list(model_keys - state_dict_keys)
            unexpected_keys = list(candidate_keys - model_keys)
            loaded_keys = list(state_dict_keys & model_keys)

            def _top_level_counts(keys):
                counts = {}
                for name in keys:
                    prefix = name.split(".", 1)[0]
                    counts[prefix] = counts.get(prefix, 0) + 1
                return dict(sorted(counts.items()))

            loaded_counts = _top_level_counts(loaded_keys)
            missing_counts = _top_level_counts(missing_keys)
            unexpected_counts = _top_level_counts(unexpected_keys)
            shape_mismatch_counts = _top_level_counts(shape_mismatch_keys)

            must_load_roi = (
                hasattr(model, "roi_net")
                and any(k.startswith("roi_net.") for k in model_keys)
                and (len(load_keys) == 0 or _match_any("roi_net.", load_keys) or _match_any("roi_net", load_keys))
            )
            roi_loaded = loaded_counts.get("roi_net", 0)
            roi_expected = sum(k.startswith("roi_net.") for k in model_keys)
            roi_available = sum(k.startswith("roi_net.") for k in candidate_keys)
            roi_missing = sum(k.startswith("roi_net.") for k in missing_keys)
            roi_shape_mismatch = sum(k.startswith("roi_net.") for k in shape_mismatch_keys)
            if must_load_roi and roi_expected > 0 and roi_loaded != roi_expected:
                raise RuntimeError(
                    "`roi_net.*` weights were not fully loaded from pretrained checkpoint. "
                    f"roi_loaded={roi_loaded}, roi_expected={roi_expected}, "
                    f"roi_missing={roi_missing}, roi_shape_mismatch={roi_shape_mismatch}, "
                    f"roi_available_in_checkpoint={roi_available}, load_keys={load_keys}. "
                    "Check that `pretrained_ckpt` was produced by the same ROI policy and that "
                    "`policy.load_parts_from_ckpt` includes `roi_net` or is `all`."
                )

            for k in missing_keys:
                filtered_state_dict[k] = model_state_dict[k]

            model.load_state_dict(filtered_state_dict, strict=True)

            # Load normalizer (do not consider `load_keys` and `skip_keys` for normalizer)
            norm_keys = set({k: None for k in state_dict.keys() if 'normalizer.' in k}.keys())
            norm_state_dict = {k[len("normalizer."):]: v for k, v in state_dict.items() if 'normalizer.' in k}
            if len(norm_state_dict) > 0:
                model.normalizer.load_state_dict(norm_state_dict)
                if hasattr(model, "_ensure_obs_normalizers"):
                    model._ensure_obs_normalizers()

            if PartialState().is_main_process:
                print(f"Load keys (len={len(load_keys)}):", load_keys[:10], "..." if len(load_keys) > 10 else "")
                print(f"Skip keys (len={len(skip_keys)}):", skip_keys[:10], "..." if len(skip_keys) > 10 else "")
                print(f"Normalizer keys (len={len(list(norm_keys))}):", list(norm_keys)[:10],
                      "..." if len(list(norm_keys)) > 10 else "")
                print(f"Missing keys (len={len(missing_keys)}):", missing_keys[:10],
                      "..." if len(missing_keys) > 10 else "")
                print(f"Unexpected keys (len={len(list(set(unexpected_keys) - norm_keys))}):",
                      list(set(unexpected_keys) - norm_keys))
                print(f"Shape mismatch keys (len={len(shape_mismatch_keys)}):", shape_mismatch_keys[:10],
                      "..." if len(shape_mismatch_keys) > 10 else "")
                print(f"Loaded keys (len={len(loaded_keys)}):")
                print(f"Loaded key groups: {loaded_counts}")
                print(f"Missing key groups: {missing_counts}")
                print(f"Unexpected key groups: {unexpected_counts}")
                print(f"Shape mismatch key groups: {shape_mismatch_counts}")
                if hasattr(model, "roi_net"):
                    print(
                        f"ROI load check: loaded={roi_loaded}, "
                        f"available_in_ckpt={roi_available}, expected={roi_expected}, "
                        f"missing={roi_missing}, shape_mismatch={roi_shape_mismatch}"
                    )

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
            r'^roi_net\..*': 'roi_net.*',
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
            ('── ROI 网络', None),
            ('roi_net.*', '✏️  finetune'),
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
            print(
                f"    {k:<{W - 2}} {fmt(v_tot):>10}  {pct:>5.1f}% | Train: {fmt(v_train):>10} ({train_pct:>5.1f}%)   {tag}")

        print(f"  {'─' * W}  {'─' * 10}  {'─' * 6}")
        print(f"  {'TOTAL (parameters)':<{W}} {fmt(total):>10}  100.0%")
        print(
            f"  {'  ├─ MoE Frozen (obs_encoder + FFN experts)':<{W}} {fmt(moe_frozen):>10}  {moe_frozen / total * 100:>5.1f}%")
        print(f"  {'  └─ MoE Finetune':<{W}} {fmt(moe_finetune):>10}  {moe_finetune / total * 100:>5.1f}%")
        print(f"  {'└─ Trainable':<{W}} {fmt(total_trainable):>10}  {total_trainable / total * 100:>5.1f}%")
        if other_keys:
            print("OTHER keys:", other_keys)

        return groups

    # ========= inference  ============
    def _ensure_obs_normalizers(self):
        for key in self.obs_shape_meta:
            if key not in self.normalizer.params_dict:
                self.normalizer[key] = EmptyNormalizer.create_identity()

    def _resize_image_obs(self, image_B_T_C_H_W: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = image_B_T_C_H_W.shape
        image_BT_C_H_W = image_B_T_C_H_W.reshape(B * T, C, H, W)
        resized_BT_C_H_W = F.interpolate(
            image_BT_C_H_W,
            size=self.image_obs_size,
            mode="bilinear",
            align_corners=False,
        )
        return resized_BT_C_H_W.reshape(B, T, C, *self.image_obs_size)

    def _maybe_drop_full_image(self, image_B_T_C_H_W: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p_drop_full_image <= 0.0:
            return image_B_T_C_H_W
        drop_mask = torch.rand(
            image_B_T_C_H_W.shape[0],
            1,
            1,
            1,
            1,
            device=image_B_T_C_H_W.device,
        ) < self.p_drop_full_image
        return image_B_T_C_H_W.masked_fill(drop_mask, 0.0)

    def _build_roi_obs(
            self,
            obs_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        image_B_T_C_H_W = obs_dict["image"]
        roi_input_B_C_T_H_W = rearrange(image_B_T_C_H_W, "b t c h w -> b c t h w")
        roi_out = self.roi_net(
            roi_input_B_C_T_H_W,
            return_grid=self.roi_coord_conv,
        )

        image_roi_B_T_C_H_W = rearrange(
            roi_out["roi_B_C_T_RH_RW"],
            "b c t h w -> b t c h w",
        )
        image_roi_B_T_C_H_W = torch.cat(
            [image_roi_B_T_C_H_W, roi_out["valid_mask_B_T_1_RH_RW"].to(
                dtype=image_roi_B_T_C_H_W.dtype,
            )],
            dim=2,
        )
        generated_obs = {
            "image": self._maybe_drop_full_image(
                self._resize_image_obs(image_B_T_C_H_W)),
            "image_roi": image_roi_B_T_C_H_W,
            "roi_param": roi_out["params_B_T_3"],
        }
        dp_obs = {}
        for key in self.obs_shape_meta:
            dp_obs[key] = generated_obs[key] if key in generated_obs else obs_dict[key]
        return dp_obs, roi_out

    def _append_roi_coord_channels(
            self,
            image_roi_B_T_C_H_W: torch.Tensor,
            roi_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "grid_B_T_RH_RW_2" not in roi_out:
            raise KeyError(
                "`roi_coord_conv=True` requires RoiNet output `grid_B_T_RH_RW_2`."
            )
        roi_coord_B_T_2_H_W = roi_out["grid_B_T_RH_RW_2"].permute(
            0,
            1,
            4,
            2,
            3,
        ).contiguous()
        if self.roi_coord_conv_detach:
            roi_coord_B_T_2_H_W = roi_coord_B_T_2_H_W.detach()
        return torch.cat([image_roi_B_T_C_H_W, roi_coord_B_T_2_H_W], dim=2)

    @staticmethod
    def _append_full_image_valid_mask(
            image_B_T_C_H_W: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask_B_T_1_H_W = torch.ones(
            *image_B_T_C_H_W.shape[:2],
            1,
            *image_B_T_C_H_W.shape[-2:],
            device=image_B_T_C_H_W.device,
            dtype=image_B_T_C_H_W.dtype,
        )
        return torch.cat([image_B_T_C_H_W, valid_mask_B_T_1_H_W], dim=2)

    def _prepare_obs_encoder_input(
            self,
            obs_dict: Dict[str, torch.Tensor],
            roi_out: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.roi_coord_conv:
            if roi_out is None:
                raise ValueError("`roi_out` is required when `roi_coord_conv=True`.")
            obs_dict = dict(obs_dict)
            obs_dict["image_roi"] = self._append_roi_coord_channels(
                obs_dict["image_roi"],
                roi_out,
            )

        if self.use_image_roi_only:
            return {key: obs_dict[key] for key in self.encoder_obs_shape_meta}

        if not self.concat_roi:
            return obs_dict

        obs_encoder_input = {}
        for key in self.encoder_obs_shape_meta:
            if key == "image":
                image_B_T_C_H_W = self._append_full_image_valid_mask(
                    obs_dict["image"]
                )
                obs_encoder_input[key] = torch.cat(
                    [image_B_T_C_H_W, obs_dict["image_roi"]],
                    dim=-1,
                )
            else:
                obs_encoder_input[key] = obs_dict[key]
        return obs_encoder_input

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
        assert 'past_action' not in obs_dict  # not implemented yet
        # normalize input
        roi_obs, roi_out = self._build_roi_obs(obs_dict)
        nobs = self.normalizer.normalize(roi_obs)
        encoder_nobs = self._prepare_obs_encoder_input(nobs, roi_out=roi_out)
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
            this_nobs = dict_apply(encoder_nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
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
            this_nobs = dict_apply(encoder_nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
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
        # print('action_pred:', action_pred.shape, )

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

    def reset(self):
        self.infer_frame_idx = 0
        self.cache_action = None

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self._ensure_obs_normalizers()

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
        optim_groups.append({
            "params": self.roi_net.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def set_roi_area_loss_step(self, global_step: int) -> None:
        self._roi_area_loss_step = int(global_step)
        if self.roi_area_loss_weight_increase_ratio is None:
            self._effective_roi_area_loss_weight = min(
                self.roi_area_loss_weight,
                self.roi_area_loss_weight_max,
            )
            return

        stage = self._roi_area_loss_step // self.roi_area_loss_weight_increase_per_steps
        self._effective_roi_area_loss_weight = min(
            self.roi_area_loss_weight + self.roi_area_loss_weight_increase_ratio * stage,
            self.roi_area_loss_weight_max,
        )

    def compute_loss(self, batch, return_roi_params: bool = False) -> Dict[str, torch.Tensor]:
        # normalize input
        assert 'valid_mask' not in batch
        gtp_open = self.training
        gtp_training = "train" if self.training else "val"
        roi_obs, roi_out = self._build_roi_obs(batch['obs'])
        nobs = self.normalizer.normalize(roi_obs)
        encoder_nobs = self._prepare_obs_encoder_input(nobs, roi_out=roi_out)
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:  # default: go here
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(encoder_nobs,
                                   lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            with gtp(f"{gtp_training}/obs_enc.forward", group=f"{gtp_training}/obs_enc"):
                nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:, start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(encoder_nobs, lambda x: x.reshape(-1, *x.shape[2:]))
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
        with gtp(f"{gtp_training}/action.forward", group=f"{gtp_training}/action"):
            pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        diffusion_loss = F.mse_loss(pred, target, reduction='none')
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = diffusion_loss.mean()

        roi_params = roi_out["params_B_T_3"]
        if self.obs_as_cond:
            roi_params = roi_params[:, :To]
        roi_area_loss = self.roi_net.area_loss(roi_params)
        effective_roi_area_loss_weight = self._effective_roi_area_loss_weight
        roi_area_weighted_loss = effective_roi_area_loss_weight * roi_area_loss
        roi_center_diversity_loss = self.roi_net.center_diversity_loss(
            roi_params,
            margin=self.roi_center_diversity_margin,
        )
        roi_center_diversity_weighted_loss = (
            self.roi_center_diversity_loss_weight * roi_center_diversity_loss
        )
        total_loss = (
            diffusion_loss
            + roi_area_weighted_loss
            + roi_center_diversity_weighted_loss
        )

        losses = {
            "diffusion_loss": diffusion_loss,
            "roi_area_loss": roi_area_loss,
            "roi_area_loss_weight": effective_roi_area_loss_weight,
            "roi_area_weighted_loss": roi_area_weighted_loss,
            "roi_center_diversity_loss": roi_center_diversity_loss,
            "roi_center_diversity_loss_weight": self.roi_center_diversity_loss_weight,
            "roi_center_diversity_weighted_loss": roi_center_diversity_weighted_loss,
            "total_loss": total_loss,
        }
        if return_roi_params:
            losses["_roi_params_B_T_3"] = roi_out["params_B_T_3"].detach()

        return losses

    def forward(self, batch, return_roi_params: bool = False):
        loss = self.compute_loss(batch, return_roi_params=return_roi_params)
        return loss
