# Adapted from RLinf and OpenPI; see licenses/ and sources.json.
"""Single raw-LIBERO preprocessing boundary for training and inference."""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from .model import IMAGE_KEYS, Observation
from .tokenizer import PaligemmaTokenizer
from diffusion_policy.common.tcl_data_util import make_normalization, transform_values


def resize_uint8(image_B_H_W_C, size=224):
    """Approximate OpenPI's antialiased JAX linear resize, then uint8 round/pad."""
    height, width = image_B_H_W_C.shape[1:3]
    ratio = max(height / size, width / size)
    out_h, out_w = int(height / ratio), int(width / ratio)
    image_B_C_H_W = image_B_H_W_C.permute(0, 3, 1, 2).float()
    image_B_C_H_W = F.interpolate(image_B_C_H_W, (out_h, out_w), mode='bilinear',
                                   align_corners=False, antialias=True)
    image_B_C_H_W = image_B_C_H_W.round().clamp(0, 255).to(torch.uint8)
    top, left = (size - out_h) // 2, (size - out_w) // 2
    image_B_C_H_W = F.pad(image_B_C_H_W, (left, size - out_w - left, top, size - out_h - top))
    return image_B_C_H_W.permute(0, 2, 3, 1)


def _augment(image_B_H_W_C, wrist, generator):
    # RLinf's per-sample recipe, using CPU RNG consistently, also when generator=None.
    # This fixes upstream's skipped randomness when no generator is provided.
    frames = []
    for image_H_W_C in image_B_H_W_C:
        image_C_H_W = (image_H_W_C.permute(2, 0, 1) + 1) / 2
        if not wrist:
            height, width = image_C_H_W.shape[1:]
            crop_h, crop_w = int(height * 0.95), int(width * 0.95)
            top = torch.randint(height - crop_h + 1, (), generator=generator).item()
            left = torch.randint(width - crop_w + 1, (), generator=generator).item()
            image_C_H_W = TF.resize(TF.crop(image_C_H_W, top, left, crop_h, crop_w),
                                     [height, width], antialias=True)
            angle = torch.rand((), generator=generator).item() * 10 - 5
            image_C_H_W = TF.rotate(image_C_H_W, angle, fill=0)
        # Preserve the selected RLinf recipe's one-sided jitter ranges.
        image_C_H_W = TF.adjust_brightness(image_C_H_W, 1 + torch.rand((), generator=generator).item() * 0.3)
        image_C_H_W = TF.adjust_contrast(image_C_H_W, 1 + torch.rand((), generator=generator).item() * 0.4)
        image_C_H_W = TF.adjust_saturation(image_C_H_W, 1 + torch.rand((), generator=generator).item() * 0.5)
        frames.append(image_C_H_W.permute(1, 2, 0) * 2 - 1)
    return torch.stack(frames)


class Pi05Preprocessor:
    """Own raw images, language and checkpointed normalization/inverse action mapping."""

    def __init__(self, tokenizer_proto, norm_stats, max_token_len=200, image_size=224, visual_input_mode='dense', foveation_split_mode='level:2', foveation_random_seed=42, render_image_orientation='rot180', normalization=None):
        if render_image_orientation not in ('raw', 'rot180'):
            raise ValueError("render_image_orientation must be 'raw' or 'rot180'")
        self.render_image_orientation = render_image_orientation
        self.tokenizer = PaligemmaTokenizer(tokenizer_proto, max_token_len)
        self.image_size = int(image_size)
        self.visual_input_mode = visual_input_mode
        self.norm_stats = copy.deepcopy(norm_stats.get('norm_stats', norm_stats))
        self.normalization = copy.deepcopy(normalization)
        if normalization is not None:
            self.normalization = make_normalization(
                {'statistics': self.norm_stats}, normalization['methods'],
                dimensions={key: field['dim'] for key, field in normalization['fields'].items()},
                saved=normalization)
            return
        for key, dim in (('state', 6), ('actions', 7)):
            for quantile in ('q01', 'q99'):
                values_D = np.asarray(self.norm_stats[key][quantile])
                if values_D.ndim != 1 or len(values_D) < dim or not np.isfinite(values_D).all():
                    raise ValueError(f'{key}.{quantile} must contain at least {dim} finite values')

    def normalize(self, values_B_T_D, key):
        """Apply saved scaling (legacy quantile by default), then pad to 32."""
        dim = values_B_T_D.shape[-1]
        if self.normalization is not None:
            field = self.normalization['fields']['action' if key == 'actions' else key]
            return F.pad(transform_values(values_B_T_D, field), (0, 32 - dim))
        low_D = torch.as_tensor(self.norm_stats[key]['q01'][:dim], device=values_B_T_D.device, dtype=torch.float32)
        high_D = torch.as_tensor(self.norm_stats[key]['q99'][:dim], device=values_B_T_D.device, dtype=torch.float32)
        values_B_T_D = (values_B_T_D.float() - low_D) / (high_D - low_D + 1e-6) * 2 - 1
        return F.pad(values_B_T_D, (0, 32 - dim))

    def decode_actions(self, actions_B_T_D):
        """Return seven raw control coordinates using the saved inverse transform."""
        if self.normalization is not None:
            return transform_values(actions_B_T_D[..., :7], self.normalization['fields']['action'], inverse=True)
        low_D = torch.as_tensor(self.norm_stats['actions']['q01'][:7], device=actions_B_T_D.device, dtype=torch.float32)
        high_D = torch.as_tensor(self.norm_stats['actions']['q99'][:7], device=actions_B_T_D.device, dtype=torch.float32)
        return (actions_B_T_D[..., :7].float() + 1) / 2 * (high_D - low_D + 1e-6) + low_D

    def prepare(self, obs, *, device, train=False, augmentation_generator=None):
        """Transform raw RGB, language and a 6/7-dimensional TCL state."""
        state_B_D = torch.as_tensor(obs['joint_state'], dtype=torch.float32, device=device)
        if state_B_D.ndim != 2 or state_B_D.shape[1] not in (6, 7, 8):
            raise ValueError('joint_state must be Bx6/Bx7 for TCL or Bx8 for LIBERO')
        batch_size = state_B_D.shape[0]
        prompts = obs['prompt']
        if isinstance(prompts, str) or len(prompts) != batch_size or not all(isinstance(p, str) for p in prompts):
            raise ValueError('prompt must be a list of B language strings')
        images = {}
        for source, name in zip(('image', 'gripper'), IMAGE_KEYS[:2]):
            image_B_H_W_C = torch.as_tensor(obs[source]).cpu()
            if (image_B_H_W_C.dtype != torch.uint8 or image_B_H_W_C.ndim != 4
                    or image_B_H_W_C.shape[0] != batch_size or image_B_H_W_C.shape[-1] != 3):
                raise ValueError(f'{source} must be raw RGB uint8 BxHxWx3')
            # Dataset and evaluator supply raw RGB; only the static camera rotates.
            if source == 'image' and self.render_image_orientation == 'rot180':
                image_B_H_W_C = image_B_H_W_C.flip((1, 2))
            image_B_H_W_C = resize_uint8(image_B_H_W_C, self.image_size)
            images[name] = image_B_H_W_C.float() / 255 * 2 - 1
        images[IMAGE_KEYS[2]] = torch.full_like(images[IMAGE_KEYS[0]], -1)
        if train:
            if augmentation_generator is not None and augmentation_generator.device.type != 'cpu':
                raise ValueError('Image augmentation uses a CPU generator')
            images = {name: _augment(image_B_H_W_C, 'wrist' in name, augmentation_generator)
                      for name, image_B_H_W_C in images.items()}
        tokens_and_masks = [self.tokenizer.tokenize(prompt) for prompt in prompts]
        tokens_B_L = torch.as_tensor(np.stack([item[0] for item in tokens_and_masks]), device=device)
        masks_B_L = torch.as_tensor(np.stack([item[1] for item in tokens_and_masks]), device=device)
        images = {name: image_B_H_W_C.to(device) for name, image_B_H_W_C in images.items()}
        return Observation(
            images=images,
            image_masks={name: torch.full((batch_size,), index < 2, dtype=torch.bool, device=device)
                         for index, name in enumerate(IMAGE_KEYS)},
            state=self.normalize(state_B_D, 'state'), tokenized_prompt=tokens_B_L,
            tokenized_prompt_mask=masks_B_L)
