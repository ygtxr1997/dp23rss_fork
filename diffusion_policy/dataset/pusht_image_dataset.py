from typing import Dict
import torch
from torchvision import transforms
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, EmptyNormalizer
from diffusion_policy.common.pymunk_util import ImageLightingEffect
from diffusion_policy.dataset.libero_dataset import RandomShiftsAug, VideoTransformTHWC


class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            shape_meta: Dict = None,
            transform_color_jitter: bool = False,
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.shape_meta = shape_meta

        # Transforms for data augmentation
        obs_h, obs_w = shape_meta["obs"]["image"]["shape"][1:]
        color_jitter = None
        if transform_color_jitter:
            # transforms.RandomResizedCrop(size=self.obs_image_shape[1:], scale=(0.68, 0.82),  # 12/16=0.75
            #                              ratio=(0.9 * obs_image_wh_ratio, 1.1 * obs_image_wh_ratio)),  # not using this would be better?
            color_jitter = transforms.ColorJitter(brightness=0.05,
                                                  contrast=0.05,
                                                  saturation=0.05,
                                                  hue=0.01)
        self.obs_image_transform = VideoTransformTHWC(
            image_size_hw=(obs_h, obs_w),
            color_jitter=color_jitter,
            resize_crop=True,
            random_shift_pad=8,
        )

        print(f"[PushTImageDataset] Loaded from: {zarr_path}, len={self.__len__()}.")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    # def get_normalizer(self, mode='limits', **kwargs):
    #     data = {
    #         'action': self.replay_buffer['action'],
    #         'agent_pos': self.replay_buffer['state'][...,:2]
    #     }
    #     normalizer = LinearNormalizer()
    #     normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
    #     normalizer['image'] = get_image_range_normalizer()
    #     return normalizer
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        obs_keys = self.shape_meta["obs"].keys()
        for obs_key in obs_keys:
            normalizer[obs_key] = EmptyNormalizer.create_identity()
        # normalizer['image'] = EmptyNormalizer.create_identity()
        # normalizer['joint_state'] = EmptyNormalizer.create_identity()
        normalizer['action'] = EmptyNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)

        # image = np.moveaxis(sample['img'],-1,1) / 255  # (B,H,W,C) -> (B,C,H,W), in [0,1]
        image = self.obs_image_transform(sample['img'])  # (T,H,W,C) in [0,255] -> (T,C,H,W) in [0,1]

        # print(sample.keys(), type(sample['img']))
        # print(sample['img'].mean(), sample['img'].max(),sample['img'].min())
        # from PIL import Image
        # s_img = Image.fromarray(sample['img'][0].astype(np.uint8))
        # s_img.save("tmp_train_b0.png")

        # Manually normalization
        agent_pos_data = agent_pos / 256. - 1. # from (0,512) to (-1,1)
        image_data = image * 2. - 1. # from [0,1] to [-1,1]
        act_data = sample['action'].astype(np.float32) / 256. - 1. # from (0,512) to (-1,1)

        data = {
            'obs': {
                'image': image_data, # T, 3, 96, 96
                'agent_pos': agent_pos_data, # T, 2
            },
            'action': act_data # T, 2
        }
        """
        [DEBUG] PushTImageDataset: Dict, keys=['obs', 'action']
        --obs: Dict, keys=['image', 'agent_pos']
        ----image, <class 'torch.Tensor'>, shape=torch.Size([10, 3, 256, 256]), min=-0.4902, max=1.0000, dtype=torch.float64
        ----agent_pos, <class 'torch.Tensor'>, shape=torch.Size([10, 2]), min=-0.1186, max=0.6088, dtype=torch.float32
        --action, <class 'torch.Tensor'>, shape=torch.Size([10, 2]), min=-0.2812, max=0.6367, dtype=torch.float32
        """
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        dict_func = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        torch_data = dict_apply(data, dict_func)
        return torch_data


class SourceTargetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 src_dataset: PushTImageDataset,
                 tgt_dataset: PushTImageDataset,
                 ):
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset

    def __len__(self):
        return min(len(self.src_dataset), len(self.tgt_dataset))

    def __getitem__(self, idx):
        src_batch = self.src_dataset[idx % len(self.src_dataset)]
        tgt_batch = self.tgt_dataset[idx % len(self.tgt_dataset)]
        return {
            "src": src_batch,
            "tgt": tgt_batch,
        }


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
