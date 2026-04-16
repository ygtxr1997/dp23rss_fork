from typing import Dict, List
import bisect
import torch
from torch.utils.data import ConcatDataset
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
            return_all_state: bool = False
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

        self.return_all_state = return_all_state
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
        all_state = sample['state'].astype(np.float32)  # (agent_posx2, block_posex3)
        all_state_data = all_state / 256. - 1. # from (0,512) to (-1,1)
        agent_pos_data = all_state_data[:, :2].astype(np.float32) # (agent_posx2, block_posex3)

        # image = np.moveaxis(sample['img'],-1,1) / 255  # (B,H,W,C) -> (B,C,H,W), in [0,1]
        image = self.obs_image_transform(sample['img'])  # (T,H,W,C) in [0,255] -> (T,C,H,W) in [0,1]

        # print(sample.keys(), type(sample['img']))
        # print(sample['img'].mean(), sample['img'].max(),sample['img'].min())
        # from PIL import Image
        # s_img = Image.fromarray(sample['img'][0].astype(np.uint8))
        # s_img.save("tmp_train_b0.png")

        # Manually normalization
        # agent_pos_data = agent_pos / 256. - 1. # from (0,512) to (-1,1), already normed in all_state_data
        image_data = image * 2. - 1. # from [0,1] to [-1,1]
        act_data = sample['action'].astype(np.float32) / 256. - 1. # from (0,512) to (-1,1)

        data = {
            'obs': {
                'image': image_data, # T, 3, 96, 96
                'agent_pos': agent_pos_data, # T, 2
            },
            'action': act_data # T, 2
        }
        if self.return_all_state:
            data['obs']['all_state'] = all_state_data  # T, 5
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


class MergedPushTImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_paths: List[str],  # 注意这里改为 List[str]
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 shape_meta: Dict = None,
                 transform_color_jitter: bool = False,
                 return_all_state: bool = False
                 ):
        super().__init__()

        # 实例化多个 PushTImageDataset
        self.datasets = []
        for path in zarr_paths:
            dataset = PushTImageDataset(
                zarr_path=path,
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                seed=seed,
                val_ratio=val_ratio,
                max_train_episodes=max_train_episodes,
                shape_meta=shape_meta,
                transform_color_jitter=transform_color_jitter,
                return_all_state=return_all_state,
            )
            self.datasets.append(dataset)

        # 使用 PyTorch 原生的 ConcatDataset 来处理 idx 的跨数据集映射
        self.concat_dataset = ConcatDataset(self.datasets)
        print("[MergedPushTImageDataset] Datasets merged, total length:", len(self.concat_dataset))

        # 暴露原有类的一些关键属性，保持接口一致性
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.shape_meta = shape_meta

    def get_validation_dataset(self):
        """
        分别获取每个子数据集的验证集，然后重新组装为一个新的 Merged 实例
        """
        val_set = copy.copy(self)
        val_set.datasets = [ds.get_validation_dataset() for ds in self.datasets]
        val_set.concat_dataset = ConcatDataset(val_set.datasets)
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        根据你目前 PushTImageDataset 的实现，正常化器使用的是 EmptyNormalizer.create_identity()。
        由于它不依赖具体的统计数据，直接返回第一个子数据集的 normalizer 即可。

        注意：如果你未来启用了基于数据统计的 Normalizer（如注释掉的那部分代码），
        你需要在这里把所有 self.datasets 的 replay_buffer 数据拼接起来再做 fit()。
        """
        return self.datasets[0].get_normalizer(**kwargs)

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # ConcatDataset 会自动帮你计算这个全局 idx 落在哪个子 dataset 上，
        # 并调用对应的子 dataset 的 __getitem__ (即触发 _sample_to_data)
        ds_idx = bisect.bisect_right(self.concat_dataset.cumulative_sizes, idx)
        start = 0 if ds_idx == 0 else self.concat_dataset.cumulative_sizes[ds_idx - 1]
        local_idx = idx - start

        item = self.datasets[ds_idx][local_idx]
        item['dataset_idx'] = torch.tensor(ds_idx, dtype=torch.long)
        item['local_idx'] = torch.tensor(local_idx, dtype=torch.long)
        """
        [DEBUG] PushTImageDataset: Dict, keys=['obs', 'action', 'dataset_idx', 'local_idx']
        --obs: Dict, keys=['image', 'agent_pos', 'all_state']
        ----image, <class 'torch.Tensor'>, shape=torch.Size([10, 3, 256, 256]), min=-1.0000, max=1.0000, dtype=torch.float32
        ----agent_pos, <class 'torch.Tensor'>, shape=torch.Size([10, 2]), min=-0.1186, max=0.6088, dtype=torch.float32
        ----all_state, <class 'torch.Tensor'>, shape=torch.Size([10, 5]), min=-1.0129, max=0.6088, dtype=torch.float32
        --action, <class 'torch.Tensor'>, shape=torch.Size([10, 2]), min=-0.2812, max=0.6367, dtype=torch.float32
        --dataset_idx, <class 'torch.Tensor'>, shape=torch.Size([]), value=0
        --local_idx, <class 'torch.Tensor'>, shape=torch.Size([]), value=0
        """
        return item


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
