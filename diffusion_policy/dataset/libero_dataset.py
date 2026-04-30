import os
import copy
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from collections.abc import Sequence
import numpy as np
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F

from robokit.datasets.libero.libero_force import LiberoH5FrameDataset, MergeDataset
from robokit.debug_utils.printer import print_batch

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, EmptyNormalizer


class RandomShiftsAug(torch.nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()

        # 1. 对图像进行填充
        padding = tuple([self.pad] * 4)
        x_padded = F.pad(x, padding, "replicate")

        # 2. 创建一个标准化的坐标网格 ([-1, 1])
        # 这个网格对应于原始图像尺寸 (h, w)
        arange_h = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        arange_w = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)

        # 使用 meshgrid 创建一个 (h, w, 2) 的网格
        grid_h, grid_w = torch.meshgrid(arange_h, arange_w, indexing='ij')
        base_grid = torch.stack((grid_w, grid_h), dim=-1)  # (x, y) 坐标
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 扩展到批次维度 (n, h, w, 2)

        # 3. 生成单个随机位移，并将其应用于所有帧
        # 在填充后的像素空间中生成随机整数位移
        # 将 size 的第一个维度从 n 改为 1，以确保所有帧使用相同的位移
        shift_xy = torch.randint(0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)

        # 4. 将像素位移转换为 [-1, 1] 坐标空间的位移
        # 注意：h 和 w 的缩放因子不同
        shift_w = shift_xy[..., 0] * 2.0 / (w + 2 * self.pad)
        shift_h = shift_xy[..., 1] * 2.0 / (h + 2 * self.pad)
        shift = torch.stack((shift_w, shift_h), dim=-1)

        # 5. 将位移应用到基础网格上
        # shift 的形状是 (1, 1, 1, 2)，会自动广播到 base_grid 的 (n, h, w, 2)
        grid = base_grid + shift

        # 6. 使用 grid_sample 进行采样
        return F.grid_sample(x_padded, grid, padding_mode="zeros", align_corners=False)


class VideoTransformTHWC:
    """
    Apply torchvision transforms to a video tensor stored as:
      input:  np.ndarray (T, H, W, C), usually uint8
      output: torch.Tensor (T, C, H, W), float32 in [0,1] (or whatever ToTensor gives)
    Supports consistent augmentation across frames via a fixed seed per call.
    """
    def __init__(self, image_size_hw, color_jitter=None,
                 resize_crop: bool = False,
                 random_shift_pad: int = None,
                 ):
        """
        image_size_hw: (H, W) output size
        color_jitter: torchvision.transforms.ColorJitter or None
        """
        tfs = [
            transforms.ToPILImage(),                 # expects (H,W,C) uint8
            transforms.Resize(image_size_hw),
        ]
        if resize_crop:
            tfs.append(transforms.RandomResizedCrop(
                size=image_size_hw,
                scale=(0.8, 1.0),  # 12/16=0.75
                ratio=(0.95, 1.0)
            ))
        self.random_shift_pad = random_shift_pad
        if random_shift_pad is not None:
            self.random_shift_aug = RandomShiftsAug(pad=random_shift_pad)
        else:
            self.random_shift_aug = lambda x: x
        if color_jitter is not None:
            tfs.append(color_jitter)
        tfs.append(transforms.ToTensor())           # -> (C,H,W), float in [0,1]
        self.frame_tf = transforms.Compose(tfs)

    def __call__(self, video_thwc: np.ndarray, consistent: bool = True, seed: int = None) -> torch.Tensor:
        assert isinstance(video_thwc, np.ndarray), type(video_thwc)
        assert video_thwc.ndim == 4, f"expected (T,H,W,C), got {video_thwc.shape}"
        T, H, W, C = video_thwc.shape
        assert C in (1, 3, 4), f"unexpected channel C={C}"

        # choose a seed per video-call (so each sample differs, but frames inside sample一致)
        if seed is None:
            seed = int(torch.randint(0, 10_000_000, (1,)).item())

        frames = []
        for t in range(T):
            frame = video_thwc[t]  # (H,W,C) np
            if consistent:
                torch.manual_seed(seed)  # reset so random jitter is identical per frame
            frames.append(self.frame_tf(frame))
        aug_result = torch.stack(frames, dim=0)  # (T,C,H,W)

        aug_result = self.random_shift_aug(aug_result)
        return aug_result


class LiberoFTDataset(BaseImageDataset):
    def __init__(
            self,
            # RoboKit Dataset
            dataset_root: str,
            dataset_subname: Union[List[str], str],
            hdf5_fns: List[str],
            # Data sequence
            horizon: int,
            pad_before: int,
            pad_after: int,
            # Data format
            shape_meta: dict,
            norm_force_type: str = "quantile",
            # Others
            seed: int = 42,
            val_ratio: float = 0.02,
            max_train_episodes: int = 90,
            max_len: Optional[int] = None,
            transform_color_jitter: bool = True,
            load_future_obs: bool = False,
            zero_force: bool = False,
    ):
        super().__init__()
        # RoboKit Dataset
        self.dataset_root = dataset_root
        self.dataset_subname = dataset_subname
        self.norm_force_type = norm_force_type
        self.hdf5_fns = hdf5_fns
        if isinstance(dataset_subname, str):
            dataset_subnames = [dataset_subname] * len(hdf5_fns)
        else:
            assert isinstance(dataset_subname, Sequence) and not isinstance(dataset_subname, (str, bytes)), \
                f"dataset_subname should be str or sequence of str, but got {type(dataset_subname)}"
            assert len(dataset_subname) == len(hdf5_fns), \
                f"len(dataset_subname)={len(dataset_subname)} must equal len(hdf5_fns)={len(hdf5_fns)}"
            dataset_subnames = list(dataset_subname)
        self.dataset_subnames = dataset_subnames
        self.resolved_dataset_roots = []
        self.resolved_dataset_subnames = []
        self.hdf5_paths = []
        self.libero_h5_datasets = []
        for subname, fn in zip(dataset_subnames, hdf5_fns):
            current_root = dataset_root
            current_subname = subname
            hdf5_path = os.path.join(current_root, current_subname, fn)
            if not os.path.exists(hdf5_path):
                current_root = os.path.join(current_root, current_subname)
                current_subname = "libero_90"
                hdf5_path = os.path.join(current_root, current_subname, fn)
            assert os.path.exists(hdf5_path), f"HDF5 file not found: {hdf5_path}"
            self.resolved_dataset_roots.append(current_root)
            self.resolved_dataset_subnames.append(current_subname)
            self.hdf5_paths.append(hdf5_path)
            self.libero_h5_datasets.append(
                LiberoH5FrameDataset(
                    hdf5_path,
                )
            )
        self.merge_dataset = MergeDataset(
            self.libero_h5_datasets,
            n_obs=pad_before + 1,
            chunk_size=horizon,
            obs_keys=["agentview_rgb", "eye_in_hand_rgb", "ee_states", "gripper_states", "wrenches"],
            max_len=max_len,
            load_future_obs=load_future_obs,
        )
        self.dataset_stats = self.merge_dataset.compute_statistics(
            keys={
                "obs": ["wrenches", "ee_states"],
                "state": [],
                "action": ["actions"],
            }
        )
        '''
        Libero Dataset Statistics: Dict, keys=['obs.wrenches', 'obs.ee_states', 'action.actions']
        --obs.wrenches: Dict, keys=['count', 'mean', 'std', 'min', 'max', 'p01', 'p99']
        ----count: <class 'int'>, value=35169
        ----mean: List, len=6, elem_type=<class 'float'>
        ------[0]: <class 'float'>, value=1.8430029145885263
        ----std: List, len=6, elem_type=<class 'float'>
        ------[0]: <class 'float'>, value=13.76074675401503
        ----min: List, len=6, elem_type=<class 'float'>
        ------[0]: <class 'float'>, value=-129.31649780273438
        ----max: List, len=6, elem_type=<class 'float'>
        ------[0]: <class 'float'>, value=358.5761413574219
        ----p01: List, len=6, elem_type=<class 'float'>
        ------[0]: <class 'float'>, value=-32.60213851928711
        ----p99: List, len=6, elem_type=<class 'float'>
        ------[0]: <class 'float'>, value=49.68210983276367
        '''
        for k, stat_dict in self.dataset_stats.items():
            self.dataset_stats[k] = {sub_k: np.array(sub_v) for sub_k, sub_v in stat_dict.items()}

        # Others
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.max_len = max_len
        self.zero_force = zero_force

        # Sampling a data sequence
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Data format and preprocessing
        self.shape_meta = shape_meta
        self.obs_image_shape = shape_meta["obs"]["image"]["shape"]  # [3, H, W]
        self.obs_gripper_shape = shape_meta["obs"]["gripper"]["shape"] if "gripper" in shape_meta["obs"] else None
        self.joint_state_shape = shape_meta["obs"]["joint_state"]["shape"]
        self.force_torque_shape = shape_meta["obs"]["force_torque"]["shape"] if "force_torque" in shape_meta[
            "obs"] else None
        self.action_shape = shape_meta["action"]["shape"]  # [7,]
        obs_image_wh_ratio = float(self.obs_image_shape[2]) / float(self.obs_image_shape[1])  # wh 4:3=16:12=12:9
        obs_h, obs_w = self.obs_image_shape[1:]  # 你原写法: (H,W)
        transform_list = [
            transforms.ToPILImage(),  # wh 16:9
            transforms.Resize(self.obs_image_shape[1:]),
        ]
        color_jitter = None
        if transform_color_jitter:
            # transforms.RandomResizedCrop(size=self.obs_image_shape[1:], scale=(0.68, 0.82),  # 12/16=0.75
            #                              ratio=(0.9 * obs_image_wh_ratio, 1.1 * obs_image_wh_ratio)),  # not using this would be better?
            color_jitter = transforms.ColorJitter(brightness=0.05,
                                                  contrast=0.05,
                                                  saturation=0.05,
                                                  hue=0.05)
        self.obs_image_transform = VideoTransformTHWC(
            image_size_hw=(obs_h, obs_w),
            color_jitter=color_jitter,
            resize_crop=True,
            random_shift_pad=4,
        )

        print(f"[diffusion_policy.libero_dataset.LiberoFTDataset] dataset loaded, len={len(self.merge_dataset)}, "
              f"hdf5_fns={self.hdf5_fns}, dataset_subnames={self.dataset_subnames}.")

    def get_validation_dataset(self):
        return self.create_val_dataset(self)

    @classmethod
    def create_val_dataset(cls, instance: 'LiberoFTDataset'):
        val_set = cls(
            dataset_root=instance.dataset_root,
            dataset_subname=instance.dataset_subname,
            hdf5_fns=instance.hdf5_fns,
            horizon=instance.horizon,
            pad_before=instance.pad_before,
            pad_after=instance.pad_after,
            shape_meta=instance.shape_meta,
            norm_force_type=instance.norm_force_type,
            seed=instance.seed,
            val_ratio=instance.val_ratio,
            max_train_episodes=instance.max_train_episodes,
            transform_color_jitter=False,  # no color jitter for val
            max_len=instance.max_len,
        )
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        obs_keys = self.shape_meta["obs"].keys()
        for obs_key in obs_keys:
            normalizer[obs_key] = EmptyNormalizer.create_identity()
        # normalizer['image'] = EmptyNormalizer.create_identity()
        # normalizer['joint_state'] = EmptyNormalizer.create_identity()
        normalizer['action'] = EmptyNormalizer.create_identity()
        return normalizer

    def _consistent_augmentations(self, frame):
        video_seed = torch.randint(0, 10000, (1,)).item()
        # Set the random seed for each frame to ensure consistent augmentation
        torch.manual_seed(video_seed)
        augmentation = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),  # Color jitter
            ]
        )
        return augmentation(frame)

    def __len__(self) -> int:
        return len(self.merge_dataset)

    def __getitem__(self, idx: int) -> dict:
        sample_dict = self.merge_dataset[idx]
        '''
        MergeDataset[0]: Dict, keys=['demo_key', 't', 'text_instruction', 'history', 'future', 'action']
        --demo_key: <class 'str'>, len=6, value='demo_0'
        --t: <class 'int'>, value=0
        --text_instruction: <class 'str'>, len=77, value='KITCHEN SCENE4 close the bottom drawer of the cabinet and open the top drawer'
        --history: Dict, keys=['obs', 'state']
        ----obs: Dict, keys=['agentview_rgb', 'eye_in_hand_rgb', 'ee_states', 'gripper_states', 'wrenches']
        ------agentview_rgb, <class 'numpy.ndarray'>, shape=(2, 128, 128, 3), min=0.0000, max=255.0000, dtype=uint8
        ------eye_in_hand_rgb, <class 'numpy.ndarray'>, shape=(2, 128, 128, 3), min=0.0000, max=237.0000, dtype=uint8
        ------ee_states, <class 'numpy.ndarray'>, shape=(2, 6), min=-0.2018, max=3.1760, dtype=float64
        ------gripper_states, <class 'numpy.ndarray'>, shape=(2, 2), min=-0.0340, max=0.0341, dtype=float64
        ------wrenches, <class 'numpy.ndarray'>, shape=(2, 6), min=-3.3128, max=1.0444, dtype=float32
        --future: Dict, keys=['obs', 'state']
        ----obs: Dict, keys=['agentview_rgb', 'eye_in_hand_rgb', 'ee_states', 'gripper_states', 'wrenches']
        ------agentview_rgb, <class 'numpy.ndarray'>, shape=(32, 128, 128, 3), min=0.0000, max=255.0000, dtype=uint8
        ------eye_in_hand_rgb, <class 'numpy.ndarray'>, shape=(32, 128, 128, 3), min=0.0000, max=238.0000, dtype=uint8
        ------ee_states, <class 'numpy.ndarray'>, shape=(32, 6), min=-0.2036, max=3.3343, dtype=float64
        ------gripper_states, <class 'numpy.ndarray'>, shape=(32, 2), min=-0.0395, max=0.0396, dtype=float64
        ------wrenches, <class 'numpy.ndarray'>, shape=(32, 6), min=-5.2036, max=1.1171, dtype=float32
        --action, <class 'numpy.ndarray'>, shape=(32, 7), min=-1.0000, max=0.9375, dtype=float64
        '''

        obs_keys = self.shape_meta["obs"].keys()
        obs_data = {k: [] for k in obs_keys}

        history_primary_rgb = sample_dict["history"]["obs"]["agentview_rgb"]  # (T,H,W,C)
        history_gripper_rgb = sample_dict["history"]["obs"]["eye_in_hand_rgb"]  # (T,H,W,C)
        history_ee_states = sample_dict["history"]["obs"]["ee_states"]  # (T,D)
        history_wrenches = sample_dict["history"]["obs"]["wrenches"]  # (T,D)
        actions = sample_dict["action"]  # (T,D)

        # Preprocess
        history_primary_rgb = self.obs_image_transform(history_primary_rgb)  # (T,C,H,W)
        history_gripper_rgb = self.obs_image_transform(history_gripper_rgb)
        history_primary_rgb = history_primary_rgb * 2. - 1.  # in [-1,1]
        history_gripper_rgb = history_gripper_rgb * 2. - 1.  # in [-1,1]
        # NOTE: Randomly zero out gripper image with 0.3 probability
        if torch.rand(1).item() < 0.3:
            history_gripper_rgb = torch.zeros_like(history_gripper_rgb)

        tcp_pose = self.norm_state_or_force(
            history_ee_states,
            norm_type="mean",
            meta_data=self.dataset_stats["obs.ee_states"],
        )

        force_torque = self.norm_state_or_force(
            history_wrenches,
            norm_type=self.norm_force_type,
            meta_data=self.dataset_stats["obs.wrenches"],
        )

        obs_data["image"] = history_primary_rgb  # (T,C,H,W)
        obs_data["gripper"] = history_gripper_rgb  # (T,C,H,W)
        obs_data["joint_state"] = tcp_pose  # (T,D)
        obs_data["force"] = force_torque  # (T,D)
        for k in obs_data.keys():
            assert obs_data[k] != [], f"obs_data[{k}] is empty!"

        if self.zero_force:
            obs_data["force"] *= 0.0

        # act_data = (actions - self.dataset_stats["action.actions"]["min"]) / (
        #         self.dataset_stats["action.actions"]["max"] - self.dataset_stats["action.actions"]["min"] + 1e-8
        # )  # in [0,1]
        mins = self.dataset_stats["action.actions"]["min"]
        maxs = self.dataset_stats["action.actions"]["max"]
        rng = maxs - mins
        eps_range = 1e-6
        safe_rng = np.where(rng < eps_range, 1.0, rng)
        act_data = (actions - mins) / safe_rng

        # 常数维设为中值 -> 最终是 0
        mask_const = (rng < eps_range)
        act_data[..., mask_const] = 0.5
        act_data = act_data * 2. - 1.  # in [-1,1]

        item_data = {
            "obs": obs_data,
            "action": act_data,
        }
        return item_data

    @staticmethod
    def norm_state_or_force(in_data: np.ndarray, norm_type: str, meta_data: dict):
        """
        `robot_obs`: (...,14)
            tcp pos (3), tcp ori (3), gripper width (1), joint_states (6) in rad, gripper_action (1)
        `force_torque`: (...,6)
        """
        D = in_data.shape[-1]
        if norm_type == "minmax":
            dataset_min = np.array(meta_data['min'])[:D]
            dataset_max = np.array(meta_data['max'])[:D]
            out_data = (in_data - dataset_min) / (dataset_max - dataset_min)  # norm here, in [0,1]
            out_data = out_data * 2. - 1.  # in [-1,1]
        elif norm_type == "mean":
            dataset_mean = np.array(meta_data['mean'])[:D]
            dataset_std = np.array(meta_data['std'])[:D]
            out_data = (in_data - dataset_mean) / dataset_std
        elif norm_type == "quantile":
            dataset_p01 = np.array(meta_data['p01'])[:D]
            dataset_p99 = np.array(meta_data['p99'])[:D]
            in_data = np.clip(in_data, dataset_p01, dataset_p99)  # different from minmax
            out_data = (in_data - dataset_p01) / (dataset_p99 - dataset_p01)  # norm here, in [0,1]
            out_data = out_data * 2. - 1.  # in [-1,1]
        else:
            assert norm_type == "identity"
            out_data = in_data
        return out_data
