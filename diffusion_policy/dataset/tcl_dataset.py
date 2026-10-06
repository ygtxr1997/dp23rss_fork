import os
import copy
import bisect
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from torch.utils.data import ConcatDataset, default_collate
from omegaconf import OmegaConf
from torchvision.transforms import transforms

from robokit.datasets.tcl_datasets import TCLDataset, TCLDatasetHDF5
from robokit.debug_utils.printer import print_batch

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, EmptyNormalizer
from diffusion_policy.dataset.libero_dataset import RandomShiftsAug as LiberoRandomShiftsAug, VideoTransformTHWC


class RandomShiftsAug(LiberoRandomShiftsAug):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return super().forward(x.unsqueeze(0))[0]
        return super().forward(x)


class TCLImageDataset(BaseImageDataset):
    def __init__(self,
                 # RoboKit Dataset
                 data_root: str,
                 # Data sequence
                 horizon: int,
                 pad_before: int,
                 pad_after: int,
                 # Data format
                 shape_meta: dict,
                 norm_force_type: str = "quantile",
                 zero_force: bool = False,
                 p_camera_drop: float = 0.,
                 p_cond_drop: float = 0.,
                 default_success_label: Optional[int] = None,
                 # Others
                 seed: int = 42,
                 val_ratio: float = 0.02,
                 max_train_episodes: int = 90,
                 transform_color_jitter: bool = True,
                 # RoboKit Dataset
                 h5_path: str = None,
                 use_h5: bool = False,
                 dataset_statistics: Optional[dict] = None,
                 # Not used
                 **kwargs
                 ):
        super().__init__()
        # RoboKit Dataset
        self.data_root = data_root
        self.shape_meta = shape_meta
        self.norm_force_type = norm_force_type
        self.zero_force = zero_force
        self.p_camera_drop = p_camera_drop
        self.p_cond_drop = p_cond_drop
        self.default_success_label = default_success_label
        self.use_success_label = "success" in self.shape_meta["obs"]

        self.load_keys = ["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text", "force_torque"]
        self.h5_path = h5_path
        self.use_h5 = use_h5
        if not use_h5:
            self.tcl_dataset = TCLDataset(data_root, use_extracted=True, load_keys=self.load_keys)
        else:
            self.tcl_dataset = TCLDatasetHDF5(
                data_root, h5_path,
                use_extracted=True, load_keys=self.load_keys)
        print("[DEBUG] type of tcl_dataset:", type(self.tcl_dataset))
        self.data_meta = self.tcl_dataset.load_meta_from_json(os.path.join(data_root, "statistics.json"))
        self.all_rel_actions = self.tcl_dataset.extracted_data["rel_actions"]
        self.all_force_torques = self.tcl_dataset.dsets["force_torque"]
        self.dataset_stats = copy.deepcopy(
            self.data_meta["stats"] if dataset_statistics is None else dataset_statistics
        )  # Keep checkpoint scaling when reopening the same dataset.
        self.dataset_total_len = self.data_meta["total_len"]
        self.dataset_action_min = np.array(self.dataset_stats["rel_actions"]["min"])
        self.dataset_action_max = np.array(self.dataset_stats["rel_actions"]["max"])

        # Calculate p01 and p99 for force_torque if available
        if dataset_statistics is None and 'force_torque' in self.dataset_stats:
            # Calculate p01 (1%) and p99 (99%) quantiles along the sample dimension (axis=0)
            p01 = np.quantile(self.all_force_torques, q=0.01, axis=0)
            p99 = np.quantile(self.all_force_torques, q=0.99, axis=0)

            # Add the calculated quantiles to the merged statistics dictionary
            self.dataset_stats['force_torque']['p01'] = p01
            self.dataset_stats['force_torque']['p99'] = p99

        self.tasks = self.tcl_dataset.tasks
        self.task_lengths = self.tcl_dataset.task_lengths
        self.ep_fns = self.tcl_dataset.ep_fns
        self.map_index_to_task_id = self.tcl_dataset.map_index_to_task_id

        # Others
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes

        # Sampling a data sequence
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.task_prefix_lengths = [0 for _ in range(len(self.task_lengths))]
        for i in range(1, len(self.task_prefix_lengths)):  # 3 means [0+1+2]
            self.task_prefix_lengths[i] = self.task_prefix_lengths[i - 1] + self.task_lengths[i - 1]

        # Data format and preprocessing
        self.obs_image_shape = shape_meta["obs"]["image"]["shape"]  # [3, H, W]
        self.obs_gripper_shape = shape_meta["obs"]["gripper"]["shape"] if "gripper" in shape_meta["obs"] else None
        self.joint_state_shape = shape_meta["obs"]["joint_state"]["shape"]
        self.force_torque_shape = shape_meta["obs"]["force_torque"]["shape"] if "force_torque" in shape_meta["obs"] else None
        self.action_shape = shape_meta["action"]["shape"]   # [7,]
        obs_image_wh_ratio = float(self.obs_image_shape[2]) / float(self.obs_image_shape[1])  # wh 4:3=16:12=12:9
        transform_list = [
            transforms.ToPILImage(),  # wh 16:9
            transforms.Resize(self.obs_image_shape[1:]),
        ]
        if transform_color_jitter:
            # transforms.RandomResizedCrop(size=self.obs_image_shape[1:], scale=(0.68, 0.82),  # 12/16=0.75
            #                              ratio=(0.9 * obs_image_wh_ratio, 1.1 * obs_image_wh_ratio)),  # not using this would be better?
            transform_list.append(transforms.ColorJitter(brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.05))
        transform_list.append(transforms.ToTensor())
        if transform_color_jitter:
            random_shift = RandomShiftsAug(pad=4)
            transform_list.append(random_shift)
        self.obs_image_transform = transforms.Compose(transform_list)  # Similar augmentation params with OCTO

        print(f"[diffusion_policy.dataset.TCLImageDataset] dataset loaded, "
              f"action_min={self.dataset_action_min}, action_max={self.dataset_action_max}, "
              f"zero_force={self.zero_force}, p_camera_drop={self.p_camera_drop}, "
              f"p_cond_drop={self.p_cond_drop}, default_success_label={self.default_success_label}.")

    def get_validation_dataset(self) -> 'TCLImageDataset':
        return self.create_val_dataset(self)

    @classmethod
    def create_val_dataset(cls, instance: 'TCLImageDataset') -> 'TCLImageDataset':
        val_set = cls(
            data_root=instance.data_root,
            horizon=instance.horizon,
            pad_before=instance.pad_before,
            pad_after=instance.pad_after,
            shape_meta=instance.shape_meta,
            norm_force_type=instance.norm_force_type,
            zero_force=instance.zero_force,
            p_camera_drop=0.0,
            p_cond_drop=0.0,
            default_success_label=instance.default_success_label,
            seed=instance.seed,
            val_ratio=instance.val_ratio,
            max_train_episodes=instance.max_train_episodes,
            use_h5=instance.use_h5,  # ori:no need to use h5
            h5_path=instance.h5_path,
            transform_color_jitter=False,
        )
        val_set.tcl_dataset.total_length = 1024
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

    def __len__(self):
        return len(self.tcl_dataset)

    def __getitem__(self, abs_idx):
        abs_idx = abs_idx % self.__len__()
        obs_data = self._get_obs_data(abs_idx)
        act_data = self._get_act_data(abs_idx)
        item_data = {
            "obs": obs_data,
            "action": act_data,
        }
        return item_data

    def _abs_idx_to_rel_idx(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        task_len = self.task_lengths[task_id]
        task_prefix_len = self.task_prefix_lengths[task_id]
        rel_idx = abs_idx - task_prefix_len
        assert 0 <= rel_idx < task_len
        return rel_idx, task_id

    def _rel_idx_to_abs_idx(self, rel_idx: int, task_id: int):
        task_prefix_len = self.task_prefix_lengths[task_id]
        task_len = self.task_lengths[task_id]
        if rel_idx < 0:
            abs_idx = None
        elif rel_idx >= task_len:
            abs_idx = None
        else:
            abs_idx = rel_idx + task_prefix_len
            assert task_prefix_len <= abs_idx < task_prefix_len + task_len
        return abs_idx

    def _get_abs_obs_indices(self, abs_now_idx: int):
        # [start_idx, end_idx)
        rel_now_idx, task_id = self._abs_idx_to_rel_idx(abs_now_idx)
        start_idx = rel_now_idx - self.pad_before
        end_idx = rel_now_idx + 1
        rel_indices = list(range(start_idx, end_idx))
        return [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]

    def _get_abs_act_indices(self, abs_now_idx: int):
        # [start_idx, end_idx)
        rel_now_idx, task_id = self._abs_idx_to_rel_idx(abs_now_idx)
        start_idx = rel_now_idx
        end_idx = rel_now_idx + self.horizon
        rel_indices = list(range(start_idx, end_idx))
        abs_indices = [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]
        action_len = len(self.all_rel_actions)  # edge safety
        return [idx if (idx is None or idx < action_len) else None for idx in abs_indices]

    def _get_obs_data_deprecated(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_obs_indices = self._get_abs_obs_indices(abs_idx)
        obs_keys = self.shape_meta["obs"].keys()
        obs_data = {
            k: [] for k in obs_keys
        }
        for idx in abs_obs_indices:
            if idx is None:
                c, h, w = self.obs_image_shape
                zero_rgb = torch.ones((c, h, w)).to(torch.float32) * -1  # all -1
                primary_rgb = zero_rgb
                gripper_rgb = zero_rgb
                tcp_pose = torch.zeros((6,)).to(torch.float32)
                force_torque = torch.zeros((6,)).to(torch.float32)
            else:
                sample_dict = self.tcl_dataset.__getitem__(idx)
                primary_rgb = sample_dict['primary_rgb']  # (H,W,C)
                gripper_rgb = sample_dict['gripper_rgb']  # (H,W,C)
                tcp_pose = joint_state = sample_dict['robot_obs'][:6]  # (6,)
                # Preprocess
                primary_rgb = self.obs_image_transform(primary_rgb)  # (C,H,W), in [0, 1]
                primary_rgb = primary_rgb * 2. - 1.  # in [-1, 1]
                tcp_pose = torch.from_numpy(tcp_pose).to(torch.float32)  # (6,)
                if "gripper" in obs_keys:
                    gripper_rgb = self.obs_image_transform(gripper_rgb)
                    gripper_rgb = gripper_rgb * 2. - 1.
                if "force" in obs_keys:
                    force_torque = sample_dict['force_torque']  # (6,)
                    force_torque = self.norm_state_or_force(
                        force_torque, self.norm_force_type, self.dataset_stats["force_torque"])
                    force_torque = torch.from_numpy(force_torque).to(torch.float32)

            # Randomly dropout camera views, robot_states
            if torch.rand(1).item() < self.p_camera_drop:
                primary_rgb = torch.zeros_like(primary_rgb)
            if "gripper" in obs_keys and torch.rand(1).item() < self.p_camera_drop:
                gripper_rgb = torch.zeros_like(gripper_rgb)
            if torch.rand(1).item() < self.p_camera_drop:
                tcp_pose = torch.zeros_like(tcp_pose)

            obs_data["image"].append(primary_rgb)
            obs_data["joint_state"].append(tcp_pose)
            if "gripper" in obs_keys:
                obs_data["gripper"].append(gripper_rgb)
            if "force" in obs_keys:
                if self.zero_force:
                    force_torque = torch.zeros_like(force_torque)
                obs_data["force"].append(force_torque)
        obs_data = {k: torch.stack(v) for k, v in obs_data.items()}
        # obs_data["image"] = torch.stack(obs_data["image"])  # should be (T,C,H,W)
        # obs_data["joint_state"] = torch.stack(obs_data["joint_state"])  # (T,6)
        return obs_data

    def _get_obs_data(self, abs_idx: int):
        abs_obs_indices = self._get_abs_obs_indices(abs_idx)
        obs_keys = self.shape_meta["obs"].keys()
        batch = self.tcl_dataset.get_item_by_indices(abs_obs_indices)

        obs_data = {k: [] for k in obs_keys}
        for t in range(len(abs_obs_indices)):
            primary_rgb = batch["primary_rgb"][t]
            gripper_rgb = batch["gripper_rgb"][t] if "gripper" in obs_keys else None
            tcp_pose = batch["robot_obs"][t][:6]

            primary_rgb = self.obs_image_transform(primary_rgb)
            primary_rgb = primary_rgb * 2. - 1.

            tcp_pose = torch.from_numpy(tcp_pose).to(torch.float32)

            if "gripper" in obs_keys:
                gripper_rgb = self.obs_image_transform(gripper_rgb)
                gripper_rgb = gripper_rgb * 2. - 1.

            if "force" in obs_keys:
                force_torque = batch["force_torque"][t]
                force_torque = self.norm_state_or_force(
                    force_torque,
                    self.norm_force_type,
                    self.dataset_stats["force_torque"],
                )
                force_torque = torch.from_numpy(force_torque).to(torch.float32)

            drop_all_cond = bool(torch.rand(1).item() < self.p_cond_drop)
            if torch.rand(1).item() < self.p_camera_drop:
                primary_rgb = torch.zeros_like(primary_rgb)
            if "gripper" in obs_keys and torch.rand(1).item() < self.p_camera_drop:
                gripper_rgb = torch.zeros_like(gripper_rgb)
            if torch.rand(1).item() < self.p_camera_drop:
                tcp_pose = torch.zeros_like(tcp_pose)
            if drop_all_cond:
                primary_rgb = torch.zeros_like(primary_rgb)
                if "gripper" in obs_keys:
                    gripper_rgb = torch.zeros_like(gripper_rgb)
                tcp_pose = torch.zeros_like(tcp_pose)

            obs_data["image"].append(primary_rgb)
            obs_data["joint_state"].append(tcp_pose)
            if "gripper" in obs_keys:
                obs_data["gripper"].append(gripper_rgb)
            if "force" in obs_keys:
                if self.zero_force:
                    force_torque = torch.zeros_like(force_torque)
                obs_data["force"].append(force_torque)

            if "success" in obs_keys:
                success_label = batch["success"][t] if "success" in batch else self.default_success_label
                if success_label is None:
                    raise ValueError("`success` is missing and `default_success_label` is None.")
                obs_data["success"].append(torch.as_tensor(success_label, dtype=torch.float32).reshape(-1))

        return {k: torch.stack(v) for k, v in obs_data.items()}

    def _get_act_data_deprecated(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_act_indices = self._get_abs_act_indices(abs_idx)
        act_data = []
        for idx in abs_act_indices:
            if idx is None:
                zero_act = np.zeros(self.action_shape).astype(np.float32)
                act_data.append(zero_act)
            else:
                rel_action = self.all_rel_actions[idx]  # (7,)
                act_data.append(rel_action)
        act_data = np.stack(act_data)  # (T,7), in [act_min, act_max]
        act_data = (act_data - self.dataset_action_min) / (self.dataset_action_max - self.dataset_action_min)  # norm here, in [0,1]
        return act_data * 2. - 1.  # in [-1,1]

    def _get_act_data(self, abs_idx: int):
        abs_act_indices = self._get_abs_act_indices(abs_idx)

        valid_positions = [i for i, idx in enumerate(abs_act_indices) if idx is not None]
        if len(valid_positions) == 0:
            raise ValueError("abs_act_indices should contain at least one valid index")

        first_pos, last_pos = valid_positions[0], valid_positions[-1]
        if any(idx is None for idx in abs_act_indices[first_pos:last_pos + 1]):
            raise ValueError("None is only allowed at the beginning or end of abs_act_indices")

        first_idx = abs_act_indices[first_pos]
        last_idx = abs_act_indices[last_pos]

        # Same padding: boundary None values repeat the nearest valid action.
        padded_indices = [
            first_idx if i < first_pos else last_idx if i > last_pos else idx
            for i, idx in enumerate(abs_act_indices)
        ]

        act_data = self.all_rel_actions[padded_indices].astype(np.float32)  # (T, action_dim)

        act_data = (act_data - self.dataset_action_min) / (
                self.dataset_action_max - self.dataset_action_min
        )
        return act_data * 2. - 1.

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


class TCLFoveationDataset(TCLImageDataset):
    """Episode-disjoint TCL windows with checkpointed splits and full video access."""

    def __init__(self, *, dataset_metadata=None, split="train", **kwargs):
        """Build deterministic episode membership or restore it from a checkpoint."""
        metadata = (OmegaConf.to_container(dataset_metadata, resolve=True)
                    if OmegaConf.is_config(dataset_metadata) else copy.deepcopy(dataset_metadata))
        super().__init__(
            **kwargs, dataset_statistics=None if metadata is None else metadata["statistics"])
        self.n_obs_steps = self.pad_before + 1
        shape_meta = (OmegaConf.to_container(self.shape_meta, resolve=True)
                      if OmegaConf.is_config(self.shape_meta) else copy.deepcopy(self.shape_meta))
        sources = dict(data_root=os.path.abspath(os.path.expanduser(self.data_root)),
                       h5_path=os.path.abspath(os.path.expanduser(self.h5_path)) if self.use_h5 else None)
        contract = dict(shape_meta=shape_meta, obs_keys=list(shape_meta["obs"]),
                        horizon=self.horizon, n_obs_steps=self.n_obs_steps,
                        pad_before=self.pad_before, pad_after=self.pad_after,
                        norm_force_type=self.norm_force_type, zero_force=self.zero_force,
                        default_success_label=self.default_success_label)
        # RoboKit calls each trajectory directory a task; files inside it are frames.
        episodes = [[name, int(length)] for name, length in zip(self.tasks, self.task_lengths)]
        assert self.task_lengths and min(self.task_lengths) > 0
        assert sum(self.task_lengths) == len(self.tcl_dataset) == len(self.all_rel_actions)
        if metadata is None:
            if len(episodes) < 2 or not 0 < self.val_ratio < 1:
                raise ValueError("TCL train/val requires >=2 episodes and 0 < val_ratio < 1")
            names = sorted(self.tasks)
            order_E = np.random.RandomState(self.seed).permutation(len(names))
            n_val = min(len(names) - 1, max(1, round(len(names) * self.val_ratio)))
            val_names = [names[i] for i in order_E[:n_val]]
            train_names = [names[i] for i in order_E[n_val:]]
            if self.max_train_episodes is not None:
                assert self.max_train_episodes > 0
                train_names = train_names[:self.max_train_episodes]
            # Statistics intentionally remain full-dataset CFGRL statistics.
            statistics = json.loads(json.dumps(
                self.dataset_stats, default=lambda value: value.tolist()))
            metadata = dict(version=1, sources=sources, contract=contract, episodes=episodes,
                            seed=self.seed, val_ratio=self.val_ratio,
                            max_train_episodes=self.max_train_episodes,
                            train=sorted(train_names), val=sorted(val_names), statistics=statistics)
        for key, value in dict(sources=sources, contract=contract, episodes=episodes,
                               seed=self.seed, val_ratio=self.val_ratio,
                               max_train_episodes=self.max_train_episodes).items():
            if metadata[key] != value:
                raise ValueError(f"TCL {key} differs from checkpoint: {value!r} != {metadata[key]!r}")
        assert metadata["train"] and metadata["val"]
        assert not set(metadata["train"]) & set(metadata["val"])
        self.dataset_metadata = metadata
        self._temporal_pairing_scalar = torch.zeros((), dtype=torch.bool).share_memory_()
        self._set_split(split)

    def _set_split(self, split):
        assert split in ("train", "val")
        self.split = split
        self._episode_ids = [self.tasks.index(name) for name in self.dataset_metadata[split]]
        self._episode_ends_E = np.cumsum([self.task_lengths[i] for i in self._episode_ids])
        self._episode_starts_E = np.concatenate(([0], self._episode_ends_E[:-1]))
        if split == "val":
            self.p_cond_drop = self.p_camera_drop = 0.0
            self.obs_image_transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize(self.obs_image_shape[1:]),
                transforms.ToTensor(),
            ])

    def get_dataset_metadata(self):
        """Return the reproducible episode split and full-dataset scaling."""
        return copy.deepcopy(self.dataset_metadata)

    def get_validation_dataset(self):
        """Share read-only source data, with separate validation sampling state."""
        val = copy.copy(self)
        val._temporal_pairing_scalar = torch.zeros((), dtype=torch.bool).share_memory_()
        val._set_split("val")
        return val

    def __len__(self):
        return int(self._episode_ends_E[-1])

    def _local_to_absolute(self, idx):
        assert 0 <= idx < len(self)
        episode = int(np.searchsorted(self._episode_ends_E, idx, side="right"))
        source_id = self._episode_ids[episode]
        abs_idx = self.task_prefix_lengths[source_id] + idx - int(self._episode_starts_E[episode])
        return abs_idx, episode

    def _get_abs_obs_indices(self, abs_now_idx):
        rel_idx, episode = self._abs_idx_to_rel_idx(abs_now_idx)
        start = self.task_prefix_lengths[episode]
        return [start + max(0, i) for i in range(rel_idx - self.pad_before, rel_idx + 1)]

    def _get_abs_act_indices(self, abs_now_idx):
        rel_idx, episode = self._abs_idx_to_rel_idx(abs_now_idx)
        start, length = self.task_prefix_lengths[episode], self.task_lengths[episode]
        # Diffusion action index n_obs_steps-1 corresponds to the current frame.
        return [start + min(length - 1, max(0, i))
                for i in range(rel_idx - self.pad_before, rel_idx - self.pad_before + self.horizon)]

    def _get_single_window(self, idx):
        abs_idx, _ = self._local_to_absolute(idx)
        return dict(obs=self._get_obs_data(abs_idx), action=self._get_act_data(abs_idx),
                    task_id=torch.tensor(0, dtype=torch.long))

    def __getitem__(self, idx):
        data = self._get_single_window(idx)
        if bool(self._temporal_pairing_scalar):
            _, episode = self._local_to_absolute(idx)
            start, end = int(self._episode_starts_E[episode]), int(self._episode_ends_E[episode])
            valid = end - start > 1
            if valid:
                pair_idx = start + np.random.randint(end - start - 1)
                pair_idx += pair_idx >= idx
                pair = self._get_single_window(pair_idx)
            else:
                pair = dict(data)
            data["temporal_pair"] = pair
            data["temporal_pair_valid"] = torch.tensor(valid, dtype=torch.bool)
        return data

    def set_temporal_pairing(self, enabled: bool):
        """Toggle training-only episode-local pairs, including in persistent workers."""
        assert not enabled or self.split == "train"
        self._temporal_pairing_scalar.fill_(enabled)

    def iter_episodes(self):
        """Yield stable source IDs, names and lengths for the selected split."""
        for episode_id in self._episode_ids:
            yield episode_id, self.tasks[episode_id], self.task_lengths[episode_id]

    def get_episode_task_name(self, episode_id):
        """Return the source file stem (or data directory name) for an episode."""
        assert episode_id in self._episode_ids
        # All episodes in this dataset instance share one source HDF5 file.
        if self.use_h5:
            return os.path.splitext(os.path.basename(self.h5_path))[0]
        return os.path.basename(os.path.normpath(self.data_root))

    def get_episode_observation_batch(self, episode_id, frame_indices):
        """Read deterministic observation windows for validation video frames."""
        assert self.split == "val" and episode_id in self._episode_ids
        assert all(0 <= i < self.task_lengths[episode_id] for i in frame_indices)
        start = self.task_prefix_lengths[episode_id]
        return default_collate([self._get_obs_data(start + int(i)) for i in frame_indices])

    def iter_task_diagnostic_batches(self, samples_per_task: int, seed: int):
        """Yield one TCL diagnostic group without advancing augmentation or pairing RNG."""
        assert samples_per_task >= 2
        indices_B = np.random.RandomState(seed).choice(
            len(self), min(samples_per_task, len(self)), replace=False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            batch = default_collate([self._get_single_window(int(i)) for i in indices_B])
        yield 0, batch


class TCLMasterSlaveDataset(BaseImageDataset):
    def __init__(
            self,
            # Master dataset config
            data_root: str,
            # Slave dataset config (must align one-to-one)
            slave_data_roots: List[str],
            slave_h5_paths: List[str],
            # Sequence config
            horizon: int,
            pad_before: int,
            pad_after: int,
            # Data format
            shape_meta: dict,
            norm_force_type: str = "quantile",
            zero_force: bool = False,
            p_camera_drop: float = 0.,
            p_cond_drop: float = 0.,
            default_success_label: Optional[int] = None,  # for `Master`
            slave_default_success_labels: Optional[Union[int, List[Optional[int]]]] = None,  # for `Slaves`
            # Others
            seed: int = 42,
            val_ratio: float = 0.02,
            max_train_episodes: int = 90,
            transform_color_jitter: bool = True,
            # H5 config for master
            h5_path: str = None,
            use_h5: bool = False,
            enable_master: bool = True,
    ):
        super().__init__()
        slave_data_roots = slave_data_roots or []
        slave_h5_paths = slave_h5_paths or []
        if len(slave_data_roots) != len(slave_h5_paths):
            raise ValueError(
                f"slave_data_roots and slave_h5_paths must have the same length, "
                f"got {len(slave_data_roots)} vs {len(slave_h5_paths)}."
            )
        if use_h5 and h5_path is None:
            raise ValueError("`h5_path` for master dataset must be provided when use_h5=True.")
        if use_h5 and any(p is None for p in slave_h5_paths):
            raise ValueError("All `slave_h5_paths` must be provided when use_h5=True.")
        if isinstance(slave_default_success_labels, list):
            if len(slave_default_success_labels) != len(slave_data_roots):
                raise ValueError(
                    f"`slave_default_success_labels` list length must match slave datasets, "
                    f"got {len(slave_default_success_labels)} vs {len(slave_data_roots)}."
                )
            resolved_slave_default_success_labels = slave_default_success_labels
        else:
            resolved_slave_default_success_labels = [slave_default_success_labels for _ in slave_data_roots]

        self.master_dataset = TCLImageDataset(
            data_root=data_root,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            shape_meta=shape_meta,
            norm_force_type=norm_force_type,
            zero_force=zero_force,
            p_camera_drop=p_camera_drop,
            p_cond_drop=p_cond_drop,
            default_success_label=default_success_label,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            transform_color_jitter=transform_color_jitter,
            h5_path=h5_path,
            use_h5=use_h5,
        )
        self.slave_datasets: List[TCLImageDataset] = []
        for slave_root, slave_h5_path, slave_default_success_label in zip(
                slave_data_roots, slave_h5_paths, resolved_slave_default_success_labels):
            slave_ds = TCLImageDataset(
                data_root=slave_root,
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                shape_meta=shape_meta,
                norm_force_type=norm_force_type,
                zero_force=zero_force,
                p_camera_drop=p_camera_drop,
                p_cond_drop=p_cond_drop,
                default_success_label=slave_default_success_label,
                seed=seed,
                val_ratio=val_ratio,
                max_train_episodes=max_train_episodes,
                transform_color_jitter=transform_color_jitter,
                h5_path=slave_h5_path,
                use_h5=use_h5,
            )
            self.slave_datasets.append(slave_ds)

        self.enable_master = enable_master
        self.datasets = ([self.master_dataset] if self.enable_master else []) + self.slave_datasets
        self.dataset_is_master = ([True] if self.enable_master else []) + [False] * len(self.slave_datasets)
        if len(self.datasets) == 0:
            raise ValueError("No dataset selected: master is disabled and no slave dataset is provided.")
        self.concat_dataset = ConcatDataset(self.datasets)
        self._sync_master_stats_to_slaves()

        # Keep compatibility with TCLImageDataset style attrs
        self.data_root = data_root
        self.h5_path = h5_path
        self.use_h5 = use_h5
        self.slave_data_roots = slave_data_roots
        self.slave_h5_paths = slave_h5_paths

        self.shape_meta = shape_meta
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.norm_force_type = norm_force_type
        self.zero_force = zero_force
        self.p_camera_drop = p_camera_drop
        self.p_cond_drop = p_cond_drop
        self.default_success_label = default_success_label
        self.slave_default_success_labels = resolved_slave_default_success_labels
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.transform_color_jitter = transform_color_jitter

        print(
            f"[diffusion_policy.dataset.TCLMasterSlaveDataset] "
            f"loaded with {'1 master + ' if self.enable_master else ''}{len(self.slave_datasets)} slaves, "
            f"total_len={len(self.concat_dataset)}."
        )

    def _sync_master_stats_to_slaves(self):
        self.dataset_stats = self.master_dataset.dataset_stats
        self.dataset_total_len = self.master_dataset.dataset_total_len
        self.dataset_action_min = self.master_dataset.dataset_action_min
        self.dataset_action_max = self.master_dataset.dataset_action_max

        for slave_ds in self.slave_datasets:
            slave_ds.dataset_stats = self.dataset_stats
            slave_ds.dataset_action_min = self.dataset_action_min
            slave_ds.dataset_action_max = self.dataset_action_max

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.master_dataset = self.master_dataset.get_validation_dataset()
        val_set.slave_datasets = [ds.get_validation_dataset() for ds in self.slave_datasets]
        val_set.datasets = ([val_set.master_dataset] if val_set.enable_master else []) + val_set.slave_datasets
        val_set.dataset_is_master = ([True] if val_set.enable_master else []) + [False] * len(val_set.slave_datasets)
        val_set.concat_dataset = ConcatDataset(val_set.datasets)
        val_set._sync_master_stats_to_slaves()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        return self.master_dataset.get_normalizer(**kwargs)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        ds_idx = bisect.bisect_right(self.concat_dataset.cumulative_sizes, idx)
        start = 0 if ds_idx == 0 else self.concat_dataset.cumulative_sizes[ds_idx - 1]
        local_idx = idx - start

        item = self.datasets[ds_idx][local_idx]
        item["dataset_idx"] = torch.tensor(ds_idx, dtype=torch.long)
        item["local_idx"] = torch.tensor(local_idx, dtype=torch.long)
        item["is_master"] = torch.tensor(self.dataset_is_master[ds_idx], dtype=torch.bool)
        return item


if __name__ == "__main__":
    from robokit.debug_utils.printer import print_batch

    dataset = TCLImageDataset(
        data_root="/home/geyuan/datasets/TCL/collected_data_0425",
        horizon=16, pad_before=1, pad_after=7,
        shape_meta={
            "obs": {
                "image": {
                    "shape": [3, 96, 96],
                    "type": "rgb"
                },
                "joint_state": {
                    "shape": [6],
                    "type": "low_dim"
                }
            },
            "action": {
                "shape": [7,]
            }
        }
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for idx, batch in enumerate(dataloader):
        print_batch(f"tcl_dataset@{idx}", batch)

        print(batch['action'])
        exit()

        if idx >= 5:
            exit()
