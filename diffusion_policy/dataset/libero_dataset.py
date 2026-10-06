import os
import copy
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import default_collate
from torchvision.transforms import transforms
import torch.nn.functional as F

from robokit.datasets.libero.libero_force import LiberoH5FrameDataset, MergeDataset

from diffusion_policy.common.libero_task_util import LIBERO_TASK_NAMES, libero_task_id
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, EmptyNormalizer


class RandomShiftsAug(torch.nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()

        padding = tuple([self.pad] * 4)
        x_padded = F.pad(x, padding, "replicate")
        padded_h = h + 2 * self.pad
        padded_w = w + 2 * self.pad

        eps_h = 1.0 / padded_h
        eps_w = 1.0 / padded_w
        arange_h = torch.linspace(
            -1.0 + eps_h,
            1.0 - eps_h,
            padded_h,
            device=x.device,
            dtype=x.dtype,
        )[:h]
        arange_w = torch.linspace(
            -1.0 + eps_w,
            1.0 - eps_w,
            padded_w,
            device=x.device,
            dtype=x.dtype,
        )[:w]

        grid_h, grid_w = torch.meshgrid(arange_h, arange_w, indexing='ij')
        base_grid = torch.stack((grid_w, grid_h), dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift_xy = torch.randint(
            0,
            2 * self.pad + 1,
            size=(1, 1, 1, 2),
            device=x.device,
        ).to(dtype=x.dtype)

        shift_w = shift_xy[..., 0] * 2.0 / padded_w
        shift_h = shift_xy[..., 1] * 2.0 / padded_h
        shift = torch.stack((shift_w, shift_h), dim=-1)
        grid = base_grid + shift

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
        self.stochastic = resize_crop or color_jitter is not None or bool(random_shift_pad)
        self.random_shift_pad = random_shift_pad
        if random_shift_pad is not None:
            self.random_shift_aug = RandomShiftsAug(pad=random_shift_pad)
        else:
            self.random_shift_aug = lambda x: x
        if color_jitter is not None:
            tfs.append(color_jitter)
        tfs.append(transforms.ToTensor())           # -> (C,H,W), float in [0,1]
        self.frame_tf = transforms.Compose(tfs)

    def __call__(self, video_T_H_W_C: np.ndarray, consistent=True, seed=None):
        """Transform a clip without resetting the caller's global RNG state."""
        assert video_T_H_W_C.ndim == 4 and video_T_H_W_C.shape[-1] == 3
        if not self.stochastic:
            return torch.stack([self.frame_tf(frame_H_W_C) for frame_H_W_C in video_T_H_W_C])
        if seed is None:
            seed = int(torch.randint(0, 10_000_000, ()).item())
        frames = []
        # Keep augmentation randomness local; frames share the same transform.
        with torch.random.fork_rng(devices=[]):
            for frame_H_W_C in video_T_H_W_C:
                if consistent:
                    torch.random.default_generator.manual_seed(seed)
                frames.append(self.frame_tf(frame_H_W_C))
            video_T_C_H_W = torch.stack(frames)
            return self.random_shift_aug(video_T_C_H_W)


class LiberoFTDataset(BaseImageDataset):
    """Episode-disjoint LIBERO replay windows with checkpointed training statistics."""

    def __init__(
        self, dataset_root, dataset_subname, hdf5_fns, horizon, pad_before,
        pad_after, shape_meta, norm_force_type="quantile", seed=42,
        val_ratio=0.02, max_train_episodes=90, max_len=None,
        transform_color_jitter=True, load_future_obs=False, zero_force=False,
        transform_resize_crop=True, random_shift_pad=4, wrist_dropout_prob=0.3,
        dataset_metadata=None, split="train", split_mode="legacy", split_stage="a",
    ):
        super().__init__()
        shape_meta = (OmegaConf.to_container(shape_meta, resolve=True)
                      if OmegaConf.is_config(shape_meta) else copy.deepcopy(shape_meta))
        self.shape_meta = shape_meta
        self.use_task_id = "task_id" in shape_meta["obs"]
        if self.use_task_id:
            assert shape_meta["obs"]["task_id"] == {"shape": [10], "type": "low_dim"}
        self.split = split
        self.norm_force_type = norm_force_type
        self.zero_force = zero_force
        self.wrist_dropout_prob = wrist_dropout_prob if split == "train" else 0.0
        self.horizon, self.pad_before, self.pad_after = horizon, pad_before, pad_after
        self.n_obs_steps = pad_before + 1
        # Persistent workers observe phase changes without rebuilding the loader.
        self._temporal_pairing_scalar = torch.zeros((), dtype=torch.bool).share_memory_()
        self._temporal_ranges_E_2 = None
        self._init_kwargs = dict(
            dataset_root=dataset_root, dataset_subname=dataset_subname,
            hdf5_fns=list(hdf5_fns), horizon=horizon, pad_before=pad_before,
            pad_after=pad_after, shape_meta=shape_meta, norm_force_type=norm_force_type,
            seed=seed, val_ratio=val_ratio, max_train_episodes=max_train_episodes,
            split_mode=split_mode, split_stage=split_stage,
            max_len=max_len, load_future_obs=load_future_obs, zero_force=zero_force,
            transform_color_jitter=transform_color_jitter,
            transform_resize_crop=transform_resize_crop, random_shift_pad=random_shift_pad,
            wrist_dropout_prob=wrist_dropout_prob,
        )
        subnames = ([dataset_subname] * len(hdf5_fns)
                    if isinstance(dataset_subname, str) else list(dataset_subname))
        assert len(subnames) == len(hdf5_fns)
        assert split in ("train", "val")
        assert split_mode in ("legacy", "two_by_two")
        assert split_stage in ("a", "b")
        if split_mode == "two_by_two" and max_train_episodes is not None:
            raise ValueError("two_by_two requires max_train_episodes=null")
        self.hdf5_paths = []
        sources = []
        for subname, filename in zip(subnames, hdf5_fns):
            if filename.endswith("_failed.hdf5"):
                raise ValueError(f"Failed replay file is not a training source: {filename}")
            path = os.path.join(dataset_root, subname, filename)
            if not os.path.isfile(path):
                path = os.path.join(dataset_root, subname, "libero_90", filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            self.hdf5_paths.append(os.path.abspath(path))
            sources.append(os.path.relpath(path, dataset_root))
        assert len(set(sources)) == len(sources), "Duplicate HDF5 sources"
        contract = dict(shape_meta=shape_meta, horizon=horizon, n_obs_steps=pad_before + 1,
                        norm_force_type=norm_force_type, zero_force=zero_force,
                        image_scaling="uint8/127.5-1", state_scaling="mean_std",
                        action_scaling="minmax", constant_epsilon=1e-6)
        if self.use_task_id:
            contract["task_id_names"] = list(LIBERO_TASK_NAMES)
        metadata = (OmegaConf.to_container(dataset_metadata, resolve=True)
                    if OmegaConf.is_config(dataset_metadata) else copy.deepcopy(dataset_metadata))
        if metadata is None and split_mode == "two_by_two":
            if split_stage != "a":
                raise ValueError("Stage B requires a two_by_two Stage A checkpoint and its statistics")
            if not 0 < val_ratio < 1:
                raise ValueError("LIBERO requires 0 < val_ratio < 1")
            # A local RNG isolates episode selection from model/augmentation randomness.
            rng = np.random.RandomState(seed)
            partitions = {name: [] for name in ("a_train", "a_val", "b_train", "b_val")}
            for source, path in sorted(zip(sources, self.hdf5_paths)):
                reader = LiberoH5FrameDataset(path)
                demo_keys = sorted(reader.demo_keys)
                reader.close()
                if not demo_keys:
                    raise ValueError(f"LIBERO task requires at least one episode: {source}")
                order_N = rng.permutation(len(demo_keys))
                episodes = [[source, demo_keys[i]] for i in order_N]
                midpoint = len(episodes) // 2
                for half, subset in (("a", episodes[:midpoint]), ("b", episodes[midpoint:])):
                    n_val = (min(len(subset) - 1, max(1, round(len(subset) * val_ratio)))
                             if len(subset) >= 2 else 0)
                    partitions[half + "_val"].extend(subset[:n_val])
                    partitions[half + "_train"].extend(subset[n_val:])
            metadata = dict(version=2, sources=sources, contract=contract, seed=seed,
                            val_ratio=val_ratio, split_mode=split_mode, split_stage=split_stage,
                            partitions=partitions, train=partitions["a_train"], val=partitions["b_val"])
        elif metadata is None:
            episodes = []
            for source, path in zip(sources, self.hdf5_paths):
                reader = LiberoH5FrameDataset(path)
                episodes.extend((source, key) for key in reader.demo_keys)
            episodes.sort()
            if len(episodes) < 2 or not 0 < val_ratio < 1:
                raise ValueError("LIBERO train/val requires >=2 episodes and 0 < val_ratio < 1")
            order_N = np.random.RandomState(seed).permutation(len(episodes))
            n_val = min(len(episodes) - 1, max(1, round(len(episodes) * val_ratio)))
            val_episodes = [episodes[i] for i in order_N[:n_val]]
            train_episodes = [episodes[i] for i in order_N[n_val:]]
            if max_train_episodes is not None:
                assert max_train_episodes > 0
                train_episodes = train_episodes[:max_train_episodes]
            metadata = dict(version=1, sources=sources, contract=contract, seed=seed,
                            train=[list(item) for item in train_episodes],
                            val=[list(item) for item in val_episodes])
        elif metadata["sources"] != sources or metadata["contract"] != contract:
            raise ValueError("LIBERO data sources/observation contract differ from checkpoint")
        if metadata.get("split_mode", "legacy") != split_mode:
            raise ValueError("LIBERO split mode differs from checkpoint; retrain Stage A for two_by_two")
        if split_mode == "two_by_two":
            for key, value in (("seed", seed), ("val_ratio", val_ratio), ("split_stage", split_stage)):
                if metadata[key] != value:
                    raise ValueError(f"LIBERO {key} differs from checkpoint: {value!r} != {metadata[key]!r}")
            partitions = metadata["partitions"]
            all_episodes = [tuple(item) for subset in partitions.values() for item in subset]
            assert len(all_episodes) == len(set(all_episodes)), "Overlapping LIBERO partitions"
            assert metadata["train"] == partitions[split_stage + "_train"]
            assert metadata["val"] == partitions[("b" if split_stage == "a" else "a") + "_val"]
            if split_stage == "b" and "statistics" not in metadata:
                raise ValueError("Stage B must reuse Stage A normalization statistics")
            counts = {name: len(metadata[name]) for name in ("train", "val")}
            if not all(counts.values()):
                raise ValueError(f"Empty LIBERO Stage {split_stage.upper()} dataset: {counts}")
            for source in sorted(sources):
                sizes = {name: sum(origin == source for origin, _ in subset)
                         for name, subset in partitions.items()}
                print(f"[LiberoFTDataset] {source}: {sizes}")
        assert not (set(map(tuple, metadata["train"])) & set(map(tuple, metadata["val"])))
        readers = []
        reader_task_ids = []
        for source_index, (source, path) in enumerate(zip(sources, self.hdf5_paths)):
            demo_keys = [key for origin, key in metadata[split] if origin == source]
            if demo_keys:
                readers.append(LiberoH5FrameDataset(path, demo_keys=demo_keys))
                reader_task_ids.append(libero_task_id(
                    source, fallback=None if self.use_task_id else 10 + source_index))
        if not readers:
            raise ValueError(f"Empty LIBERO {split} split")
        obs_meta = shape_meta["obs"]
        obs_keys = ["agentview_rgb"]
        if "gripper" in obs_meta:
            obs_keys.append("eye_in_hand_rgb")
        stats_keys = []
        if "joint_state" in obs_meta:
            obs_keys.append("ee_states")
            stats_keys.append("ee_states")
        if "force" in obs_meta or "force_torque" in obs_meta:
            obs_keys.append("wrenches")
            stats_keys.append("wrenches")
        self.merge_dataset = MergeDataset(
            readers, n_obs=pad_before + 1, chunk_size=horizon, obs_keys=obs_keys,
            state_keys=[], max_len=max_len, load_future_obs=load_future_obs,
        )
        self._reader_task_ids_R = np.asarray(reader_task_ids, dtype=np.int64)
        if "statistics" not in metadata:
            assert split == "train", "Validation must reuse training statistics"
            metadata["statistics"] = self.merge_dataset.compute_statistics(
                keys={"obs": stats_keys, "state": [], "action": ["actions"]})
        # Statistics open HDF5 handles; workers should lazily open their own handles.
        for reader in readers:
            reader.close()
        self.dataset_metadata = metadata
        self.dataset_stats = metadata["statistics"]
        augment = split == "train"
        color_jitter = (transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
                        if augment and transform_color_jitter else None)
        self.image_transforms = {
            key: VideoTransformTHWC(
                image_size_hw=value["shape"][1:], color_jitter=color_jitter,
                resize_crop=augment and transform_resize_crop,
                random_shift_pad=random_shift_pad if augment and random_shift_pad else None,
            ) for key, value in obs_meta.items() if value["type"] == "rgb"
        }
        print(f"[LiberoFTDataset] split={split}, episodes={len(metadata[split])}, "
              f"windows={len(self)}, sources={sources}")

    def get_dataset_metadata(self):
        """Return the exact split, preprocessing contract and training statistics."""
        return copy.deepcopy(self.dataset_metadata)

    def get_validation_dataset(self):
        """Build held-out windows without fitting statistics or applying augmentation."""
        return self.create_val_dataset(self)

    @classmethod
    def create_val_dataset(cls, instance):
        """Reuse a training dataset's saved split and statistics."""
        return cls(**instance._init_kwargs, dataset_metadata=instance.dataset_metadata, split="val")

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Dataset and socket adapter own scaling; the policy normalizer is identity."""
        normalizer = LinearNormalizer()
        for key in (*self.shape_meta["obs"], "action"):
            normalizer[key] = EmptyNormalizer.create_identity()
        return normalizer

    def __len__(self):
        return len(self.merge_dataset)

    def set_temporal_pairing(self, enabled: bool) -> None:
        """Toggle episode-local pairing; prepare ranges before starting workers."""
        if enabled and self._temporal_ranges_E_2 is None:
            ranges = []
            for offset, reader in zip(self.merge_dataset.prefix, self.merge_dataset.datasets):
                for start, length in zip(reader.prefix, reader.demo_lengths):
                    start = offset + start
                    end = min(start + length, len(self))
                    if start < end:
                        ranges.append((start, end))
            # File-local demo ranges remain distinct; max_len limits both A and B.
            self._temporal_ranges_E_2 = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
        self._temporal_pairing_scalar.fill_(enabled)

    def __getitem__(self, idx):
        data = self._get_single_window(idx)
        if bool(self._temporal_pairing_scalar):
            episode = int(np.searchsorted(self._temporal_ranges_E_2[:, 1], idx, side="right"))
            start, end = map(int, self._temporal_ranges_E_2[episode])
            pair_valid = end - start > 1
            if pair_valid:
                # Uniformly skip A without rejection sampling or resetting worker RNG.
                pair_idx = start + np.random.randint(end - start - 1)
                pair_idx += pair_idx >= idx
                pair_data = self._get_single_window(pair_idx)
            else:
                # Reuse augmented A as a placeholder, before adding nested pair fields.
                pair_data = dict(data)
            data["temporal_pair"] = pair_data
            data["temporal_pair_valid"] = torch.tensor(pair_valid, dtype=torch.bool)
        return data

    def _get_single_window(self, idx):
        data = self._sample_to_data(self.merge_dataset[idx])
        reader = int(np.searchsorted(self.merge_dataset.prefix, idx, side="right")) - 1
        task_id = int(self._reader_task_ids_R[reader])
        data["task_id"] = torch.tensor(task_id, dtype=torch.long)
        if self.use_task_id:
            task_T_C = torch.zeros((self.n_obs_steps, 10), dtype=torch.float32)
            task_T_C[:, task_id] = 1.0
            data["obs"]["task_id"] = task_T_C
        return data

    def iter_task_diagnostic_batches(self, samples_per_task: int, seed: int):
        """Yield fixed task-local batches without Temporal pairing or RNG side effects."""
        if samples_per_task < 2:
            raise ValueError("topology_samples_per_task must be at least 2")
        # Merge sources sharing an official task identity before sampling.
        ranges_by_task = {}
        for start, reader, task_id in zip(
            self.merge_dataset.prefix, self.merge_dataset.datasets, self._reader_task_ids_R
        ):
            end = min(start + len(reader), len(self))
            if start < end:
                ranges_by_task.setdefault(int(task_id), []).append((start, end))
        for task_id, ranges in sorted(ranges_by_task.items()):
            lengths_R = np.asarray([end - start for start, end in ranges], dtype=np.int64)
            ends_R = lengths_R.cumsum()
            rng = np.random.RandomState(seed + task_id)
            local_indices_B = rng.choice(int(ends_R[-1]), min(samples_per_task, int(ends_R[-1])), replace=False)
            readers_B = np.searchsorted(ends_R, local_indices_B, side="right")
            offsets_R = np.asarray([start for start, _ in ranges]) - (ends_R - lengths_R)
            indices_B = local_indices_B + offsets_R[readers_B]
            # Restore CPU augmentation RNG before yielding control to the workspace.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + task_id)
                batch = default_collate([self._get_single_window(int(idx)) for idx in indices_B])
            yield task_id, batch

    def _sample_to_data(self, sample):
        history = sample["history"]["obs"]
        obs = {}
        for key, source in (("image", "agentview_rgb"), ("gripper", "eye_in_hand_rgb")):
            if key in self.shape_meta["obs"]:
                image_T_C_H_W = self.image_transforms[key](history[source]) * 2.0 - 1.0
                if key == "gripper" and self.wrist_dropout_prob > 0:
                    if torch.rand(()).item() < self.wrist_dropout_prob:
                        image_T_C_H_W = torch.zeros_like(image_T_C_H_W)
                obs[key] = image_T_C_H_W
        for key, source in (("joint_state", "ee_states"), ("force", "wrenches"),
                            ("force_torque", "wrenches")):
            if key in self.shape_meta["obs"]:
                obs[key] = self.norm_state_or_force(
                    history[source], "mean" if key == "joint_state" else self.norm_force_type,
                    self.dataset_stats[f"obs.{source}"],
                ).astype(np.float32)
                if self.zero_force and key in ("force", "force_torque"):
                    obs[key] *= 0
        action_T_D = self.normalize_action(sample["action"], self.dataset_stats["action.actions"])
        return {"obs": obs, "action": action_T_D.astype(np.float32)}

    @staticmethod
    def normalize_action(action_T_D, statistics):
        """Map raw actions to [-1,1], placing constant dimensions at zero."""
        min_D = np.asarray(statistics["min"])
        range_D = np.asarray(statistics["max"]) - min_D
        constant_D = range_D < 1e-6
        action_T_D = 2 * (action_T_D - min_D) / np.where(constant_D, 1.0, range_D) - 1
        return np.where(constant_D, 0.0, action_T_D)

    @staticmethod
    def unnormalize_action(action_B_K_D, statistics):
        """Restore raw control actions, including constant training dimensions."""
        min_D = np.asarray(statistics["min"])
        range_D = np.asarray(statistics["max"]) - min_D
        action_B_K_D = (action_B_K_D + 1) * range_D / 2 + min_D
        return np.where(range_D < 1e-6, min_D, action_B_K_D).astype(np.float32)

    @staticmethod
    def norm_state_or_force(in_data, norm_type, meta_data):
        """Scale low-dimensional observations with training-only statistics."""
        state_T_D = in_data
        dim = state_T_D.shape[-1]
        if norm_type == "identity":
            return state_T_D
        if norm_type == "mean":
            mean_D = np.asarray(meta_data["mean"])[:dim]
            std_D = np.asarray(meta_data["std"])[:dim]
            return (state_T_D - mean_D) / np.maximum(std_D, 1e-6)
        if norm_type == "minmax":
            low_D, high_D = np.asarray(meta_data["min"])[:dim], np.asarray(meta_data["max"])[:dim]
        else:
            assert norm_type == "quantile"
            low_D, high_D = np.asarray(meta_data["p01"])[:dim], np.asarray(meta_data["p99"])[:dim]
            state_T_D = np.clip(state_T_D, low_D, high_D)
        return 2 * (state_T_D - low_D) / np.maximum(high_D - low_D, 1e-6) - 1
