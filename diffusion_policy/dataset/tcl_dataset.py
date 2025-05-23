import os
import copy
import numpy as np
import torch
from torchvision.transforms import transforms

from robokit.data.tcl_datasets import TCLDataset, TCLDatasetHDF5

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, EmptyNormalizer


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
                 # Others
                 seed: int = 42,
                 val_ratio: float = 0.02,
                 max_train_episodes: int = 90,
                 transform_color_jitter: bool = True,
                 # RoboKit Dataset
                 h5_path: str = None,
                 use_h5: bool = False,
                 ):
        super().__init__()
        # RoboKit Dataset
        self.data_root = data_root
        self.shape_meta = shape_meta
        self.load_keys = ["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text"]
        self.h5_path = h5_path
        self.use_h5 = use_h5
        if not use_h5:
            self.tcl_dataset = TCLDataset(data_root, use_extracted=True, load_keys=self.load_keys)
        else:
            self.tcl_dataset = TCLDatasetHDF5(
                data_root, h5_path,
                use_extracted=True, load_keys=self.load_keys)
        self.data_meta = self.tcl_dataset.load_statistics_from_json(os.path.join(data_root, "statistics.json"))
        self.all_rel_actions = self.tcl_dataset.extracted_data["rel_actions"]
        self.dataset_min = np.array(self.data_meta["min"])
        self.dataset_max = np.array(self.data_meta["max"])

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
        self.obs_image_transform = transforms.Compose(transform_list)  # Similar augmentation params with OCTO

        print(f"[TCLImageDataset] dataset loaded, "
              f"action_min={self.dataset_min}, action_max={self.dataset_max}")

    def get_validation_dataset(self):
        return self.create_val_dataset(self)

    @classmethod
    def create_val_dataset(cls, instance: 'TCLImageDataset'):
        val_set = cls(
            data_root=instance.data_root,
            horizon=instance.horizon,
            pad_before=instance.pad_before,
            pad_after=instance.pad_after,
            shape_meta=instance.shape_meta,
            seed=instance.seed,
            val_ratio=instance.val_ratio,
            max_train_episodes=instance.max_train_episodes,
            use_h5=False,  # no need to use h5
        )
        val_set.tcl_dataset.total_length = 64
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
        return [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]

    def _get_obs_data(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_obs_indices = self._get_abs_obs_indices(abs_idx)
        # print(f"[DEBUG] _get_obs_data: abs_idx={abs_idx}, task_id={task_id}, "
        #       f"task_1st_ep={self.task_prefix_lengths[task_id]}, "
        #       f"task_last_ep={self.task_prefix_lengths[task_id] + self.task_lengths[task_id]}"
        #       )
        # print(f"[DEBUG] _get_obs_data: abs_obs_indices={abs_obs_indices} ")
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
            obs_data["image"].append(primary_rgb)
            obs_data["joint_state"].append(tcp_pose)
            if "gripper" in obs_keys:
                obs_data["gripper"].append(gripper_rgb)
        obs_data = {k: torch.stack(v) for k, v in obs_data.items()}
        # obs_data["image"] = torch.stack(obs_data["image"])  # should be (T,C,H,W)
        # obs_data["joint_state"] = torch.stack(obs_data["joint_state"])  # (T,6)
        return obs_data

    def _get_act_data(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_act_indices = self._get_abs_act_indices(abs_idx)
        # print(f"[DEBUG] _get_act_data: abs_idx={abs_idx}, task_id={task_id}, "
        #       f"task_1st_ep={self.task_prefix_lengths[task_id]}, "
        #       f"task_last_ep={self.task_prefix_lengths[task_id] + self.task_lengths[task_id]}"
        #       )
        # print(f"[DEBUG] _get_obs_data: abs_obs_indices={abs_act_indices} ")
        act_data = []
        for idx in abs_act_indices:
            if idx is None:
                zero_act = np.zeros(self.action_shape).astype(np.float32)
                act_data.append(zero_act)
            else:
                rel_action = self.all_rel_actions[idx]  # (7,)
                act_data.append(rel_action)
        act_data = np.stack(act_data)  # (T,7), in [act_min, act_max]
        act_data = (act_data - self.dataset_min) / (self.dataset_max - self.dataset_min)  # norm here, in [0,1]
        return act_data * 2. - 1.  # in [-1,1]


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
