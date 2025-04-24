import os
import copy
import numpy as np

from robokit.data.tcl_datasets import TCLDataset

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


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
                 ):
        super().__init__()
        # RoboKit Dataset
        self.data_root = data_root
        self.tcl_dataset = TCLDataset(data_root)
        self.data_meta = self.tcl_dataset.load_statistics_from_json(os.path.join(data_root, "statistics.json"))

        self.tasks = self.tcl_dataset.tasks
        self.task_lengths = self.tcl_dataset.task_lengths
        self.ep_fns = self.tcl_dataset.ep_fns
        self.map_index_to_task_id = self.tcl_dataset.map_index_to_task_id

        # Sampling a data sequence
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.task_prefix_lengths = [0 for _ in range(len(self.task_lengths))]
        for i in range(1, len(self.task_prefix_lengths)):
            self.task_prefix_lengths[i] = self.task_prefix_lengths[i - 1] + self.task_lengths[i - 1]

        # Data format and preprocessing
        self.obs_image_shape = shape_meta["obs"]["img"]["shape"]  # [3, 96, 96]
        self.action_shape = shape_meta["action"]["shape"]   # [7,]
        self.obs_image_transform = torch.  # TODO: need to be updated

    def get_validation_dataset(self):
        pass

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        pass

    def __len__(self):
        pass

    def __getitem__(self, abs_idx):
        pass

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

    def _get_abs_obs_indices(self, rel_now_idx: int, task_id: int):
        # [start_idx, end_idx)
        start_idx = rel_now_idx - self.pad_before
        end_idx = rel_now_idx + 1
        rel_indices = list(range(start_idx, end_idx))
        return [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]

    def _get_abs_act_indices(self, rel_now_idx: int, task_id: int):
        # [start_idx, end_idx)
        start_idx = rel_now_idx
        end_idx = rel_now_idx + self.horizon
        rel_indices = list(range(start_idx, end_idx))
        return [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]

    def _get_obs_data(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_obs_indices = self._get_abs_obs_indices(abs_idx, task_id)
        obs_data = {
            "primary_rgb": []
        }
        for idx in abs_obs_indices:
            if idx is None:
                zero_rgb = np.zeros(self.obs_image_shape).astype(np.uint8)
                primary_rgb = zero_rgb
            else:
                primary_rgb = self.tcl_dataset.__getitem__(idx)
            # Preprocess

            obs_data["primary_rgb"].append(primary_rgb)
        obs_data["primary_rgb"] = np.concatenate(obs_data["primary_rgb"])  # should be (T,C,H,W)
        return obs_data

    def _get_act_data(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_act_indices = self._get_abs_act_indices(abs_idx, task_id)
        act_data = []
        for idx in abs_act_indices:
            if idx is None:
                zero_act = np.zeros(self.action_shape).astype(np.float32)
                act_data.append(zero_act)



if __name__ == "__main__":
    dataset = TCLImageDataset(
        data_root="/home/geyuan/datasets/TCL/collected_data_0422",
    )
