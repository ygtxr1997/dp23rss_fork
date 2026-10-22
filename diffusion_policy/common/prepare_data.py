"""Prepare immutable train/validation assets for TCL or LIBERO data."""

import copy
import hashlib
import json
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.tcl_data_util import action_bounds, quantile_stats
from diffusion_policy.common.libero_task_util import libero_task_language
from diffusion_policy.dataset.libero_dataset import LiberoFTDataset
from diffusion_policy.dataset.tcl_dataset import TCLFoveationDataset
from robokit.datasets.tcl_datasets import TCLDatasetHDF5


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _statistics(samples):
    """Compute stable float64 statistics for low-dimensional replay arrays."""
    values_N_D = np.concatenate(samples, axis=0).astype(np.float64)
    return dict(count=len(values_N_D), mean=values_N_D.mean(axis=0).tolist(),
                std=values_N_D.std(axis=0).tolist(),
                q01=np.quantile(values_N_D, 0.01, axis=0).tolist(),
                q99=np.quantile(values_N_D, 0.99, axis=0).tolist())


def _merge_full_statistics(summary, values_N_D, dim):
    """Merge a frame block using float64 parallel variance (population std)."""
    values_N_D = np.asarray(values_N_D, dtype=np.float64)
    if values_N_D.ndim != 2 or values_N_D.shape[1] != dim or not np.isfinite(values_N_D).all():
        raise ValueError(f'Expected finite statistics block [N, {dim}]')
    count = len(values_N_D)
    mean_D = values_N_D.mean(axis=0)
    m2_D = ((values_N_D - mean_D) ** 2).sum(axis=0)
    min_D, max_D = values_N_D.min(axis=0), values_N_D.max(axis=0)
    if summary is None:
        return dict(count=count, mean=mean_D, m2=m2_D, min=min_D, max=max_D)
    total = summary['count'] + count
    delta_D = mean_D - summary['mean']
    return dict(count=total, mean=summary['mean'] + delta_D * count / total,
                m2=summary['m2'] + m2_D + delta_D ** 2 * summary['count'] * count / total,
                min=np.minimum(summary['min'], min_D), max=np.maximum(summary['max'], max_D))


def _prepare_tcl(cfg: DictConfig):
    """Prepare TCL frame split and raw state/action/optional force quantiles."""
    output = Path(cfg.asset_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Data assets already exist; use a new asset_dir: {output}")
    roots = [str(Path(value).expanduser().resolve()) for value in cfg.data_roots]
    h5_paths = [str(Path(value).expanduser().resolve()) for value in cfg.h5_paths]
    if len(roots) != len(h5_paths) or not roots:
        raise ValueError("data_roots and h5_paths must be nonempty and same length")
    state_blocks, action_blocks, source_metadata = [], [], []
    full_statistics = {}
    force_blocks = []
    prepare_force_stats = bool(cfg.prepare_force_stats)
    load_keys = ["rel_actions", "robot_obs"]
    if prepare_force_stats:
        load_keys.append("force_torque")
    for source_id, (root, h5_path) in enumerate(zip(roots, h5_paths)):
        dataset = TCLDatasetHDF5(root, h5_path, use_extracted=True,
                                 load_keys=load_keys)
        # Extrema/moments cover every frame; quantiles below retain their old sample.
        action_data = dataset.extracted_data["rel_actions"]
        if len(action_data) != len(dataset):
            raise ValueError(f"Extracted actions do not cover every frame in {h5_path}")
        dimensions = dict(state=int(cfg.state_dim), actions=7)
        if prepare_force_stats:
            dimensions['force_torque'] = 6
        for start in range(0, len(dataset), 8192):
            end = min(start + 8192, len(dataset))
            batch = dataset.get_item_by_indices(list(range(start, end)))
            blocks = dict(state=np.asarray(batch['robot_obs'])[:, :int(cfg.state_dim)],
                          actions=np.asarray(action_data[start:end]))
            if prepare_force_stats:
                blocks['force_torque'] = np.asarray(batch['force_torque'])
            for key, values_N_D in blocks.items():
                full_statistics[key] = _merge_full_statistics(full_statistics.get(key), values_N_D, dimensions[key])
        total = len(dataset)
        count = min(int(cfg.stats_max_frames_per_task), total)
        indices = np.random.RandomState(int(cfg.stats_seed) + source_id).choice(
            total, count, replace=False)
        batch = dataset.get_item_by_indices(indices.tolist())
        state_blocks.append(np.asarray(batch["robot_obs"], dtype=np.float32)[:, :int(cfg.state_dim)])
        action_blocks.append(np.asarray(batch["rel_actions"], dtype=np.float32))
        if prepare_force_stats:
            if "force_torque" not in batch:
                raise ValueError(f"TCL force statistics require force_torque in {h5_path}")
            # Use the same sampled frames as state/actions; preserve raw units.
            force_torque_N_6 = np.asarray(batch["force_torque"], dtype=np.float32)
            if force_torque_N_6.shape != (count, 6) or not np.isfinite(force_torque_N_6).all():
                raise ValueError(f"Expected finite force_torque [N, 6] in {h5_path}")
            force_blocks.append(force_torque_N_6)
        source_metadata.append(dict(root=root, h5_path=h5_path,
                                   task_dir=Path(root).name, total_len=int(total)))
        # RoboKit exposes the HDF5 handle, not a dataset-level close() method.
        handle = getattr(dataset, "hf", None)
        if handle is not None and handle.id.valid:
            handle.close()
    sampled = dict(state=state_blocks, actions=action_blocks)
    if prepare_force_stats:
        sampled['force_torque'] = force_blocks
    statistics = {}
    for key, blocks in sampled.items():
        values_N_D = np.concatenate(blocks, axis=0)
        summary = full_statistics[key]
        statistics[key] = dict(
            min=summary['min'].tolist(), max=summary['max'].tolist(),
            mean=summary['mean'].tolist(),
            std=np.sqrt(summary['m2'] / summary['count']).tolist(),
            q01=np.quantile(values_N_D, .01, axis=0).tolist(),
            q99=np.quantile(values_N_D, .99, axis=0).tolist(),
            count=summary['count'], quantile_count=len(values_N_D))
    dataset = TCLFoveationDataset(
        data_roots=roots, h5_paths=h5_paths, output_mode="raw",
        task_mapping_path=str(Path(cfg.task_mapping_path).expanduser()),
        state_dim=int(cfg.state_dim), horizon=int(cfg.horizon),
        val_ratio=float(cfg.val_ratio), seed=int(cfg.seed),
        stats_seed=int(cfg.stats_seed),
        stats_max_frames_per_task=int(cfg.stats_max_frames_per_task),
        use_h5=bool(cfg.use_h5), split="train")
    metadata = dataset.get_dataset_metadata()
    dataset.close()
    metadata["sources"] = [dict(item, task_id=metadata["task_mapping"][item["task_dir"]])
                            for item in source_metadata]
    metadata["statistics"] = statistics
    metadata["statistics_scope"] = dict(
        min_max_mean_std="all configured sources, all frames; population std (ddof=0)",
        quantiles="sampled frames per source without replacement",
        stats_seed=int(cfg.stats_seed), source_seed="stats_seed + source_id",
        stats_max_frames_per_task=int(cfg.stats_max_frames_per_task))
    action_bounds(metadata)
    quantile_stats(statistics, int(cfg.state_dim))
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[data] prepared TCL assets: {output}")


def _prepare_libero(cfg: DictConfig):
    """Prepare LIBERO replay split and Pi0.5 raw statistics."""
    root = Path(cfg.dataset_root).expanduser().resolve()
    output = Path(cfg.asset_dir).expanduser().resolve()
    if any((output / name).exists() for name in ("norm_stats.json", "dataset_metadata.json")):
        raise FileExistsError(f"Data assets already exist; use a new asset_dir: {output}")
    episodes, sources, source_info = [], [], {}
    image_shape = None
    state_samples, action_samples = [], []
    for filename in cfg.hdf5_fns:
        if filename.endswith("_failed.hdf5"):
            raise ValueError(f"Failed replay file is not a training source: {filename}")
        path = root / cfg.dataset_subname / filename
        source = path.relative_to(root).as_posix()
        sources.append(source)
        with h5py.File(path, "r") as h5:
            data = h5["data"]
            task_name = Path(_text(data.attrs["bddl_file_name"])).stem
            lengths = {}
            for demo_key in sorted(data.keys()):
                demo = data[demo_key]
                length = len(demo["actions"])
                if length == 0:
                    raise ValueError(f"Empty episode: {source}/{demo_key}")
                for key in ("obs/agentview_rgb", "obs/eye_in_hand_rgb"):
                    frames = demo[key]
                    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != length:
                        raise ValueError(f"{source}/{demo_key}/{key}: expected uint8 [T,H,W,3]")
                    if image_shape is None:
                        image_shape = list(frames.shape[1:])
                    if list(frames.shape[1:]) != image_shape:
                        raise ValueError(f"Inconsistent camera shape: {source}/{demo_key}/{key}")
                lengths[demo_key] = length
                episodes.append([source, demo_key])
            source_info[source] = dict(task_name=task_name,
                                       prompt=libero_task_language(task_name),
                                       episode_lengths=lengths)
    if len(set(sources)) != len(sources) or len(episodes) < 2:
        raise ValueError("LIBERO sources must be unique and contain at least two episodes")
    episodes.sort()
    order_N = np.random.RandomState(int(cfg.seed)).permutation(len(episodes))
    n_val = min(len(episodes) - 1, max(1, round(len(episodes) * float(cfg.val_ratio))))
    train_episodes = [episodes[i] for i in order_N[n_val:]]
    val_episodes = [episodes[i] for i in order_N[:n_val]]
    for source, demo_key in train_episodes:
        with h5py.File(root / source, "r") as h5:
            demo = h5[f"data/{demo_key}"]
            state_samples.append(np.concatenate((demo["obs/ee_states"][...],
                                                  demo["obs/gripper_states"][...]), axis=-1))
            action_samples.append(np.asarray(demo["actions"], dtype=np.float32))
    metadata = dict(version=1, sources=sources, source_info=source_info,
                    contract=dict(action_horizon=10, state_dim=8, action_dim=7),
                    image_shape=image_shape, seed=int(cfg.seed),
                    val_ratio=float(cfg.val_ratio),
                    train=train_episodes, val=val_episodes,
                    statistics={"state": _statistics(state_samples),
                                "actions": _statistics(action_samples)})
    stats_text = json.dumps({"norm_stats": metadata["statistics"]}, indent=2) + "\n"
    metadata["norm_stats_sha256"] = hashlib.sha256(stats_text.encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "norm_stats.json").write_text(stats_text)
    (output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if int(cfg.batch_size) > 0:
        dataset = LiberoFTDataset(
            str(root), cfg.dataset_subname, list(cfg.hdf5_fns), horizon=10,
            pad_before=0, pad_after=0, shape_meta={"obs": {}, "action": {"shape": [7]}},
            output_mode="raw", asset_dir=str(output), split="train")
        validation = dataset.get_validation_dataset()
        for split, subset in (("train", dataset), ("val", validation)):
            batch = next(iter(DataLoader(subset, batch_size=int(cfg.batch_size),
                                         num_workers=int(cfg.num_workers))))
            print(f"[data] LIBERO {split}: action={tuple(batch['action'].shape)}")
            subset.close()
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    print(f"[data] prepared LIBERO assets: {output}")


@hydra.main(version_base=None, config_path="../../configs", config_name="pi05_libero_data")
def main(cfg: DictConfig):
    """Prepare one domain's immutable assets before training starts."""
    if "data_roots" in cfg:
        _prepare_tcl(cfg)
    else:
        _prepare_libero(cfg)


if __name__ == "__main__":
    main()
