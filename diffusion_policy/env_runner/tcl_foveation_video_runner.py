"""Offline validation trajectories for TCL foveation training."""

from pathlib import Path

from diffusion_policy.common.foveation_video_util import (
    make_foveation_video_log,
    observation_rgb_to_uint8,
    render_foveation_episode_video,
)
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class TCLFoveationVideoRunner(BaseImageRunner):
    """Render validation episodes through the shared recorded-observation renderer."""

    def __init__(self, output_dir, dataset, max_episodes=None, fps=20, crf=22,
                 foveation_vis_batch_size=16, seed=42, save_foveation_video=True,
                 episodes_per_task=2, log_wandb=True):
        """Reuse an existing validation dataset and configure full-episode encoding."""
        super().__init__(output_dir)
        assert dataset.split == "val"
        assert max_episodes is None or max_episodes > 0
        self.dataset = dataset
        self.image_keys = [key for key, meta in dataset.shape_meta["obs"].items()
                           if meta.get("type") == "rgb"]
        self.max_episodes = max_episodes
        self.fps, self.crf = fps, crf
        self.batch_size = foveation_vis_batch_size
        self.seed = seed
        self.save_foveation_video = save_foveation_video
        assert episodes_per_task > 0
        self.episodes_per_task = episodes_per_task
        self.log_wandb = log_wandb
        self.epoch = self.global_step = 0

    def set_context(self, dataset_metadata, epoch, global_step):
        """Set the checkpoint context used to name this evaluation's artifacts."""
        self.epoch, self.global_step = epoch, global_step

    def run(self, policy):
        """Write full val trajectories and return videos plus frame-weighted patch counts."""
        if not self.save_foveation_video:
            return {}
        if self.dataset._multi_source:
            return self._run_recordings(policy)
        media_dir = Path(self.output_dir) / "media" / "val_foveation" / f"epoch_{self.epoch:04d}"
        media_dir.mkdir(parents=True, exist_ok=True)
        log = {}
        logged_tasks = set()
        totals = {key: dict(sum=0, min=float("inf"), max=0) for key in self.image_keys}
        episode_count = frame_count = 0
        for ordinal, (episode_id, name, length) in enumerate(self.dataset.iter_episodes()):
            if self.max_episodes is not None and ordinal >= self.max_episodes:
                break
            # Stable source ID disambiguates names; source directories are not task labels.
            stem = f"episode_{episode_id:04d}_{Path(name).name}"
            raw_path = media_dir / f"{stem}.mp4"
            tree_path = media_dir / f"{stem}_foveation.mp4"

            def get_batch(indices_B):
                obs = self.dataset.get_episode_observation_batch(episode_id, indices_B)
                raw_by_key = {key: observation_rgb_to_uint8(obs[key], normalized=True)
                              for key in self.image_keys}
                return obs, raw_by_key

            counts_by_key = render_foveation_episode_video(
                policy, self.image_keys, length, get_batch, raw_path, tree_path,
                batch_size=self.batch_size, fps=self.fps, crf=self.crf,
                seed=self.seed + episode_id)
            for key, counts_F in counts_by_key.items():
                totals[key]["sum"] += int(counts_F.sum())
                totals[key]["min"] = min(totals[key]["min"], int(counts_F.min()))
                totals[key]["max"] = max(totals[key]["max"], int(counts_F.max()))
            task_name = self.dataset.get_episode_task_name(episode_id)
            # Keep all local videos and statistics, but upload only the first episode per task.
            if self.log_wandb and task_name not in logged_tasks:
                caption = (f"val task={task_name}, episode={name}, "
                           f"frames={length}, epoch={self.epoch}")
                log.update(make_foveation_video_log(
                    raw_path, tree_path,
                    raw_key=f"vis/val_foveation_{task_name}/raw",
                    tree_key=f"vis/val_foveation_{task_name}/tree", caption=caption,
                    tree_caption="recorded-observation routing; " + caption))
                logged_tasks.add(task_name)
            episode_count += 1
            frame_count += length
        log.update({"val/video_episodes": episode_count, "val/video_frames": frame_count})
        for key, stats in totals.items():
            log[f"val/foveation/{key}/num_patches_mean"] = stats["sum"] / frame_count
            log[f"val/foveation/{key}/num_patches_min"] = stats["min"]
            log[f"val/foveation/{key}/num_patches_max"] = stats["max"]
        return log

    def _run_recordings(self, policy):
        # Full recordings are for visualization, not held-out validation episodes.
        media_dir = (Path(self.output_dir) / "media" / "tcl_recordings"
                     / f"epoch_{self.epoch:04d}_step_{self.global_step:08d}")
        media_dir.mkdir(parents=True, exist_ok=True)
        image_keys = ["image", "gripper"]
        log = {}
        for record_id, task_name, ordinal, length in self.dataset.iter_recorded_episodes(self.episodes_per_task):
            record = self.dataset.episode_records[record_id]
            source = self.dataset.sources[record["source_id"]]
            stem = f"{task_name}_record_{record_id:06d}"
            raw_path, tree_path = media_dir / f"{stem}.mp4", media_dir / f"{stem}_tree.mp4"

            def get_batch(indices_B):
                obs = self.dataset.get_recorded_observation_batch(record_id, indices_B)
                raw_by_key = {
                    key: observation_rgb_to_uint8(
                        obs[key], normalized=self.dataset.output_mode != "raw")
                    for key in image_keys}
                return obs, raw_by_key

            render_foveation_episode_video(
                policy, image_keys, length, get_batch, raw_path, tree_path,
                batch_size=self.batch_size, fps=self.fps, crf=self.crf,
                seed=self.seed + record_id,
                raw_rotate_keys=(),
                tree_rotate_keys=("image",) if policy.render_image_orientation == "rot180" else ())
            if self.log_wandb:
                caption = (f"offline recording; may contain training frames; task={task_name}, "
                           f"source={source['root']}, episode={record['episode_id']}, "
                           f"epoch={self.epoch}, step={self.global_step}")
                log.update(make_foveation_video_log(
                    raw_path, tree_path, raw_key=f"vis/{task_name}/raw_{ordinal}",
                    tree_key=f"vis/{task_name}/tree_{ordinal}", caption=caption))
        return log
