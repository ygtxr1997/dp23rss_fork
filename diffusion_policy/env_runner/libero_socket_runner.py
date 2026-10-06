"""Training-side HTTP/process helper; all simulator rollout lives in LIBERO-FT."""

import copy
import ctypes
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import random
import shutil
import signal
import socket
import subprocess
import threading
import time
from contextlib import nullcontext

import cv2
import wandb
import numpy as np
import torch
from tqdm import tqdm
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
import uvicorn

from diffusion_policy.common.foveation_video_util import render_foveation_episode_video
from diffusion_policy.common.libero_task_util import LIBERO_TASK_NAMES, libero_task_id
from diffusion_policy.dataset.libero_dataset import LiberoFTDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from robokit.connects.protocols import StepRequestEvent, StepRequestFromEvaluator, StepRequestFromPolicy


class LiberoSocketRunner(BaseImageRunner):
    """Serve the current EMA and manage an external evaluator, never a local simulator."""

    def __init__(
        self, output_dir, libero_root, dataset_root, dataset_subname, hdf5_fns,
        shape_meta, n_obs_steps, n_action_steps, policy_config,
        conda_env="libero", conda_executable=None, host="127.0.0.1", port=0,
        n_test=100, max_steps=400, seed=42, test_start_seed=130000,
        image_format="PNG", startup_timeout=60, request_timeout=120,
        eval_timeout=7200, shutdown_timeout=30, n_envs=1,
        egl_device_id=None, render_gpu_device_id=None,
        log_patch_counts=False, save_foveation_video=False, n_test_vis=4,
        fps=20, crf=22, foveation_vis_batch_size=16, vis_rot_image=False,
        egl_device_ids=None, render_gpu_device_ids=None, auto_render_device=True,
    ):
        super().__init__(output_dir)
        self.libero_root = str(Path(libero_root).expanduser().resolve())
        subnames = ([dataset_subname] * len(hdf5_fns)
                    if isinstance(dataset_subname, str) else list(dataset_subname))
        assert len(hdf5_fns) > 0 and len(subnames) == len(hdf5_fns)
        self.tasks = [dict(
            name=Path(filename).stem.removesuffix("_demo_wrench"),
            hdf5_path=str((Path(dataset_root).expanduser() / subname / filename).resolve()),
            suite_name=Path(subname).name,
        ) for subname, filename in zip(subnames, hdf5_fns)]
        assert len({task["name"] for task in self.tasks}) == len(self.tasks), "Duplicate task names"
        self.shape_meta = OmegaConf.to_container(shape_meta, resolve=True) if OmegaConf.is_config(shape_meta) else shape_meta
        self.policy_config = OmegaConf.to_container(policy_config, resolve=True) if OmegaConf.is_config(policy_config) else policy_config
        obs_meta = self.shape_meta["obs"]
        self.use_task_id = "task_id" in obs_meta
        assert set(obs_meta) - {"task_id"} in ({"image", "joint_state"}, {"image", "gripper", "joint_state"})
        if self.use_task_id:
            assert obs_meta["task_id"] == {"shape": [10], "type": "low_dim"}
            for task in self.tasks:
                task["task_id"] = libero_task_id(task["name"])
        assert list(obs_meta["joint_state"]["shape"]) == [6]
        assert list(self.shape_meta["action"]["shape"]) == [7]
        self.image_keys = [key for key in ("image", "gripper") if key in obs_meta]
        camera_map = {"image": "agentview_image", "gripper": "robot0_eye_in_hand_image"}
        image_shape = list(obs_meta["image"]["shape"])
        assert len(image_shape) == 3 and image_shape[0] == 3
        assert all(type(size) is int and size > 0 for size in image_shape[1:])
        assert all(list(obs_meta[key]["shape"]) == image_shape for key in self.image_keys)
        # Render at the configured policy input size, overriding HDF5 camera metadata.
        self.spec = dict(camera_names=[camera_map[key] for key in self.image_keys],
                         camera_hw=image_shape[1:], n_obs_steps=int(n_obs_steps),
                         state_dim=6, action_dim=7, max_cache_action=int(n_action_steps))
        self.conda_env, self.conda_executable = conda_env, conda_executable
        self.host, self.port = host, int(port)
        self.n_test, self.max_steps = int(n_test), int(max_steps)
        self.n_envs = int(n_envs)
        self.egl_device_id = None if egl_device_id is None else int(egl_device_id)
        self.render_gpu_device_id = None if render_gpu_device_id is None else int(render_gpu_device_id)
        self.egl_device_ids = egl_device_ids
        self.render_gpu_device_ids = render_gpu_device_ids
        self.auto_render_device = bool(auto_render_device)
        self.rank = 0
        self.seed, self.test_start_seed = int(seed), int(test_start_seed)
        self.image_format = image_format
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self.eval_timeout, self.shutdown_timeout = eval_timeout, shutdown_timeout
        assert self.host == "127.0.0.1", "Training evaluator runs on the rank-0 host"
        assert self.n_test > 0 and self.max_steps > 0
        assert self.n_envs >= 1
        assert self.egl_device_id is None or self.egl_device_id >= 0
        assert self.render_gpu_device_id is None or self.render_gpu_device_id >= -1
        assert self.spec["n_obs_steps"] > 0 and self.spec["max_cache_action"] > 0
        assert min(startup_timeout, request_timeout, eval_timeout, shutdown_timeout) > 0
        self.log_patch_counts = bool(log_patch_counts)
        self.save_foveation_video = bool(save_foveation_video)
        self.n_test_vis, self.fps, self.crf = int(n_test_vis), int(fps), int(crf)
        self.foveation_vis_batch_size = int(foveation_vis_batch_size)
        self.vis_rot_image = bool(vis_rot_image)
        assert self.n_test_vis >= 0 and self.fps > 0 and self.foveation_vis_batch_size > 0
        self.dataset_metadata = None

    def set_context(self, dataset_metadata, epoch, global_step):
        """Bind checkpointed preprocessing and identify the current EMA evaluation."""
        if self.use_task_id and dataset_metadata["contract"].get("task_id_names") != list(LIBERO_TASK_NAMES):
            raise ValueError("LIBERO task identity mapping differs from checkpoint")
        self.dataset_metadata = copy.deepcopy(dataset_metadata)
        self.epoch, self.global_step = int(epoch), int(global_step)

    def configure_distributed(self, rank, local_rank, world_size):
        """Select automatic matching or explicit per-rank rendering devices."""
        if self.port != 0:
            raise ValueError("Distributed rollout requires task.env_runner.port=0")
        self.rank = rank
        if self._uses_auto_render_device():
            return
        for key in ("egl_device_ids", "render_gpu_device_ids"):
            mapping = getattr(self, key)
            if mapping is None or len(mapping) != world_size:
                raise ValueError(f"task.env_runner.{key} must contain one device ID per local rank")
            minimum = 0 if key == "egl_device_ids" else -1
            if any(type(value) is not int or value < minimum for value in mapping):
                raise ValueError(f"task.env_runner.{key} must contain integer IDs >= {minimum}")
        self.egl_device_id = self.egl_device_ids[local_rank]
        self.render_gpu_device_id = self.render_gpu_device_ids[local_rank]

    def _uses_auto_render_device(self):
        return self.auto_render_device and all(value is None for value in (
            self.egl_device_id, self.render_gpu_device_id,
            self.egl_device_ids, self.render_gpu_device_ids))

    @staticmethod
    def _policy_render_identity(policy):
        device = torch.device(policy.device)
        if device.type != "cuda":
            raise ValueError("Automatic EGL matching requires a CUDA policy; configure rendering manually")
        # The driver resolves the same logical ordinal as PyTorch, including CVD reordering.
        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cuda.cuDeviceGetPCIBusId.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
        cuda.cuDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
        ordinal = device.index if device.index is not None else torch.cuda.current_device()
        handle = ctypes.c_int()
        bus_id = ctypes.create_string_buffer(32)
        gpu_uuid = ctypes.create_string_buffer(16)
        for name, args in (
            ("cuInit", (0,)),
            ("cuDeviceGet", (ctypes.byref(handle), ordinal)),
            ("cuDeviceGetPCIBusId", (bus_id, len(bus_id), handle)),
            ("cuDeviceGetUuid", (ctypes.byref(gpu_uuid), handle)),
        ):
            status = getattr(cuda, name)(*args)
            if status != 0:
                raise RuntimeError(f"{name} failed with CUDA status {status} for {device}")
        # A physical UUID also identifies NVIDIA EGL devices that expose no DRM node.
        return dict(pci_bus_id=bus_id.value.decode("ascii").lower(), gpu_uuid=gpu_uuid.raw.hex())

    def _predict(self, policy, request, task_id):
        raw = request.decode_to_raw()
        video_B_VT_H_W_C = raw["gt_video"]
        batch_size = video_B_VT_H_W_C.shape[0]
        assert 1 <= batch_size <= min(self.n_envs, self.n_test)
        n_obs = self.spec["n_obs_steps"]
        height, width = self.spec["camera_hw"]
        assert raw["num_camera_views"] == len(self.image_keys)
        assert video_B_VT_H_W_C.shape == (batch_size, len(self.image_keys) * n_obs, height, width, 3)
        state_B_T_D = raw["tcp_state"]
        assert state_B_T_D is not None and state_B_T_D.shape == (batch_size, n_obs, 6)
        obs = self._prepare_policy_obs(video_B_VT_H_W_C, state_B_T_D, policy, task_id)
        prediction = policy.predict_action(obs, return_foveation_outputs=self.log_patch_counts)
        action_B_K_D = prediction["action"].detach().cpu().numpy()
        action_B_K_D = LiberoFTDataset.unnormalize_action(
            action_B_K_D, self.dataset_metadata["statistics"]["action.actions"])
        assert action_B_K_D.shape == (batch_size, self.spec["max_cache_action"], 7)
        assert np.isfinite(action_B_K_D).all(), "Non-finite policy action"
        response = StepRequestFromPolicy.encode_from_raw(
            action_B_K_D, max_cache_action=self.spec["max_cache_action"]).model_dump(mode="json")
        if self.log_patch_counts:
            outputs = prediction.get("_foveation_outputs", {})
            if set(outputs) != set(self.image_keys):
                raise ValueError("Policy did not return the requested foveation outputs")
            response["num_patches"] = {
                key: outputs[key]["num_patches_B_T"].detach().cpu().long().tolist()
                for key in self.image_keys
            }
        return response

    def _prepare_policy_obs(self, video_B_VT_H_W_C, state_B_T_D, policy, task_id):
        # Online requests and recorded windows share the same normalization.
        n_obs = self.spec["n_obs_steps"]
        obs = {}
        for view, key in enumerate(self.image_keys):
            image_B_T_H_W_C = video_B_VT_H_W_C[:, view * n_obs:(view + 1) * n_obs]
            obs[key] = torch.as_tensor(image_B_T_H_W_C, device=policy.device, dtype=policy.dtype).permute(0, 1, 4, 2, 3) / 127.5 - 1
        state_B_T_D = LiberoFTDataset.norm_state_or_force(
            state_B_T_D, "mean", self.dataset_metadata["statistics"]["obs.ee_states"])
        obs["joint_state"] = torch.as_tensor(state_B_T_D, device=policy.device, dtype=policy.dtype)
        if self.use_task_id:
            task_B_T_C = torch.zeros((*state_B_T_D.shape[:2], 10), device=policy.device, dtype=policy.dtype)
            task_B_T_C[..., task_id] = 1.0
            obs["task_id"] = task_B_T_C
        return obs

    def _render_episode(self, policy, episode, media_dir, task_id):
        path = Path(episode["observation_path"])
        camera_map = {"image": "agentview_image", "gripper": "robot0_eye_in_hand_image"}
        with np.load(path) as recording:
            states_F_D = recording["joint_state"].astype(np.float32)
            frames = {}
            height, width = self.spec["camera_hw"]
            for key in self.image_keys:
                frames[key] = np.stack([
                    cv2.resize(frame_H_W_C, (width, height), interpolation=cv2.INTER_LINEAR)
                    for frame_H_W_C in recording[camera_map[key]]
                ])
        frame_count = len(states_F_D)
        assert all(len(frames[key]) == frame_count for key in self.image_keys)
        assert frame_count == episode["steps"] + 1, "Recording must include reset and terminal frames"
        seed = episode["seed"]
        raw_path = media_dir / f"episode_{episode['episode']:04d}_seed_{seed}.mp4"
        tree_path = raw_path.with_stem(raw_path.stem + "_foveation")
        def get_batch(indices_B):
            windows_B_T = np.maximum(
                indices_B[:, None] + np.arange(1 - self.spec["n_obs_steps"], 1), 0)
            video_B_VT_H_W_C = np.concatenate(
                [frames[key][windows_B_T] for key in self.image_keys], axis=1)
            obs = self._prepare_policy_obs(
                video_B_VT_H_W_C, states_F_D[windows_B_T], policy, task_id)
            return obs, {key: frames[key][indices_B] for key in self.image_keys}

        render_foveation_episode_video(
            policy, self.image_keys, frame_count, get_batch, raw_path, tree_path,
            batch_size=self.foveation_vis_batch_size, fps=self.fps, crf=self.crf,
            rotate_keys=("image",) if self.vis_rot_image else ())
        caption = f"episode={episode['episode']}, seed={seed}, success={episode['success']}"
        videos = {
            f"test/sim_video_{seed}": dict(path=str(raw_path), format="mp4", caption=caption),
            f"vis/test_foveation/video_{seed}": dict(
                path=str(tree_path), format="mp4", caption="recorded-observation routing; " + caption),
        }
        path.unlink()  # Failed routing/encoding leaves the NPZ available for diagnosis.
        return videos

    def _create_app(self, policy, completed_episodes, task_id):
        app = FastAPI()
        reported_episodes = set()

        @app.get("/health")
        async def health():
            return {"ready": True}

        @app.get("/init")
        async def init():
            return self.spec

        @app.get("/reset")
        async def reset():
            policy.reset()
            return self.spec

        @app.post("/episode_end")
        async def episode_end(result: dict):
            # An individual environment finishing must not reset other batch slots.
            episode = result.get("episode")
            if self.n_envs == 1 or type(episode) is not int or not 0 <= episode < self.n_test:
                raise HTTPException(status_code=400, detail="Invalid parallel episode notification")
            if episode in reported_episodes:
                raise HTTPException(status_code=400, detail="Duplicate episode notification")
            reported_episodes.add(episode)
            completed_episodes.put(result)
            return {"recorded": True}

        @app.post("/step")
        async def step(request: StepRequestFromEvaluator):
            # End events intentionally carry no images; decode only actual steps.
            if request.event == StepRequestEvent.ROLLOUT_END:
                policy.reset()
                if self.n_envs == 1:
                    completed_episodes.put(None)
                return {"ended": True}
            device_context = torch.cuda.device(policy.device) if policy.device.type == "cuda" else nullcontext()
            # Async handlers run serially on this server thread; no threadpool model access.
            with device_context, torch.inference_mode():
                return self._predict(policy, request, task_id)

        return app

    def _wait_for_evaluator(self, process, completed_episodes, task_name):
        deadline = time.monotonic() + self.eval_timeout
        task_label = task_name if len(task_name) <= 25 else task_name[:22] + "..."
        # The HTTP thread reports episode ends; only the training thread draws tqdm.
        with tqdm(total=self.n_test, desc=f"LIBERO EMA epoch={self.epoch} task={task_label}",
                  unit="episode", mininterval=1.0) as progress:
            while True:
                returncode = process.poll()
                while True:
                    try:
                        completed_episodes.get_nowait()
                    except Empty:
                        break
                    progress.update(1)
                if returncode is not None:
                    return returncode
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(process.args, self.eval_timeout)
                time.sleep(0.1)

    def _evaluator_environment(self):
        env = os.environ.copy()
        distributed_keys = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                            "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
                            "ROLE_WORLD_SIZE", "PYTHONHOME"}
        for key in list(env):
            if key in distributed_keys or key.startswith(("TORCHELASTIC_", "ACCELERATE_")):
                env.pop(key)
        env["PYTHONPATH"] = self.libero_root
        env.setdefault("MUJOCO_GL", "egl")
        if self._uses_auto_render_device():
            # This CPU evaluator calls the remote policy; EGL selects its GPU independently.
            # Empty CVD also avoids robosuite 1.4's CUDA/EGL ordinal equality assertion.
            env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", CUDA_VISIBLE_DEVICES="")
            env.pop("MUJOCO_EGL_DEVICE_ID", None)
        elif self.egl_device_id is not None:
            # EGL numbering need not match CUDA ordinals; use only an explicit mapping.
            env["MUJOCO_GL"] = "egl"
            env["MUJOCO_EGL_DEVICE_ID"] = str(self.egl_device_id)
        return env

    def _stop_process(self, process):
        if process is None:
            return
        # The conda wrapper and its Python child share a new POSIX process group.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + self.shutdown_timeout
        while time.monotonic() < deadline:
            process.poll()  # Reap the wrapper if it already exited.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()

    def _eval_dir(self):
        return Path(self.output_dir).resolve() / "eval" / f"epoch_{self.epoch:04d}"

    def prepare_run(self):
        """Clear the global summary and check all sources, on rank zero only."""
        eval_dir = self._eval_dir()
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "summary.json").unlink(missing_ok=True)
        for task in self.tasks:
            if not Path(task["hdf5_path"]).is_file():
                raise FileNotFoundError(task["hdf5_path"])

    def run(self, policy):
        """Evaluate all tasks serially and aggregate episode success."""
        self.prepare_run()
        return self.summarize_results(self.run_tasks(policy, range(len(self.tasks))))

    def run_tasks(self, policy, task_indices):
        """Run a rank's task queue and return only serializable results/media paths."""
        task_indices = list(task_indices)
        if not task_indices:
            return []
        if self.dataset_metadata is None:
            raise ValueError("Set LIBERO dataset metadata before evaluation")
        conda = self.conda_executable or os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not conda:
            raise FileNotFoundError("Set task.env_runner.conda_executable or install conda on PATH")
        payloads = []
        for index in task_indices:
            task = self.tasks[index]
            task_dir = self._eval_dir() / task["name"] if len(self.tasks) > 1 else self._eval_dir()
            results, task_log = self._run_task(policy, task, task_dir, conda)
            payloads.append(dict(
                name=task["name"], log=task_log,
                n_episodes=len(results["episodes"]),
                n_success=sum(int(episode["success"]) for episode in results["episodes"]),
                patch_views=results["patch_counts"]["views"] if self.log_patch_counts else {},
            ))
        return payloads

    def summarize_results(self, payloads):
        """Validate complete task coverage and publish metrics/media on rank zero."""
        by_name = {payload["name"]: payload for payload in payloads}
        if len(by_name) != len(payloads) or set(by_name) != {task["name"] for task in self.tasks}:
            raise ValueError("LIBERO rollout has duplicate, missing or unexpected tasks")
        if any(payload["n_episodes"] != self.n_test for payload in payloads):
            raise ValueError("LIBERO rollout has incomplete task episodes")
        log, task_summaries, patch_totals = {}, {}, {}
        multi_task = len(self.tasks) > 1
        for task in self.tasks:
            name = task["name"]
            payload = by_name[name]
            # Only the logger rank creates W&B objects; Gloo carries paths, not videos.
            for key, value in payload["log"].items():
                is_video = isinstance(value, dict)
                if is_video:
                    value = wandb.Video(value["path"], format=value["format"], caption=value["caption"])
                # Keep video keys stable when switching between single and multiple tasks.
                if multi_task or is_video:
                    group, metric = key.split("/", 1)
                    key = f"{group}/{name}/{metric}"
                log[key] = value
            score = payload["n_success"] / payload["n_episodes"]
            log[f"test/{name}/mean_score"] = score
            task_summaries[name] = dict(
                n_episodes=payload["n_episodes"], n_success=payload["n_success"], mean_score=score)
            if self.log_patch_counts:
                for key, view in payload["patch_views"].items():
                    total = patch_totals.setdefault(key, dict(patch_sum=0, observation_count=0))
                    for field in total:
                        total[field] += view[field]
        n_episodes = sum(task["n_episodes"] for task in task_summaries.values())
        n_success = sum(task["n_success"] for task in task_summaries.values())
        log.update({"test/mean_score": n_success / n_episodes, "test/n_episodes": n_episodes})
        if self.log_patch_counts:
            # Weight by actual observations: tasks can have different episode lengths.
            for key, view in patch_totals.items():
                log[f"test/num_patches_mean_{key}"] = view["patch_sum"] / view["observation_count"]
            log["test/num_patches_mean"] = (
                sum(view["patch_sum"] for view in patch_totals.values())
                / sum(view["observation_count"] for view in patch_totals.values()))
            log["test/total_num_patches_mean"] = sum(
                log[f"test/num_patches_mean_{key}"] for key in patch_totals)
        summary = dict(epoch=self.epoch, global_step=self.global_step, tasks=task_summaries,
                       n_episodes=n_episodes, n_success=n_success, mean_score=log["test/mean_score"])
        summary_path = self._eval_dir() / "summary.json"
        temporary_path = summary_path.with_suffix(".tmp.json")
        temporary_path.write_text(json.dumps(summary, indent=2))
        temporary_path.replace(summary_path)
        return log

    def _run_task(self, policy, task, eval_dir, conda):
        # Each task owns its server/process and restores the caller's policy/RNG state.
        render_identity = self._policy_render_identity(policy) if self._uses_auto_render_device() else {}
        eval_dir.mkdir(parents=True, exist_ok=True)
        result_path = eval_dir / "results.json"
        result_path.unlink(missing_ok=True)  # A failed retry must not reuse a previous result.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_rng_N = torch.get_rng_state()
        cuda_rng_N = torch.cuda.get_rng_state(policy.device) if policy.device.type == "cuda" else None
        was_training = policy.training
        process = server = thread = listener = None
        completed_episodes = SimpleQueue()
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.random.default_generator.manual_seed(self.seed)
            if cuda_rng_N is not None:
                with torch.cuda.device(policy.device):
                    torch.cuda.manual_seed(self.seed)
            policy.eval()
            policy.reset()
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind((self.host, self.port))
            listener.listen()
            policy_url = f"http://{self.host}:{listener.getsockname()[1]}"
            server = uvicorn.Server(uvicorn.Config(
                self._create_app(policy, completed_episodes, task.get("task_id")), log_level="warning", workers=1,
                timeout_graceful_shutdown=self.shutdown_timeout,
            ))
            thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
            thread.start()
            deadline = time.monotonic() + self.startup_timeout
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError("LIBERO policy HTTP service failed to start")
                time.sleep(0.05)
            config = dict(
                output_dir=str(eval_dir), hdf5_path=task["hdf5_path"], suite_name=task["suite_name"],
                libero_root=self.libero_root, policy_url=policy_url, policy_spec=self.spec,
                n_test=self.n_test, n_envs=self.n_envs, max_steps=self.max_steps, seed=self.seed,
                test_start_seed=self.test_start_seed, image_format=self.image_format,
                request_timeout=self.request_timeout, epoch=self.epoch, rank=self.rank,
                global_step=self.global_step, weights="ema", policy_config=self.policy_config,
                render_gpu_device_id=self.render_gpu_device_id,
                render_device_pci_bus_id=render_identity.get("pci_bus_id"),
                render_device_uuid=render_identity.get("gpu_uuid"),
                log_patch_counts=self.log_patch_counts,
                observation_dir=str(eval_dir / "observations") if self.save_foveation_video else None,
                n_test_vis=self.n_test_vis if self.save_foveation_video else 0,
            )
            evaluator_env = self._evaluator_environment()
            config["render_environment"] = {
                key: evaluator_env.get(key) for key in
                ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "PYOPENGL_PLATFORM",
                 "__EGL_VENDOR_LIBRARY_FILENAMES")
            }
            config_path = eval_dir / "eval_config.json"
            config_path.write_text(json.dumps(config, indent=2))
            command = [conda, "run", "--no-capture-output", "-n", self.conda_env,
                       "python", "-u", "eval_socket.py", "--eval-config", str(config_path)]
            print(f"[LiberoSocketRunner] rank={self.rank}, task={task['name']}, EMA epoch={self.epoch}, n_envs={self.n_envs}, API={policy_url}, log={eval_dir / 'evaluator.log'}")
            print(f"[LiberoSocketRunner] policy_device={policy.device}, target_device={render_identity}, render_environment={config['render_environment']}, render_gpu_device_id={self.render_gpu_device_id}")
            with (eval_dir / "evaluator.log").open("w") as log_file:
                process = subprocess.Popen(
                    command, cwd=self.libero_root, env=evaluator_env,
                    stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
                )
                returncode = self._wait_for_evaluator(process, completed_episodes, task["name"])
            if returncode != 0:
                raise RuntimeError(f"LIBERO evaluator exited {returncode}; see {eval_dir / 'evaluator.log'}")
            results = json.loads(result_path.read_text())
            episodes = results["episodes"]
            if len(episodes) != self.n_test or results["n_episodes"] != self.n_test:
                raise ValueError("LIBERO evaluator returned incomplete episodes")
            assert [item["episode"] for item in episodes] == list(range(self.n_test))
            assert results["epoch"] == self.epoch and results["global_step"] == self.global_step
            assert results.get("n_envs", 1) == self.n_envs
            log = {"test/mean_score": float(np.mean([item["success"] for item in episodes])),
                   "test/n_episodes": len(episodes)}
            if self.log_patch_counts:
                stats = results["patch_counts"]
                if set(stats["views"]) != set(self.image_keys):
                    raise ValueError("Evaluator returned incomplete patch statistics")
                log["test/num_patches_mean"] = stats["mean"]
                log["test/total_num_patches_mean"] = stats["total_mean"]
                for key, view in stats["views"].items():
                    log[f"test/num_patches_mean_{key}"] = view["mean"]
        finally:
            try:
                self._stop_process(process)
            finally:
                if server is not None:
                    server.should_exit = True
                if thread is not None:
                    thread.join(timeout=self.shutdown_timeout)
                    if thread.is_alive():
                        server.force_exit = True
                        thread.join(timeout=self.shutdown_timeout)
                if listener is not None:
                    listener.close()
                if thread is not None and thread.is_alive():
                    # Never resume training while a request could still access the EMA.
                    raise RuntimeError("Policy HTTP thread did not stop; terminate this training job")
                policy.reset()
                policy.train(was_training)
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.set_rng_state(torch_rng_N)
                if cuda_rng_N is not None:
                    torch.cuda.set_rng_state(cuda_rng_N, policy.device)
        if self.save_foveation_video:
            media_dir = eval_dir / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            policy.eval()
            try:
                for episode in episodes[:self.n_test_vis]:
                    log.update(self._render_episode(policy, episode, media_dir, task.get("task_id")))
            finally:
                policy.train(was_training)
                random.setstate(python_state)
                np.random.set_state(numpy_state)
        return results, log
