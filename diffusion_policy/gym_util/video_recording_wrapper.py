import gym
import numpy as np
from pathlib import Path
from diffusion_policy.real_world.video_recorder import VideoRecorder

class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            steps_per_render=1,
            record_observations=False,
            rgb_obs_keys=('image',),
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder
        self.record_observations = record_observations
        self.rgb_obs_keys = tuple(rgb_obs_keys)
        self.recorded_observations = []
        self.video_frame_indices = []

        self.step_count = 0

    def set_record_observations(self, enabled: bool):
        """Enable observation capture for the next episode and clear old data."""
        self.record_observations = enabled
        self.recorded_observations = []
        self.video_frame_indices = []

    def _record_observation(self, obs):
        if not self.record_observations or self.file_path is None:
            return
        recorded_obs = {}
        for key, value_X in obs.items():
            # X denotes the observation's dimensions; match vector-env casting.
            value_X = np.asarray(value_X, dtype=self.observation_space[key].dtype)
            if key in self.rgb_obs_keys and value_X.dtype != np.uint8:
                # Recover byte RGB from [0, 1] observations to limit episode memory.
                value_X = np.rint(value_X * 255).clip(0, 255).astype(np.uint8)
            recorded_obs[key] = value_X.copy()
        self.recorded_observations.append(recorded_obs)

    def pop_recorded_observations(self):
        """Return stacked dict observations and video indices, then clear capture."""
        obs = {}
        if self.recorded_observations:
            obs = {
                key: np.stack([item[key] for item in self.recorded_observations])
                for key in self.recorded_observations[0]
            }
        indices_F = np.asarray(self.video_frame_indices, dtype=np.int64)
        self.recorded_observations = []
        self.video_frame_indices = []
        return obs, indices_F

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
        self.recorded_observations = []
        self.video_frame_indices = []
        # Reset supplies window history, but is not a frame in the sim video.
        self._record_observation(obs)
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        self._record_observation(result[0])
        if self.file_path is not None \
            and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)

            frame = self.env.render(
                mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            self.video_recoder.write_frame(frame)
            if self.record_observations:
                self.video_frame_indices.append(len(self.recorded_observations) - 1)
        return result
    
    def render(self, mode='rgb_array', **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path


class ROIVideoSaver:
    def __init__(self,
                 fps: int,
                 codec: str = 'h264',
                 input_pix_fmt: str = 'rgb24',
                 save_crop_side_by_side: bool = False,
                 **video_recorder_kwargs):
        if "save_crop_side_by_side" in video_recorder_kwargs:
            save_crop_side_by_side = video_recorder_kwargs.pop("save_crop_side_by_side")
        self.video_recorder = VideoRecorder.create_h264(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            **video_recorder_kwargs
        )
        self.save_crop_side_by_side = bool(save_crop_side_by_side)
        self.frames = []
        self.roi_params = []

    @staticmethod
    def _roi_box_xyxy(image_H_W_C: np.ndarray,
                      params_3: np.ndarray):
        height, width = image_H_W_C.shape[:2]
        center_x, center_y, scale = np.asarray(params_3, dtype=np.float32).tolist()

        x0 = int(round((center_x - scale + 1.0) * 0.5 * width))
        x1 = int(round((center_x + scale + 1.0) * 0.5 * width))
        y0 = int(round((center_y - scale + 1.0) * 0.5 * height))
        y1 = int(round((center_y + scale + 1.0) * 0.5 * height))

        x0, x1 = sorted((max(0, min(width - 1, x0)), max(0, min(width - 1, x1))))
        y0, y1 = sorted((max(0, min(height - 1, y0)), max(0, min(height - 1, y1))))
        return x0, y0, x1, y1

    @staticmethod
    def _draw_roi_box(image_H_W_C: np.ndarray,
                      params_3: np.ndarray,
                      thickness: int = 3) -> np.ndarray:
        canvas = image_H_W_C.copy()
        height, width = canvas.shape[:2]
        x0, y0, x1, y1 = ROIVideoSaver._roi_box_xyxy(canvas, params_3)
        color = np.array([255, 0, 0], dtype=np.uint8)
        canvas[y0:min(height, y0 + thickness), x0:x1 + 1] = color
        canvas[max(0, y1 - thickness + 1):y1 + 1, x0:x1 + 1] = color
        canvas[y0:y1 + 1, x0:min(width, x0 + thickness)] = color
        canvas[y0:y1 + 1, max(0, x1 - thickness + 1):x1 + 1] = color
        return canvas

    @staticmethod
    def _crop_roi_to_frame_size(image_H_W_C: np.ndarray,
                                params_3: np.ndarray) -> np.ndarray:
        height, width = image_H_W_C.shape[:2]
        x0, y0, x1, y1 = ROIVideoSaver._roi_box_xyxy(image_H_W_C, params_3)
        crop = image_H_W_C[y0:y1 + 1, x0:x1 + 1]
        if crop.size == 0:
            crop = np.zeros_like(image_H_W_C)
        crop_h, crop_w = crop.shape[:2]
        y_idx = np.linspace(0, crop_h - 1, height).round().astype(np.int64)
        x_idx = np.linspace(0, crop_w - 1, width).round().astype(np.int64)
        return crop[y_idx[:, None], x_idx[None, :]].astype(image_H_W_C.dtype, copy=False)

    def _make_output_frame(self,
                           image_H_W_C: np.ndarray,
                           roi_param_3: np.ndarray) -> np.ndarray:
        boxed = self._draw_roi_box(image_H_W_C, roi_param_3)
        if not self.save_crop_side_by_side:
            return boxed

        roi_crop = self._crop_roi_to_frame_size(image_H_W_C, roi_param_3)
        return np.concatenate([boxed, roi_crop], axis=1)

    @staticmethod
    def derive_roi_video_path(video_path: str) -> str:
        path = Path(video_path)
        return str(path.with_name(f"{path.stem}_roi{path.suffix}"))

    def append_frame(self,
                     image_H_W_C: np.ndarray,
                     roi_param_3: np.ndarray):
        image_H_W_C = np.asarray(image_H_W_C)
        roi_param_3 = np.asarray(roi_param_3, dtype=np.float32)
        if image_H_W_C.ndim != 3:
            raise ValueError(f"Expected image to be (H,W,C), got {image_H_W_C.shape}")
        if roi_param_3.shape != (3,):
            raise ValueError(f"Expected roi_param to be (3,), got {roi_param_3.shape}")
        self.frames.append(image_H_W_C.copy())
        self.roi_params.append(roi_param_3.copy())

    def append_window(self,
                      image_B_T_H_W_C: np.ndarray,
                      roi_param_B_T_3: np.ndarray):
        image_B_T_H_W_C = np.asarray(image_B_T_H_W_C)
        roi_param_B_T_3 = np.asarray(roi_param_B_T_3, dtype=np.float32)
        if image_B_T_H_W_C.ndim != 4:
            raise ValueError(f"Expected image window to be (T,H,W,C), got {image_B_T_H_W_C.shape}")
        if roi_param_B_T_3.ndim != 2 or roi_param_B_T_3.shape[-1] != 3:
            raise ValueError(f"Expected roi_param window to be (T,3), got {roi_param_B_T_3.shape}")
        if image_B_T_H_W_C.shape[0] != roi_param_B_T_3.shape[0]:
            raise ValueError(
                f"ROI window shape mismatch: image frames {image_B_T_H_W_C.shape[0]} vs roi_param {roi_param_B_T_3.shape[0]}"
            )
        for frame, params in zip(image_B_T_H_W_C, roi_param_B_T_3):
            self.append_frame(frame, params)

    def has_data(self) -> bool:
        return len(self.frames) > 0

    def reset(self):
        self.frames = []
        self.roi_params = []

    def save(self, file_path: str):
        if not self.has_data():
            return None

        file_path = str(file_path)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        self.video_recorder.stop()
        self.video_recorder.start(file_path)
        for frame, params in zip(self.frames, self.roi_params):
            self.video_recorder.write_frame(self._make_output_frame(frame, params))
        self.video_recorder.stop()
        return file_path

    def save_like(self, video_path: str):
        return self.save(self.derive_roi_video_path(video_path))
