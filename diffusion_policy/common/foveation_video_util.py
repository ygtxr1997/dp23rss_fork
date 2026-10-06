"""Shared rendering for recorded-observation routing videos."""

from contextlib import ExitStack
import random

import cv2
import numpy as np
import torch

def draw_patch_history(
        frame_H_W_C, steps_F, counts_F, first_step, last_step, min_patches, max_patches):
    """Overlay a fixed-axis history of the trees displayed so far in this episode."""
    height, width = frame_H_W_C.shape[:2]
    panel_w, panel_h = max(1, round(width * 0.30)), max(1, round(height * 0.20))
    margin = max(1, round(min(height, width) * 0.01))
    roi_H_W_C = frame_H_W_C[margin:margin + panel_h, margin:margin + panel_w]
    # Draw on a larger canvas for legible antialiased text in the small inset.
    panel_H_W_C = cv2.resize(roi_H_W_C, (300, 200))
    panel_H_W_C = np.rint(panel_H_W_C * 0.45).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (235, 235, 235)
    color = (255, 220, 80)  # RGB, like the video frame.
    left, right, top, bottom = 72, 288, 52, 152
    cv2.putText(panel_H_W_C, f'patches: {int(counts_F[-1])}', (8, 27),
                font, 1.0, white, 2, cv2.LINE_AA)
    cv2.line(panel_H_W_C, (left, top), (left, bottom), white, 2, cv2.LINE_AA)
    cv2.line(panel_H_W_C, (left, bottom), (right, bottom), white, 2, cv2.LINE_AA)
    for count, y in ((max_patches, top), (min_patches, bottom)):
        cv2.putText(panel_H_W_C, str(count), (3, y + 7),
                    font, 0.9, white, 2, cv2.LINE_AA)
    cv2.putText(panel_H_W_C, str(first_step), (left - 5, 188),
                font, 0.9, white, 2, cv2.LINE_AA)
    last_label = str(last_step)
    text_width = cv2.getTextSize(last_label, font, 0.9, 2)[0][0]
    cv2.putText(panel_H_W_C, last_label, (right - text_width, 188),
                font, 0.9, white, 2, cv2.LINE_AA)
    cv2.putText(panel_H_W_C, 'step', (142, 188),
                font, 0.85, white, 2, cv2.LINE_AA)
    x_F = left + (steps_F - first_step) / max(1, last_step - first_step) * (right - left)
    if max_patches == min_patches:
        y_F = np.full_like(x_F, (top + bottom) / 2)
    else:
        y_F = bottom - (counts_F - min_patches) / (max_patches - min_patches) * (bottom - top)
    points_F_2 = np.rint(np.stack((x_F, y_F), axis=-1)).astype(np.int32)
    cv2.polylines(panel_H_W_C, [points_F_2], False, color, 4, cv2.LINE_AA)
    cv2.circle(panel_H_W_C, tuple(points_F_2[-1]), 4, color, -1, cv2.LINE_AA)
    roi_H_W_C[:] = cv2.resize(panel_H_W_C, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    return frame_H_W_C


def render_foveation_episode_video(
        policy, image_keys, frame_count, get_batch, raw_path, tree_path,
        batch_size=16, fps=20, crf=22, rotate_keys=(), seed=None):
    """Write videos from get_batch(indices_B) -> (obs, {camera: uint8[B,H,W,3]})."""
    # Import encoding only when writing video; overlay-only consumers do not need it.
    from diffusion_policy.real_world.video_recorder import VideoRecorder

    assert frame_count > 0 and batch_size > 0 and fps > 0
    counts_by_key = {key: np.empty(frame_count, dtype=np.int64) for key in image_keys}
    device = torch.device(policy.device)
    rng_devices = [device] if device.type == "cuda" else []
    with ExitStack() as stack:
        stack.callback(random.setstate, random.getstate())
        stack.callback(np.random.set_state, np.random.get_state())
        stack.enter_context(torch.random.fork_rng(devices=rng_devices))
        stack.enter_context(torch.inference_mode())
        stack.callback(policy.train, policy.training)
        policy.eval()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
        recorders = []
        for path in (raw_path, tree_path):
            recorder = VideoRecorder.create_h264(
                fps=fps, codec="h264", input_pix_fmt="rgb24", crf=crf,
                thread_type="FRAME", thread_count=1)
            stack.callback(recorder.stop)
            recorder.start(str(path))
            recorders.append(recorder)
        for start in range(0, frame_count, batch_size):
            indices_B = np.arange(start, min(start + batch_size, frame_count))
            obs, raw_by_key = get_batch(indices_B)
            obs = {key: value_B_T_X.to(device=device) for key, value_B_T_X in obs.items()}
            outputs = policy.get_foveation_outputs(obs)
            visualizations = {}
            for key in image_keys:
                last = {name: value_B_T_X[:, -1:] for name, value_B_T_X in outputs[key].items()}
                visualizations[key] = policy.foveation_net.vis_ar_tokens(last)[:, 0]
                counts_by_key[key][indices_B] = last["num_patches_B_T"][:, 0].cpu().numpy()
            for offset, index in enumerate(indices_B):
                raw_views, tree_views = [], []
                for key in image_keys:
                    raw_H_W_C = raw_by_key[key][offset].copy()
                    tree_H_W_C = visualizations[key][offset].copy()
                    if key in rotate_keys:
                        raw_H_W_C = np.flip(raw_H_W_C, axis=(0, 1)).copy()
                        tree_H_W_C = np.flip(tree_H_W_C, axis=(0, 1)).copy()
                    tree_H_W_C = draw_patch_history(
                        tree_H_W_C, np.arange(index + 1), counts_by_key[key][:index + 1],
                        0, frame_count - 1, policy.foveation_net.min_num_patches,
                        policy.foveation_net.max_num_patches)
                    raw_views.append(raw_H_W_C)
                    tree_views.append(tree_H_W_C)
                recorders[0].write_frame(np.concatenate(raw_views, axis=1))
                recorders[1].write_frame(np.concatenate(tree_views, axis=1))
    return counts_by_key
