"""Shared TCL image geometry for training and inference."""

import numpy as np
import torch
import torch.nn.functional as F


def resolve_tcl_image_geometry(shape_meta, image_resize_wh_ratio=None):
    """Resolve per-camera content (H, W); an empty mapping selects legacy geometry."""
    content_hw_by_key = {}
    if image_resize_wh_ratio is None:
        return content_hw_by_key
    ratio_w, ratio_h = map(float, image_resize_wh_ratio.split(":"))
    if not (np.isfinite(ratio_w) and np.isfinite(ratio_h) and ratio_w >= ratio_h > 0):
        raise ValueError("image_resize_wh_ratio must be finite W:H with W >= H > 0")
    for key in ("image", "gripper"):
        channels, height, width = shape_meta["obs"][key]["shape"]
        content_height = round(width * ratio_h / ratio_w)
        if channels != 3 or height != width or not 1 <= content_height <= height:
            raise ValueError("RGB shape_meta must be square with a nonempty ratio-controlled image")
        content_hw_by_key[key] = (content_height, width)
    return content_hw_by_key


def prepare_tcl_rgb(rgb_H_W_C, content_hw=None, output_mode="normalized"):
    """Return resized uint8 HWC pixels or letterboxed float32 CHW pixels in [-1, 1]."""
    if output_mode not in ("raw", "normalized"):
        raise ValueError("output_mode must be raw or normalized")
    rgb_H_W_C = np.asarray(rgb_H_W_C, dtype=np.uint8)
    size = 256  # Preserve the legacy letterbox size when no ratio is configured.
    if content_hw is not None:
        size = content_hw[1]
        # Fix content geometry before collate; Pi0.5 owns the later raw-mode padding.
        if rgb_H_W_C.shape[:2] != content_hw:
            image_1_C_H_W = torch.from_numpy(rgb_H_W_C).permute(2, 0, 1).unsqueeze(0).float()
            image_1_C_H_W = F.interpolate(
                image_1_C_H_W, content_hw, mode="bilinear", align_corners=False, antialias=True)
            rgb_H_W_C = image_1_C_H_W[0].round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    if output_mode == "raw":
        return rgb_H_W_C

    height, width = rgb_H_W_C.shape[:2]
    scale = min(float(size) / height, float(size) / width)
    resized_hw = (max(1, round(height * scale)), max(1, round(width * scale)))
    image_1_C_H_W = torch.from_numpy(rgb_H_W_C).permute(2, 0, 1).unsqueeze(0).float()
    if resized_hw != (height, width):
        image_1_C_H_W = F.interpolate(image_1_C_H_W, resized_hw, mode="bilinear", align_corners=False)
    canvas_1_C_H_W = torch.zeros((1, 3, size, size), dtype=torch.float32)
    top = (size - resized_hw[0]) // 2
    left = (size - resized_hw[1]) // 2
    canvas_1_C_H_W[:, :, top:top + resized_hw[0], left:left + resized_hw[1]] = image_1_C_H_W
    return canvas_1_C_H_W[0] / 127.5 - 1.0
