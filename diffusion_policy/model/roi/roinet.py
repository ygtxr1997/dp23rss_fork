import math
import time
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from robokit.debug_utils.time_profiler import global_time_profiler as gtp


Size2D = Union[int, Tuple[int, int]]
BlocksPerStage = Union[int, Tuple[int, ...]]


def _as_hw(size: Size2D, name: str) -> Tuple[int, int]:
    if isinstance(size, int):
        height = size
        width = size
    else:
        if len(size) != 2:
            raise ValueError(f"{name} must be an int or a pair of ints.")
        height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} values must be positive.")
    return height, width


def _as_blocks_per_stage(
    blocks_per_stage: BlocksPerStage,
    num_stages: int,
) -> Tuple[int, ...]:
    if isinstance(blocks_per_stage, int):
        stage_blocks = (int(blocks_per_stage),) * num_stages
    else:
        stage_blocks = tuple(int(x) for x in blocks_per_stage)
        if len(stage_blocks) != num_stages:
            raise ValueError("blocks_per_stage length must match num_stages.")
    if any(x <= 0 for x in stage_blocks):
        raise ValueError("blocks_per_stage values must be positive.")
    return stage_blocks


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = max(1, num_channels // 16)
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            _make_group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            _make_group_norm(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _make_group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


class RoiNet(nn.Module):
    """Lightweight CNN spatial transformer for learnable ROI crops.

    The input tensor convention is fixed to (B, C, T, H, W). The localization
    CNN predicts one normalized center_x, center_y, scale triplet per frame,
    then affine_grid + grid_sample crops the original image to a fixed ROI size.
    Samples outside the source image are masked out instead of moving the
    predicted center back inside the image.
    """

    _PRESETS = {
        "tiny": {
            "base_channels": 48,
            "num_stages": 3,
            "blocks_per_stage": 2,
            "locator_input_size": (96, 96),
            "hidden_dim": 256,
        },
        "small": {
            "base_channels": 96,
            "num_stages": 6,
            "blocks_per_stage": 2,
            "locator_input_size": (96, 96),
            "hidden_dim": 384,
        },
        "medium": {
            "base_channels": 128,
            "num_stages": 8,
            "blocks_per_stage": 2,
            "locator_input_size": (160, 160),
            "hidden_dim": 512,
        },
    }
    _LOCATOR_TYPES = {"residual", "resnet"}
    _PREDICT_TYPES = {"xys", "logit_scale", "logit_logit"}
    _RESNET_BLOCKS_PER_STAGE_BY_PRESET = {
        "tiny": (1, 1, 1, 1),    # ResNet-10-style: stem + 4 BasicBlock stages.
        "small": (2, 2, 2, 2),   # ResNet-18-style.
        "medium": (3, 4, 6, 3),  # ResNet-34-style.
    }

    def __init__(
        self,
        in_channels: int = 3,
        roi_size: Size2D = (256, 256),
        preset: str = "small",
        locator_type: str = "residual",
        base_channels: Optional[int] = None,
        num_stages: Optional[int] = None,
        blocks_per_stage: Optional[BlocksPerStage] = None,
        locator_input_size: Optional[Size2D] = None,
        hidden_dim: Optional[int] = None,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
        init_scale: Optional[float] = None,
        init_head_weight_std: float = 1e-4,
        init_center_raw_std: Optional[float] = None,
        init_patch_logit_raw_std: Optional[float] = None,
        init_scale_logit_raw_std: Optional[float] = None,
        detach_input_grad: bool = True,
        padding_mode: str = "border",
        align_corners: bool = False,
        return_grid: bool = False,
        return_features: bool = False,
        predict_type: str = "xys",
        patchify_wh: int = 16,
        scale_num_bins: int = 64,
    ) -> None:
        super().__init__()
        if preset not in self._PRESETS:
            valid_presets = ", ".join(sorted(self._PRESETS))
            raise ValueError(f"Unknown preset '{preset}'. Valid presets: {valid_presets}.")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if not (0.0 < min_scale <= max_scale <= 1.0):
            raise ValueError("Expected 0 < min_scale <= max_scale <= 1.0.")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'.")
        if locator_type not in self._LOCATOR_TYPES:
            valid_locator_types = ", ".join(sorted(self._LOCATOR_TYPES))
            raise ValueError(
                f"Unknown locator_type '{locator_type}'. "
                f"Valid locator types: {valid_locator_types}."
            )
        if predict_type not in self._PREDICT_TYPES:
            valid_predict_types = ", ".join(sorted(self._PREDICT_TYPES))
            raise ValueError(
                f"Unknown predict_type '{predict_type}'. "
                f"Valid predict types: {valid_predict_types}."
            )
        if locator_type == "resnet" and num_stages is not None:
            raise ValueError("num_stages is not supported when locator_type='resnet'.")
        if locator_type == "resnet" and blocks_per_stage is not None:
            raise ValueError(
                "blocks_per_stage is not supported when locator_type='resnet'."
            )

        preset_config = self._PRESETS[preset]
        self.in_channels = int(in_channels)
        self.roi_h, self.roi_w = _as_hw(roi_size, "roi_size")
        self.preset = preset
        self.locator_type = locator_type
        self.base_channels = int(
            preset_config["base_channels"] if base_channels is None else base_channels
        )
        if self.locator_type == "residual":
            self.num_stages = int(
                preset_config["num_stages"] if num_stages is None else num_stages
            )
            self.blocks_per_stage = _as_blocks_per_stage(
                preset_config["blocks_per_stage"]
                if blocks_per_stage is None
                else blocks_per_stage,
                self.num_stages,
            )
        else:
            self.blocks_per_stage = self._RESNET_BLOCKS_PER_STAGE_BY_PRESET[preset]
            self.num_stages = len(self.blocks_per_stage)
        locator_size = (
            preset_config["locator_input_size"]
            if locator_input_size is None
            else locator_input_size
        )
        self.locator_input_h, self.locator_input_w = _as_hw(
            locator_size, "locator_input_size"
        )
        self.hidden_dim = int(
            preset_config["hidden_dim"] if hidden_dim is None else hidden_dim
        )
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.init_scale = self._resolve_init_scale(init_scale)
        self.init_head_weight_std = float(init_head_weight_std)
        self.init_center_raw_std = (
            None if init_center_raw_std is None else float(init_center_raw_std)
        )
        self.init_patch_logit_raw_std = (
            None
            if init_patch_logit_raw_std is None
            else float(init_patch_logit_raw_std)
        )
        self.init_scale_logit_raw_std = (
            None
            if init_scale_logit_raw_std is None
            else float(init_scale_logit_raw_std)
        )
        self.detach_input_grad = bool(detach_input_grad)
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)
        self.return_grid = bool(return_grid)
        self.return_features = bool(return_features)
        self.predict_type = predict_type
        self.patchify_wh = int(patchify_wh)
        self.scale_num_bins = int(scale_num_bins)

        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if self.num_stages <= 0:
            raise ValueError("num_stages must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.init_head_weight_std < 0.0:
            raise ValueError("init_head_weight_std must be non-negative.")
        if self.init_center_raw_std is not None and self.init_center_raw_std < 0.0:
            raise ValueError("init_center_raw_std must be non-negative.")
        if (
            self.init_patch_logit_raw_std is not None
            and self.init_patch_logit_raw_std < 0.0
        ):
            raise ValueError("init_patch_logit_raw_std must be non-negative.")
        if (
            self.init_scale_logit_raw_std is not None
            and self.init_scale_logit_raw_std < 0.0
        ):
            raise ValueError("init_scale_logit_raw_std must be non-negative.")
        if self.patchify_wh <= 0:
            raise ValueError("patchify_wh must be positive.")
        if self.scale_num_bins < 2:
            raise ValueError("scale_num_bins must be at least 2.")
        self.patch_grid_h = self.locator_input_h // self.patchify_wh
        self.patch_grid_w = self.locator_input_w // self.patchify_wh
        if self.predict_type in {"logit_scale", "logit_logit"}:
            if (
                self.locator_input_h % self.patchify_wh != 0
                or self.locator_input_w % self.patchify_wh != 0
            ):
                raise ValueError(
                    "locator_input_size must be divisible by patchify_wh "
                    "when predict_type uses logits for center prediction."
                )
            if self.patch_grid_h <= 0 or self.patch_grid_w <= 0:
                raise ValueError(
                    "patchify_wh must not exceed locator_input_size "
                    "when predict_type uses logits for center prediction."
                )
        self.patch_grid_size = self.patch_grid_h * self.patch_grid_w

        if self.locator_type == "residual":
            locator_layers, self.locator_feature_dim = (
                self._make_residual_locator_layers()
            )
        else:
            locator_layers, self.locator_feature_dim = self._make_resnet_locator_layers()
        self.locator_cnn = nn.Sequential(
            *locator_layers,
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1),
        )
        if self.predict_type == "xys":
            locator_head_out_dim = 3
        elif self.predict_type == "logit_scale":
            locator_head_out_dim = self.patch_grid_size + 1
        elif self.predict_type == "logit_logit":
            locator_head_out_dim = self.patch_grid_size + self.scale_num_bins
        else:
            raise RuntimeError(f"Unsupported predict_type {self.predict_type}.")
        self.locator_head = nn.Sequential(
            nn.Linear(self.locator_feature_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, locator_head_out_dim),
        )
        self.register_buffer(
            "_patch_centers_G_2",
            self._make_patch_centers(),
            persistent=False,
        )
        self.register_buffer(
            "_scale_bins_S",
            self._make_scale_bins(),
            persistent=False,
        )
        self._init_locator_head()

    def _make_residual_locator_layers(self):
        locator_layers = []
        block_in_channels = self.in_channels
        for stage_idx, num_stage_blocks in enumerate(self.blocks_per_stage):
            # Each stage starts with one spatial downsample, then repeats
            # residual blocks at the same H/W like a compact UNet-style encoder.
            block_out_channels = self.base_channels * min(2 ** stage_idx, 4)
            for block_idx in range(num_stage_blocks):
                stride = 4 if block_idx == 0 else 2
                block = _ResidualBlock(
                    in_channels=block_in_channels,
                    out_channels=block_out_channels,
                    stride=stride,
                )
                locator_layers.append(block)
                block_in_channels = block_out_channels

        return locator_layers, block_in_channels

    def _make_resnet_locator_layers(self):
        locator_layers = [
            nn.Conv2d(
                self.in_channels,
                self.base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            _make_group_norm(self.base_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
        block_in_channels = self.base_channels
        for stage_idx, num_stage_blocks in enumerate(self.blocks_per_stage):
            block_out_channels = self.base_channels * (2 ** stage_idx)
            for block_idx in range(num_stage_blocks):
                stride = 1 if stage_idx == 0 or block_idx > 0 else 2
                block = _ResidualBlock(
                    in_channels=block_in_channels,
                    out_channels=block_out_channels,
                    stride=stride,
                )
                locator_layers.append(block)
                block_in_channels = block_out_channels

        return locator_layers, block_in_channels

    def _resolve_init_scale(self, init_scale: Optional[float]) -> float:
        scale_range = self.max_scale - self.min_scale
        if scale_range == 0.0:
            if init_scale is not None and float(init_scale) != self.min_scale:
                raise ValueError("init_scale must equal min_scale when scale range is zero.")
            return self.min_scale

        if init_scale is None:
            target_init_scale = min(max(0.98, self.min_scale), self.max_scale)
        else:
            target_init_scale = float(init_scale)
            if not (self.min_scale <= target_init_scale <= self.max_scale):
                raise ValueError("Expected min_scale <= init_scale <= max_scale.")

        # Keep the sigmoid parameterization away from exact 0/1 saturation.
        scale_eps = 1e-4
        init_scale_ratio = (target_init_scale - self.min_scale) / scale_range
        init_scale_ratio = min(max(init_scale_ratio, scale_eps), 1.0 - scale_eps)
        return self.min_scale + scale_range * init_scale_ratio

    def _make_patch_centers(self) -> torch.Tensor:
        if self.patch_grid_h <= 0 or self.patch_grid_w <= 0:
            return torch.empty(0, 2)
        y_centers = (
            (torch.arange(self.patch_grid_h, dtype=torch.float32) + 0.5)
            / float(self.patch_grid_h)
            * 2.0
            - 1.0
        )
        x_centers = (
            (torch.arange(self.patch_grid_w, dtype=torch.float32) + 0.5)
            / float(self.patch_grid_w)
            * 2.0
            - 1.0
        )
        grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)

    def _make_scale_bins(self) -> torch.Tensor:
        return torch.linspace(
            self.min_scale,
            self.max_scale,
            steps=self.scale_num_bins,
            dtype=torch.float32,
        )

    def _make_scale_init_logits(self) -> torch.Tensor:
        if self.max_scale == self.min_scale:
            return torch.zeros(self.scale_num_bins, dtype=torch.float32)

        scale_ratio = (self.init_scale - self.min_scale) / (
            self.max_scale - self.min_scale
        )
        positions = torch.linspace(0.0, 1.0, steps=self.scale_num_bins)
        centered_positions = positions - positions.mean()
        if abs(scale_ratio - 0.5) < 1e-6:
            return torch.zeros_like(positions)

        low = -200.0
        high = 200.0
        for _ in range(80):
            mid = (low + high) * 0.5
            probs = torch.softmax(centered_positions * mid, dim=0)
            mean = torch.sum(probs * positions).item()
            if mean < scale_ratio:
                low = mid
            else:
                high = mid
        slope = (low + high) * 0.5
        return centered_positions * slope

    def _init_locator_head(self) -> None:
        last_linear = self.locator_head[-1]

        with torch.no_grad():
            last_linear.weight.zero_()
            last_linear.bias.zero_()

            if self.predict_type == "xys":
                center_weight_std = self._resolve_head_weight_std(
                    self.init_center_raw_std
                )
                if center_weight_std > 0.0:
                    nn.init.normal_(
                        last_linear.weight[:2],
                        mean=0.0,
                        std=center_weight_std,
                    )
            elif self.predict_type in {"logit_scale", "logit_logit"}:
                patch_logit_weight_std = self._resolve_head_weight_std(
                    self.init_patch_logit_raw_std
                )
                if patch_logit_weight_std > 0.0:
                    nn.init.normal_(
                        last_linear.weight[: self.patch_grid_size],
                        mean=0.0,
                        std=patch_logit_weight_std,
                    )
                if self.predict_type == "logit_logit":
                    scale_logit_weight_std = self._resolve_head_weight_std(
                        self.init_scale_logit_raw_std
                    )
                    if scale_logit_weight_std > 0.0:
                        nn.init.normal_(
                            last_linear.weight[self.patch_grid_size :],
                            mean=0.0,
                            std=scale_logit_weight_std,
                        )
            else:
                raise RuntimeError(f"Unsupported predict_type {self.predict_type}.")

            if self.max_scale > self.min_scale:
                if self.predict_type == "logit_logit":
                    last_linear.bias[self.patch_grid_size :] = (
                        self._make_scale_init_logits().to(
                            device=last_linear.bias.device,
                            dtype=last_linear.bias.dtype,
                        )
                    )
                else:
                    init_scale_ratio = (
                        (self.init_scale - self.min_scale)
                        / (self.max_scale - self.min_scale)
                    )
                    scale_bias_idx = 2 if self.predict_type == "xys" else -1
                    last_linear.bias[scale_bias_idx] = math.log(
                        init_scale_ratio / (1.0 - init_scale_ratio)
                    )

    def _resolve_head_weight_std(self, raw_std: Optional[float]) -> float:
        if raw_std is None:
            return self.init_head_weight_std
        return raw_std / math.sqrt(float(self.hidden_dim))

    def _constrain_raw_scale(self, raw_scale_BT: torch.Tensor) -> torch.Tensor:
        # Smoothly map raw scale to [min_scale, max_scale]. The final-layer bias
        # is initialized so raw output starts near self.init_scale, usually 0.98.
        if self.max_scale == self.min_scale:
            return torch.full_like(raw_scale_BT, fill_value=self.min_scale)
        scale_ratio_BT = torch.sigmoid(raw_scale_BT)
        return self.min_scale + (self.max_scale - self.min_scale) * scale_ratio_BT

    def _constrain_raw_params(self, raw_params_BT_3: torch.Tensor) -> torch.Tensor:
        if raw_params_BT_3.shape[-1] != 3:
            raise ValueError("Expected raw xys params with last dimension 3.")
        raw_center_x_BT = raw_params_BT_3[:, 0]
        raw_center_y_BT = raw_params_BT_3[:, 1]
        raw_scale_BT = raw_params_BT_3[:, 2]

        scale_BT = self._constrain_raw_scale(raw_scale_BT)
        center_x_BT = torch.tanh(raw_center_x_BT)
        center_y_BT = torch.tanh(raw_center_y_BT)

        params_BT_3 = torch.stack(
            (center_x_BT, center_y_BT, scale_BT),
            dim=-1,
        )
        return params_BT_3

    def _constrain_logit_scale_params(
        self,
        raw_logit_scale_BT_G1: torch.Tensor,
    ) -> torch.Tensor:
        expected_dim = self.patch_grid_size + 1
        if raw_logit_scale_BT_G1.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected raw logit_scale params with last dimension {expected_dim}."
            )

        logits_BT_G = raw_logit_scale_BT_G1[:, : self.patch_grid_size]
        raw_scale_BT = raw_logit_scale_BT_G1[:, -1]

        probs_BT_G = torch.softmax(logits_BT_G, dim=-1)
        centers_BT_2 = probs_BT_G @ self._patch_centers_G_2.to(
            device=raw_logit_scale_BT_G1.device,
            dtype=raw_logit_scale_BT_G1.dtype,
        )
        scale_BT = self._constrain_raw_scale(raw_scale_BT)
        return torch.cat((centers_BT_2, scale_BT.unsqueeze(-1)), dim=-1)

    def _constrain_logit_logit_params(
        self,
        raw_logit_logit_BT_GS: torch.Tensor,
    ) -> torch.Tensor:
        expected_dim = self.patch_grid_size + self.scale_num_bins
        if raw_logit_logit_BT_GS.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected raw logit_logit params with last dimension {expected_dim}."
            )

        center_logits_BT_G = raw_logit_logit_BT_GS[:, : self.patch_grid_size]
        scale_logits_BT_S = raw_logit_logit_BT_GS[:, self.patch_grid_size :]

        center_probs_BT_G = torch.softmax(center_logits_BT_G, dim=-1)
        centers_BT_2 = center_probs_BT_G @ self._patch_centers_G_2.to(
            device=raw_logit_logit_BT_GS.device,
            dtype=raw_logit_logit_BT_GS.dtype,
        )
        scale_probs_BT_S = torch.softmax(scale_logits_BT_S, dim=-1)
        scale_BT = scale_probs_BT_S @ self._scale_bins_S.to(
            device=raw_logit_logit_BT_GS.device,
            dtype=raw_logit_logit_BT_GS.dtype,
        )
        return torch.cat((centers_BT_2, scale_BT.unsqueeze(-1)), dim=-1)

    def _constrain_raw_head_output(
        self,
        raw_head_output_BT_D: torch.Tensor,
    ) -> torch.Tensor:
        if self.predict_type == "xys":
            return self._constrain_raw_params(raw_head_output_BT_D)
        if self.predict_type == "logit_scale":
            return self._constrain_logit_scale_params(raw_head_output_BT_D)
        if self.predict_type == "logit_logit":
            return self._constrain_logit_logit_params(raw_head_output_BT_D)
        raise RuntimeError(f"Unsupported predict_type {self.predict_type}.")

    @staticmethod
    def area_loss(
        params_or_output: Union[Dict[str, torch.Tensor], torch.Tensor],
        reduction: str = "mean",
    ) -> torch.Tensor:
        if isinstance(params_or_output, dict):
            params_B_T_3 = params_or_output["params_B_T_3"]
        else:
            params_B_T_3 = params_or_output
        if params_B_T_3.shape[-1] != 3:
            raise ValueError("Expected ROI params with last dimension 3.")

        # Current RoiNet uses isotropic scale, so percentage area is scale^2.
        area_percent = params_B_T_3[..., 2].square()
        if reduction == "none":
            return area_percent
        if reduction == "mean":
            return area_percent.mean()
        if reduction == "sum":
            return area_percent.sum()
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")

    @staticmethod
    def center_diversity_loss(
        params_or_output: Union[Dict[str, torch.Tensor], torch.Tensor],
        margin: float = 0.25,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if isinstance(params_or_output, dict):
            params_B_T_3 = params_or_output["params_B_T_3"]
        else:
            params_B_T_3 = params_or_output
        if params_B_T_3.shape[-1] != 3:
            raise ValueError("Expected ROI params with last dimension 3.")
        if params_B_T_3.ndim != 3:
            raise ValueError("Expected ROI params with shape (B, T, 3).")
        if margin < 0.0:
            raise ValueError("margin must be non-negative.")
        if eps < 0.0:
            raise ValueError("eps must be non-negative.")

        centers_B_T_2 = params_B_T_3[..., :2]
        batch_size = centers_B_T_2.shape[0]
        if batch_size < 2:
            return centers_B_T_2.sum() * 0.0

        center_diff_B_B_T_2 = (
            centers_B_T_2[:, None, :, :] - centers_B_T_2[None, :, :, :]
        )
        center_dist_B_B_T = torch.sqrt(
            center_diff_B_B_T_2.square().sum(dim=-1) + eps
        )
        pair_mask_B_B = torch.triu(
            torch.ones(
                batch_size,
                batch_size,
                device=centers_B_T_2.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        pair_dist_P_T = center_dist_B_B_T[pair_mask_B_B]
        return torch.relu(float(margin) - pair_dist_P_T).square().mean()

    def _make_theta(self, params_BT_3: torch.Tensor) -> torch.Tensor:
        center_x_BT = params_BT_3[:, 0]
        center_y_BT = params_BT_3[:, 1]
        scale_BT = params_BT_3[:, 2]

        theta_BT_2_3 = params_BT_3.new_zeros((params_BT_3.shape[0], 2, 3))
        theta_BT_2_3[:, 0, 0] = scale_BT
        theta_BT_2_3[:, 1, 1] = scale_BT
        theta_BT_2_3[:, 0, 2] = center_x_BT
        theta_BT_2_3[:, 1, 2] = center_y_BT
        return theta_BT_2_3

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        return_grid: Optional[bool] = None,
        return_features: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        # Input dimension convention is fixed: B=batch, C=channels,
        # T=time, H/W=source image size.
        gtp_open = self.training
        gtp_training = "train" if self.training else "val"
        with gtp(f"{gtp_training}/roi.preprocess", group=f"{gtp_training}/roi"):
            if x_B_C_T_H_W.ndim != 5:
                raise ValueError(
                    f"Expected x_B_C_T_H_W to have shape (B, C, T, H, W), "
                    f"got {tuple(x_B_C_T_H_W.shape)}."
                )
            if not torch.is_floating_point(x_B_C_T_H_W):
                raise TypeError("RoiNet expects a floating point image tensor.")

            B, C, T, H, W = x_B_C_T_H_W.shape
            if C != self.in_channels:
                raise ValueError(
                    f"Expected {self.in_channels} input channels, got {C}."
                )
            BT = B * T
            should_return_grid = self.return_grid if return_grid is None else return_grid
            should_return_features = (
                self.return_features if return_features is None else return_features
            )

            # Convert B,C,T,H,W -> B*T,C,H,W so each frame is localized and sampled
            # independently by ordinary 2D CNN and grid_sample operators.
            x_BT_C_H_W = (
                x_B_C_T_H_W.permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(BT, C, H, W)
            )
            image_for_roi = x_BT_C_H_W.detach() if self.detach_input_grad else x_BT_C_H_W

        # The localization CNN sees only a low-resolution proxy for speed; the
        # final differentiable crop below samples directly from the original frame.
        if self.detach_input_grad:
            with gtp(f"{gtp_training}/roi.interpolate", group=f"{gtp_training}/roi"):
                with torch.no_grad():
                    locator_x_BT_C_LH_LW = F.interpolate(
                        image_for_roi,
                        size=(self.locator_input_h, self.locator_input_w),
                        mode="bilinear",
                        align_corners=False,
                    )
        else:
            locator_x_BT_C_LH_LW = F.interpolate(
                image_for_roi,
                size=(self.locator_input_h, self.locator_input_w),
                mode="bilinear",
                align_corners=False,
            )
        with gtp(f"{gtp_training}/roi.locator_cnn", group=f"{gtp_training}/roi"):
            locator_features_BT_D = self.locator_cnn(locator_x_BT_C_LH_LW)
        with gtp(f"{gtp_training}/roi.locator_head", group=f"{gtp_training}/roi"):
            raw_head_output_BT_D = self.locator_head(locator_features_BT_D)

        # Map unconstrained network outputs into normalized ROI parameters.
        params_BT_3 = self._constrain_raw_head_output(raw_head_output_BT_D)

        # theta maps output normalized coordinates to input normalized coordinates.
        # With center=(0,0), scale=1 this is an identity full-frame resize.
        theta_BT_2_3 = self._make_theta(params_BT_3)

        with gtp(f"{gtp_training}/roi.affine_grid_sample", group=f"{gtp_training}/roi"):
            # Build the sampling grid and crop from the original full-resolution frames.
            grid_BT_RH_RW_2 = F.affine_grid(
                theta_BT_2_3,
                size=(BT, C, self.roi_h, self.roi_w),
                align_corners=self.align_corners,
            )
            valid_mask_BT_1_RH_RW = (
                (grid_BT_RH_RW_2[..., 0] >= -1.0)
                & (grid_BT_RH_RW_2[..., 0] <= 1.0)
                & (grid_BT_RH_RW_2[..., 1] >= -1.0)
                & (grid_BT_RH_RW_2[..., 1] <= 1.0)
            ).unsqueeze(1)
            roi_BT_C_RH_RW = F.grid_sample(
                image_for_roi,
                grid_BT_RH_RW_2,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            roi_BT_C_RH_RW = roi_BT_C_RH_RW * valid_mask_BT_1_RH_RW.to(
                dtype=roi_BT_C_RH_RW.dtype
            )

        with gtp(f"{gtp_training}/roi.postprocess", group=f"{gtp_training}/roi"):
            roi_B_C_T_RH_RW = (
                roi_BT_C_RH_RW.view(B, T, C, self.roi_h, self.roi_w)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
            )
            theta_B_T_2_3 = theta_BT_2_3.view(B, T, 2, 3)
            params_B_T_3 = params_BT_3.view(B, T, 3)
            valid_mask_B_T_1_RH_RW = valid_mask_BT_1_RH_RW.view(
                B,
                T,
                1,
                self.roi_h,
                self.roi_w,
            )

        out = {
            "roi_B_C_T_RH_RW": roi_B_C_T_RH_RW,
            "theta_B_T_2_3": theta_B_T_2_3,
            "params_B_T_3": params_B_T_3,
            "valid_mask_B_T_1_RH_RW": valid_mask_B_T_1_RH_RW,
        }
        if should_return_grid:
            out["grid_B_T_RH_RW_2"] = grid_BT_RH_RW_2.view(
                B,
                T,
                self.roi_h,
                self.roi_w,
                2,
            )
        if should_return_features:
            out["locator_features_B_T_D"] = locator_features_BT_D.view(
                B,
                T,
                self.locator_feature_dim,
            )
        return out

    def count_parameters(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (p for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)

    def parameter_summary(self) -> Dict[str, int]:
        locator_cnn_params = sum(p.numel() for p in self.locator_cnn.parameters())
        locator_head_params = sum(p.numel() for p in self.locator_head.parameters())
        return {
            "total": self.count_parameters(trainable_only=False),
            "trainable": self.count_parameters(trainable_only=True),
            "locator_cnn": locator_cnn_params,
            "locator_head": locator_head_params,
        }

    @classmethod
    def speed_test(
        cls,
        batch_size: int = 1,
        channels: int = 3,
        time_steps: int = 1,
        height: int = 2160,
        width: int = 3840,
        roi_size: Size2D = (256, 256),
        iterations: int = 50,
        warmup: int = 10,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        **roinet_kwargs,
    ) -> Dict[str, object]:
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        if warmup < 0:
            raise ValueError("warmup must be non-negative.")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        if "in_channels" in roinet_kwargs and roinet_kwargs["in_channels"] != channels:
            raise ValueError("channels must match roinet_kwargs['in_channels'].")
        roinet_kwargs["in_channels"] = channels

        model = cls(roi_size=roi_size, **roinet_kwargs).to(device=device, dtype=dtype)
        model.eval()
        x_B_C_T_H_W = torch.randn(
            batch_size,
            channels,
            time_steps,
            height,
            width,
            device=device,
            dtype=dtype,
        )

        def _sync_if_cuda() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        with torch.no_grad():
            for _ in range(warmup):
                model(x_B_C_T_H_W)
            _sync_if_cuda()
            start_time = time.perf_counter()
            out = None
            for _ in range(iterations):
                out = model(x_B_C_T_H_W)
            _sync_if_cuda()
            elapsed_s = time.perf_counter() - start_time

        ms_per_iter = elapsed_s * 1000.0 / iterations
        frames_per_iter = batch_size * time_steps
        fps = frames_per_iter * 1000.0 / ms_per_iter
        return {
            "ms_per_iter": ms_per_iter,
            "fps": fps,
            "parameter_count": model.count_parameters(),
            "input_shape": tuple(x_B_C_T_H_W.shape),
            "output_shape": tuple(out["roi_B_C_T_RH_RW"].shape),
            "device": str(device),
            "dtype": str(dtype),
        }

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"roi_size=({self.roi_h}, {self.roi_w}), "
            f"preset='{self.preset}', "
            f"locator_type='{self.locator_type}', "
            f"locator_input_size=({self.locator_input_h}, {self.locator_input_w}), "
            f"base_channels={self.base_channels}, "
            f"num_stages={self.num_stages}, "
            f"blocks_per_stage={self.blocks_per_stage}, "
            f"hidden_dim={self.hidden_dim}, "
            f"scale_range=({self.min_scale}, {self.max_scale}), "
            f"init_scale={self.init_scale}, "
            f"init_head_weight_std={self.init_head_weight_std}, "
            f"init_center_raw_std={self.init_center_raw_std}, "
            f"init_patch_logit_raw_std={self.init_patch_logit_raw_std}, "
            f"init_scale_logit_raw_std={self.init_scale_logit_raw_std}, "
            f"detach_input_grad={self.detach_input_grad}, "
            f"predict_type='{self.predict_type}', "
            f"patchify_wh={self.patchify_wh}, "
            f"scale_num_bins={self.scale_num_bins}"
        )
