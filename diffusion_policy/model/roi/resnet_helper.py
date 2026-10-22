from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, ResNet


class ResNetHelper(nn.Module):
    """Encode video frames with a from-scratch torchvision ResNet."""

    MODEL_DIMS: Dict[str, int] = {
        "resnet10": 512,
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }

    def __init__(
        self,
        model_name: str = "resnet18",
        out_dim: int = 768,
        project_output: bool = True,
    ) -> None:
        super().__init__()
        if model_name not in self.MODEL_DIMS:
            raise ValueError(f"{model_name} not in {tuple(self.MODEL_DIMS)}")
        if out_dim <= 0:
            raise ValueError("out_dim must be positive.")

        self.model_name = model_name
        self.out_dim = int(out_dim)
        self.project_output = bool(project_output)
        self.backbone = (
            ResNet(block=BasicBlock, layers=[1, 1, 1, 1])
            if model_name == "resnet10"
            else getattr(models, model_name)(weights=None)
        )
        self.backbone.fc = nn.Identity()
        self.output_projection = (
            nn.Conv2d(
                self.MODEL_DIMS[model_name],
                self.out_dim,
                kernel_size=1,
            )
            if self.project_output
            else nn.Identity()
        )

    def _reshape_frames(
        self,
        x_B_C_T_H_W: torch.Tensor,
        spatial_divisor: Optional[int],
    ) -> Tuple[torch.Tensor, int, int]:
        B, C, T, H, W = x_B_C_T_H_W.shape
        if C != 3:
            raise ValueError("ResNetHelper expects three-channel RGB input.")
        if spatial_divisor is not None and (H % spatial_divisor != 0 or W % spatial_divisor != 0):
            raise ValueError(
                "ResNet input height and width must be divisible by "
                f"{spatial_divisor}."
            )

        if x_B_C_T_H_W.dtype == torch.uint8:
            x_B_C_T_H_W = x_B_C_T_H_W.float().mul(2.0 / 255.0).sub(1.0)
        else:
            x_B_C_T_H_W = x_B_C_T_H_W.float()
        x_BT_C_H_W = (
            x_B_C_T_H_W.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(B * T, C, H, W)
        )
        return x_BT_C_H_W, B, T

    def _forward_backbone_stages(
        self,
        x_BT_C_H_W: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        features_BT_Cr_Hr_Wr = self.backbone.conv1(x_BT_C_H_W)
        features_BT_Cr_Hr_Wr = self.backbone.bn1(features_BT_Cr_Hr_Wr)
        features_BT_Cr_Hr_Wr = self.backbone.relu(features_BT_Cr_Hr_Wr)
        features_BT_Cr_Hr_Wr = self.backbone.maxpool(features_BT_Cr_Hr_Wr)
        layer1_BT_C_H_W = self.backbone.layer1(features_BT_Cr_Hr_Wr)
        layer2_BT_C_H_W = self.backbone.layer2(layer1_BT_C_H_W)
        layer3_BT_C_H_W = self.backbone.layer3(layer2_BT_C_H_W)
        layer4_BT_C_H_W = self.backbone.layer4(layer3_BT_C_H_W)
        return {
            "layer1": layer1_BT_C_H_W,
            "layer2": layer2_BT_C_H_W,
            "layer3": layer3_BT_C_H_W,
            "layer4": layer4_BT_C_H_W,
        }

    def forward_stages(
        self,
        x_B_C_T_H_W: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return raw layer1-layer4 maps with shape (B, T, C, H, W)."""
        x_BT_C_H_W, B, T = self._reshape_frames(
            x_B_C_T_H_W,
            spatial_divisor=16,
        )
        stages = self._forward_backbone_stages(x_BT_C_H_W)
        return {
            name: features_BT_C_H_W.reshape(
                B,
                T,
                *features_BT_C_H_W.shape[1:],
            )
            for name, features_BT_C_H_W in stages.items()
        }

    def forward(self, x_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        """Return per-frame spatial tokens with shape (B, T, N, D)."""
        x_BT_C_H_W, B, T = self._reshape_frames(
            x_B_C_T_H_W,
            # Pooled routing only needs a valid convolutional feature map;
            # unlike the node-aligned FPN it does not require a fixed grid.
            spatial_divisor=None,
        )
        stages = self._forward_backbone_stages(x_BT_C_H_W)
        # Keep layer4 spatial structure so the pooled router receives image tokens.
        features_BT_D_Hr_Wr = self.output_projection(stages["layer4"])

        grid_h, grid_w = features_BT_D_Hr_Wr.shape[-2:]
        feature_dim = features_BT_D_Hr_Wr.shape[1]
        tokens_B_T_N_D = (
            features_BT_D_Hr_Wr.flatten(start_dim=2)
            .transpose(1, 2)
            .reshape(B, T, grid_h * grid_w, feature_dim)
        )
        return tokens_B_T_N_D
