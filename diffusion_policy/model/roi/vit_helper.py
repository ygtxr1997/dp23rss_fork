from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class QuadtreeViTEncoder(nn.Module):
    """Encode every internal quadtree node with globally attended ViT tokens."""

    def __init__(
        self,
        in_channels: int,
        input_hw: Tuple[int, int],
        level_grid_hw: Tuple[Tuple[int, int], ...],
        token_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.input_hw = tuple(int(size) for size in input_hw)
        self.level_grid_hw = tuple(
            tuple(int(size) for size in grid_hw) for grid_hw in level_grid_hw
        )
        self.token_dim = int(token_dim)

        if not self.level_grid_hw:
            raise ValueError("QuadtreeViTEncoder requires an internal node level.")
        for level in range(len(self.level_grid_hw) - 1):
            coarse_h, coarse_w = self.level_grid_hw[level]
            fine_h, fine_w = self.level_grid_hw[level + 1]
            if (fine_h, fine_w) != (2 * coarse_h, 2 * coarse_w):
                raise ValueError("Adjacent quadtree grids must differ by exactly 2x.")

        input_h, input_w = self.input_hw
        fine_grid_h, fine_grid_w = self.level_grid_hw[-1]
        if (
            input_h < fine_grid_h
            or input_w < fine_grid_w
            or input_h % fine_grid_h != 0
            or input_w % fine_grid_w != 0
        ):
            raise ValueError(
                "The global-level input must split exactly into the finest "
                "internal node grid."
            )
        patch_h = input_h // fine_grid_h
        patch_w = input_w // fine_grid_w
        self.patch_hw = (patch_h, patch_w)
        self.patch_embedding = nn.Conv2d(
            self.in_channels,
            self.token_dim,
            kernel_size=self.patch_hw,
            stride=self.patch_hw,
        )

        # Each merge has its own parameters because every target level has a
        # different quadtree scale, while child order stays TL, TR, BL, BR.
        self.token_merges = nn.ModuleDict(
            {
                str(level): nn.Sequential(
                    nn.LayerNorm(4 * self.token_dim),
                    nn.Linear(4 * self.token_dim, self.token_dim),
                )
                for level in range(len(self.level_grid_hw) - 1)
            }
        )
        self.position_embeddings = nn.ParameterDict(
            {
                str(level): nn.Parameter(
                    torch.empty(1, grid_h, grid_w, self.token_dim)
                )
                for level, (grid_h, grid_w) in enumerate(self.level_grid_hw)
            }
        )
        self.level_embedding_1_L_D = nn.Parameter(
            torch.empty(1, len(self.level_grid_hw), self.token_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(num_heads),
            dim_feedforward=int(round(self.token_dim * float(mlp_ratio))),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.token_dim),
        )

        self.level_num_nodes = tuple(
            grid_h * grid_w for grid_h, grid_w in self.level_grid_hw
        )
        offsets = [0]
        for num_nodes in self.level_num_nodes:
            offsets.append(offsets[-1] + num_nodes)
        self.level_offsets = tuple(offsets)

        for position_embedding_1_H_W_D in self.position_embeddings.values():
            nn.init.trunc_normal_(position_embedding_1_H_W_D, std=0.02)
        nn.init.trunc_normal_(self.level_embedding_1_L_D, std=0.02)

    @staticmethod
    def _merge_children(
        child_features_BT_D_H2_W2: torch.Tensor,
        merge: nn.Module,
    ) -> torch.Tensor:
        BT, D, H2, W2 = child_features_BT_D_H2_W2.shape
        # Reshape makes the last four D-blocks row-major: TL, TR, BL, BR.
        children_BT_H_W_4D = (
            child_features_BT_D_H2_W2.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(BT, H2 // 2, 2, W2 // 2, 2, D)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .reshape(BT, H2 // 2, W2 // 2, 4 * D)
        )
        parent_features_BT_H_W_D = merge(children_BT_H_W_4D)
        return parent_features_BT_H_W_D.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Return internal node maps as ``level: (B, T, D, H, W)``."""
        if x_B_C_T_H_W.ndim != 5:
            raise ValueError("Expected ViT input with shape (B, C, T, H, W).")
        B, C, T, H, W = x_B_C_T_H_W.shape
        if C != self.in_channels or (H, W) != self.input_hw:
            raise ValueError(
                f"Expected input shape (B, {self.in_channels}, T, "
                f"{self.input_hw[0]}, {self.input_hw[1]})."
            )

        x_BT_C_H_W = (
            x_B_C_T_H_W.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(B * T, C, H, W)
        )
        finest_level = len(self.level_grid_hw) - 1
        level_features_BT_D_H_W = {
            finest_level: self.patch_embedding(x_BT_C_H_W)
        }
        for level in range(finest_level - 1, -1, -1):
            level_features_BT_D_H_W[level] = self._merge_children(
                level_features_BT_D_H_W[level + 1],
                self.token_merges[str(level)],
            )

        level_tokens_BT_N_D = []
        for level, (grid_h, grid_w) in enumerate(self.level_grid_hw):
            features_BT_D_H_W = level_features_BT_D_H_W[level]
            if features_BT_D_H_W.shape[-2:] != (grid_h, grid_w):
                raise ValueError(
                    f"ViT node level {level} does not match grid {(grid_h, grid_w)}."
                )
            tokens_BT_N_D = (
                features_BT_D_H_W.permute(0, 2, 3, 1)
                .contiguous()
                .reshape(B * T, grid_h * grid_w, self.token_dim)
            )
            position_1_N_D = self.position_embeddings[str(level)].reshape(
                1,
                grid_h * grid_w,
                self.token_dim,
            )
            tokens_BT_N_D = (
                tokens_BT_N_D
                + position_1_N_D
                + self.level_embedding_1_L_D[:, level : level + 1]
            )
            level_tokens_BT_N_D.append(tokens_BT_N_D)

        # Frames are folded into the batch, so attention is global in space and
        # across node levels but never mixes different time steps.
        tokens_BT_M_D = torch.cat(level_tokens_BT_N_D, dim=1)
        encoded_BT_M_D = self.transformer(tokens_BT_M_D)

        encoded_levels_B_T_D_H_W: Dict[int, torch.Tensor] = {}
        for level, (grid_h, grid_w) in enumerate(self.level_grid_hw):
            start = self.level_offsets[level]
            end = self.level_offsets[level + 1]
            encoded_levels_B_T_D_H_W[level] = (
                encoded_BT_M_D[:, start:end]
                .reshape(B, T, grid_h, grid_w, self.token_dim)
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )
        return encoded_levels_B_T_D_H_W


class NodeAlignedViTRouter(nn.Module):
    """Fuse condition into aligned ViT nodes and predict per-level split logits."""

    def __init__(
        self,
        level_grid_hw: Tuple[Tuple[int, int], ...],
        node_dim: int,
        hidden_dim: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.level_grid_hw = tuple(
            tuple(int(size) for size in grid_hw) for grid_hw in level_grid_hw
        )
        self.node_dim = int(node_dim)
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        if not self.level_grid_hw:
            raise ValueError("NodeAlignedViTRouter requires an internal node level.")

        self.condition_projection = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.node_projections = nn.ModuleDict(
            {
                str(level): nn.Linear(self.node_dim, self.hidden_dim)
                for level in range(len(self.level_grid_hw))
            }
        )
        self.split_heads = nn.ModuleDict(
            {
                str(level): nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, 1),
                )
                for level in range(len(self.level_grid_hw))
            }
        )

    def forward(
        self,
        level_features_B_T_D_H_W: Dict[int, torch.Tensor],
        cond_B_T_D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits in internal-level-major, node-row-major order."""
        reference_B_T_D_H_W = level_features_B_T_D_H_W[0]
        if reference_B_T_D_H_W.ndim != 5:
            raise ValueError("Expected ViT node features with shape (B, T, D, H, W).")
        B, T = reference_B_T_D_H_W.shape[:2]

        if cond_B_T_D is None:
            condition_B_T_D = reference_B_T_D_H_W.new_zeros(
                (B, T, self.hidden_dim)
            )
        else:
            if cond_B_T_D.shape != (B, T, self.cond_dim):
                raise ValueError(
                    f"Expected cond_B_T_D with shape {(B, T, self.cond_dim)}."
                )
            cond_B_T_D = cond_B_T_D.to(
                device=reference_B_T_D_H_W.device,
                dtype=reference_B_T_D_H_W.dtype,
            )
            condition_B_T_D = self.condition_projection(cond_B_T_D)

        logits_B_T_M = []
        for level, (grid_h, grid_w) in enumerate(self.level_grid_hw):
            expected_shape = (B, T, self.node_dim, grid_h, grid_w)
            features_B_T_D_H_W = level_features_B_T_D_H_W[level]
            if features_B_T_D_H_W.shape != expected_shape:
                raise ValueError(
                    f"Expected node level {level} features with shape "
                    f"{expected_shape}, got {tuple(features_B_T_D_H_W.shape)}."
                )
            features_B_T_H_W_D = features_B_T_D_H_W.permute(0, 1, 3, 4, 2)
            fused_B_T_H_W_D = self.node_projections[str(level)](
                features_B_T_H_W_D
            ) + condition_B_T_D[:, :, None, None, :]
            logits_B_T_H_W_1 = self.split_heads[str(level)](fused_B_T_H_W_D)
            logits_B_T_M.append(logits_B_T_H_W_1.reshape(B, T, grid_h * grid_w))
        return torch.cat(logits_B_T_M, dim=-1)
