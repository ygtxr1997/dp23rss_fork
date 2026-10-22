"""Router-free Stage A sampling with FoveationNet's quadtree packing semantics."""
import ast
import math
import random
import re

import torch
import torch.nn.functional as F


class FixedTopologyPatchSampler:
    """Pack legal 56/112/224 leaves, sampling one tree per observation window."""

    def __init__(self, split_mode, seed=42):
        self.rng = random.Random(seed)
        self.modes = self._parse_modes(split_mode)
        self.last_mode_indices = []

    @staticmethod
    def _parse_modes(split_mode):
        if isinstance(split_mode, str):
            split_mode = split_mode.strip()
            entries = ast.literal_eval(split_mode) if split_mode.startswith('[') else [split_mode]
        else:
            entries = list(split_mode)
        modes = []
        number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
        for entry in entries:
            if '%' in entry:
                weight_text, mode = entry.split('%', 1)
                weight = float(weight_text)
            else:
                weight, mode = 100.0, entry
            mode = mode.strip().strip("'\"")
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('Split percentages must be finite and non-negative')
            if mode.startswith('level:'):
                value = int(mode.removeprefix('level:'))
                if value not in (0, 1, 2):
                    raise ValueError('Leaf level must be 0, 1 or 2')
                kind = 'level'
            elif mode.startswith('random:'):
                text = mode.removeprefix('random:')
                match = re.fullmatch(rf'\s*({number})\s*-\s*({number})\s*', text)
                value = tuple(float(item) for item in (match.groups() if match else text.split(',')))
                if any(not 0 <= ratio <= 1 for ratio in value) or (match and value[0] > value[1]):
                    raise ValueError('Split ratios must satisfy 0 <= min <= max <= 1')
                kind = 'range' if match else 'ratios'
            else:
                raise ValueError(f'Unsupported Stage A split mode: {mode}')
            modes.append((weight, kind, value, mode))
        if not modes or abs(sum(mode[0] for mode in modes) - 100) > 1e-6:
            raise ValueError('Split percentages must sum to 100%')
        return modes

    def _sample_leaves(self, kind, value):
        if kind == 'level':
            return [(value, index) for index in range((4 * 2 ** value) ** 2)]
        ratio = self.rng.uniform(*value) if kind == 'range' else self.rng.choice(value)
        # FoveationNet counts splits among 16 + 64 internal nodes, rounding half up.
        leaves = {(0, index) for index in range(16)}
        frontier = sorted(leaves)
        for _ in range(int(ratio * 80 + 0.5)):
            level, index = frontier.pop(self.rng.randrange(len(frontier)))
            leaves.remove((level, index))
            row, col = divmod(index, 4 * 2 ** level)
            child_grid = 4 * 2 ** (level + 1)
            children = [(level + 1, (2 * row + dr) * child_grid + 2 * col + dc)
                        for dr in range(2) for dc in range(2)]
            leaves.update(children)
            if level == 0:
                frontier.extend(children)
        # Global bank ordering: level first, then row-major within each level.
        return sorted(leaves)

    def __call__(self, images_B_C_T_H_W):
        """Return packed patch metadata with a tree shared across the T frames."""
        B, C, T, H, W = images_B_C_T_H_W.shape
        if (C, H, W) != (3, 224, 224):
            raise ValueError('Stage A expects Bx3xTx224x224 images')
        modes = range(len(self.modes))
        self.last_mode_indices = (self.rng.choices(modes, [m[0] for m in self.modes], k=B)
                                  if len(self.modes) > 1 else [0] * B)
        trees = [self._sample_leaves(*self.modes[index][1:3]) for index in self.last_mode_indices]
        max_leaves = max(map(len, trees))
        indices_B_N = torch.full((B, max_leaves), -1, dtype=torch.long)
        levels_B_N = torch.zeros_like(indices_B_N)
        for b, leaves in enumerate(trees):
            indices_B_N[b, :len(leaves)] = torch.tensor([index + (0, 16, 80)[level]
                                                        for level, index in leaves])
            levels_B_N[b, :len(leaves)] = torch.tensor([level for level, _ in leaves])
        indices_B_T_N = indices_B_N.to(images_B_C_T_H_W.device)[:, None].expand(-1, T, -1)
        levels_B_T_N = levels_B_N.to(images_B_C_T_H_W.device)[:, None].expand(-1, T, -1)
        valid_B_T_N = indices_B_T_N >= 0

        # Match FoveationNet's successive area downsampling; fine patches are untouched.
        image_BT_C_H_W = images_B_C_T_H_W.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        pyramid = [image_BT_C_H_W]
        for _ in range(2):
            pyramid.append(F.interpolate(pyramid[-1], scale_factor=0.5, mode='area'))
        banks = []
        for image_BT_C_H_W in reversed(pyramid):
            grid = image_BT_C_H_W.shape[-1] // 14
            patches_BT_H_W_C_P_P = image_BT_C_H_W.reshape(B * T, C, grid, 14, grid, 14).permute(0, 2, 4, 1, 3, 5)
            banks.append(patches_BT_H_W_C_P_P.reshape(B, T, grid * grid, C, 14, 14))
        bank_B_T_K_C_P_P = torch.cat(banks, dim=2)
        gather_B_T_N_C_P_P = indices_B_T_N.clamp_min(0)[..., None, None, None].expand(-1, -1, -1, C, 14, 14)
        patches_B_T_N_C_P_P = bank_B_T_K_C_P_P.gather(2, gather_B_T_N_C_P_P)
        patches_B_T_N_C_P_P = patches_B_T_N_C_P_P * valid_B_T_N[..., None, None, None]
        return dict(patches_B_T_N_C_P_P=patches_B_T_N_C_P_P,
                    patch_bank_indices_B_T_N=indices_B_T_N, levels_B_T_N=levels_B_T_N,
                    token_valid_mask_B_T_N=valid_B_T_N)
