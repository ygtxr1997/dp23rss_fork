import os
from typing import Any

import torch
import torchvision
from PIL import Image

from diffusion_policy.model.roi.foveation_net import FoveationNet
from diffusion_policy.model.roi.jepa_helper import JEPAHelper


def _print_tensor(name: str, value: Any) -> None:
    """Print only tensor metadata so debugging does not trigger large copies."""
    if isinstance(value, torch.Tensor):
        print(
            f"{name}: shape={value.shape}, dtype={value.dtype}, "
            f"device={value.device}"
        )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _print_tensor(f"{name}[{index}]", item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _print_tensor(f"{name}.{key}", item)


class DebugFoveationNet(FoveationNet):
    """FoveationNet with shape prints around the non-module forward stages."""

    def _build_image_pyramid(self, x_high_B_C_T_H_W):
        _print_tensor("[2] x_high_B_C_T_H_W", x_high_B_C_T_H_W)
        image_pyramid = super()._build_image_pyramid(x_high_B_C_T_H_W)
        _print_tensor("[2] image_pyramid", image_pyramid)
        return image_pyramid

    def _patchify_pyramid(self, image_pyramid):
        patch_levels, patch_bank_B_T_K_C_P_P = super()._patchify_pyramid(
            image_pyramid
        )
        _print_tensor("[2] patch_levels", patch_levels)
        _print_tensor("[2] patch_bank_B_T_K_C_P_P", patch_bank_B_T_K_C_P_P)
        return patch_levels, patch_bank_B_T_K_C_P_P

    def _encode_global_images(self, global_images_B_C_T_H_W):
        _print_tensor("[3] global_images_B_C_T_H_W", global_images_B_C_T_H_W)
        global_features_B_T_N_D = super()._encode_global_images(
            global_images_B_C_T_H_W
        )
        _print_tensor("[3] global_features_B_T_N_D", global_features_B_T_N_D)
        return global_features_B_T_N_D

    def _build_router_condition(self, cond_B_T_D, reference_B_T_N_D):
        _print_tensor("[3] cond_B_T_D", cond_B_T_D)
        router_condition_B_T_D = super()._build_router_condition(
            cond_B_T_D,
            reference_B_T_N_D,
        )
        _print_tensor("[3] router_condition_B_T_D", router_condition_B_T_D)
        return router_condition_B_T_D

    def _sample_split_decisions(self, split_logits_B_T_M):
        _print_tensor("[3] split_logits_B_T_M", split_logits_B_T_M)
        split_soft_B_T_M, split_hard_B_T_M = super()._sample_split_decisions(
            split_logits_B_T_M
        )
        _print_tensor("[3] split_soft_B_T_M", split_soft_B_T_M)
        _print_tensor("[3] split_hard_B_T_M", split_hard_B_T_M)
        return split_soft_B_T_M, split_hard_B_T_M

    def _build_soft_and_hard_leaf_masks(
        self,
        split_soft_B_T_M,
        split_hard_B_T_M,
    ):
        masks = super()._build_soft_and_hard_leaf_masks(
            split_soft_B_T_M,
            split_hard_B_T_M,
        )
        _print_tensor("[4] soft_leaf_probs_B_T_K", masks[0])
        _print_tensor("[4] hard_leaf_mask_B_T_K", masks[1])
        _print_tensor("[4] st_leaf_mask_B_T_K", masks[2])
        return masks

    def _pack_selected_patches(
        self,
        patch_bank_B_T_K_C_P_P,
        hard_leaf_mask_B_T_K,
        st_leaf_mask_B_T_K,
    ):
        packed = super()._pack_selected_patches(
            patch_bank_B_T_K_C_P_P,
            hard_leaf_mask_B_T_K,
            st_leaf_mask_B_T_K,
        )
        _print_tensor("[5] packed", packed)
        return packed

    def _build_patch_metadata(
        self,
        selected_indices_B_T_N,
        token_valid_mask_B_T_N,
    ):
        metadata = super()._build_patch_metadata(
            selected_indices_B_T_N,
            token_valid_mask_B_T_N,
        )
        _print_tensor("[5] metadata", metadata)
        return metadata


def de_foveation_net() -> None:
    """
    Run one complete FoveationNet forward pass.

    Usage:
    export PYTHONPATH=~/code/dp23rss_fork/
    export TORCH_HOME=/home/geyuan/pretrained/torch
    CUDA_VISIBLE_DEVICES=0 python debug/de_foveation_net.py
    """
    torch.manual_seed(2)

    # Keep these values together so they are easy to edit during debugging.
    device = torch.device("cuda")
    torch_home = "/home/geyuan/pretrained/torch"  # where you downloaded JEPA ckpt
    model_name = "vjepa2_1_vit_base_384"
    batch_size = 1
    num_frames = 1
    input_hw = (384, 384)
    num_levels = 3
    global_level = 0
    router_cond_dim = 512

    model = DebugFoveationNet(
        input_hw=input_hw,
        num_levels=num_levels,
        global_level=global_level,
        global_encoder_type=model_name,
        torch_home=torch_home,
        global_feature_dim=JEPAHelper.MODEL_DIMS[model_name],
        router_layers=4,
        router_hidden_dim=512,
        router_heads=8,
        router_cond_dim=router_cond_dim,
        patch_size=16,
        freeze_global_encoder=True,
        freeze_router=False,
        target_num_patches=0.5,
        budget_loss_weight=0.01,
    ).to(device)
    model.eval()

    # The lowest pyramid level is 256x256 here, matching de_jepahelper's
    # confirmed T=1 JEPA input. Use uint8 to exercise the full input path.
    x_B_C_T_H_W = torch.randint(
        low=0,
        high=256,
        size=(batch_size, 3, num_frames, *input_hw),
        dtype=torch.uint8,
        device=device,
    )
    frames_T_H_W_C, _, info = torchvision.io.read_video(
        # "/home/geyuan/code/dp23rss_fork/debug/libero384_roi.mp4",
        "/home/geyuan/code/dp23rss_fork/debug/libero_button.png",
        pts_unit="sec",
    )  # uint8, [0,255]
    # frame_C_H_W = torchvision.io.read_image(
    #     "/home/geyuan/code/dp23rss_fork/debug/libero_button.png",
    # )  # uint8, [0,255]
    # frames_T_H_W_C = frame_C_H_W.permute(1, 2, 0).unsqueeze(0)  # [T,H,W,C]
    x_B_C_T_H_W = frames_T_H_W_C[:1].permute(3, 0, 1, 2).unsqueeze(0).to(device='cuda')

    cond_B_T_D = torch.randn(
        batch_size,
        num_frames,
        router_cond_dim,
        device=device,
    )
    _print_tensor("[1] x_B_C_T_H_W", x_B_C_T_H_W)
    _print_tensor("[1] cond_B_T_D", cond_B_T_D)

    # Hooks expose the raw JEPA token shape and every Router block output.
    hook_handles = []
    hook_handles.append(
        model.global_encoder.register_forward_hook(
            lambda module, inputs, output: _print_tensor(
                "[hook] raw_jepa_features_B_L_D",
                output,
            )
        )
    )
    for block_index, block in enumerate(model.router.blocks):
        hook_handles.append(
            block.register_forward_hook(
                lambda module, inputs, output, index=block_index: _print_tensor(
                    f"[hook] router_block_{index}_tokens_BT_N_D",
                    output,
                )
            )
        )

    try:
        with torch.no_grad():
            foveation_out = model(x_B_C_T_H_W, cond_B_T_D)
    finally:
        for handle in hook_handles:
            handle.remove()

    _print_tensor("[out] foveation_out", foveation_out)
    print("num_patches_B_T:", foveation_out["num_patches_B_T"].cpu())
    print("budget_loss:", foveation_out["budget_loss"].item())

    # Render the selected native-resolution patches with one color per
    # pyramid level.  Save the first batch/frame as a PNG for quick inspection.
    vis_B_T_H_W_C = model.vis_ar_tokens(
        foveation_out,
        alpha=0.1,
        border_width=1,
    )
    vis_path = os.path.join(os.path.dirname(__file__), "tmp_vis_ar_patches.png")
    Image.fromarray(vis_B_T_H_W_C[0, 0]).save(vis_path)
    print("vis_ar_tokens:", vis_B_T_H_W_C.shape, vis_B_T_H_W_C.dtype)
    print("saved visualization:", vis_path)
    print("OK!")


if __name__ == "__main__":
    de_foveation_net()
