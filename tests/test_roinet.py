import os
import sys

import torch
import torch.nn as nn
import pytest

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.model.roi.roinet import (
    RoiNet,
    _ResidualBlock,
)


def test_roinet_forward_shapes_and_optional_outputs():
    torch.manual_seed(0)
    model = RoiNet(in_channels=3)
    x_B_C_T_H_W = torch.randn(2, 3, 4, 480, 640)

    out = model(
        x_B_C_T_H_W,
        return_grid=True,
        return_features=True,
    )

    assert set(out.keys()) == {
        "roi_B_C_T_RH_RW",
        "theta_B_T_2_3",
        "params_B_T_3",
        "valid_mask_B_T_1_RH_RW",
        "grid_B_T_RH_RW_2",
        "locator_features_B_T_D",
    }
    assert out["roi_B_C_T_RH_RW"].shape == (2, 3, 4, 256, 256)
    assert out["theta_B_T_2_3"].shape == (2, 4, 2, 3)
    assert out["params_B_T_3"].shape == (2, 4, 3)
    assert out["valid_mask_B_T_1_RH_RW"].shape == (2, 4, 1, 256, 256)
    assert out["grid_B_T_RH_RW_2"].shape == (2, 4, 256, 256, 2)
    assert out["locator_features_B_T_D"].shape == (
        2,
        4,
        model.locator_feature_dim,
    )


def test_roinet_locator_uses_residual_conv_blocks():
    torch.manual_seed(0)
    x_B_C_T_H_W = torch.randn(2, 3, 4, 96, 96)

    model = RoiNet(
        in_channels=3,
        preset="tiny",
        locator_type="residual",
        locator_input_size=(32, 32),
    )

    out = model(x_B_C_T_H_W)
    locator_blocks = [
        module
        for module in model.locator_cnn
        if isinstance(module, _ResidualBlock)
    ]
    locator_norms = [
        module
        for module in model.locator_cnn.modules()
        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm))
    ]

    assert out["roi_B_C_T_RH_RW"].shape == (2, 3, 4, 256, 256)
    assert len(locator_blocks) == sum(model.blocks_per_stage)
    assert all(isinstance(module, nn.GroupNorm) for module in locator_norms)
    assert model.parameter_summary()["total"] > 0


def test_roinet_locator_uses_resnet_blocks_with_group_norm_and_silu():
    torch.manual_seed(0)
    x_B_C_T_H_W = torch.randn(2, 3, 4, 96, 96)

    model = RoiNet(
        in_channels=3,
        roi_size=(64, 64),
        preset="tiny",
        locator_type="resnet",
        base_channels=8,
        locator_input_size=(32, 32),
        hidden_dim=16,
    )

    out = model(
        x_B_C_T_H_W,
        return_features=True,
    )
    locator_blocks = [
        module
        for module in model.locator_cnn.modules()
        if isinstance(module, _ResidualBlock)
    ]
    locator_norms = [
        module
        for module in model.locator_cnn.modules()
        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm))
    ]
    locator_activations = [
        module
        for module in model.locator_cnn.modules()
        if isinstance(module, nn.SiLU)
    ]

    assert model.locator_type == "resnet"
    assert model.num_stages == 4
    assert model.blocks_per_stage == (1, 1, 1, 1)
    assert model.locator_feature_dim == 64
    assert out["roi_B_C_T_RH_RW"].shape == (2, 3, 4, 64, 64)
    assert out["locator_features_B_T_D"].shape == (2, 4, 64)
    assert len(locator_blocks) == 4
    assert len(locator_norms) > 0
    assert all(isinstance(module, nn.GroupNorm) for module in locator_norms)
    assert len(locator_activations) > 0
    assert model.parameter_summary()["total"] > 0


def test_roinet_resnet_preset_configures_depth():
    tiny = RoiNet(
        in_channels=3,
        preset="tiny",
        locator_type="resnet",
        base_channels=8,
        hidden_dim=16,
    )
    small = RoiNet(
        in_channels=3,
        preset="small",
        locator_type="resnet",
        base_channels=8,
        hidden_dim=16,
    )
    medium = RoiNet(
        in_channels=3,
        preset="medium",
        locator_type="resnet",
        base_channels=8,
        hidden_dim=16,
    )

    assert tiny.blocks_per_stage == (1, 1, 1, 1)
    assert small.blocks_per_stage == (2, 2, 2, 2)
    assert medium.blocks_per_stage == (3, 4, 6, 3)
    assert sum(tiny.blocks_per_stage) == 4
    assert sum(small.blocks_per_stage) == 8
    assert sum(medium.blocks_per_stage) == 16
    assert tiny.locator_feature_dim == 64
    assert small.locator_feature_dim == 64
    assert medium.locator_feature_dim == 64


def test_roinet_resnet_rejects_residual_depth_args():
    with pytest.raises(ValueError, match="num_stages"):
        RoiNet(locator_type="resnet", num_stages=4)

    with pytest.raises(ValueError, match="blocks_per_stage"):
        RoiNet(locator_type="resnet", blocks_per_stage=(2, 2, 2, 2))


def test_roinet_default_output_omits_large_optional_tensors():
    model = RoiNet(in_channels=3, preset="tiny", locator_input_size=(32, 32))
    x_B_C_T_H_W = torch.randn(1, 3, 2, 64, 80)

    out = model(x_B_C_T_H_W)

    assert set(out.keys()) == {
        "roi_B_C_T_RH_RW",
        "theta_B_T_2_3",
        "params_B_T_3",
        "valid_mask_B_T_1_RH_RW",
    }


def test_roinet_locator_configures_blocks_per_stage():
    model = RoiNet(
        in_channels=3,
        preset="tiny",
        locator_type="residual",
        num_stages=3,
        blocks_per_stage=(2, 3, 1),
        locator_input_size=(64, 64),
    )

    stage_strides = [
        module.main[0].stride
        for module in model.locator_cnn
        if hasattr(module, "main")
    ]

    assert model.num_stages == 3
    assert model.blocks_per_stage == (2, 3, 1)
    assert stage_strides == [
        (4, 4),
        (2, 2),
        (4, 4),
        (2, 2),
        (2, 2),
        (4, 4),
    ]


def test_roinet_initializes_to_near_full_crop_with_trainable_center_room():
    torch.manual_seed(2)
    model = RoiNet(in_channels=3, preset="tiny", locator_input_size=(32, 32))
    x_B_C_T_H_W = torch.randn(2, 3, 4, 64, 80)

    out = model(x_B_C_T_H_W)

    torch.testing.assert_close(
        out["theta_B_T_2_3"][..., 0, 1],
        torch.zeros(2, 4),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        out["theta_B_T_2_3"][..., 1, 0],
        torch.zeros(2, 4),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        out["params_B_T_3"][..., 2],
        torch.full((2, 4), model.init_scale),
        rtol=0.0,
        atol=1e-3,
    )
    assert out["params_B_T_3"][..., :2].abs().max().item() < 1e-3
    assert model.init_scale < model.max_scale


def test_roinet_xys_large_center_raw_std_spreads_initial_centers():
    torch.manual_seed(12)
    model = RoiNet(
        in_channels=3,
        roi_size=(32, 32),
        preset="tiny",
        base_channels=8,
        num_stages=2,
        locator_input_size=(32, 32),
        hidden_dim=32,
        init_scale=0.5,
        init_center_raw_std=0.9,
    )
    x_B_C_T_H_W = torch.randn(4, 3, 4, 64, 80)

    out = model(x_B_C_T_H_W)

    assert out["params_B_T_3"][..., :2].abs().max().item() > 0.25
    torch.testing.assert_close(
        out["params_B_T_3"][..., 2],
        torch.full((4, 4), model.init_scale),
        rtol=0.0,
        atol=1e-3,
    )


def test_roinet_backward_reaches_input_and_model_params():
    torch.manual_seed(1)
    model = RoiNet(
        in_channels=3,
        roi_size=(32, 32),
        preset="tiny",
        base_channels=8,
        num_stages=2,
        locator_input_size=(32, 32),
        hidden_dim=16,
        detach_input_grad=False,
    )
    x_B_C_T_H_W = torch.randn(2, 3, 2, 64, 80, requires_grad=True)

    out = model(x_B_C_T_H_W)
    out["roi_B_C_T_RH_RW"].mean().backward()

    assert x_B_C_T_H_W.grad is not None
    assert torch.isfinite(x_B_C_T_H_W.grad).all()
    param_grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(param_grads) > 0
    assert any(grad.abs().sum().item() > 0.0 for grad in param_grads)
    locator_cnn_grads = [
        p.grad
        for p in model.locator_cnn.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(locator_cnn_grads) > 0
    assert any(grad.abs().sum().item() > 0.0 for grad in locator_cnn_grads)


def test_roinet_constrained_params_stay_in_bounds():
    model = RoiNet(in_channels=3, min_scale=0.25, max_scale=0.9)
    raw_params_BT_3 = torch.randn(128, 3) * 8.0

    params_BT_3 = model._constrain_raw_params(raw_params_BT_3)
    center_x_BT = params_BT_3[:, 0]
    center_y_BT = params_BT_3[:, 1]
    scale_BT = params_BT_3[:, 2]

    assert torch.all(scale_BT >= model.min_scale)
    assert torch.all(scale_BT <= model.max_scale)
    assert torch.all(center_x_BT >= -1.0)
    assert torch.all(center_x_BT <= 1.0)
    assert torch.all(center_y_BT >= -1.0)
    assert torch.all(center_y_BT <= 1.0)


def test_roinet_logit_scale_forward_shapes_and_head_dim():
    torch.manual_seed(3)
    model = RoiNet(
        in_channels=3,
        roi_size=(32, 32),
        preset="tiny",
        base_channels=8,
        num_stages=2,
        locator_input_size=(32, 48),
        hidden_dim=16,
        predict_type="logit_scale",
        patchify_wh=16,
    )
    x_B_C_T_H_W = torch.randn(2, 3, 2, 64, 80)

    out = model(x_B_C_T_H_W)

    assert model.patch_grid_h == 2
    assert model.patch_grid_w == 3
    assert model.locator_head[-1].out_features == 7
    assert out["roi_B_C_T_RH_RW"].shape == (2, 3, 2, 32, 32)
    assert out["params_B_T_3"].shape == (2, 2, 3)
    assert out["valid_mask_B_T_1_RH_RW"].shape == (2, 2, 1, 32, 32)
    assert torch.all(out["params_B_T_3"][..., :2] >= -1.0)
    assert torch.all(out["params_B_T_3"][..., :2] <= 1.0)
    assert torch.all(out["params_B_T_3"][..., 2] >= model.min_scale)
    assert torch.all(out["params_B_T_3"][..., 2] <= model.max_scale)


def test_roinet_logit_scale_initializes_center_and_scale():
    torch.manual_seed(4)
    model = RoiNet(
        in_channels=3,
        roi_size=(16, 16),
        preset="tiny",
        base_channels=8,
        num_stages=1,
        locator_input_size=(32, 32),
        hidden_dim=16,
        predict_type="logit_scale",
        patchify_wh=16,
    )
    x_B_C_T_H_W = torch.randn(2, 3, 3, 32, 32)

    out = model(x_B_C_T_H_W)

    assert out["params_B_T_3"][..., :2].abs().max().item() < 1e-3
    torch.testing.assert_close(
        out["params_B_T_3"][..., 2],
        torch.full((2, 3), model.init_scale),
        rtol=0.0,
        atol=1e-3,
    )


def test_roinet_logit_scale_patch_logit_raw_std_spreads_initial_centers():
    torch.manual_seed(22)
    model = RoiNet(
        in_channels=3,
        roi_size=(24, 24),
        preset="tiny",
        base_channels=8,
        num_stages=2,
        locator_input_size=(96, 96),
        hidden_dim=32,
        predict_type="logit_scale",
        patchify_wh=16,
        init_scale=0.5,
        init_patch_logit_raw_std=2.0,
    )
    x_B_C_T_H_W = torch.randn(4, 3, 4, 96, 96)

    out = model(x_B_C_T_H_W)
    centers_B_T_2 = out["params_B_T_3"][..., :2]

    assert model.patch_grid_h == 6
    assert model.patch_grid_w == 6
    assert centers_B_T_2.abs().max().item() > 0.1
    assert centers_B_T_2.std().item() > 0.04
    torch.testing.assert_close(
        out["params_B_T_3"][..., 2],
        torch.full((4, 4), model.init_scale),
        rtol=0.0,
        atol=1e-3,
    )


def test_roinet_logit_scale_backward_reaches_locator_params():
    torch.manual_seed(5)
    model = RoiNet(
        in_channels=3,
        roi_size=(24, 24),
        preset="tiny",
        base_channels=8,
        num_stages=2,
        locator_input_size=(32, 32),
        hidden_dim=16,
        predict_type="logit_scale",
        patchify_wh=16,
        detach_input_grad=False,
    )
    x_B_C_T_H_W = torch.randn(2, 3, 2, 48, 48, requires_grad=True)

    out = model(x_B_C_T_H_W)
    out["roi_B_C_T_RH_RW"].mean().backward()

    assert x_B_C_T_H_W.grad is not None
    assert torch.isfinite(x_B_C_T_H_W.grad).all()
    param_grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(param_grads) > 0
    assert any(grad.abs().sum().item() > 0.0 for grad in param_grads)


def test_roinet_logit_logit_forward_shapes_and_head_dim():
    torch.manual_seed(23)
    model = RoiNet(
        in_channels=3,
        roi_size=(32, 32),
        preset="tiny",
        base_channels=8,
        num_stages=2,
        locator_input_size=(32, 48),
        hidden_dim=16,
        min_scale=0.25,
        max_scale=0.9,
        predict_type="logit_logit",
        patchify_wh=16,
        scale_num_bins=64,
    )
    x_B_C_T_H_W = torch.randn(2, 3, 2, 64, 80)

    out = model(x_B_C_T_H_W)

    assert model.patch_grid_h == 2
    assert model.patch_grid_w == 3
    assert model.locator_head[-1].out_features == 70
    assert out["roi_B_C_T_RH_RW"].shape == (2, 3, 2, 32, 32)
    assert out["params_B_T_3"].shape == (2, 2, 3)
    assert out["valid_mask_B_T_1_RH_RW"].shape == (2, 2, 1, 32, 32)
    assert torch.all(out["params_B_T_3"][..., :2] >= -1.0)
    assert torch.all(out["params_B_T_3"][..., :2] <= 1.0)
    assert torch.all(out["params_B_T_3"][..., 2] >= model.min_scale)
    assert torch.all(out["params_B_T_3"][..., 2] <= model.max_scale)
    torch.testing.assert_close(
        model._scale_bins_S,
        torch.linspace(model.min_scale, model.max_scale, steps=64),
    )


def test_roinet_logit_logit_initializes_center_and_scale():
    torch.manual_seed(24)
    model = RoiNet(
        in_channels=3,
        roi_size=(16, 16),
        preset="tiny",
        base_channels=8,
        num_stages=1,
        locator_input_size=(32, 32),
        hidden_dim=16,
        min_scale=0.2,
        max_scale=1.0,
        init_scale=0.85,
        predict_type="logit_logit",
        patchify_wh=16,
    )
    x_B_C_T_H_W = torch.randn(2, 3, 3, 32, 32)

    out = model(x_B_C_T_H_W)

    assert out["params_B_T_3"][..., :2].abs().max().item() < 1e-3
    torch.testing.assert_close(
        out["params_B_T_3"][..., 2],
        torch.full((2, 3), model.init_scale),
        rtol=0.0,
        atol=1e-3,
    )


def test_roinet_logit_logit_scale_logit_raw_std_spreads_initial_scales():
    torch.manual_seed(25)
    model = RoiNet(
        in_channels=3,
        roi_size=(24, 24),
        preset="tiny",
        base_channels=8,
        num_stages=2,
        locator_input_size=(96, 96),
        hidden_dim=32,
        min_scale=0.2,
        max_scale=1.0,
        init_scale=0.5,
        predict_type="logit_logit",
        patchify_wh=16,
        init_scale_logit_raw_std=4.0,
    )
    x_B_C_T_H_W = torch.randn(4, 3, 4, 96, 96)

    out = model(x_B_C_T_H_W)
    scales_B_T = out["params_B_T_3"][..., 2]

    assert torch.all(scales_B_T >= model.min_scale)
    assert torch.all(scales_B_T <= model.max_scale)
    assert (scales_B_T - model.init_scale).abs().max().item() > 0.01
    assert scales_B_T.std().item() > 0.005


def test_roinet_logit_logit_backward_reaches_locator_params():
    torch.manual_seed(26)
    model = RoiNet(
        in_channels=3,
        roi_size=(24, 24),
        preset="tiny",
        base_channels=8,
        num_stages=2,
        locator_input_size=(32, 32),
        hidden_dim=16,
        predict_type="logit_logit",
        patchify_wh=16,
        detach_input_grad=False,
    )
    x_B_C_T_H_W = torch.randn(2, 3, 2, 48, 48, requires_grad=True)

    out = model(x_B_C_T_H_W)
    out["roi_B_C_T_RH_RW"].mean().backward()

    assert x_B_C_T_H_W.grad is not None
    assert torch.isfinite(x_B_C_T_H_W.grad).all()
    param_grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(param_grads) > 0
    assert any(grad.abs().sum().item() > 0.0 for grad in param_grads)


def test_roinet_logit_meanstd_samples_scale_only_in_train_mode():
    torch.manual_seed(27)
    model = RoiNet(
        in_channels=3,
        roi_size=(24, 24),
        preset="tiny",
        base_channels=8,
        num_stages=2,
        locator_input_size=(32, 48),
        hidden_dim=16,
        min_scale=0.05,
        max_scale=0.8,
        init_scale=0.2,
        predict_type="logit_meanstd",
        patchify_wh=16,
    )
    x_B_C_T_H_W = torch.randn(2, 3, 2, 48, 64)

    assert model.patch_grid_h == 2
    assert model.patch_grid_w == 3
    assert model.locator_head[-1].out_features == 8

    model.train()
    out_a = model(x_B_C_T_H_W)
    out_b = model(x_B_C_T_H_W)

    expected_keys = {
        "roi_B_C_T_RH_RW",
        "theta_B_T_2_3",
        "params_B_T_3",
        "params_mean_B_T_3",
        "raw_scale_mu_B_T",
        "raw_scale_logstd_B_T",
        "raw_scale_std_B_T",
        "valid_mask_B_T_1_RH_RW",
    }
    assert set(out_a.keys()) == expected_keys
    assert out_a["params_B_T_3"].shape == (2, 2, 3)
    assert out_a["params_mean_B_T_3"].shape == (2, 2, 3)
    assert not torch.allclose(
        out_a["params_B_T_3"][..., 2],
        out_b["params_B_T_3"][..., 2],
    )
    assert torch.all(out_a["params_B_T_3"][..., 2] >= model.min_scale)
    assert torch.all(out_a["params_B_T_3"][..., 2] <= model.max_scale)
    assert torch.all(out_a["params_mean_B_T_3"][..., 2] >= model.min_scale)
    assert torch.all(out_a["params_mean_B_T_3"][..., 2] <= model.max_scale)
    torch.testing.assert_close(
        out_a["params_mean_B_T_3"][..., 2],
        torch.full((2, 2), model.init_scale),
        rtol=0.0,
        atol=1e-3,
    )

    model.eval()
    out_eval_a = model(x_B_C_T_H_W)
    out_eval_b = model(x_B_C_T_H_W)
    torch.testing.assert_close(out_eval_a["params_B_T_3"], out_eval_b["params_B_T_3"])
    torch.testing.assert_close(
        out_eval_a["params_B_T_3"],
        out_eval_a["params_mean_B_T_3"],
    )


def test_roinet_logit_meanstd_backward_reaches_mu_and_logstd_head_params():
    torch.manual_seed(28)
    model = RoiNet(
        in_channels=3,
        roi_size=(24, 24),
        preset="tiny",
        base_channels=8,
        num_stages=2,
        locator_input_size=(32, 32),
        hidden_dim=16,
        min_scale=0.05,
        max_scale=0.8,
        init_scale=0.2,
        predict_type="logit_meanstd",
        patchify_wh=16,
        detach_input_grad=False,
    )
    model.train()
    x_B_C_T_H_W = torch.randn(2, 3, 2, 48, 48, requires_grad=True)

    out = model(x_B_C_T_H_W)
    (
        out["roi_B_C_T_RH_RW"].mean()
        + out["params_mean_B_T_3"][..., 2].mean()
        + out["raw_scale_std_B_T"].mean()
    ).backward()

    last_linear = model.locator_head[-1]
    mu_idx = model.patch_grid_size
    logstd_idx = model.patch_grid_size + 1
    assert last_linear.weight.grad is not None
    assert last_linear.bias.grad is not None
    assert last_linear.weight.grad[mu_idx].abs().sum().item() > 0.0
    assert last_linear.weight.grad[logstd_idx].abs().sum().item() > 0.0
    assert last_linear.bias.grad[mu_idx].abs().item() > 0.0
    assert last_linear.bias.grad[logstd_idx].abs().item() > 0.0


def test_roinet_out_of_bounds_crop_masks_invalid_samples_without_moving_center():
    model = RoiNet(
        in_channels=1,
        roi_size=(8, 8),
        preset="tiny",
        base_channels=8,
        num_stages=1,
        locator_input_size=(16, 16),
        hidden_dim=16,
        min_scale=1.0,
        max_scale=1.0,
        padding_mode="border",
    )
    with torch.no_grad():
        model.locator_head[-1].weight.zero_()
        model.locator_head[-1].bias.zero_()
        model.locator_head[-1].bias[0] = torch.atanh(torch.tensor(0.8))
    x_B_C_T_H_W = torch.ones(1, 1, 1, 16, 16)

    out = model(x_B_C_T_H_W)
    valid_mask = out["valid_mask_B_T_1_RH_RW"]
    roi = out["roi_B_C_T_RH_RW"]

    torch.testing.assert_close(
        out["params_B_T_3"][..., 0],
        torch.full((1, 1), 0.8),
    )
    assert valid_mask.logical_not().any()
    assert torch.all(roi[~valid_mask.permute(0, 2, 1, 3, 4).expand_as(roi)] == 0.0)


def test_roinet_rejects_invalid_predict_config():
    with pytest.raises(ValueError, match="predict_type"):
        RoiNet(predict_type="bad")

    with pytest.raises(ValueError, match="patchify_wh"):
        RoiNet(patchify_wh=0)

    with pytest.raises(ValueError, match="divisible"):
        RoiNet(
            predict_type="logit_scale",
            locator_input_size=(30, 32),
            patchify_wh=16,
        )

    with pytest.raises(ValueError, match="divisible"):
        RoiNet(
            predict_type="logit_logit",
            locator_input_size=(30, 32),
            patchify_wh=16,
        )

    with pytest.raises(ValueError, match="scale_num_bins"):
        RoiNet(scale_num_bins=0)


def test_roinet_rejects_negative_init_randomization_stds():
    with pytest.raises(ValueError, match="init_head_weight_std"):
        RoiNet(init_head_weight_std=-1e-4)

    with pytest.raises(ValueError, match="init_center_raw_std"):
        RoiNet(init_center_raw_std=-0.1)

    with pytest.raises(ValueError, match="init_patch_logit_raw_std"):
        RoiNet(
            predict_type="logit_scale",
            locator_input_size=(32, 32),
            init_patch_logit_raw_std=-0.1,
        )

    with pytest.raises(ValueError, match="init_scale_logit_raw_std"):
        RoiNet(
            predict_type="logit_logit",
            locator_input_size=(32, 32),
            init_scale_logit_raw_std=-0.1,
        )


def test_roinet_area_loss_returns_percentage_area():
    params_B_T_3 = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
            [[0.0, 0.0, 0.25], [0.0, 0.0, 0.75]],
        ],
        requires_grad=True,
    )
    expected_area_B_T = torch.tensor(
        [[1.0, 0.25], [0.0625, 0.5625]],
        dtype=params_B_T_3.dtype,
    )

    area_B_T = RoiNet.area_loss(params_B_T_3, reduction="none")
    torch.testing.assert_close(area_B_T, expected_area_B_T)
    torch.testing.assert_close(
        RoiNet.area_loss(params_B_T_3, reduction="sum", mode="square"),
        expected_area_B_T.sum(),
    )
    torch.testing.assert_close(
        RoiNet.area_loss({"params_B_T_3": params_B_T_3}),
        expected_area_B_T.mean(),
    )

    RoiNet.area_loss(params_B_T_3).backward()
    assert params_B_T_3.grad is not None
    assert torch.isfinite(params_B_T_3.grad).all()


def test_roinet_area_loss_identity_mode_returns_scale():
    params_B_T_3 = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
            [[0.0, 0.0, 0.25], [0.0, 0.0, 0.75]],
        ],
        requires_grad=True,
    )
    expected_scale_B_T = torch.tensor(
        [[1.0, 0.5], [0.25, 0.75]],
        dtype=params_B_T_3.dtype,
    )

    scale_B_T = RoiNet.area_loss(
        params_B_T_3,
        reduction="none",
        mode="identity",
    )

    torch.testing.assert_close(scale_B_T, expected_scale_B_T)
    torch.testing.assert_close(
        RoiNet.area_loss(params_B_T_3, mode="identity"),
        expected_scale_B_T.mean(),
    )
    torch.testing.assert_close(
        RoiNet.area_loss(params_B_T_3, reduction="sum", mode="identity"),
        expected_scale_B_T.sum(),
    )
    RoiNet.area_loss(params_B_T_3, mode="identity").backward()
    assert params_B_T_3.grad is not None
    assert torch.isfinite(params_B_T_3.grad).all()


def test_roinet_area_loss_upper_mode_penalizes_only_large_scale():
    params_B_T_3 = torch.tensor(
        [
            [[0.0, 0.0, 0.4], [0.0, 0.0, 0.7]],
            [[0.0, 0.0, 0.8], [0.0, 0.0, 1.0]],
        ],
        requires_grad=True,
    )
    expected_B_T = torch.tensor(
        [[0.0, 0.01], [0.04, 0.16]],
        dtype=params_B_T_3.dtype,
    )

    upper_B_T = RoiNet.area_loss(
        params_B_T_3,
        reduction="none",
        mode="upper",
        upper_bound=0.6,
    )

    torch.testing.assert_close(upper_B_T, expected_B_T)
    torch.testing.assert_close(
        RoiNet.area_loss(params_B_T_3, mode="upper", upper_bound=0.6),
        expected_B_T.mean(),
    )
    RoiNet.area_loss(params_B_T_3, mode="upper", upper_bound=0.6).backward()
    assert params_B_T_3.grad is not None
    assert torch.isfinite(params_B_T_3.grad).all()


def test_roinet_area_loss_range_mode_penalizes_only_out_of_range_scale():
    params_B_T_3 = torch.tensor(
        [
            [[0.0, 0.0, 0.03], [0.0, 0.0, 0.2]],
            [[0.0, 0.0, 0.8], [0.0, 0.0, 0.9]],
        ],
        requires_grad=True,
    )
    expected_B_T = torch.tensor(
        [[0.0004, 0.0], [0.0, 0.01]],
        dtype=params_B_T_3.dtype,
    )

    range_B_T = RoiNet.area_loss(
        params_B_T_3,
        reduction="none",
        mode="range",
        lower_bound=0.05,
        upper_bound=0.8,
    )

    torch.testing.assert_close(range_B_T, expected_B_T)
    torch.testing.assert_close(
        RoiNet.area_loss(
            params_B_T_3,
            mode="range",
            lower_bound=0.05,
            upper_bound=0.8,
        ),
        expected_B_T.mean(),
    )
    RoiNet.area_loss(
        params_B_T_3,
        mode="range",
        lower_bound=0.05,
        upper_bound=0.8,
    ).backward()
    assert params_B_T_3.grad is not None
    assert torch.isfinite(params_B_T_3.grad).all()


def test_roinet_area_loss_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        RoiNet.area_loss(torch.zeros(2, 1, 3), mode="invalid")


def test_roinet_area_loss_upper_mode_requires_upper_bound():
    with pytest.raises(ValueError, match="upper_bound"):
        RoiNet.area_loss(torch.zeros(2, 1, 3), mode="upper")


def test_roinet_area_loss_range_mode_requires_bounds():
    with pytest.raises(ValueError, match="lower_bound"):
        RoiNet.area_loss(torch.zeros(2, 1, 3), mode="range", upper_bound=0.8)

    with pytest.raises(ValueError, match="lower_bound"):
        RoiNet.area_loss(
            torch.zeros(2, 1, 3),
            mode="range",
            lower_bound=0.9,
            upper_bound=0.8,
        )


def test_roinet_center_diversity_loss_compares_batch_not_time():
    params_B_T_3 = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        ],
        requires_grad=True,
    )

    loss = RoiNet.center_diversity_loss(params_B_T_3, margin=0.25, eps=1e-6)

    torch.testing.assert_close(loss, torch.tensor((0.25 - 1e-3) ** 2))
    loss.backward()
    assert params_B_T_3.grad is not None
    assert torch.isfinite(params_B_T_3.grad).all()


def test_roinet_center_diversity_loss_returns_zero_for_single_batch():
    params_B_T_3 = torch.tensor([[[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]]])

    loss = RoiNet.center_diversity_loss(params_B_T_3, margin=0.25)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_roinet_center_diversity_loss_zero_when_pairs_exceed_margin():
    params_B_T_3 = torch.tensor(
        [
            [[-0.5, 0.0, 1.0]],
            [[0.5, 0.0, 1.0]],
        ]
    )

    loss = RoiNet.center_diversity_loss(params_B_T_3, margin=0.25)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_roinet_center_diversity_loss_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        RoiNet.center_diversity_loss(torch.zeros(2, 3))

    with pytest.raises(ValueError, match="margin"):
        RoiNet.center_diversity_loss(torch.zeros(2, 1, 3), margin=-0.1)


def test_roinet_scale_diversity_loss_compares_batch_not_time():
    params_B_T_3 = torch.tensor(
        [
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.9]],
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.9]],
        ],
        requires_grad=True,
    )

    loss = RoiNet.scale_diversity_loss(params_B_T_3, margin=0.05, eps=1e-6)

    torch.testing.assert_close(loss, torch.tensor((0.05 - 1e-3) ** 2))
    loss.backward()
    assert params_B_T_3.grad is not None
    assert torch.isfinite(params_B_T_3.grad).all()


def test_roinet_scale_diversity_loss_returns_zero_for_single_batch():
    params_B_T_3 = torch.tensor([[[0.0, 0.0, 0.5], [0.0, 0.0, 0.9]]])

    loss = RoiNet.scale_diversity_loss(params_B_T_3, margin=0.05)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_roinet_scale_diversity_loss_zero_when_pairs_exceed_margin():
    params_B_T_3 = torch.tensor(
        [
            [[0.0, 0.0, 0.4]],
            [[0.0, 0.0, 0.6]],
        ]
    )

    loss = RoiNet.scale_diversity_loss(params_B_T_3, margin=0.05)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_roinet_scale_diversity_loss_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        RoiNet.scale_diversity_loss(torch.zeros(2, 3))

    with pytest.raises(ValueError, match="margin"):
        RoiNet.scale_diversity_loss(torch.zeros(2, 1, 3), margin=-0.1)


def test_roinet_scale_batch_std_loss_penalizes_low_batch_std_per_time():
    params_B_T_3 = torch.tensor(
        [
            [[0.0, 0.0, 0.49], [0.0, 0.0, 0.5]],
            [[0.0, 0.0, 0.51], [0.0, 0.0, 0.7]],
        ],
        requires_grad=True,
    )

    loss = RoiNet.scale_batch_std_loss(
        params_B_T_3,
        std_target=0.15,
        eps=0.0,
    )

    torch.testing.assert_close(loss, torch.tensor((0.14 ** 2 + 0.05 ** 2) / 2.0))
    loss.backward()
    assert params_B_T_3.grad is not None
    assert torch.isfinite(params_B_T_3.grad).all()


def test_roinet_scale_batch_std_loss_zero_when_std_exceeds_target():
    params_B_T_3 = torch.tensor(
        [
            [[0.0, 0.0, 0.3]],
            [[0.0, 0.0, 0.7]],
        ]
    )

    loss = RoiNet.scale_batch_std_loss(params_B_T_3, std_target=0.1, eps=0.0)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_roinet_scale_batch_std_loss_returns_zero_for_single_batch():
    params_B_T_3 = torch.tensor([[[0.0, 0.0, 0.5], [0.0, 0.0, 0.9]]])

    loss = RoiNet.scale_batch_std_loss(params_B_T_3, std_target=0.1)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_roinet_scale_batch_mean_std_target_loss_matches_formula():
    params_B_T_3 = torch.tensor(
        [
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]],
            [[0.0, 0.0, 0.3], [0.0, 0.0, 0.6]],
        ],
        requires_grad=True,
    )

    loss = RoiNet.scale_batch_mean_std_target_loss(
        params_B_T_3,
        mean_target=0.2,
        std_target=0.1,
        eps=0.0,
    )

    expected = torch.tensor((0.0 ** 2 + 0.0 ** 2 + 0.2 ** 2 + 0.1 ** 2) / 2.0)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert params_B_T_3.grad is not None
    assert torch.isfinite(params_B_T_3.grad).all()


def test_roinet_scale_batch_std_loss_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        RoiNet.scale_batch_std_loss(torch.zeros(2, 3))

    with pytest.raises(ValueError, match="std_target"):
        RoiNet.scale_batch_std_loss(torch.zeros(2, 1, 3), std_target=-0.1)


def test_roinet_speed_test_smoke():
    result = RoiNet.speed_test(
        batch_size=1,
        channels=3,
        time_steps=1,
        height=48,
        width=64,
        roi_size=(24, 24),
        iterations=1,
        warmup=0,
        device="cpu",
        preset="tiny",
        base_channels=8,
        num_stages=1,
        locator_input_size=(24, 24),
        hidden_dim=16,
    )

    assert result["ms_per_iter"] > 0.0
    assert result["fps"] > 0.0
    assert result["parameter_count"] > 0
    assert result["input_shape"] == (1, 3, 1, 48, 64)
    assert result["output_shape"] == (1, 3, 1, 24, 24)
    assert result["device"] == "cpu"
