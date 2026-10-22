"""Independent camera routing with shared projected low-dimensional conditions."""

import copy
from typing import Dict, Optional, Sequence

import torch
from torch import nn

from diffusion_policy.model.roi.foveation_net import FoveationNet


class MultiViewFoveationNet(nn.Module):
    """Organize single-camera trees, interventions and local budget statistics."""

    def __init__(self, view_keys: Sequence[str], template: FoveationNet):
        """Clone the initialized template without sharing parameters across cameras."""
        super().__init__()
        self.view_keys = tuple(view_keys)
        self.nets = nn.ModuleDict({
            key: template if index == 0 else copy.deepcopy(template)
            for index, key in enumerate(self.view_keys)
        })
        # Frozen pretrained encoders are camera-agnostic and should occupy one
        # parameter/buffer set.  Keep the shared module registered only under
        # the first camera so state_dict() serializes it once; camera routers
        # remain independent.
        # FPN encoders were trained per camera in DP; freezing must not merge them.
        if (template.freeze_global_encoder and template.global_encoder is not None
                and template.router_type != "node_aligned_fpn"):
            shared_encoder = self.nets[self.view_keys[0]].global_encoder
            for key in self.view_keys[1:]:
                net = self.nets[key]
                net._modules.pop('global_encoder', None)
                object.__setattr__(net, 'global_encoder', shared_encoder)
        # Geometry and targets are common configuration, not a fallback camera.
        self.num_levels = template.num_levels
        self.num_bank_patches = template.num_bank_patches
        self.min_num_patches = template.min_num_patches
        self.max_num_patches = template.max_num_patches
        self.resolved_target_num_patches = template.resolved_target_num_patches

    def resolve_target_num_patches(self, target_num_patches):
        """Resolve a common ratio/absolute target using the shared tree geometry."""
        return FoveationNet.resolve_target_num_patches(self, target_num_patches)

    def set_target_num_patches(self, target_num_patches):
        """Apply the same target schedule to every independently budgeted camera."""
        self.resolved_target_num_patches = self.resolve_target_num_patches(target_num_patches)
        for net in self.nets.values():
            net.set_target_num_patches(target_num_patches)
        return self.resolved_target_num_patches

    def set_router_temperature(self, router_temp: float) -> None:
        """Set all single-camera routing temperatures."""
        for net in self.nets.values():
            net.set_router_temperature(router_temp)

    def forward(
        self,
        images_by_view: Dict[str, torch.Tensor],
        cond_B_T_D: Optional[torch.Tensor] = None,
        *,
        return_routing_state: bool = False,
    ):
        """Return per-camera outputs and statistics with V in view_keys order."""
        routing_by_view = {}
        for key, net in self.nets.items():
            image_B_T_C_H_W = images_by_view[key]
            routing_by_view[key] = net(
                image_B_T_C_H_W.permute(0, 2, 1, 3, 4).contiguous(),
                cond_B_T_D=cond_B_T_D,
                return_routing_state=return_routing_state,
            )
        outputs = list(routing_by_view.values())
        budget_loss_V = torch.stack([out["budget_loss"] for out in outputs])
        hard_count_V = torch.stack([out["num_patches_B_T"].float().mean() for out in outputs])
        random_fraction_V = torch.stack([
            out["random_topology_mask_B_T"].float().mean() for out in outputs
        ])
        aux = {
            "budget_loss": budget_loss_V.mean(),
            "num_patches_mean": hard_count_V.mean(),
            "random_topology_fraction": random_fraction_V.mean(),
        }
        for index, key in enumerate(self.view_keys):
            aux[f"views/{key}/budget_loss"] = budget_loss_V[index]
            aux[f"views/{key}/num_patches_mean"] = hard_count_V[index]
            aux[f"views/{key}/random_topology_fraction"] = random_fraction_V[index]
        metric_sums, metric_counts = {}, {}
        for metric, field in (
            ("num_patches_mean", "num_patches_B_T"),
            ("expected_num_patches_mean", "expected_num_patches_B_T"),
            ("random_topology_fraction", "random_topology_mask_B_T"),
        ):
            metric_sums[metric] = torch.stack([out[field].detach().double().sum() for out in outputs])
            metric_counts[metric] = metric_sums[metric].new_tensor([out[field].numel() for out in outputs])
        # Budget losses are per-forward objectives; weight diagnostics by frame count.
        metric_counts["budget_loss"] = metric_counts["num_patches_mean"]
        metric_sums["budget_loss"] = budget_loss_V.detach().double() * metric_counts["budget_loss"]
        aux["_view_metric_sums"] = metric_sums
        aux["_view_metric_counts"] = metric_counts
        # Same configuration means each camera emits the same budget fields.
        for name in outputs[0]:
            if name.startswith("_budget_"):
                aux[name] = torch.stack([out[name] for out in outputs])
        return routing_by_view, aux

    def sample_paired_interventions(
        self, routing_by_view, temporal_pair_valid_P: Optional[torch.Tensor] = None,
    ):
        """Sample one camera/frame per window or pair, preserving [all A, all B].

        Temporal matching is intentionally left to the consumer after packing,
        because this module only owns node interventions and camera grouping.
        """
        reference_B_T_M = routing_by_view[self.view_keys[0]]["split_hard_B_T_M"]
        B, T, _ = reference_B_T_M.shape
        device = reference_B_T_M.device
        P = B if temporal_pair_valid_P is None else temporal_pair_valid_P.shape[0]
        selected_view_P = torch.randint(len(self.view_keys), (P,), device=device)
        selected_frame_P = torch.randint(T, (P,), device=device)
        selected_view_B = selected_view_P if temporal_pair_valid_P is None else selected_view_P.repeat(2)
        selected_frame_B = selected_frame_P if temporal_pair_valid_P is None else selected_frame_P.repeat(2)
        selected_logits_B = routing_by_view[self.view_keys[0]]["split_logits_B_T_M"].new_zeros(B)
        base_split_B = torch.zeros(B, dtype=torch.bool, device=device)
        interventions_by_view = {}
        for view, key in enumerate(self.view_keys):
            pair_indices_R = torch.nonzero(selected_view_P == view, as_tuple=True)[0]
            if pair_indices_R.numel() == 0:
                continue
            if temporal_pair_valid_P is None:
                batch_indices_R = pair_indices_R
                pair_valid_R = None
            else:
                # Group whole pairs; never independently regroup A and B windows.
                batch_indices_R = torch.cat([pair_indices_R, pair_indices_R + P])
                pair_valid_R = temporal_pair_valid_P[pair_indices_R]
            frame_indices_R = selected_frame_B[batch_indices_R]
            out = routing_by_view[key]
            net = self.nets[key]
            base_split_R_M = out["split_hard_B_T_M"][batch_indices_R, frame_indices_R]
            node_R, stop_R_M, split_R_M = net.sample_paired_splits(base_split_R_M, pair_valid_R)
            logits_R = out["split_logits_B_T_M"][batch_indices_R, frame_indices_R, node_R]
            if net.split_action_value == 0:
                logits_R = -logits_R
            # index_copy retains gradients back to the selected single-camera routers.
            selected_logits_B = selected_logits_B.index_copy(0, batch_indices_R, logits_R)
            base_split_B[batch_indices_R] = base_split_R_M.gather(1, node_R[:, None]).squeeze(1)
            interventions_by_view[key] = {
                "batch_indices_R": batch_indices_R,
                "frame_indices_R": frame_indices_R,
                "selected_node_R": node_R,
                "stop_R_M": stop_R_M,
                "split_R_M": split_R_M,
            }
        return {
            "selected_view_B": selected_view_B,
            "selected_frame_B": selected_frame_B,
            "selected_logits_B": selected_logits_B,
            "base_split_B": base_split_B,
            "by_view": interventions_by_view,
        }
