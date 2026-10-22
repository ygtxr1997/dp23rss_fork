# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP32-master AdamW adapted from the pinned RLinf source in sources.json."""

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class FP32MasterAdamW(Optimizer):
    """AdamW with FP32 state and master weights for low-precision parameters.

    The optimizer keeps the model parameters in their original dtype so the
    forward path is unchanged. For FP16/BF16 parameters, updates are applied to
    an FP32 master copy and then cast back to the model parameter.
    """

    _FP32_STATE_KEYS = ("fp32_master_param", "exp_avg", "exp_avg_sq")

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _new_fp32_tensor_like(param_D: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(
            param_D, dtype=torch.float32, memory_format=torch.preserve_format
        )

    def _initialize_state(self, param_D: torch.Tensor) -> dict[str, Any]:
        if not param_D.is_floating_point():
            raise TypeError(
                "FP32MasterAdamW only supports floating-point parameters, got "
                f"{param_D.dtype}."
            )

        state = self.state[param_D]
        if state:
            return state

        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = self._new_fp32_tensor_like(param_D)
        state["exp_avg_sq"] = self._new_fp32_tensor_like(param_D)
        if param_D.dtype != torch.float32:
            state["fp32_master_param"] = param_D.detach().to(torch.float32).clone()
        return state

    @torch.no_grad()
    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """Apply one AdamW update through FP32 master parameters."""
        loss_1 = None
        if closure is not None:
            with torch.enable_grad():
                loss_1 = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param_D in group["params"]:
                grad_D = param_D.grad
                if grad_D is None:
                    continue
                if grad_D.is_sparse:
                    raise RuntimeError(
                        "FP32MasterAdamW does not support sparse gradients."
                    )

                state = self._initialize_state(param_D)
                state["step"].add_(1)
                step = int(state["step"].item())

                update_param_D = state.get("fp32_master_param", param_D)
                grad_fp32_D = grad_D.detach().to(torch.float32)
                exp_avg_D = state["exp_avg"]
                exp_avg_sq_D = state["exp_avg_sq"]

                update_param_D.mul_(1.0 - lr * weight_decay)
                exp_avg_D.lerp_(grad_fp32_D, 1.0 - beta1)
                exp_avg_sq_D.mul_(beta2).addcmul_(grad_fp32_D, grad_fp32_D, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
                denom_D = exp_avg_sq_D.sqrt().div_(bias_correction2_sqrt).add_(eps)
                update_param_D.addcdiv_(exp_avg_D, denom_D, value=-(lr / bias_correction1))

                if update_param_D is not param_D:
                    param_D.copy_(update_param_D)

        return loss_1

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state without casting FP32 tensors to parameter dtype."""
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups) or any(
            len(saved_group["params"]) != len(current_group["params"])
            for saved_group, current_group in zip(
                saved_groups, self.param_groups, strict=False
            )
        ):
            raise ValueError(
                "Cannot load FP32MasterAdamW state with a different parameter topology."
            )

        saved_to_current = {
            param_id: param_D
            for saved_group, current_group in zip(
                saved_groups, self.param_groups, strict=True
            )
            for param_id, param_D in zip(
                saved_group["params"], current_group["params"], strict=True
            )
        }
        fp32_state: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
        state_without_fp32 = {}
        for param_id, saved_state in state_dict["state"].items():
            stripped_state = dict(saved_state)
            param_D = saved_to_current.get(param_id)
            if param_D is not None:
                extracted = {}
                for key in (*self._FP32_STATE_KEYS, "step"):
                    value = stripped_state.pop(key, None)
                    if isinstance(value, torch.Tensor):
                        target_device = param_D.device if key != "step" else value.device
                        target_dtype = torch.float32 if key != "step" else value.dtype
                        extracted[key] = (
                            value.detach()
                            .to(device=target_device, dtype=target_dtype)
                            .clone()
                        )
                if extracted:
                    fp32_state[param_D] = extracted
            state_without_fp32[param_id] = stripped_state

        state_for_super = dict(state_dict)
        state_for_super["state"] = state_without_fp32
        super().load_state_dict(state_for_super)
        for param_D, saved_state in fp32_state.items():
            self.state[param_D].update(saved_state)
