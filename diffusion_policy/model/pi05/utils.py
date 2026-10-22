# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small eager kernels shared by the extracted dense model."""
import torch
import torch.nn.functional as F


def _str_to_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def gelu_glu(gate_B_T_M: torch.Tensor, value_B_T_M: torch.Tensor) -> torch.Tensor:
    """Apply the upstream exact GELU gated activation."""
    return F.gelu(gate_B_T_M) * value_B_T_M
