import os
from typing import Optional, Union

import torch
import torch.nn as nn


class JEPAHelper(nn.Module):
    MODEL_NAMES = [
        "vjepa2_1_vit_base_384",
        "vjepa2_1_vit_large_384",
        "vjepa2_1_vit_giant_384",
        "vjepa2_1_vit_gigantic_384",
    ]
    MODEL_DIMS = {
        "vjepa2_1_vit_base_384": 768,
        "vjepa2_1_vit_large_384": 1024,
        "vjepa2_1_vit_giant_384": 1408,
        "vjepa2_1_vit_gigantic_384": 1664,
    }

    def __init__(
        self,
        model_name: str = "vjepa2_1_vit_base_384",
        torch_home: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"{model_name} not in {self.MODEL_NAMES}")
        self.model_name = model_name
        self.jepa_out_dim = self.MODEL_DIMS[model_name]

        if torch_home is None:
            processor, jepa_model, jepa_predictor = self._download_models(model_name)
        else:
            local_jepa_dir = os.path.join(torch_home, "hub/facebookresearch_vjepa2_main/")
            torch.hub.set_dir(os.path.join(torch_home, "hub"))
            print(f"[DEBUG][JEPAHelper] Using torch_home={torch_home}")
            processor = torch.hub.load(
                local_jepa_dir,
                "vjepa2_preprocessor",
                source="local",
            )
            jepa_model, jepa_predictor = torch.hub.load(
                local_jepa_dir,
                model_name,
                source="local",
            )

        # Keep the preprocessor available for the legacy preprocess_frames()
        # helper, but do not register it as a child module or serialize it.
        object.__setattr__(self, "processor", processor)
        self.jepa_model = jepa_model

        norm_params_2_C = processor.normalize.detach().clone().float()
        self.register_buffer(
            "norm_mean_1_C_1_1_1",
            norm_params_2_C[0].view(1, 3, 1, 1, 1),
        )
        self.register_buffer(
            "norm_std_1_C_1_1_1",
            norm_params_2_C[1].view(1, 3, 1, 1, 1),
        )

        # Predictor weights are not used by FoveationNet and can be released
        # immediately after torch hub returns the encoder.
        del jepa_predictor
        del norm_params_2_C

        if device is not None:
            self.to(device=device)
        self.count_parameters()

    @classmethod
    def _download_models(cls, model_name: str):
        assert model_name in cls.MODEL_NAMES, f"{model_name} not in {cls.MODEL_NAMES}"
        print(
            f"[JEPAHelper] Using `TORCH_HOME`: "
            f"{os.environ.get('TORCH_HOME', None)}"
        )
        processor = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_preprocessor",
        )
        jepa_model, jepa_predictor = torch.hub.load(
            "facebookresearch/vjepa2",
            model_name,
        )
        print(f"[JEPAHelper] Model `{model_name}` downloaded successfully.")
        return processor, jepa_model, jepa_predictor

    def count_parameters(self) -> int:
        total_params = sum(p.numel() for p in self.jepa_model.parameters())
        print(
            f"[JEPAHelper] Total parameters in `{self.model_name}` model: "
            f"{total_params / 1e9:.2f}B"
        )
        return total_params

    def preprocess_frames(self, frames_T_H_W_C: torch.Tensor) -> torch.Tensor:
        x_C_T_H_W = self.processor(
            frames_T_H_W_C.permute(0, 3, 1, 2)
        )[0]
        return x_C_T_H_W

    def norm_batch_input(self, x_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        if x_B_C_T_H_W.dtype == torch.uint8:
            x_B_C_T_H_W = x_B_C_T_H_W.float().div(255.0)
        else:
            # FoveationNet's floating image convention is [-1, 1].
            x_B_C_T_H_W = x_B_C_T_H_W.float().add(1.0).mul(0.5)
        return (
            x_B_C_T_H_W - self.norm_mean_1_C_1_1_1
        ) / self.norm_std_1_C_1_1_1

    def encode(self, x_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        x_B_C_T_H_W = self.norm_batch_input(x_B_C_T_H_W)
        return self.jepa_model(x_B_C_T_H_W)

    def forward(self, x_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        return self.encode(x_B_C_T_H_W)
