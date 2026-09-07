import torch
from torch.nn.modules.batchnorm import _BatchNorm

class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
        exclude_prefixes=(),
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
            exclude_prefixes: parameter/buffer name prefixes omitted from updates.
        """

        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        # Optional shared/frozen modules can be left out of EMA updates.
        self.exclude_prefixes = tuple(str(prefix) for prefix in exclude_prefixes)

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        bn_param_names = set()
        for module_name, module in new_model.named_modules():
            if isinstance(module, _BatchNorm):
                prefix = f"{module_name}." if module_name else ""
                bn_param_names.update(prefix + name for name, _ in module.named_parameters(recurse=False))

        new_params = dict(new_model.named_parameters())
        ema_params = dict(self.averaged_model.named_parameters())
        if new_params.keys() != ema_params.keys():
            missing = sorted(set(new_params.keys()) ^ set(ema_params.keys()))
            raise RuntimeError(f"EMA parameter key mismatch: {missing[:10]}")

        for key, ema_param in ema_params.items():
            if key.startswith(self.exclude_prefixes):
                continue
            param = new_params[key]
            value = param.detach().to(device=ema_param.device, dtype=ema_param.dtype)
            if key in bn_param_names or not param.requires_grad:
                ema_param.copy_(value)
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(value, alpha=1 - self.decay)

        new_buffers = dict(new_model.named_buffers())
        ema_buffers = dict(self.averaged_model.named_buffers())
        if new_buffers.keys() != ema_buffers.keys():
            missing = sorted(set(new_buffers.keys()) ^ set(ema_buffers.keys()))
            raise RuntimeError(f"EMA buffer key mismatch: {missing[:10]}")

        for key, ema_buffer in ema_buffers.items():
            if key.startswith(self.exclude_prefixes):
                continue
            value = new_buffers[key].detach().to(device=ema_buffer.device, dtype=ema_buffer.dtype)
            ema_buffer.copy_(value)

        self.optimization_step += 1
