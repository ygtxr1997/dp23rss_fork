"""Small state, sampling and EMA helpers for dense pi0.5 SFT."""
import copy
from contextlib import contextmanager
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class Pi05BatchSampler(Sampler):
    """Shard shuffled global batches; checkpoint consumption, never prefetch progress."""

    def __init__(self, size, global_batch, micro_batch, rank, world_size, seed):
        if global_batch <= 0 or micro_batch <= 0 or global_batch % (micro_batch * world_size):
            raise ValueError('global_batch must be divisible by micro_batch * world_size')
        self.size, self.global_batch, self.micro_batch = size, global_batch, micro_batch
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.accumulation = global_batch // (micro_batch * world_size)
        self.updates_per_epoch = size // global_batch
        if not self.updates_per_epoch:
            raise ValueError('Training dataset is smaller than one global batch')
        self.epoch, self.update = 0, 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order_N = torch.randperm(self.size, generator=generator)
        # Each global batch is [accumulation, world_size, micro_batch].
        order_U_A_R_B = order_N[:self.updates_per_epoch * self.global_batch].reshape(
            self.updates_per_epoch, self.accumulation, self.world_size, self.micro_batch)
        for update_A_R_B in order_U_A_R_B[self.update:]:
            for batch_R_B in update_A_R_B:
                yield batch_R_B[self.rank].tolist()

    def __len__(self):
        return (self.updates_per_epoch - self.update) * self.accumulation

    def advance(self):
        """Commit one successfully completed optimizer update."""
        self.update += 1
        if self.update == self.updates_per_epoch:
            self.epoch += 1
            self.update = 0

    def state_dict(self):
        """Return the next update's deterministic sampling position."""
        return dict(epoch=self.epoch, update=self.update)

    def load_state_dict(self, state):
        """Restore a position under the workspace's checked batch/data contract."""
        self.epoch, self.update = state['epoch'], state['update']
        if self.epoch < 0 or not 0 <= self.update < self.updates_per_epoch:
            raise ValueError('Invalid saved sampler position')


def seed_training(seed):
    """Seed all random sources used by the current training rank."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def capture_rng():
    """Capture CPU and local CUDA RNG using weights-only-loadable values."""
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    """Restore RNG after construction, validation or checkpoint loading."""
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def cpu_state(value):
    """Copy nested tensors to CPU without changing their dtype."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(item) for item in value)
    return copy.deepcopy(value)


def atomic_save(payload, path):
    """Replace a checkpoint only after the new file has been written successfully."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def master_parameter(optimizer, param_D):
    """Return FP32 master weights, or the parameter before lazy state allocation."""
    return optimizer.state.get(param_D, {}).get('fp32_master_param', param_D)


class Pi05EMA:
    """Keep FP32 shadows on rank zero without a second resident model."""

    def __init__(self, policy, decay):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0, 1)')
        self.decay, self.updates = decay, 0
        self.exclude_prefixes = ()
        self.shadow = {name: param_D.detach().float().clone()
                       for name, param_D in policy.named_parameters()}
        self.buffer_shadow = {name: buffer_D.detach().cpu().clone()
                              for name, buffer_D in policy.named_buffers()}

    def set_excluded_prefixes(self, prefixes):
        """Keep inactive phase branches unchanged during subsequent EMA updates."""
        self.exclude_prefixes = tuple(prefixes)

    def _excluded(self, name):
        return name.startswith(self.exclude_prefixes)

    @torch.no_grad()
    def update(self, policy, optimizer):
        """Average optimizer master weights once per completed update."""
        for name, param_D in policy.named_parameters():
            if self._excluded(name):
                continue
            self.shadow[name].lerp_(master_parameter(optimizer, param_D).float(), 1 - self.decay)
        for name, buffer_D in policy.named_buffers():
            if not self._excluded(name):
                self.buffer_shadow[name].copy_(buffer_D.detach().cpu())
        self.updates += 1

    def state_dict(self):
        """Return full-precision EMA state for exact training continuation."""
        return dict(decay=self.decay, updates=self.updates, shadow=self.shadow,
                    buffer_shadow=self.buffer_shadow, exclude_prefixes=self.exclude_prefixes)

    def load_state_dict(self, state):
        """Restore shadows without rounding through live model dtypes."""
        if state['decay'] != self.decay or state['shadow'].keys() != self.shadow.keys():
            raise ValueError('EMA configuration or parameter topology differs')
        for name, shadow_D in self.shadow.items():
            shadow_D.copy_(state['shadow'][name])
        saved_buffers = state.get('buffer_shadow', {})
        for name, buffer_D in self.buffer_shadow.items():
            if name in saved_buffers:
                buffer_D.copy_(saved_buffers[name])
        self.updates = state['updates']
        self.exclude_prefixes = tuple(state.get('exclude_prefixes', ()))

    @torch.no_grad()
    def copy_prefix_to_policy(self, policy, prefixes):
        """Copy EMA parameters and buffers for selected module prefixes to a live policy."""
        prefixes = tuple(prefixes)
        for name, param_D in policy.named_parameters():
            if name.startswith(prefixes):
                param_D.copy_(self.shadow[name].to(device=param_D.device, dtype=param_D.dtype))
        for name, buffer_D in policy.named_buffers():
            if name.startswith(prefixes) and name in self.buffer_shadow:
                buffer_D.copy_(self.buffer_shadow[name].to(device=buffer_D.device, dtype=buffer_D.dtype))

    @contextmanager
    def apply(self, policy):
        """Temporarily install EMA; keep the live backup in host memory."""
        live = {name: param_D.detach().cpu().clone() for name, param_D in policy.named_parameters()}
        live_buffers = {name: buffer_D.detach().cpu().clone()
                        for name, buffer_D in policy.named_buffers()}
        try:
            with torch.no_grad():
                for name, param_D in policy.named_parameters():
                    param_D.copy_(self.shadow[name].to(device=param_D.device, dtype=param_D.dtype))
                for name, buffer_D in policy.named_buffers():
                    if name in self.buffer_shadow:
                        buffer_D.copy_(self.buffer_shadow[name].to(device=buffer_D.device,
                                                                   dtype=buffer_D.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, param_D in policy.named_parameters():
                    param_D.copy_(live[name])
                for name, buffer_D in policy.named_buffers():
                    buffer_D.copy_(live_buffers[name].to(device=buffer_D.device,
                                                         dtype=buffer_D.dtype))

    def inference_state(self, policy, metadata):
        """Export EMA in model dtypes with the existing self-contained policy format."""
        state = {}
        for name, value in policy.state_dict().items():
            if name in self.shadow:
                state[name] = self.shadow[name].to(device='cpu', dtype=value.dtype)
            elif name in self.buffer_shadow:
                state[name] = self.buffer_shadow[name].to(device='cpu', dtype=value.dtype)
            else:
                state[name] = cpu_state(value)
        state['_extra_state']['assets']['training'] = copy.deepcopy(metadata)
        return state


@contextmanager
def distributed_ema(policy, ema, evaluation):
    """Borrow each rank's model for EMA inference, backing live tensors up on CPU."""
    rng, was_training = capture_rng(), policy.training
    live = {}
    tensors = dict(policy.named_parameters())
    tensors.update(dict(policy.named_buffers()))
    def install():
        # No second GPU model: only rank zero owns FP32 shadows; peers receive inference dtype.
        for name, value_D in tensors.items():
            live[name] = value_D.detach().cpu().clone()
            if ema is not None and name in ema.shadow:
                with torch.no_grad():
                    value_D.copy_(ema.shadow[name])
            elif ema is not None and name in ema.buffer_shadow:
                with torch.no_grad():
                    value_D.copy_(ema.buffer_shadow[name].to(device=value_D.device, dtype=value_D.dtype))
        policy.eval()
    try:
        evaluation.call(install)
        with torch.no_grad():
            if evaluation.world_size > 1:
                for value_D in tensors.values():
                    if value_D.numel():
                        synced_D = value_D.detach().contiguous()
                        dist.broadcast(synced_D, src=0,
                                       group=None if synced_D.is_cuda else evaluation.group)
                        value_D.copy_(synced_D)
        torch.cuda.synchronize(policy.device)
        yield
    finally:
        with torch.no_grad():
            for name, value_D in tensors.items():
                if name in live:
                    value_D.copy_(live[name])
        policy.train(was_training)
        restore_rng(rng)


@torch.no_grad()
def clip_gradients(parameters, max_norm):
    """Compute the norm in FP32, then clip gradients in their original dtypes."""
    params = [param_D for param_D in parameters if param_D.grad is not None]
    norms_P = torch.stack([torch.linalg.vector_norm(param_D.grad, dtype=torch.float32) for param_D in params])
    norm_1 = torch.linalg.vector_norm(norms_P)
    if not torch.isfinite(norm_1):
        raise FloatingPointError('Nonfinite pi0.5 gradient norm')
    scale_1 = (max_norm / (norm_1 + 1e-6)).clamp(max=1)
    for param_D in params:
        param_D.grad.mul_(scale_1)
    return norm_1.item()


def _module_group(name):
    if name.startswith('model.img.'):
        return 'vision_projection' if '.head.' in name else 'vision'
    if 'embedder' in name:
        return 'embedding'
    if name.startswith('model.llm.'):
        expert = any(part in name for part in ('norms.1.', 'mlps.1.', '_proj.1.'))
        return 'action_expert' if expert else 'language_backbone'
    return name.split('.')[1]


@torch.no_grad()
def begin_update_diagnostics(policy, optimizer, ema):
    """Record module gradient norms and one active coordinate per module."""
    report, probes = {}, {}
    unused = []
    for name, param_D in policy.named_parameters():
        if param_D.grad is None:
            unused.append(name)
            continue
        group = _module_group(name)
        item = report.setdefault(group, dict(gradient_squared_norm=0.0, tensors=0))
        norm = torch.linalg.vector_norm(param_D.grad, dtype=torch.float32).item()
        item['gradient_squared_norm'] += norm * norm
        item['tensors'] += 1
        if group not in probes and norm > 0:
            # One-time diagnostic; a maximum-gradient coordinate avoids untouched embedding rows.
            index = param_D.grad.detach().reshape(-1).abs().argmax().item()
            probes[group] = (name, index,
                             master_parameter(optimizer, param_D).reshape(-1)[index].float().item(),
                             ema.shadow[name].reshape(-1)[index].item())
    return dict(modules=report, unused_parameters=unused), probes


def finish_update_diagnostics(policy, optimizer, ema, report, probes):
    """Complete the report with actual master and EMA coordinate changes."""
    parameters = dict(policy.named_parameters())
    for item in report['modules'].values():
        item['gradient_norm'] = item.pop('gradient_squared_norm') ** 0.5
    for group, (name, index, before, ema_before) in probes.items():
        item = report['modules'][group]
        item['parameter'] = name
        item['coordinate'] = index
        item['master_delta'] = master_parameter(optimizer, parameters[name]).reshape(-1)[index].item() - before
        item['ema_delta'] = ema.shadow[name].reshape(-1)[index].item() - ema_before
    report['optimizer_state_dtypes'] = sorted({str(value_D.dtype)
        for state in optimizer.state.values() for value_D in state.values()
        if isinstance(value_D, torch.Tensor)})
    report['ema_updates'] = ema.updates
    return report
