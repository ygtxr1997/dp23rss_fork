"""Shared pi0.5 task evaluation and rank-zero logging, using the existing runner."""
from datetime import timedelta
from pathlib import Path
import math
import time

import hydra
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from diffusion_policy.common.pi05_training_util import capture_rng, restore_rng


class Pi05Evaluation:
    """Coordinate task queues and errors on CPU while each GPU serves its policy."""

    def __init__(self, runner_cfg, world_size, rank, local_rank, task_count):
        self.cfg = runner_cfg
        self.world_size, self.rank, self.local_rank = world_size, rank, local_rank
        timeout = 3600 + math.ceil(task_count / world_size) * (
            runner_cfg.startup_timeout + runner_cfg.eval_timeout + 3 * runner_cfg.shutdown_timeout)
        self.group = (dist.new_group(backend='gloo', timeout=timedelta(seconds=timeout))
                      if world_size > 1 else None)

    def gather(self, value):
        """Gather small CPU payloads, including errors, without a waiting GPU kernel."""
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def call(self, function):
        """Finish a local stage on every rank before propagating any caught error."""
        result, error = None, None
        try:
            result = function()
        except BaseException as exc:
            error = exc
        errors = self.gather(None if error is None else f'rank {self.rank}: {type(error).__name__}: {error}')
        if any(item is not None for item in errors):
            raise RuntimeError('; '.join(item for item in errors if item is not None)) from error
        return result

    def run(self, policy, output_dir, epoch, step, metadata, dataset_metadata=None):
        """Run disjoint task queues; only rank zero builds videos and the summary."""
        started = time.perf_counter()
        metadata = dict(metadata, evaluation=OmegaConf.to_container(self.cfg, resolve=True))

        def prepare():
            runner = hydra.utils.instantiate(self.cfg, output_dir=str(output_dir),
                                            policy_config=policy.get_extra_state()['config'])
            if self.world_size > 1:
                runner.configure_distributed(self.rank, self.local_rank, self.world_size)
            runner.set_context(dataset_metadata, epoch, step, weights=metadata['weights'],
                               checkpoint_metadata=metadata)
            if self.rank == 0:
                runner.prepare_run()
            return runner
        runner = self.call(prepare)
        paths = self.gather(str(Path(output_dir).resolve()))
        if len(set(paths)) != 1:
            raise ValueError('All evaluation ranks must share the same output directory')
        indices = list(range(self.rank, len(runner.tasks), self.world_size))
        print(f'[pi0.5 rollout] rank={self.rank}, device={policy.device}, tasks={indices}', flush=True)
        if policy.device.type == 'cuda':
            torch.cuda.synchronize(policy.device)
        payloads = self.call(lambda: runner.run_tasks(policy, indices))
        gathered = self.gather(payloads)
        log = self.call(lambda: runner.summarize_results([
            task for queue in gathered for task in queue]) if self.rank == 0 else {})
        if self.rank == 0:
            log['test/seconds'] = time.perf_counter() - started
        return log

    def close(self):
        """Release the auxiliary CPU communication group."""
        if self.group is not None:
            dist.destroy_process_group(self.group)


def init_wandb(cfg, output_dir, config, metadata):
    """Initialize one run without changing the policy's random streams."""
    rng = capture_rng()
    try:
        import wandb
        options = OmegaConf.to_container(cfg, resolve=True)
        options['name'] = options['name'] or f'{Path(output_dir).parent.name}-{Path(output_dir).name}'
        run = wandb.init(dir=str(output_dir), config=config, **options)
        run.config.update(dict(provenance=metadata))
        run.define_metric('global_step')
        run.define_metric('*', step_metric='global_step')
        return run
    finally:
        restore_rng(rng)


def log_wandb(run, metrics, step):
    """Submit one complete optimizer step while preserving training RNG."""
    rng = capture_rng()
    try:
        run.log(dict(metrics, global_step=step), step=step)
    finally:
        restore_rng(rng)
