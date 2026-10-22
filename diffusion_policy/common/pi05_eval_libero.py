"""Evaluate converted or trained EMA pi0.5 policies through LIBERO-FT HTTP."""
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import re

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist

from diffusion_policy.common.pi05_eval_util import Pi05Evaluation, init_wandb, log_wandb
from diffusion_policy.model.pi05.checkpoint import sha256_file
from diffusion_policy.model.pi05.config import Pi05Config
from diffusion_policy.policy.pi05_policy import Pi05Policy


def _parse_log_time(log_time):
    log_time = str(log_time)
    if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}[-/][0-9]{2}-[0-9]{2}-[0-9]{2}', log_time) is None:
        raise ValueError('log_time must use YYYY-MM-DD-HH-MM-SS or YYYY-MM-DD/HH-MM-SS')
    return datetime.strptime(log_time.replace('/', '-'), '%Y-%m-%d-%H-%M-%S')


def _eval_name(log_time, eval_time, directory=False):
    if log_time is None:
        return f'eval-{eval_time}'
    run_time = _parse_log_time(log_time)
    if directory:
        return f"eval-{run_time.strftime('%Y-%m-%d/%H-%M-%S')}/{eval_time}"
    return f"eval-{run_time.strftime('%Y-%m-%d-%H-%M-%S')}_{eval_time}"


# Resolve before Hydra creates its run directory; both names share the same eval_time.
OmegaConf.register_new_resolver('pi05_eval_name', _eval_name, replace=True)


def _load_converted(asset_path, device):
    asset_dir = Path(asset_path).expanduser().resolve()
    for name in ("model.safetensors", "tokenizer.model", "norm_stats.json", "config.json", "manifest.json"):
        if not (asset_dir / name).is_file():
            raise FileNotFoundError(asset_dir / name)
    model_config = Pi05Config(**json.loads((asset_dir / "config.json").read_text()))
    if model_config.num_steps != 5 or model_config.n_action_steps != 5:
        raise ValueError("This LIBERO evaluation requires Euler 5 steps and execution chunks of 5")
    manifest = json.loads((asset_dir / "manifest.json").read_text())
    # Small asset checks prevent accidentally pairing weights with different preprocessing.
    for name in ("tokenizer.model", "norm_stats.json", "config.json"):
        if sha256_file(asset_dir / name) != manifest["output_sha256"][name]:
            raise ValueError(f"Asset differs from conversion manifest: {name}")
    policy = Pi05Policy(
        model_path=asset_dir, tokenizer_path=asset_dir / "tokenizer.model",
        norm_stats_path=asset_dir / "norm_stats.json", config=asdict(model_config),
    ).to(device=device).eval()
    # Pi05Policy already hashes the loaded weights; avoid another multi-GB pass here.
    weights_sha256 = policy.get_extra_state()["assets"]["base_weights_sha256"]
    if weights_sha256 != manifest["output_sha256"]["model.safetensors"]:
        raise ValueError("Weights differ from conversion manifest")
    return policy, dict(weights='pretrained', global_step=0, asset_dir=str(asset_dir),
                        source_uri=manifest['source_uri'], output_sha256=manifest['output_sha256'],
                        purpose='official LIBERO checkpoint interface check')


def _load_policy(cfg, device):
    if cfg.checkpoint is None and cfg.get('log_time') is not None:
        checkpoint_type = cfg.get('checkpoint_type', 'best')
        if checkpoint_type not in ('best', 'latest'):
            raise ValueError('checkpoint_type must be best or latest')
        log_time = str(cfg.log_time)
        run_time = _parse_log_time(log_time)
        output_root = Path(__file__).resolve().parents[2] / 'outputs'
        patterns = (run_time.strftime('%Y-%m-%d/%H-%M-%S'),
                    f"*{run_time.strftime('%Y-%m-%d-%H-%M-%S')}*")
        # Evaluation outputs reuse training timestamps but are never training candidates.
        candidates = sorted({path for pattern in patterns for path in output_root.rglob(pattern)
                             if path.is_dir() and path.relative_to(output_root).parts[0] != 'pi05_libero_eval'})
        if not candidates:
            raise FileNotFoundError(f'No run directory matches log_time={log_time} under {output_root}')
        if len(candidates) > 1:
            matches = '\n'.join(str(path) for path in candidates)
            raise ValueError(f'Multiple run directories match log_time={log_time}; '
                             f'specify checkpoint explicitly:\n{matches}')
        cfg.checkpoint = str(candidates[0] / 'checkpoints' / f'{checkpoint_type}_ema.pt')
    if cfg.checkpoint is None:
        return _load_converted(cfg.asset_dir, device)
    path = Path(cfg.checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg.checkpoint = str(path)
    if int(os.environ.get('RANK', 0)) == 0:
        print(f'[pi0.5 eval] Loading checkpoint: {path}', flush=True)
    policy = Pi05Policy.from_checkpoint(path, device=device)
    metadata = dict(policy.get_extra_state()['assets'].get('training', {}))
    if metadata.get('weights') != 'ema':
        raise ValueError('checkpoint must be an exported SFT EMA policy; use asset_dir for converted weights')
    metadata.update(checkpoint=str(path), checkpoint_sha256=sha256_file(path),
                    purpose='project SFT EMA evaluation')
    return policy, metadata


@hydra.main(version_base=None, config_path="../../configs", config_name="pi05_libero_eval")
def main(cfg: DictConfig):
    """Load embedded assets and evaluate disjoint task queues on the visible GPUs."""
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank) if world > 1 else torch.device(cfg.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl')
    evaluation, wandb_run = None, None
    try:
        if cfg.all_tasks:
            # The data preparation config is the canonical ten-task inventory.
            data_cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / 'configs/pi05_libero_data.yaml')
            cfg.runner.hdf5_fns = data_cfg.hdf5_fns
            cfg.runner.dataset_subname = data_cfg.dataset_subname
        evaluation = Pi05Evaluation(cfg.runner, world, rank, local_rank, len(cfg.runner.hdf5_fns))
        output = Path(cfg.output_dir).expanduser().resolve()
        evaluation.call(lambda: output.mkdir(parents=True, exist_ok=True) if rank == 0 else None)
        policy, metadata = evaluation.call(lambda: _load_policy(cfg, device))
        if policy.config.num_steps != 5 or policy.config.n_action_steps != 5:
            raise ValueError('LIBERO evaluation requires Euler 5 steps and execution chunks of 5')
        def initialize_logging():
            if rank == 0:
                OmegaConf.save(cfg, output / 'config.yaml', resolve=True)
                return init_wandb(cfg.logging, output, OmegaConf.to_container(cfg, resolve=True), metadata)
        wandb_run = evaluation.call(initialize_logging)
        step = metadata['global_step']
        log = evaluation.run(policy, output, 0, step, metadata)
        evaluation.call(lambda: log_wandb(wandb_run, log, step) if rank == 0 else None)
        if rank == 0:
            print(f"[pi0.5] completed {log['test/n_episodes']} episodes; success rate={log['test/mean_score']:.3f}")
            print(f"[pi0.5] results: {output / 'eval' / 'epoch_0000' / 'summary.json'}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if evaluation is not None:
            evaluation.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
