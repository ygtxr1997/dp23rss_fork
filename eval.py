"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""
import random
import sys

import numpy as np

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from omegaconf import OmegaConf, DictConfig
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-f', '--from_config', default='')
def main(checkpoint, output_dir, device, from_config=''):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    # cfg.task.env_runner.test_start_seed = 400
    if from_config != '':  # if load from a specific config
        force_cfg = OmegaConf.load(from_config)
        force_cfg.task.env_runner.test_start_seed = cfg.task.env_runner.test_start_seed
        cfg = force_cfg
        exclude_keys = [
            'model',  # skip model
            'ema_model',  # skip ema_model
            'optimizer',  # skip optimizer
        ]
    else:
        exclude_keys = None
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=exclude_keys, include_keys=None)
    print(f"[eval] workspace loaded from: {checkpoint}. exclude_keys={exclude_keys}")

    # Note: the normalizer of policy has a very strange loading logic!
    norm_state_dict = {k[len("normalizer."):]: v for k, v in payload['state_dicts']['model'].items()
                       if 'normalizer.' in k}
    workspace.model.normalizer.load_state_dict(norm_state_dict)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        norm_state_dict = {k[len("normalizer."):]: v for k, v in payload['state_dicts']['ema_model'].items()
                           if 'normalizer.' in k}
        workspace.ema_model.normalizer.load_state_dict(norm_state_dict)
        # policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    eval_results = []
    for idx in range(100):
        cfg.task.env_runner.n_train = 0
        cfg.task.env_runner.n_test = 10
        # cfg.task.env_runner.max_steps = 110  # for quick debug
        reset_to_state = np.array([  # agent,block,block_rot
            256+100, 256-100,
            256-100, 256+50,
            np.pi / 4
        ])
        # reset_to_state = None
        domain_shift = "size"  # in 'none', 'orange', 'texture', 'light', 'size'
        save_name = f"size_hdfree_vis@{idx:02d}"  # in 'baseline', 'orange', 'orange_hdfree', 'texture', 'light', 'size_'
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir,
            reset_to_state=reset_to_state,
            domain_shift=domain_shift,
            save_name=save_name,
        )
        runner_log = env_runner.run(policy)

        # dump log to json
        json_log = dict()
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = value
        eval_results.append(json_log["test/mean_score"])

        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

        tmp_results = np.array(eval_results)
        eval_mean = np.mean(tmp_results)
        eval_std = np.std(tmp_results)
        print(f"[shift:{domain_shift}-save:{save_name}][{idx}/100]: "
              f"{eval_results[-1]:.4f}, mean={eval_mean:.4f}, std={eval_std:.4f}")

        # (50,8,2), len=38
        print(len(env_runner.cache_actions))

    eval_results = np.array(eval_results)
    eval_mean = np.mean(eval_results)
    eval_std = np.std(eval_results)
    print(f"[Final]: mean={eval_mean:.4f}, std={eval_std:.4f}")

if __name__ == '__main__':
    main()
