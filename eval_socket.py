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
from diffusion_policy.env_runner.pusht_image_socket_runner import PushTImageSocketRunner

from robokit.debug_utils.curves import H5CurveStorage


'''
Example usage:
conda activate robodiff
cd code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork
yes | CUDA_VISIBLE_DEVICES=2 python eval_socket.py  \
    -c "None"  \
    -o data/pusht_eval_output  \
    -p 6062  \
    -s rainbow  \
    -a 1  \
    -r -1  \
    --max_repeats 30  \
    --close_online
'''
h5_suffix = '_tmp'  # just set for debug. `3e-5`, `layer28`, `ex_lora`, `ex_kv`

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-f', '--from_config', default='')
@click.option("-p", "--port", default=6060, help="Port for socket communication")
@click.option('-s', '--domain_shift', default='none',
              help="in `none`, `orange`, `texture`, `light`, `size`, `rainbow`")
@click.option('-a', '--acc_seed', default=1.0, type=float, help='how much to change seed after an eval')
@click.option('--close_online', is_flag=True, default=False, help='whether to eval the baseline')
@click.option('-r', '--reset_each', default=-1, help='reset model after each n eval')
@click.option('-n', '--num_envs', default=1, help='num of parallel envs')
@click.option('-m', '--max_repeats', default=200, help='num of repat time, will be x1.5 if acc_seed=0.66')
def main(checkpoint, output_dir, device, from_config='', port=6060, domain_shift='none',
         acc_seed: float = 1.0, close_online=False, reset_each=-1, num_envs: int = 1,
         max_repeats: int = 200,
         ):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    h5_data_helper = H5CurveStorage(filepath=os.path.join(output_dir, 'eval_results.h5'))

    # run eval
    env_runner_config = OmegaConf.create({
        "_target_": "diffusion_policy.env_runner.pusht_image_socket_runner.PushTImageSocketRunner",
        "fps": 10,
        "legacy_test": True,
        "max_steps": 300,  # ori:300, quick_debug:110
        "n_action_steps": 1,  # ori:8
        "n_envs": num_envs,  # ori:8
        "n_obs_steps": 4*5,  # Ts can be larger than v2, ori:4+1,
        "n_test": num_envs*1,  # ori:8*1
        "n_test_vis": 1,  # ori:4
        "n_train": 0,  # ori: 6
        "n_train_vis": 0,  # ori: 2
        "past_action": False,
        "test_start_seed": 4300000, # now:4300050, ori:4300000, NOTE: start from +50?
        "train_start_seed": 0,
        "domain_shift": domain_shift,  # `none`, `orange`, `texture`, `light`, `size`
        "render_size": 256,  # ori:96
        # socket related
        "policy_url": f"http://localhost:{port}",
        "send_per_frames": 12,  # will be modified in init() and reset()
    })
    eval_results = []
    repeat_times = max_repeats  # ori:50
    if 0.0 < acc_seed < 1.0:
        repeat_times = int(repeat_times * (1.0 / acc_seed))

    stage_flag = 0  # cold-start at the very beginning
    seed_accumulate: float = 0.0
    for idx in range(repeat_times):
        # cfg.task.env_runner.n_train = 0
        # cfg.task.env_runner.n_test = 10
        # cfg.task.env_runner.max_steps = 110  # for quick debug
        # reset_to_state = np.array([  # agent,block,block_rot
        #     256 + 100, 256 - 100,
        #     256 - 100, 256 + 50,
        #     np.pi / 4
        # ])
        reset_to_state = None
        # domain_shift = "size"  # in 'none', 'orange', 'texture', 'light', 'size'
        domain_shift = env_runner_config.domain_shift
        # save_name = f"size_hdfree_vis@{idx:02d}"  # in 'baseline', 'orange', 'orange_hdfree', 'texture', 'light', 'size_'
        save_name = f"tmp_expert_vis@{idx:02d}"
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=output_dir,
        #     reset_to_state=reset_to_state,
        #     domain_shift=domain_shift,
        #     save_name=save_name,
        # )
        env_runner: PushTImageSocketRunner = hydra.utils.instantiate(
            env_runner_config,
            output_dir=output_dir,
            reset_to_state=reset_to_state,
            save_name=save_name,
        )

        # runner_log = env_runner.run(policy)
        env_runner.init_socket("push the T-block into the green area.")
        if reset_each > 0 and idx % reset_each == 0:
            env_runner.send_reset()  # NOTE: send reset before each evaluator?
            stage_flag = 0  # reset to cold-start when seed changes
        runner_log = env_runner.run(
            device=device,
            close_online=close_online,
            stage_flag=stage_flag,
        )
        if not close_online:
            # stage_flag = 1  # NOTE: after first eval, set to 1-online next time
            stage_flag = 2  # after first eval, set to 2-copy weights next time

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
        print(f"[shift:{domain_shift}-seed:{env_runner_config.test_start_seed}"
              f"-save:{save_name}][{idx}/{repeat_times}]: "
              f"{eval_results[-1]:.4f}, mean={eval_mean:.4f}, std={eval_std:.4f}")

        # (50,8,2), len=38
        # print(len(env_runner.cache_actions_B_T_D))
        h5_save_key = f"{domain_shift}_ol={not close_online}{h5_suffix}"
        h5_data_helper.update_or_append(
            key=h5_save_key,
            index=idx,
            value=eval_results[-1],
        )
        print(f"[DEBUG] h5_data[{h5_save_key}]:", len(h5_data_helper.get(key=h5_save_key)))

        # Check if need to update seed at the end of an eval
        if acc_seed >= 1.0 or acc_seed == 0.0:
            env_runner_config.test_start_seed += env_runner_config.n_envs ** int(acc_seed)
            # NOTE: make sure each eval uses different seeds, acc_seed=0 the added seed is n_envs^0=1
        elif 0.0 < acc_seed < 1.0:
            seed_accumulate += acc_seed
            if seed_accumulate >= 1.0:
                add_seed = int(seed_accumulate)
                env_runner_config.test_start_seed += add_seed
                seed_accumulate = 0.0  # reset
        else:
            pass  # do not change seed

    eval_results = np.array(eval_results)
    eval_mean = np.mean(eval_results)
    eval_std = np.std(eval_results)
    print(f"[Final]: mean={eval_mean:.4f}, std={eval_std:.4f}")


if __name__ == '__main__':
    main()
