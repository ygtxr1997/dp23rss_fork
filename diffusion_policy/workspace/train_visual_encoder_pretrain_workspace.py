if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import copy
import random
import pathlib
import numpy as np
import hydra
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from datetime import timedelta
from contextlib import nullcontext

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.policy.visual_encoder_pretrain_policy import VisualEncoderPretrainPolicy
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainVisualEncoderPretrainWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # =========================================================
        # 提前锁定多卡通信环境，防止稍后 load_weight 污染 CUDA 上下文
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs, ddp_kwargs],
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every
        )
        # =========================================================

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: VisualEncoderPretrainPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: VisualEncoderPretrainPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0
        print(f"[TrainVisualEncoderPretrainWorkspace] Initialized workspace")

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = self.accelerator
        device = accelerator.device

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                if accelerator.is_main_process:
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # configure ema (EMA 模型通常不需要放入 DDP 包装中，手动放到 device 即可)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )
        self.lr_scheduler = lr_scheduler

        # 让 Accelerator 接管模型、优化器和 DataLoader ###
        # 被接管后，accelerator 会自动处理 .to(device) 和 DDP 的包装
        self.model, self.optimizer, train_dataloader, val_dataloader, self.lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, train_dataloader, val_dataloader, self.lr_scheduler
        )

        # configure ema (EMA 模型通常不需要放入 DDP 包装中，手动放到 device 即可)
        ema: EMAModel = None
        if cfg.training.use_ema:
            self.ema_model.to(device)
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env and wandb
        wandb_run = None
        topk_manager = None

        if accelerator.is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update({"output_dir": self.output_dir})

            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

            log_path = os.path.join(self.output_dir, 'logs.json.txt')
            log_ctx = JsonLogger(log_path)
        else:
            log_ctx = nullcontext(None)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        disable_tqdm = not accelerator.is_main_process
        with log_ctx as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                train_loss_sum = torch.tensor(0.0, device=device)
                train_state_loss_sum = torch.tensor(0.0, device=device)
                train_action_loss_sum = torch.tensor(0.0, device=device)
                train_state_mae_sum = torch.tensor(0.0, device=device)
                train_action_mae_sum = torch.tensor(0.0, device=device)
                train_batches = torch.tensor(0, device=device)
                self.optimizer.zero_grad(set_to_none=True)

                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                        disable=disable_tqdm,) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        losses_log = {}

                        with accelerator.accumulate(self.model):
                            losses = self.model(batch)
                            raw_loss = losses['total_loss']
                            accelerator.backward(raw_loss)

                            if accelerator.sync_gradients:
                                self.optimizer.step()
                                self.lr_scheduler.step()
                                self.optimizer.zero_grad(set_to_none=True)

                            for k, v in losses.items():
                                if k == 'total_loss': continue
                                losses_log[f"train_step/{k}"] = v.item() if isinstance(v, torch.Tensor) else v

                        # ### [修改点 6] 获取真实的 model 进行 EMA 更新 ###
                        if cfg.training.use_ema and accelerator.sync_gradients:
                            ema.step(accelerator.unwrap_model(self.model))

                        # ### [修改点 7] 仅主进程进行 Logging ###
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)

                        train_loss_sum += raw_loss.detach()
                        train_state_loss_sum += losses["state_loss"].detach() \
                            if "state_loss" in losses else losses["agent_pos_loss"].detach()
                        train_action_loss_sum += losses["action_loss"].detach()
                        train_state_mae_sum += losses["state_mae"].detach() \
                            if "state_mae" in losses else losses["agent_pos_mae"].detach()
                        train_action_mae_sum += losses["action_mae"].detach()
                        train_batches += 1

                        if accelerator.is_main_process:
                            step_log = {
                                'train_step/loss': raw_loss_cpu,
                                'train_step/lr': self.lr_scheduler.get_last_lr()[0],
                                'meta/global_step': self.global_step,
                                'meta/epoch': self.epoch,
                                **losses_log
                            }

                            is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                            if not is_last_batch:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)

                        self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # ========= 评测阶段 (仅让主进程执行 Rollout 和保存，避免冲突) =========
                # 同步所有 GPU，确保大家都跑完了这个 Epoch
                accelerator.wait_for_everyone()

                train_stats = torch.stack([
                    train_loss_sum,
                    train_state_loss_sum,
                    train_action_loss_sum,
                    train_state_mae_sum,
                    train_action_mae_sum,
                    train_batches.float()
                ])
                train_stats = accelerator.reduce(train_stats, reduction="sum")

                if accelerator.is_main_process:
                    if train_stats[5] > 0:
                        step_log["train/loss"] = (train_stats[0] / train_stats[5]).item()
                        step_log["train/state_loss"] = (train_stats[1] / train_stats[5]).item()
                        step_log["train/action_loss"] = (train_stats[2] / train_stats[5]).item()
                        step_log["train/state_mae"] = (train_stats[3] / train_stats[5]).item()
                        step_log["train/action_mae"] = (train_stats[4] / train_stats[5]).item()
                    else:
                        step_log["train/loss"] = 0.0
                        step_log["train/state_loss"] = 0.0
                        step_log["train/action_loss"] = 0.0
                        step_log["train/state_mae"] = 0.0
                        step_log["train/action_mae"] = 0.0

                    # Unwrap model to pass to runner
                    unwrapped_policy = accelerator.unwrap_model(self.model)
                    policy = unwrapped_policy
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()
                    # NOTE: There is no external env evaluator for this pretraining task

                # run validation (所有卡都可以参与计算验证集，但最简单的做法是只让主卡算
                if (self.epoch % cfg.training.val_every) == 0:
                    policy_val = self.ema_model if cfg.training.use_ema else accelerator.unwrap_model(self.model)
                    policy_val.eval()

                    val_loss_sum = torch.tensor(0.0, device=device)
                    val_state_loss_sum = torch.tensor(0.0, device=device)
                    val_action_loss_sum = torch.tensor(0.0, device=device)
                    val_state_mae_sum = torch.tensor(0.0, device=device)
                    val_action_mae_sum = torch.tensor(0.0, device=device)
                    val_batches = torch.tensor(0, device=device)
                    with torch.no_grad():
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                                disable=disable_tqdm) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                losses = policy_val.compute_loss(batch)
                                batch_loss = losses["total_loss"]
                                batch_loss = batch_loss.detach() if isinstance(batch_loss, torch.Tensor) else torch.tensor(
                                    batch_loss, device=device, dtype=torch.float32)
                                val_loss_sum += batch_loss
                                val_state_loss_sum += losses["state_loss"].detach() \
                                    if "state_loss" in losses else losses["agent_pos_loss"].detach()
                                val_action_loss_sum += losses["action_loss"].detach()
                                val_state_mae_sum += losses["state_mae"].detach() \
                                    if "state_mae" in losses else losses["agent_pos_mae"].detach()
                                val_action_mae_sum += losses["action_mae"].detach()
                                val_batches += 1

                                if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break

                    stats = torch.stack([
                        val_loss_sum,
                        val_state_loss_sum,
                        val_action_loss_sum,
                        val_state_mae_sum,
                        val_action_mae_sum,
                        val_batches.float()
                    ])
                    stats = accelerator.reduce(stats, reduction="sum")

                    if accelerator.is_main_process and stats[5] > 0:
                        step_log["val/loss"] = (stats[0] / stats[5]).item()
                        step_log["val/state_loss"] = (stats[1] / stats[5]).item()
                        step_log["val/action_loss"] = (stats[2] / stats[5]).item()
                        step_log["val/state_mae"] = (stats[3] / stats[5]).item()
                        step_log["val/action_mae"] = (stats[4] / stats[5]).item()

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()

                        # DDP 下不建议保存 snapshot；只在单进程时保留
                        if cfg.checkpoint.save_last_snapshot and accelerator.num_processes == 1:
                            self.save_snapshot()

                        metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                        # keep compatibility with checkpoint format strings like "{epoch}" / "{global_step}"
                        metric_dict.setdefault("epoch", self.epoch)
                        metric_dict.setdefault("global_step", self.global_step)
                        if cfg.checkpoint.topk.monitor_key in metric_dict:
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                            if topk_ckpt_path is not None:
                                self.save_checkpoint(path=topk_ckpt_path)

                if accelerator.is_main_process:
                    policy.train()
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)

                self.epoch += 1
                accelerator.wait_for_everyone()  # 确保下一轮开始前所有进程同步


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainVisualEncoderPretrainWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
