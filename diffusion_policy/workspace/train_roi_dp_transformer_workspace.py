if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np

from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from datetime import timedelta
from contextlib import nullcontext

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.roi_dp_transformer_hybrid_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        if getattr(cfg.policy, 'use_da', False):
            raise ValueError(
                "TrainDiffusionTransformerHybridWorkspace no longer supports "
                "`policy.use_da=True`."
            )

        # =========================================================
        # 提前锁定多卡通信环境，防止稍后 load_weight 污染 CUDA 上下文
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[kwargs, ddp_kwargs])
        # =========================================================

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.pretrained_ckpt = cfg.pretrained_ckpt if hasattr(cfg, 'pretrained_ckpt') else None
        if self.pretrained_ckpt:
            if not os.path.exists(self.pretrained_ckpt):
                print(f"[Warning] => {self.pretrained_ckpt} does not exist, skipping.")
            else:
                self.model.load_weight_from_ckpt(self.pretrained_ckpt, use_ema=True)  # NOTE: ori:False

        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        else:
            print(f"[TrainDiffusionTransformerHybridWorkspace] training from scratch")

        # freeze some parameters if needed
        if getattr(cfg.policy, 'en_freeze_obs_encoder', False):
            print("[TrainDiffusionTransformerHybridWorkspace] Freezing obs_encoder...")
            if hasattr(self.model, 'freeze_obs_encoder'):
                self.model.freeze_obs_encoder()
        self.model.print_training_status()

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    @staticmethod
    def _image_tensor_to_uint8(image_C_H_W: torch.Tensor) -> np.ndarray:
        image = image_C_H_W.detach().float().cpu()
        if image.min() < 0:
            image = (image + 1.0) * 0.5
        image = image.clamp(0.0, 1.0)
        image = image.permute(1, 2, 0).numpy()
        return (image * 255.0).round().astype(np.uint8)

    @staticmethod
    def _draw_roi_box(
            image_H_W_C: np.ndarray,
            params_3: torch.Tensor,
            thickness: int = 3
    ) -> np.ndarray:
        canvas = image_H_W_C.copy()
        height, width = canvas.shape[:2]
        center_x, center_y, scale = params_3.detach().float().cpu().tolist()

        x0 = int(round((center_x - scale + 1.0) * 0.5 * width))
        x1 = int(round((center_x + scale + 1.0) * 0.5 * width))
        y0 = int(round((center_y - scale + 1.0) * 0.5 * height))
        y1 = int(round((center_y + scale + 1.0) * 0.5 * height))

        x0, x1 = sorted((max(0, min(width - 1, x0)), max(0, min(width - 1, x1))))
        y0, y1 = sorted((max(0, min(height - 1, y0)), max(0, min(height - 1, y1))))
        color = np.array([255, 0, 0], dtype=np.uint8)
        canvas[y0:min(height, y0 + thickness), x0:x1 + 1] = color
        canvas[max(0, y1 - thickness + 1):y1 + 1, x0:x1 + 1] = color
        canvas[y0:y1 + 1, x0:min(width, x0 + thickness)] = color
        canvas[y0:y1 + 1, max(0, x1 - thickness + 1):x1 + 1] = color
        return canvas

    def _make_roi_vis_images(
            self,
            batch: dict,
            params_B_T_3: torch.Tensor,
            max_batch: int = 4
    ) -> list:
        image_B_T_C_H_W = batch['obs']['image']
        B, T = image_B_T_C_H_W.shape[:2]
        num_batch = min(max_batch, B)
        num_steps = min(params_B_T_3.shape[1], T)
        images = []
        for batch_idx in range(num_batch):
            for step_idx in range(num_steps):
                image = self._image_tensor_to_uint8(
                    image_B_T_C_H_W[batch_idx, step_idx])
                params = params_B_T_3[batch_idx, step_idx]
                boxed_image = self._draw_roi_box(image, params)
                center_x, center_y, scale = params.detach().float().cpu().tolist()
                caption = (
                    f"b={batch_idx}, t={step_idx}, "
                    f"center=({center_x:.3f}, {center_y:.3f}), scale={scale:.3f}"
                )
                images.append(wandb.Image(boxed_image, caption=caption))
        return images

    @staticmethod
    def _roi_state_max_abs_diff(policy_a, policy_b) -> float:
        state_a = policy_a.roi_net.state_dict()
        state_b = policy_b.roi_net.state_dict()
        max_diff = 0.0
        for name, value_a in state_a.items():
            if name not in state_b or not torch.is_tensor(value_a):
                continue
            value_b = state_b[name]
            if not torch.is_tensor(value_b):
                continue
            diff = (
                value_a.detach().float()
                - value_b.detach().to(device=value_a.device).float()
            ).abs().max()
            max_diff = max(max_diff, float(diff.item()))
        return max_diff

    @staticmethod
    def _roi_output_stats(policy, batch: dict) -> dict:
        was_training = policy.training
        policy.eval()
        try:
            with torch.no_grad():
                _, roi_out = policy._build_roi_obs(batch["obs"])
                params = roi_out["params_B_T_3"].detach().float()
                scale = params[..., 2]
                center_abs = params[..., :2].abs()
                return {
                    "scale_mean": float(scale.mean().item()),
                    "scale_min": float(scale.min().item()),
                    "scale_max": float(scale.max().item()),
                    "center_abs_mean": float(center_abs.mean().item()),
                }
        finally:
            policy.train(was_training)

    def run(self):  # Main Entry
        cfg = copy.deepcopy(self.cfg)

        # =========================================================
        # 直接获取在 __init__ 中安全初始化好的 accelerator
        accelerator = self.accelerator
        device = accelerator.device
        # =========================================================

        # resume training
        if cfg.training.resume:
            # 注意：多卡模式下，只需主进程打印
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file() and accelerator.is_main_process:
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
            # 模型加载需在所有进程同步执行
            if lastest_ckpt_path.is_file():
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, drop_last=False, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
                               // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )

        self.model, self.optimizer, train_dataloader, val_dataloader, self.lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, train_dataloader, val_dataloader, self.lr_scheduler
        )

        # configure ema (EMA 模型通常不需要放入 DDP 包装中，手动放到 device 即可)
        ema: EMAModel = None
        if cfg.training.use_ema:
            self.ema_model.to(device)
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            # EMAModel itself is not checkpointed; keep its decay schedule aligned after resume.
            ema.optimization_step = int(self.global_step)
            ema.decay = ema.get_decay(ema.optimization_step)

        # configure env (只有主进程需要跑仿真 Rollout)
        env_runner = None
        wandb_run = None
        topk_manager = None

        if accelerator.is_main_process:
            env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)

            # To avoid time inconsistency between ckpt dir and wandb run name
            run_dir = pathlib.Path(self.output_dir)  # data/outputs/YYYY.MM.DD/HH.MM.SS_xxx
            logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            logging_cfg["name"] = f"{run_dir.parent.name}-{run_dir.name}"  # YYYY.MM.DD - HH.MM.SS_xxx

            # Save dataset statistics if supported
            if hasattr(dataset, "compute_statistics_and_save_json"):
                stats = dataset.compute_statistics_and_save_json(os.path.join(run_dir, "statistics.json"))
                from robokit.debug_utils.printer import beautiful_print
                print(f"[DEBUG] statistics saved. Content is:")
                beautiful_print(stats)

            print(f"[DEBUG] out_dir is: {run_dir}")

            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **logging_cfg
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

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        # ### [修改点 4] 仅主进程开启 tqdm ###
        disable_tqdm = not accelerator.is_main_process
        with log_ctx as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                train_losses = list()
                train_vis_batch = None
                train_vis_roi_params = None

                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=disable_tqdm) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        losses_log = {}

                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        with accelerator.accumulate(self.model):
                            want_train_vis = (
                                train_vis_batch is None
                                and (self.epoch % cfg.training.sample_every) == 0
                            )
                            losses: dict = self.model(batch, return_roi_params=want_train_vis)
                            if want_train_vis:
                                train_vis_batch = batch
                                train_vis_roi_params = losses["_roi_params_B_T_3"]
                            raw_loss = losses['total_loss']
                            accelerator.backward(raw_loss)

                            if accelerator.sync_gradients:
                                self.optimizer.step()
                                self.lr_scheduler.step()
                                self.optimizer.zero_grad()
                                if cfg.training.use_ema:
                                    ema.step(accelerator.unwrap_model(self.model))

                            for k, v in losses.items():
                                if k == 'total_loss' or k.startswith("_"):
                                    continue
                                losses_log[k] = v.item() if isinstance(v, torch.Tensor) else v

                        # ### [修改点 7] 仅主进程进行 Logging ###
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)

                        if accelerator.is_main_process:
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': self.lr_scheduler.get_last_lr()[0],
                                **losses_log
                            }

                            is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                            if not is_last_batch:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)

                        self.global_step += 1

                # ========= 评测阶段 (仅让主进程执行 Rollout 和保存，避免冲突) =========
                # 同步所有 GPU，确保大家都跑完了这个 Epoch
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    step_log['train_loss'] = np.mean(train_losses)

                    # Unwrap model to pass to runner
                    unwrapped_policy = accelerator.unwrap_model(self.model)
                    policy = unwrapped_policy
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()

                    # run rollout
                    if cfg.training.rollout_every != -1 and ((self.epoch) % cfg.training.rollout_every) == 0:
                        runner_log = env_runner.run(policy)
                        step_log.update(runner_log)

                # run validation (所有卡都可以参与计算验证集，但最简单的做法是只让主卡算，或者算完 gather。此处保持原逻辑但在主卡上收集)
                if (self.epoch % cfg.training.val_every) == 0:
                    policy_live = accelerator.unwrap_model(self.model)
                    policy_live_was_training = policy_live.training
                    policy_val = self.ema_model if cfg.training.use_ema else policy_live
                    # Validation uses EMA by default; debug logs below compare live/EMA ROI behavior on the same batch.
                    policy_val.eval()

                    val_loss_sums = {
                        "total_loss": torch.tensor(0.0, device=device),
                        "diffusion_loss": torch.tensor(0.0, device=device),
                        "roi_area_loss": torch.tensor(0.0, device=device),
                        "roi_area_weighted_loss": torch.tensor(0.0, device=device),
                    }
                    val_batches = torch.tensor(0, device=device)
                    val_vis_batch = None
                    val_vis_roi_params = None

                    try:
                        with torch.no_grad():
                            with tqdm.tqdm(
                                    val_dataloader,
                                    desc=f"Validation epoch {self.epoch}",
                                    leave=False,
                                    mininterval=cfg.training.tqdm_interval_sec,
                                    disable=disable_tqdm,
                            ) as tepoch:
                                for batch in tepoch:
                                    want_val_vis = val_vis_batch is None
                                    if want_val_vis:
                                        val_vis_batch = batch
                                    losses = policy_val.compute_loss(
                                        batch,
                                        return_roi_params=want_val_vis,
                                    )
                                    if want_val_vis:
                                        val_vis_roi_params = losses["_roi_params_B_T_3"]
                                    if isinstance(losses, torch.Tensor):
                                        losses = {"total_loss": losses}
                                    for key in val_loss_sums:
                                        if key in losses:
                                            value = losses[key]
                                            if not isinstance(value, torch.Tensor):
                                                value = torch.tensor(value, device=device)
                                            val_loss_sums[key] += value.detach()
                                    val_batches += 1
                    finally:
                        policy_live.train(policy_live_was_training)
                        if policy_live_was_training and getattr(cfg.policy, 'en_freeze_obs_encoder', False):
                            policy_live.freeze_obs_encoder()

                    stats = torch.stack([
                        val_loss_sums["total_loss"],
                        val_loss_sums["diffusion_loss"],
                        val_loss_sums["roi_area_loss"],
                        val_loss_sums["roi_area_weighted_loss"],
                        val_batches.float(),
                    ])
                    stats = accelerator.reduce(stats, reduction="sum")

                    if accelerator.is_main_process and stats[-1] > 0:
                        val_count = stats[-1]
                        step_log["val_loss"] = (stats[0] / val_count).item()
                        step_log["val_diffusion_loss"] = (stats[1] / val_count).item()
                        step_log["val_roi_area_loss"] = (stats[2] / val_count).item()
                        step_log["val_roi_area_weighted_loss"] = (stats[3] / val_count).item()
                        if val_vis_batch is not None and val_vis_roi_params is not None:
                            step_log["vis/val_roi_box"] = self._make_roi_vis_images(
                                val_vis_batch, val_vis_roi_params)
                        if cfg.training.use_ema and val_vis_batch is not None:
                            live_roi_stats = self._roi_output_stats(policy_live, val_vis_batch)
                            ema_roi_stats = self._roi_output_stats(self.ema_model, val_vis_batch)
                            for key, value in live_roi_stats.items():
                                step_log[f"debug/live_roi_{key}"] = value
                            for key, value in ema_roi_stats.items():
                                step_log[f"debug/ema_roi_{key}"] = value
                            step_log["debug/ema_live_roi_state_max_abs_diff"] = (
                                self._roi_state_max_abs_diff(policy_live, self.ema_model)
                            )
                            step_log["debug/ema_live_roi_scale_mean_abs_diff"] = abs(
                                live_roi_stats["scale_mean"] - ema_roi_stats["scale_mean"]
                            )
                            if ema is not None:
                                step_log["debug/ema_decay"] = float(ema.decay)
                                step_log["debug/ema_optimization_step"] = int(ema.optimization_step)

                # run diffusion sampling
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        obs_dict = train_sampling_batch['obs']
                        gt_action = train_sampling_batch['action']
                        result = policy.predict_action(obs_dict)
                        mse = torch.nn.functional.mse_loss(result['action_pred'], gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        if train_vis_batch is not None and train_vis_roi_params is not None:
                            step_log["vis/train_roi_box"] = self._make_roi_vis_images(
                                train_vis_batch, train_vis_roi_params)

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
                        # NOTE: why no val_loss? {'train_loss': 0.44689754078118626, 'global_step': 282, 'epoch': 0, 'lr': 5.66e-05, 'train_action_mse_error': 0.2813175352475294}
                        monitor_key = cfg.checkpoint.topk.monitor_key
                        assert monitor_key in metric_dict, f"skip topk ckpt: monitor_key `{monitor_key}` not in step_log."
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                live_model = accelerator.unwrap_model(self.model)
                live_model.train()
                if getattr(cfg.policy, 'en_freeze_obs_encoder', False):
                    live_model.freeze_obs_encoder()
                if cfg.training.use_ema:
                    self.ema_model.eval()

                if accelerator.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)

                self.epoch += 1
                accelerator.wait_for_everyone()  # 确保下一轮开始前所有进程同步


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
