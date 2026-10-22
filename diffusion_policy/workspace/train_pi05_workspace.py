"""Single-machine dense pi0.5 SFT with optional complete training recovery."""
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from diffusion_policy.common.tcl_data_util import (
    PI05_NORM, load_tcl_statistics, checkpoint_normalization, make_normalization, norm_methods,
)

from diffusion_policy.common.pi05_training_util import (
    Pi05BatchSampler, Pi05EMA, atomic_save, begin_update_diagnostics, capture_rng,
    clip_gradients, distributed_ema, finish_update_diagnostics, restore_rng, seed_training,
)
from diffusion_policy.common.foveation_training_util import (
    cosine_schedule_value, projected_dual_ascent,
)
from diffusion_policy.common.pi05_eval_util import Pi05Evaluation, init_wandb, log_wandb
from diffusion_policy.common.foveation_video_util import make_foveation_vis_log
from diffusion_policy.common.libero_task_util import LIBERO_TASK_NAMES
from diffusion_policy.dataset.libero_dataset import LiberoFTDataset
from diffusion_policy.dataset.tcl_dataset import TCLFoveationDataset
from diffusion_policy.env_runner.tcl_foveation_video_runner import TCLFoveationVideoRunner
from diffusion_policy.model.pi05.checkpoint import sha256_file
from diffusion_policy.model.pi05.config import Pi05Config
from diffusion_policy.model.pi05.optimizer import FP32MasterAdamW
from diffusion_policy.policy.pi05_policy import Pi05Policy
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class TrainPi05Workspace(BaseWorkspace):
    """Own DDP updates, rank-zero EMA and explicit versioned training checkpoints."""

    def _barrier(self):
        if self.world_size > 1:
            # Tell NCCL which local CUDA device owns this rank; without this,
            # a barrier may infer the device from an incomplete process-group map.
            dist.barrier(device_ids=[self.device.index])

    def _signature(self):
        cfg = self.cfg
        # Operational options may change; settings affecting numerical continuation may not.
        return dict(world_size=self.world_size, stage_b=bool(cfg.training.stage_b),
                    model=asdict(Pi05Config(**cfg.model)),
                    global_batch=cfg.training.global_batch, micro_batch=cfg.training.micro_batch,
                    seed=cfg.training.seed, gradient_clip=cfg.training.gradient_clip,
                    optimizer=OmegaConf.to_container(cfg.optimizer, resolve=True),
                    ema_decay=cfg.ema_decay,
                    stage_b_options={key: cfg.training.get(key) for key in (
                        'downstream_finetune_start_step', 'target_num_patches_start',
                        'target_num_patches_decay_steps', 'paired_loss_weight',
                        'temporal_loss_weight', 'budget_lambda_init', 'budget_lambda_lr',
                        'temporal_adaptive', 'paired_node_sampling', 'pairing_seed')},
                    validation=dict(seed=cfg.validation.seed, batch_size=cfg.validation.batch_size,
                                    max_batches=cfg.validation.max_batches))

    def _initialize(self):
        cfg = self.cfg
        self.is_tcl = str(cfg.get('dataset_type', 'libero')) == 'tcl'
        self.stage_b = bool(cfg.training.stage_b)
        if self.stage_b and cfg.pretrained_ckpt_mode not in ('best', 'latest', None):
            raise ValueError('pretrained_ckpt_mode must be best, latest, or null')
        if self.stage_b:
            if cfg.training.target_num_patches_start is None:
                raise ValueError('StageB requires training.target_num_patches_start')
            if cfg.model.foveation_target_num_patches is None:
                raise ValueError('StageB requires model.foveation_target_num_patches')
            if cfg.training.paired_node_sampling != 'reachable':
                raise ValueError('Only training.paired_node_sampling=reachable is supported')
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if not torch.cuda.is_available():
            raise RuntimeError('pi0.5 SFT requires CUDA; use the server training environment')
        torch.cuda.set_device(local_rank)
        self.device = torch.device('cuda', local_rank)
        if self.world_size > 1:
            # Explicit device_id avoids NCCL's "device used by this process is
            # currently unknown" warning and prevents an ambiguous barrier map.
            dist.init_process_group('nccl', timeout=timedelta(
                seconds=cfg.training.ddp_timeout_seconds), device_id=self.device)
        if cfg.model.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('The selected device does not support bf16')
        if (cfg.training.max_steps <= 0 or cfg.validation.every <= 0
                or cfg.validation.max_batches <= 0 or cfg.validation.batch_size <= 0
                or cfg.checkpoint.full_every <= 0 or cfg.training.log_every <= 0
                or cfg.training.gradient_clip <= 0 or cfg.optimizer.warmup_steps < 0):
            raise ValueError('Invalid pi0.5 step, validation, checkpoint or optimizer configuration')
        if cfg.checkpoint.monitor not in ('success_rate', 'val_loss'):
            raise ValueError('checkpoint.monitor must be success_rate or val_loss')
        if cfg.checkpoint.monitor == 'success_rate' and not cfg.rollout.enabled:
            raise ValueError('success_rate selection requires rollout.enabled=true')
        if cfg.rollout.enabled and cfg.rollout.every <= 0:
            raise ValueError('rollout.every must be positive')
        self.first_rollout_step = cfg.rollout.get('first_step', 0)
        if (type(self.first_rollout_step) is not int or self.first_rollout_step < 0):
            raise ValueError('rollout.first_step must be a nonnegative integer')
        self.sample_every = int(cfg.training.get('sample_every', 0))
        self.sample_batch_size = int(cfg.training.get('sample_batch_size', 8))
        if self.sample_every < 0 or self.sample_batch_size <= 0:
            raise ValueError('sample_every must be nonnegative and sample_batch_size positive')
        self.topology_samples_per_task = int(cfg.training.get('topology_samples_per_task', 64))
        self.sample_diagnostic_batch_size = int(cfg.training.get('sample_diagnostic_batch_size', 8))
        if self.topology_samples_per_task < 2 or self.sample_diagnostic_batch_size <= 0:
            raise ValueError('Require topology_samples_per_task >= 2 and sample_diagnostic_batch_size > 0')
        self.sample_enabled = cfg.model.visual_input_mode == 'mixed_foveation' and self.sample_every > 0
        self.offline_video_enabled = (self.is_tcl
            and cfg.get("offline_video", {}).get("enabled", False)
            and cfg.model.visual_input_mode == "mixed_foveation")
        if self.offline_video_enabled and cfg.rollout.every <= 0:
            raise ValueError("TCL offline video requires rollout.every > 0")
        self.train_sampling_obs = None
        self.train_sampling_action = None
        seed_training(cfg.training.seed + self.rank)
        self.output = Path(self.output_dir).expanduser().resolve()
        self.checkpoints = self.output / 'checkpoints'
        if self.rank == 0:
            self.output.mkdir(parents=True, exist_ok=True)
            # Avoid silently mixing runs, including continuation into an old log directory.
            if (self.output / 'train.jsonl').exists():
                raise FileExistsError('Use a new OUTPUT_DIR for each run, including resume')
            OmegaConf.save(cfg, self.output / 'config.yaml', resolve=True)
        self._barrier()

        self.norm = norm_methods(OmegaConf.select(cfg, 'task.dataset.norm'), PI05_NORM)
        if not self.is_tcl and self.norm != PI05_NORM:
            raise ValueError('LIBERO Pi0.5 only supports its existing quantile normalization')
        if cfg.get('asset_dir') is not None:
            cfg.asset_dir = str(Path(cfg.asset_dir).expanduser().resolve())
            if self.rank == 0:
                OmegaConf.save(cfg, self.output / 'config.yaml', resolve=True)
        payload = None
        if cfg.resume:
            # Memory mapping avoids materializing all ranks' full optimizer/EMA files in RAM.
            payload = torch.load(Path(cfg.resume).expanduser(), map_location='cpu',
                                 weights_only=True, mmap=True)
            if payload.get('format') != 'pi05_training_v1':
                raise ValueError('resume requires latest_full.ckpt; EMA files are inference-only')
            saved_signature = dict(payload['signature'])
            saved_signature['model'] = asdict(Pi05Config(**saved_signature['model']))
            if saved_signature != self._signature():
                raise ValueError('Resume requires the same world/batch/model/optimizer/seed/validation settings')
            self.dataset_metadata = payload['dataset_metadata']
            extra = payload['policy']['_extra_state']
            if asdict(Pi05Config(**extra['config'])) != self._signature()['model']:
                raise ValueError('Saved model configuration differs from training configuration')
            self.policy = Pi05Policy(config=extra['config'], assets=extra['assets'])
            self.policy.load_state_dict(payload['policy'], strict=True)
            self.dataset = self._make_dataset(dataset_metadata=self.dataset_metadata)
            self.provenance = payload['provenance']
        else:
            inherited_metadata = None
            if self.stage_b and self.is_tcl:
                pretrained_path = self._resolve_pretrained_ckpt(cfg.pretrained_ckpt, cfg.pretrained_ckpt_mode)
                extra = self._read_pretrained_extra(pretrained_path)
                inherited_metadata = extra['assets'].get('dataset_metadata')
            self.dataset = self._make_dataset(asset_dir=cfg.asset_dir, dataset_metadata=inherited_metadata)
            self.dataset_metadata = self.dataset.get_dataset_metadata()
            if self.stage_b:
                if not self.is_tcl:
                    pretrained_path = self._resolve_pretrained_ckpt(
                        cfg.pretrained_ckpt, cfg.pretrained_ckpt_mode)
                    extra = self._read_pretrained_extra(pretrained_path)
                self._validate_stage_a_config(extra.get('config'), cfg.model)
                assets = extra['assets']
                if self.is_tcl:
                    if inherited_metadata is None:
                        statistics, source = load_tcl_statistics(self.dataset_metadata, cfg.asset_dir, self.norm, int(cfg.state_dim))
                        for key, values in statistics.items():
                            self.dataset_metadata.setdefault('statistics', {}).setdefault(key, {}).update(values)
                        print(f'[TCL] legacy Stage A metadata fallback: {source}')
                    checkpoint_normalization(self.dataset_metadata, assets, self.norm, int(cfg.state_dim))
                self.policy = Pi05Policy(assets=assets, config=cfg.model)
                self._load_stage_a_weights(self.policy, pretrained_path)
                artifact_path = cfg.get('foveation_pretrained_path')
                if artifact_path is None:
                    raise ValueError('StageB requires foveation_pretrained_path for DP routing initialization')
                self.policy.load_foveation_artifact(artifact_path)
                if self.is_tcl and self.policy.foveation_artifact['task_mapping'] != \
                        self.dataset_metadata['task_mapping']:
                    raise ValueError('DP routing artifact task mapping differs from TCL dataset mapping')
                self.provenance = dict(stage_a_checkpoint=str(pretrained_path),
                                       stage_a_checkpoint_sha256=sha256_file(pretrained_path),
                                       stage_a_config=extra.get('config'),
                                       stage_a_metadata=extra.get('assets', {}).get('training'),
                                       stage_b_config=OmegaConf.to_container(cfg, resolve=True),
                                       foveation_pretrained=dict(
                                           encoder_type=cfg.model.foveation_global_encoder_type,
                                           feature_dim=int(cfg.model.foveation_global_feature_dim),
                                           resize=cfg.model.foveation_global_encoder_resize,
                                           torch_home=cfg.model.foveation_torch_home,
                                           frozen=bool(cfg.model.foveation_freeze_global_encoder)),
                                       stage_b_initialization=self.policy.stage_b_init_report)
            else:
                base = Path(cfg.base_dir).expanduser().resolve()
                manifest = json.loads((base / 'manifest.json').read_text())
                for name in ('config.json', 'tokenizer.model'):
                    if sha256_file(base / name) != manifest['output_sha256'][name]:
                        raise ValueError(f'Base asset differs from manifest: {name}')
                base_config = json.loads((base / 'config.json').read_text())
                base_config['gradient_checkpointing'] = cfg.model.gradient_checkpointing
                base_model_config = asdict(Pi05Config(**base_config))
                sft_model_config = asdict(Pi05Config(**cfg.model))
                for key in ('attention_backend', 'image_size', 'visual_input_mode',
                            'action_horizon', 'n_action_steps',
                            'foveation_split_mode', 'foveation_random_seed',
                            'foveation_router_type', 'render_image_orientation',
                            'foveation_global_encoder_type', 'foveation_global_feature_dim',
                            'foveation_global_encoder_resize', 'foveation_torch_home',
                            'foveation_freeze_global_encoder',
                            'foveation_position_encoding', 'foveation_level_embedding',
                            'foveation_level_adapter', 'foveation_patch_level_adapter',
                            'foveation_patch_level_adapter_channels',
                            'foveation_apt_position_encoding'):
                    base_model_config.pop(key, None)
                    sft_model_config.pop(key, None)
                if base_model_config != sft_model_config:
                    raise ValueError('Base model configuration differs from SFT configuration')
                policy_kwargs = dict(norm_stats_path=Path(cfg.asset_dir).expanduser() / 'norm_stats.json')
                if self.is_tcl:
                    statistics, source = load_tcl_statistics(self.dataset_metadata, cfg.asset_dir, self.norm, int(cfg.state_dim))
                    for key, values in statistics.items():
                        self.dataset_metadata.setdefault('statistics', {}).setdefault(key, {}).update(values)
                    normalization = make_normalization(self.dataset_metadata, self.norm, defaults=PI05_NORM,
                                                       dimensions=dict(state=int(cfg.state_dim), action=7))
                    policy_kwargs = dict(assets=dict(normalization=normalization,
                                                     tokenizer_proto=(base / 'tokenizer.model').read_bytes(),
                                                     norm_stats={'norm_stats': statistics},
                                                     norm_stats_source=str(Path(cfg.asset_dir).expanduser() / source)
                                                     if source == 'dataset_metadata.json' else source))
                    print(f'[TCL] Stage A statistics: {source} ({cfg.asset_dir}); norm={self.norm}')
                self.policy = Pi05Policy(model_path=base, tokenizer_path=base / 'tokenizer.model',
                                         config=cfg.model, **policy_kwargs)
                base_hash = self.policy.get_extra_state()['assets']['base_weights_sha256']
                if base_hash != manifest['output_sha256']['model.safetensors']:
                    raise ValueError('Base model weights differ from manifest')
                metadata_bytes = json.dumps(self.dataset_metadata, sort_keys=True).encode()
                self.provenance = dict(base_weights_sha256=base_hash,
                                       dataset_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest())
                if not self.is_tcl:
                    self.provenance['norm_stats_sha256'] = self.dataset_metadata['norm_stats_sha256']
        if self.is_tcl:
            normalization = checkpoint_normalization(self.dataset_metadata, self.policy._assets,
                                                       self.norm, int(cfg.state_dim))
            self.policy._assets['normalization'] = copy.deepcopy(normalization)
            self.policy.preprocessor.normalization = copy.deepcopy(normalization)
            self.dataset.normalization = copy.deepcopy(normalization)
            self.provenance['dataset_metadata_sha256'] = hashlib.sha256(
                json.dumps(self.dataset_metadata, sort_keys=True).encode()).hexdigest()
            self.policy._assets['dataset_metadata'] = copy.deepcopy(self.dataset_metadata)
            self.dataset.dataset_metadata = copy.deepcopy(self.dataset_metadata)
            inheritance = 'resume checkpoint' if cfg.resume else ('Stage A checkpoint' if self.stage_b else 'prepared assets')
            print(f'[TCL] normalization={self.norm}; inherited from {inheritance}; router TCP=DP artifact contract')
            if self.rank == 0:
                (self.output / 'dataset_metadata.json').write_text(json.dumps(self.dataset_metadata, indent=2) + '\n')
        self.policy.to(device=self.device).train()
        self.phase = 'router_only' if self.stage_b else 'native'
        if self.stage_b:
            # Register both future branches with DDP before applying phase masks.
            self.policy.requires_grad_(True)
            self.policy.set_target_num_patches(cfg.training.target_num_patches_start)
            for net in self.policy.foveation_net.nets.values():
                net.budget_lambda = float(cfg.training.budget_lambda_init)
        self.optimizer = FP32MasterAdamW(self.policy.parameters(), lr=cfg.optimizer.lr,
                                         betas=tuple(cfg.optimizer.betas), eps=cfg.optimizer.eps,
                                         weight_decay=cfg.optimizer.weight_decay)
        # LambdaLR index is the number of completed updates: first update uses index zero.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min((step + 1) / (cfg.optimizer.warmup_steps + 1), 1.0))
        self.global_step, self.best_loss = 0, float('inf')
        self.best_metric = None
        self.wandb_run = None
        self.evaluation = None
        self.ema = Pi05EMA(self.policy, cfg.ema_decay) if self.rank == 0 else None
        self.sampler = Pi05BatchSampler(len(self.dataset), cfg.training.global_batch,
                                        cfg.training.micro_batch, self.rank, self.world_size, cfg.training.seed)
        self.loader_generator = torch.Generator().manual_seed(cfg.training.seed + self.rank + 10000)
        self.loader = DataLoader(self.dataset, batch_sampler=self.sampler,
                                 num_workers=cfg.dataloader.num_workers, generator=self.loader_generator,
                                 persistent_workers=False,
                                 multiprocessing_context='spawn' if cfg.dataloader.num_workers else None)
        # Some language output branches do not contribute to action loss. Keep their true graph semantics.
        self.model = (DistributedDataParallel(self.policy, device_ids=[local_rank],
                                              find_unused_parameters=True, broadcast_buffers=False,
                                              gradient_as_bucket_view=True)
                      if self.world_size > 1 else self.policy)
        if self.stage_b:
            self.policy.set_stage_b_phase(self.phase)
        self._set_ema_phase()
        runtime = None
        if payload is not None:
            self.optimizer.load_state_dict(payload['optimizer'])
            self.scheduler.load_state_dict(payload['scheduler'])
            self.global_step, self.best_loss = payload['global_step'], payload['best_loss']
            selection = payload.get('selection', dict(monitor='val_loss', best=self.best_loss))
            if selection['monitor'] == cfg.checkpoint.monitor:
                self.best_metric = selection['best']
            self.sampler.load_state_dict(payload['sampler'])
            runtime = payload['rank_states'][self.rank]
            if self.ema is not None:
                self.ema.load_state_dict(payload['ema'])
            if (self.sampler.epoch * self.sampler.updates_per_epoch + self.sampler.update != self.global_step
                    or payload['ema']['updates'] != self.global_step
                    or self.scheduler.last_epoch != self.global_step):
                raise ValueError('Checkpoint sampler/scheduler/EMA steps disagree')
            if cfg.training.max_steps <= self.global_step:
                raise ValueError('training.max_steps must exceed the restored global_step')
            if self.stage_b and cfg.training.downstream_finetune_start_step is not None \
                    and self.global_step >= int(cfg.training.downstream_finetune_start_step):
                self.phase = 'downstream'
            stage_b_state = payload.get('stage_b_state', {})
            if self.stage_b and stage_b_state:
                self.phase = stage_b_state.get('phase', self.phase)
                if stage_b_state.get('target_num_patches') is not None:
                    self.policy.set_target_num_patches(stage_b_state['target_num_patches'])
                for net, value in zip(self.policy.foveation_net.nets.values(),
                                      stage_b_state.get('budget_lambda', [])):
                    net.budget_lambda = float(value)
                pairing = stage_b_state.get('pairing')
                if pairing is not None:
                    self.dataset.set_temporal_pairing(
                        pairing.get('enabled', False), pairing.get('seed'), pairing.get('epoch'))
                if stage_b_state.get('router_temperature') is not None:
                    self.policy.set_router_temperature(stage_b_state['router_temperature'])
            if self.stage_b:
                self.policy.set_stage_b_phase(self.phase)
                self._set_ema_phase()
                if self.phase == 'downstream':
                    self.dataset.set_temporal_pairing(False)
                    self._clear_router_optimizer_state()
        # Construct readers/DDP/iterator before restoring the training random streams.
        if runtime is not None:
            self.loader_generator.set_state(runtime['loader_generator'])
            self.policy.set_split_sampler_state(runtime.get('split_sampler'))
        self.dataset.set_pairing_epoch(self.sampler.epoch)
        self.iterator = iter(self.loader)
        self._budget_stats_V_2 = torch.zeros(
            (len(self.policy.foveation_net.view_keys), 2),
            device=self.device, dtype=torch.float64,
        ) if self.stage_b else None
        if runtime is not None:
            # Mid-epoch reconstruction must not consume an additional loader seed. At an
            # epoch boundary the next iterator also consumes one seed in uninterrupted runs.
            if self.sampler.update != 0:
                self.loader_generator.set_state(runtime['loader_generator'])
            restore_rng(runtime['rng'])
        del payload
        if self.rank == 0:
            self.validation_dataset = self.dataset.get_validation_dataset()
            generator = torch.Generator().manual_seed(cfg.validation.seed)
            count = min(len(self.validation_dataset), cfg.validation.batch_size * cfg.validation.max_batches)
            indices_N = torch.randperm(len(self.validation_dataset), generator=generator)[:count]
            self.validation_loader = DataLoader(Subset(self.validation_dataset, indices_N.tolist()),
                                                batch_size=cfg.validation.batch_size, num_workers=0,
                                                generator=torch.Generator().manual_seed(cfg.validation.seed))
            # Dataset construction must not perturb the resumed main-process random stream.
            if runtime is not None:
                restore_rng(runtime['rng'])
            self.versions = {name: importlib.metadata.version(name) for name in
                        ('torch', 'torchvision', 'numpy', 'h5py', 'hydra-core', 'sentencepiece', 'robokit')}
            self.sources = json.loads((Path(__file__).resolve().parents[1] / 'model/pi05/sources.json').read_text())
            (self.output / 'run_metadata.json').write_text(json.dumps(dict(
                signature=self._signature(), provenance=self.provenance, versions=self.versions,
                sources=self.sources,
                resume=cfg.resume, accumulation=self.sampler.accumulation,
                train_windows=len(self.dataset), val_windows=len(self.validation_dataset),
                trainable_parameters=sum(param_D.numel() for param_D in self.policy.parameters()),
            ), indent=2) + '\n')
            print(f'[pi0.5 SFT] world={self.world_size}, micro={cfg.training.micro_batch}, '
                  f'accumulation={self.sampler.accumulation}, global_batch={cfg.training.global_batch}; '
                  f'step={self.global_step}, output={self.output}', flush=True)
        runner_cfg = OmegaConf.create(OmegaConf.to_container(cfg.runner, resolve=True))
        if runner_cfg.hdf5_fns is None and not self.is_tcl:
            sources = [Path(source) for source in self.dataset_metadata['sources']]
            runner_cfg.dataset_subname = [str(source.parent) for source in sources]
            runner_cfg.hdf5_fns = [source.name for source in sources]
        if self.is_tcl and runner_cfg.hdf5_fns is None:
            runner_cfg.hdf5_fns = []
        self.runner_cfg = runner_cfg
        self.evaluation = Pi05Evaluation(runner_cfg, self.world_size, self.rank, local_rank,
                                         len(runner_cfg.hdf5_fns) if cfg.rollout.enabled else 0)
        self.evaluation.call(self._init_logging)

    def _make_dataset(self, *, asset_dir=None, dataset_metadata=None):
        """Instantiate the configured LIBERO or raw multi-task TCL dataset."""
        cfg = self.cfg
        if self.is_tcl:
            source_cfg = dataset_metadata.get('sources') if dataset_metadata is not None else None
            data_roots = cfg.data_roots if source_cfg is None else [item['root'] for item in source_cfg]
            h5_paths = cfg.h5_paths if source_cfg is None else [item['h5_path'] for item in source_cfg]
            return TCLFoveationDataset(
                data_roots=data_roots,
                h5_paths=h5_paths,
                output_mode='raw', norm=self.norm,
                image_resize_wh_ratio=OmegaConf.select(cfg, 'task.dataset.image_resize_wh_ratio'),
                shape_meta={
                    'obs': {key: {'shape': [3, int(cfg.model.image_size), int(cfg.model.image_size)],
                                  'type': 'rgb'} for key in ('image', 'gripper')},
                    'action': {'shape': [7]},
                },
                task_mapping_path=cfg.task_mapping_path,
                asset_dir=asset_dir,
                dataset_metadata=dataset_metadata,
                split='train',
                horizon=int(cfg.model.action_horizon),
                state_dim=int(cfg.state_dim),
                val_ratio=float(cfg.val_ratio),
                seed=int(cfg.dataset_seed),
                stats_seed=int(cfg.stats_seed),
                stats_max_frames_per_task=int(cfg.stats_max_frames_per_task),
                use_h5=bool(cfg.use_h5),
                temporal_pairing=self.stage_b and bool(cfg.training.temporal_adaptive),
                pairing_seed=int(cfg.training.pairing_seed),
            )
        if dataset_metadata is not None:
            return LiberoFTDataset(
                cfg.dataset_root,
                cfg.runner.get('dataset_subname', 'libero_force_256/libero_10'),
                cfg.runner.get('hdf5_fns'),
                horizon=int(cfg.model.get('action_horizon', 10)), pad_before=0, pad_after=0,
                shape_meta={'obs': {}, 'action': {'shape': [7]}},
                output_mode='raw', asset_dir=asset_dir,
                dataset_metadata=dataset_metadata, split='train',
                temporal_pairing=self.stage_b and bool(cfg.training.temporal_adaptive),
                pairing_seed=int(cfg.training.pairing_seed))
        return LiberoFTDataset(
            cfg.dataset_root,
            cfg.runner.get('dataset_subname', 'libero_force_256/libero_10'),
            cfg.runner.get('hdf5_fns'),
            horizon=int(cfg.model.get('action_horizon', 10)), pad_before=0, pad_after=0,
            shape_meta={'obs': {}, 'action': {'shape': [7]}},
            output_mode='raw', asset_dir=asset_dir, split='train',
            temporal_pairing=self.stage_b and bool(cfg.training.temporal_adaptive),
            pairing_seed=int(cfg.training.pairing_seed))

    @staticmethod
    def _resolve_pretrained_ckpt(path, mode):
        """Resolve a Stage A file or run directory to one EMA checkpoint."""
        if path is None:
            raise ValueError('stageB requires pretrained_ckpt')
        path = Path(path).expanduser()
        if path.is_dir():
            if mode is None:
                raise ValueError('A run directory requires pretrained_ckpt_mode=best or latest')
            filename = 'best_ema.pt' if mode == 'best' else 'latest_ema.pt'
            path = path / 'checkpoints' / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()

    @staticmethod
    def _read_pretrained_extra(path):
        """Read self-contained policy metadata from a StageA EMA checkpoint."""
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if payload.get('format') == 'pi05_training_v1':
            raise ValueError('pretrained_ckpt accepts Stage A EMA files; use resume for full checkpoints')
        state = payload
        if '_extra_state' not in state:
            raise ValueError('Stage A checkpoint must contain policy _extra_state')
        extra = state['_extra_state']
        training_metadata = extra.get('assets', {}).get('training')
        if training_metadata is not None and training_metadata.get('weights') not in (None, 'ema'):
            raise ValueError('pretrained_ckpt must point to an EMA policy checkpoint')
        return extra

    @staticmethod
    def _validate_stage_a_config(source_config, target_config):
        """Check that StageA and StageB share all non-router model contracts."""
        if source_config is None:
            raise ValueError('Stage A EMA metadata must include the saved model config')
        source = asdict(Pi05Config(**source_config))
        target = asdict(Pi05Config(**target_config))
        if source['visual_input_mode'] != 'mixed_foveation' or target['visual_input_mode'] != 'mixed_foveation':
            raise ValueError('StageB initialization requires a mixed-foveation StageA EMA')
        if source['foveation_split_mode'] is None or source['foveation_router_type'] != 'none':
            raise ValueError('StageA EMA must use a fixed or random topology with router_type=none')
        if target['foveation_split_mode'] is not None or target['foveation_router_type'] != 'node_aligned_fpn':
            raise ValueError('StageB initialization requires node_aligned_fpn with no fixed split mode')
        stage_specific = {
            'foveation_split_mode', 'foveation_router_type',
            'foveation_global_encoder_type', 'foveation_global_feature_dim',
            'foveation_global_encoder_resize', 'foveation_torch_home',
            'foveation_freeze_global_encoder',
            'foveation_router_layers', 'foveation_router_hidden_dim',
            'foveation_router_heads', 'foveation_router_mlp_ratio',
            'foveation_router_dropout', 'foveation_router_cond_dim',
            'foveation_router_temperature', 'foveation_target_num_patches',
            'foveation_random_topology_prob',
            'foveation_budget_loss_weight', 'foveation_budget_loss_type',
            'foveation_budget_base_cost_weight', 'foveation_budget_probability_type',
            'foveation_budget_elu_tau',
        }
        for key in stage_specific:
            source.pop(key, None)
            target.pop(key, None)
        if source != target:
            raise ValueError('Stage A and StageB model contracts differ outside router settings')

    @staticmethod
    def _load_stage_a_weights(policy, path):
        """Load Stage A EMA weights with an explicit StageB extension whitelist."""
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if payload.get('format') == 'pi05_training_v1':
            raise ValueError('Stage A initialization requires an EMA policy, not a full checkpoint')
        state = payload
        current = policy.state_dict()
        loaded, archived_source_only, unexpected = [], [], []
        allowed_missing_prefixes = (
            'foveation_net.nets.image.global_encoder.',
            'foveation_net.nets.gripper.global_encoder.',
            'foveation_net.nets.image.router.',
            'foveation_net.nets.gripper.router.',
            'router_condition_projection.',
        )
        source_only_prefixes = (
            'foveation_net.nets.image.fixed_split_hard_1_1_M',
            'foveation_net.nets.gripper.fixed_split_hard_1_1_M',
        )
        for name, value in state.items():
            if name == '_extra_state':
                continue
            if name in current and current[name].shape == value.shape:
                current[name].copy_(value)
                loaded.append(name)
            elif name in current:
                unexpected.append(f'{name} (shape {tuple(value.shape)} != {tuple(current[name].shape)})')
            elif name.startswith(source_only_prefixes):
                digest = hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                archived_source_only.append(dict(name=name, shape=tuple(value.shape), sha256=digest))
            else:
                unexpected.append(name)
        missing = [name for name in current if name not in loaded]
        disallowed_missing = [name for name in missing
                              if not name.startswith(allowed_missing_prefixes)
                              and name != '_extra_state']
        if unexpected or disallowed_missing:
            raise ValueError(
                'Stage A EMA does not match the StageB extension contract: '
                f'unexpected={unexpected[:5]}, missing={disallowed_missing[:5]}')
        policy.load_state_dict(current, strict=True)
        source_config = state['_extra_state']['config']
        policy.stage_b_init_report = dict(
            loaded_keys=loaded,
            allowed_missing=[name for name in missing if name not in disallowed_missing],
            archived_source_only=archived_source_only,
            unexpected_keys=unexpected,
            checkpoint=str(path),
            source_config=source_config,
        )

    def _init_logging(self):
        if self.rank == 0:
            self.wandb_run = init_wandb(self.cfg.logging, self.output,
                OmegaConf.to_container(self.cfg, resolve=True),
                dict(self.provenance, effective_runner=OmegaConf.to_container(self.runner_cfg)))

    def _log(self, record):
        if self.rank == 0:
            with (self.output / 'train.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(f'[pi0.5 SFT] {json.dumps(record, allow_nan=False)}', flush=True)

    def _foveation_metrics(self):
        """Return scalar foveation metrics using the diffusion-policy names."""
        stats = self.policy.last_visual_stats
        if not stats:
            return {}
        camera_tokens = stats.get('camera_tokens', {})
        expected_camera_tokens = stats.get('expected_camera_tokens', {})
        budget_losses = stats.get('budget_loss', {})
        random_fractions = stats.get('random_topology_fraction', {})
        metrics = {}
        token_means = []
        total_token_mean = 0.0
        random_values = []
        for key, values in camera_tokens.items():
            # Routing stores counts as nested [B, T] lists, including when T=1.
            counts_B_T = torch.tensor(values, dtype=torch.float64, device='cpu')
            mean = counts_B_T.mean().item()
            metrics[f'views/{key}/num_patches_mean'] = mean
            token_means.append(mean)
            total_token_mean += mean
            if key in expected_camera_tokens:
                metrics[f'views/{key}/expected_num_patches_mean'] = float(
                    torch.tensor(expected_camera_tokens[key], dtype=torch.float64).mean().item())
            if key in budget_losses:
                metrics[f'views/{key}/budget_loss'] = float(budget_losses[key])
            if key in random_fractions:
                fraction = float(random_fractions[key])
                metrics[f'views/{key}/random_topology_fraction'] = fraction
                random_values.append(fraction)
        if token_means:
            metrics['num_patches_mean'] = sum(token_means) / len(token_means)
            metrics['total_num_patches_mean'] = total_token_mean
        if random_values:
            metrics['random_topology_fraction'] = sum(random_values) / len(random_values)
        return metrics

    @staticmethod
    def _topology_diversity(foveation_outputs, num_bank_patches):
        """Measure mean pairwise leaf-set Jaccard distance across sampled trees."""
        distance_sum_scalar = None
        num_distances = 0
        for foveation_out in foveation_outputs.values():
            indices_B_T_N = foveation_out['patch_bank_indices_B_T_N']
            valid_B_T_N = foveation_out['token_valid_mask_B_T_N']
            batch_size, time_steps, _ = indices_B_T_N.shape
            if batch_size < 2:
                continue
            leaf_counts_B_T_K = torch.zeros(
                (batch_size, time_steps, num_bank_patches), dtype=torch.long,
                device=indices_B_T_N.device)
            leaf_counts_B_T_K.scatter_add_(
                2, indices_B_T_N.clamp_min(0), valid_B_T_N.to(dtype=torch.long))
            leaf_float_B_T_K = (leaf_counts_B_T_K > 0).float()
            intersection_T_B_B = torch.einsum('btk,ctk->tbc', leaf_float_B_T_K, leaf_float_B_T_K)
            leaf_count_T_B = leaf_float_B_T_K.sum(-1).transpose(0, 1)
            union_T_B_B = (leaf_count_T_B[:, :, None] + leaf_count_T_B[:, None, :]
                           - intersection_T_B_B)
            distance_T_B_B = 1.0 - intersection_T_B_B / union_T_B_B.clamp_min(1.0)
            pair_mask_B_B = torch.triu(
                torch.ones((batch_size, batch_size), dtype=torch.bool,
                           device=indices_B_T_N.device), diagonal=1)
            distances_T_P = distance_T_B_B[:, pair_mask_B_B]
            distance_sum_scalar = distances_T_P.sum() if distance_sum_scalar is None else (
                distance_sum_scalar + distances_T_P.sum())
            num_distances += distances_T_P.numel()
        if distance_sum_scalar is None or num_distances == 0:
            return 0.0
        return float((distance_sum_scalar / num_distances).item())

    @torch.no_grad()
    def _validate(self):
        if self.rank != 0:
            return {}
        rng, was_training = capture_rng(), self.policy.training
        split_sampler = self.policy.get_split_sampler_state()
        try:
            seed_training(self.cfg.validation.seed)
            self.policy.eval()
            with self.ema.apply(self.policy):
                total, count = 0.0, 0
                visual_sums = {}
                for batch in self.validation_loader:
                    # Validation measures native flow matching even while the
                    # live StageB phase trains the differentiable paired loss.
                    loss_B = self.policy(batch, paired=False, reduction='none')
                    total += loss_B.sum().item()
                    count += len(loss_B)
                    for key, value in self._foveation_metrics().items():
                        visual_sums[key] = visual_sums.get(key, 0.0) + value * len(loss_B)
            loss = total / count
            if not torch.isfinite(torch.tensor(loss)):
                raise FloatingPointError('Nonfinite EMA validation loss')
            self.best_loss = min(loss, self.best_loss)
            metrics = {
                'diffusion_loss': loss,
                'budget_loss': 0.0,
                'total_loss': loss,
            }
            metrics.update({key: value / count for key, value in visual_sums.items()})
            metrics = {f'val/{key}': value for key, value in metrics.items()}
            metrics['val/loss'] = loss  # Checkpoint compatibility with older configs.
            self._log(dict(event='validation', epoch=self.sampler.epoch,
                           global_step=self.global_step, **metrics))
            return metrics
        finally:
            self.policy.train(was_training)
            self.policy.set_split_sampler_state(split_sampler)
            restore_rng(rng)

    def _rollout(self):
        output = self.output / 'rollouts' / f'step_{self.global_step:08d}'
        metadata = dict(weights='ema', global_step=self.global_step, **self.provenance)
        split_sampler = self.policy.get_split_sampler_state()
        with distributed_ema(self.policy, self.ema, self.evaluation):
            try:
                log = self.evaluation.run(self.policy, output, self.sampler.epoch,
                                          self.global_step, metadata, self.dataset_metadata)
            finally:
                self.policy.set_split_sampler_state(split_sampler)
        if self.rank == 0:
            scalars = {key: value for key, value in log.items() if isinstance(value, (int, float))}
            self._log(dict(event='rollout', epoch=self.sampler.epoch,
                           global_step=self.global_step, **scalars))
        return log

    @torch.no_grad()
    def _offline_video(self):
        if self.rank != 0:
            return {}
        rng, was_training = capture_rng(), self.policy.training
        split_sampler = self.policy.get_split_sampler_state()
        visual_stats = self.policy.last_visual_stats
        try:
            self.policy.eval()
            options = OmegaConf.to_container(self.cfg.offline_video, resolve=True)
            options.pop('enabled')
            runner = TCLFoveationVideoRunner(
                str(self.output), self.validation_dataset,
                seed=int(self.cfg.training.seed),
                log_wandb=self.wandb_run is not None and self.cfg.logging.mode != 'disabled',
                **options)
            runner.set_context(self.dataset_metadata, self.sampler.epoch, self.global_step)
            with self.ema.apply(self.policy):
                return runner.run(self.policy)
        finally:
            self.policy.last_visual_stats = visual_stats
            self.policy.set_split_sampler_state(split_sampler)
            self.policy.train(was_training)
            restore_rng(rng)

    @torch.no_grad()
    def _task_sample_diagnostics(self):
        diagnostic_seed = int(self.cfg.training.seed)
        sampling_seed = diagnostic_seed
        if self.cfg.training.get('sample_resample', True):
            sampling_seed += self.global_step
        resolution_enabled = self.cfg.training.get('sample_resolution_advantage', True)
        metrics, task_distances, task_losses = {}, [], []
        for task_id, batch in self.dataset.iter_task_diagnostic_batches(
                self.topology_samples_per_task, sampling_seed):
            task_name = (next(name for name, value in self.dataset.task_mapping.items()
                              if value == task_id) if self.is_tcl else LIBERO_TASK_NAMES[task_id])
            batch_size = len(batch['action'])
            with torch.random.fork_rng(devices=[self.device]):
                torch.random.default_generator.manual_seed(diagnostic_seed + task_id)
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(diagnostic_seed + task_id)
                if resolution_enabled:
                    diagnostics = self.policy.compute_resolution_diagnostics(
                        batch, seed=diagnostic_seed + task_id,
                        batch_size=self.sample_diagnostic_batch_size)
                    outputs = diagnostics['foveation_outputs']
                    loss_L = diagnostics['loss_B_L'].mean(dim=0)
                    task_losses.append(loss_L)
                    if self.is_tcl:
                        metrics.update({
                            f"sample/resolution_advantage_{level}_to_{level + 1}_{task_name}": value
                            for level, value in enumerate((loss_L[:-1] - loss_L[1:]).tolist())
                        })
                    del diagnostics
                else:
                    obs = dict(batch['obs'])
                    if self.is_tcl:
                        obs['task_id'] = batch['task_id']
                    outputs = self.policy.get_foveation_outputs(obs)
                if batch_size >= 2:
                    distance = self._topology_diversity(outputs, self.policy.foveation_net.num_bank_patches)
                    task_distances.append(distance)
                    metrics[f'sample/topology_div_{task_name}'] = distance
            del outputs, batch
        if task_distances:
            metrics['sample/topology_div'] = sum(task_distances) / len(task_distances)
        else:
            print('[topology_div] No task has at least two diagnostic windows; skipping metric')
        if task_losses:
            # Equal task weights, even when a task provides fewer diagnostic windows.
            loss_L = torch.stack(task_losses).mean(dim=0)
            advantage_L = loss_L[:-1] - loss_L[1:]
            metrics.update({
                f'sample/resolution_advantage_{level}_to_{level + 1}': value
                for level, value in enumerate(advantage_L.tolist())
            })
        return metrics

    @torch.no_grad()
    def _sample(self):
        if self.rank != 0:
            return {}
        rng, was_training = capture_rng(), self.policy.training
        visual_stats = self.policy.last_visual_stats
        try:
            # Reuse the first training images; optionally resample their trees each time.
            seed = self.cfg.training.seed
            if self.cfg.training.get('sample_resample', True):
                seed += self.global_step
            torch.random.default_generator.manual_seed(seed)
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(seed)
            self.policy.eval()
            with self.ema.apply(self.policy):
                result = self.policy.predict_action(self.train_sampling_obs, return_foveation_outputs=True)
                outputs = result['_foveation_outputs']
                metrics = {'sample/batch_size': len(self.train_sampling_obs['prompt'])}
                for key, output in outputs.items():
                    counts_B_T = output['num_patches_B_T'].float()
                    valid_B_T_N = output['token_valid_mask_B_T_N']
                    metrics[f'sample/num_patches_mean_{key}'] = counts_B_T.mean().item()
                    metrics[f'sample/padding_fraction_{key}'] = 1 - valid_B_T_N.float().mean().item()
                target_action_B_T_D = self.train_sampling_action.to(self.device)
                metrics['sample/train_action_mse_error'] = (
                    result['action_pred'] - target_action_B_T_D
                ).square().mean().item()
                metrics.update(self._task_sample_diagnostics())
            # JSON logs contain only numbers; WandB media are added afterwards.
            self._log(dict(event='sample', epoch=self.sampler.epoch,
                           global_step=self.global_step, **metrics))
            if self.wandb_run is not None and self.cfg.logging.mode != 'disabled':
                metrics.update(make_foveation_vis_log(
                    self.policy, outputs, max_batch=self.sample_batch_size,
                    vis_rot_image=self.policy.render_image_orientation == 'raw'))
            return metrics
        finally:
            self.policy.last_visual_stats = visual_stats
            self.policy.train(was_training)
            restore_rng(rng)

    def _export(self, metrics):
        if self.rank != 0:
            return
        monitor = self.cfg.checkpoint.monitor
        value = metrics.get('test/mean_score' if monitor == 'success_rate' else 'val/total_loss')
        if value is None and monitor != 'success_rate':
            value = metrics.get('val/loss')
        improved = value is not None and (self.best_metric is None or
            (value > self.best_metric if monitor == 'success_rate' else value < self.best_metric))
        if improved:
            self.best_metric = value
        metadata = dict(weights='ema', global_step=self.global_step, ema_updates=self.ema.updates,
                        monitor=monitor, metric=value, best_metric=self.best_metric, **self.provenance)
        if 'val/total_loss' in metrics:
            metadata['val_loss'] = metrics['val/total_loss']
        if 'test/mean_score' in metrics:
            metadata.update(success_rate=metrics['test/mean_score'],
                            n_episodes=metrics['test/n_episodes'],
                            evaluation=OmegaConf.to_container(self.runner_cfg, resolve=True))
        state = self.ema.inference_state(self.policy, metadata)
        atomic_save(state, self.checkpoints / 'latest_ema.pt')
        if improved:
            atomic_save(state, self.checkpoints / 'best_ema.pt')
        self._log(dict(event='checkpoint', **metadata, improved=improved))

    def save_checkpoint(self, path=None, **kwargs):
        """Synchronously save complete training state; every rank must enter this call."""
        if kwargs:
            raise TypeError('pi0.5 full checkpoints do not support generic BaseWorkspace save options')
        self._barrier()
        runtime = dict(rng=capture_rng(), loader_generator=self.loader_generator.get_state(),
                       split_sampler=self.policy.get_split_sampler_state())
        rank_states = [None] * self.world_size if self.rank == 0 else None
        if self.world_size > 1:
            dist.gather_object(runtime, rank_states, dst=0)
        else:
            rank_states = [runtime]
        target = Path(path) if path is not None else self.checkpoints / 'latest_full.ckpt'
        if self.rank == 0:
            # Training is stopped on all ranks: serialize directly, avoiding a full host-memory clone.
            atomic_save(dict(format='pi05_training_v1', signature=self._signature(),
                             cfg=OmegaConf.to_container(self.cfg, resolve=True),
                             policy=self.policy.state_dict(), optimizer=self.optimizer.state_dict(),
                             scheduler=self.scheduler.state_dict(), ema=self.ema.state_dict(),
                             global_step=self.global_step, best_loss=self.best_loss,
                             stage_b_state=dict(phase=self.phase,
                                                target_num_patches=(self.policy.foveation_net.resolved_target_num_patches
                                                                    if self.stage_b else None),
                                                target_num_patches_step=(self.global_step if self.stage_b else None),
                                                router_temperature=(self.policy.foveation_net.nets['image'].router_temperature
                                                                    if self.stage_b else None),
                                                budget_lambda=[net.budget_lambda for net in self.policy.foveation_net.nets.values()]
                                                if self.stage_b else [],
                                                pairing=self.dataset.get_pairing_state()),
                             selection=dict(monitor=self.cfg.checkpoint.monitor, best=self.best_metric),
                             sampler=self.sampler.state_dict(), rank_states=rank_states,
                             dataset_metadata=self.dataset_metadata, provenance=self.provenance,
                             versions=self.versions, sources=self.sources), target)
            print(f'[pi0.5 SFT] full checkpoint: {target}', flush=True)
        self._barrier()
        return str(target)

    def _reduce_sum(self, value_X):
        """Sum detached scalar statistics over all training ranks."""
        if self.world_size > 1:
            dist.all_reduce(value_X, op=dist.ReduceOp.SUM)
        return value_X

    def _globalize_elu_budget(self, losses):
        """Replace local ELU budget terms with a globally averaged surrogate."""
        if '_budget_normalized_mean' not in losses:
            return
        policy = self.policy
        local_mean_V = losses['_budget_normalized_mean']
        expected_sum_V = self._reduce_sum(losses['_budget_expected_sum'].detach().double())
        observation_count_V = self._reduce_sum(losses['_budget_observation_count'].detach().double())
        budget_losses_V = []
        for view, net in enumerate(policy.foveation_net.nets.values()):
            budget_range = float(net.max_num_patches - net.min_num_patches)
            global_mean = ((expected_sum_V[view] / observation_count_V[view].clamp_min(1.0))
                           - net.min_num_patches) / budget_range
            mean_proxy = global_mean.to(local_mean_V.dtype) + (
                local_mean_V[view] - local_mean_V[view].detach())
            normalized_target = (net.resolved_target_num_patches - net.min_num_patches) / budget_range
            budget_losses_V.append(
                net.upper_bound_budget_loss(
                    mean_proxy, normalized_target, net.budget_loss_type,
                    net.budget_base_cost_weight, net.budget_lambda, net.budget_elu_tau,
                ) * net.budget_loss_weight
            )
        budget_loss = torch.stack(budget_losses_V).mean()
        losses['total_loss'] = losses['total_loss'] - losses['budget_loss'] + budget_loss
        losses['budget_loss'] = budget_loss
        losses['_budget_expected_sum'] = expected_sum_V.to(losses['_budget_expected_sum'])
        losses['_budget_observation_count'] = observation_count_V.to(losses['_budget_observation_count'])

    @staticmethod
    def _accumulate_view_metrics(accumulator, sums_V, counts_V):
        """Accumulate per-camera metric numerators and denominators."""
        for name, values_V in sums_V.items():
            accumulator.setdefault(name, [None, None])
            current_sum_V, current_count_V = accumulator[name]
            current_sum_V = values_V.detach().double() if current_sum_V is None else current_sum_V + values_V.detach().double()
            count_V = counts_V[name].detach().double()
            current_count_V = count_V if current_count_V is None else current_count_V + count_V
            accumulator[name] = [current_sum_V, current_count_V]

    def _global_view_metrics(self, accumulator):
        """Reduce accumulated camera sums and return global and view metrics."""
        metrics = {}
        for name, (sums_V, counts_V) in accumulator.items():
            sums_V = self._reduce_sum(sums_V)
            counts_V = self._reduce_sum(counts_V)
            values_V = sums_V / counts_V.clamp_min(1.0)
            for index, key in enumerate(self.policy.foveation_net.view_keys):
                metrics[f'views/{key}/{name}'] = float(values_V[index].item())
            total_sum = sums_V.sum()
            total_count = counts_V.sum()
            metrics[name] = float((total_sum / total_count.clamp_min(1.0)).item())
        return metrics

    def _update_budget_lambda(self, budget_stats_V_2):
        """Update both camera multipliers from one optimizer step's global counts."""
        metrics = {}
        for view, (key, net) in enumerate(self.policy.foveation_net.nets.items()):
            expected_mean = (budget_stats_V_2[view, 0] /
                             budget_stats_V_2[view, 1].clamp_min(1.0)).item()
            target = float(net.resolved_target_num_patches)
            metrics[f'train/views/{key}/budget_lambda'] = float(net.budget_lambda)
            metrics[f'train/views/{key}/expected_num_patches_mean'] = expected_mean
            metrics[f'train/views/{key}/budget_gap'] = expected_mean - target
            net.budget_lambda = projected_dual_ascent(
                net.budget_lambda, expected_mean, target,
                self.cfg.training.budget_lambda_lr,
                net.max_num_patches - net.min_num_patches,
            )
        return metrics

    @torch.no_grad()
    def _switch_to_downstream(self):
        """Install rank-zero EMA routing and freeze its optimizer branch once."""
        self.optimizer.zero_grad(set_to_none=True)
        prefixes = ('foveation_net.', 'router_condition_projection.')
        if self.ema is not None:
            self.ema.copy_prefix_to_policy(self.policy, prefixes)
        if self.world_size > 1:
            for name, value in self.policy.state_dict().items():
                if name.startswith(prefixes):
                    dist.broadcast(value, src=0)
        self.policy.set_stage_b_phase('downstream')
        self._set_ema_phase()
        self.dataset.set_temporal_pairing(False)
        self.dataset.set_pairing_epoch(self.sampler.epoch)
        self._clear_router_optimizer_state()
        # The old iterator may already contain prefetched paired windows.
        self.iterator = iter(self.loader)

    def _clear_router_optimizer_state(self):
        """Discard stale moments and gradients for the frozen routing branch."""
        for module in (self.policy.foveation_net,
                       getattr(self.policy, 'router_condition_projection', None)):
            if module is None:
                continue
            for param in module.parameters():
                self.optimizer.state.pop(param, None)
                param.grad = None

    def _set_ema_phase(self):
        """Select the inactive parameter branch for the current StageB phase."""
        if self.ema is None:
            return
        if self.phase == 'router_only':
            self.ema.set_excluded_prefixes(('model.',))
        elif self.phase == 'downstream':
            self.ema.set_excluded_prefixes(('foveation_net.', 'router_condition_projection.'))
        else:
            self.ema.set_excluded_prefixes(())

    def _update_stage_b_state(self):
        """Advance target/lambda state and perform the one-way downstream switch."""
        cfg = self.cfg
        self._last_budget_metrics = {}
        if self.phase == 'router_only':
            start = self.policy.foveation_net.resolve_target_num_patches(
                cfg.training.target_num_patches_start)
            target = self.policy.foveation_net.resolve_target_num_patches(
                cfg.model.foveation_target_num_patches)
            if hasattr(self, '_budget_stats_V_2') and getattr(self, '_has_budget_stats', False):
                global_budget_stats_V_2 = self._reduce_sum(self._budget_stats_V_2)
                self._last_budget_metrics = self._update_budget_lambda(global_budget_stats_V_2)
            if hasattr(self, '_budget_stats_V_2'):
                self._budget_stats_V_2.zero_()
            next_target = cosine_schedule_value(
                start, target, self.global_step, cfg.training.target_num_patches_decay_steps)
            self.policy.set_target_num_patches(next_target)
        switch_step = cfg.training.downstream_finetune_start_step
        if switch_step is not None and self.global_step >= int(switch_step) and self.phase == 'router_only':
            self.phase = 'downstream'
            self._switch_to_downstream()

    def run(self):
        """Run accumulated full-parameter updates, EMA validation and optional recovery saves."""
        cfg = self.cfg
        try:
            self._initialize()
            while self.global_step < cfg.training.max_steps:
                self.optimizer.zero_grad(set_to_none=True)
                loss_1 = torch.zeros((), device=self.device)
                log_step = (self.global_step == 0
                            or (self.global_step + 1) % cfg.training.log_every == 0)
                visual_sums, visual_count = {}, 0
                view_metric_accumulator = {}
                temporal_view_valid_V = torch.zeros(
                    len(self.policy.foveation_net.view_keys), device=self.device, dtype=torch.float64
                ) if self.stage_b else None
                temporal_view_matched_V = torch.zeros_like(temporal_view_valid_V) \
                    if temporal_view_valid_V is not None else None
                paired_scalar_sums, paired_scalar_count = {}, 0
                has_budget_stats = False
                if self.stage_b and self.phase == 'router_only':
                    self._budget_stats_V_2.zero_()
                for micro_step in range(self.sampler.accumulation):
                    batch = next(self.iterator)
                    if self.rank == 0 and self.sample_enabled and self.train_sampling_obs is None:
                        # Keep one CPU window for the DP-compatible sample action metric.
                        self.train_sampling_obs = {
                            key: (value_B_X[:self.sample_batch_size].detach().cpu().clone()
                                  if torch.is_tensor(value_B_X) else value_B_X[:self.sample_batch_size])
                            for key, value_B_X in batch['obs'].items()
                        }
                        self.train_sampling_action = batch['action'][:self.sample_batch_size].detach().cpu().clone()
                    synchronizing = micro_step == self.sampler.accumulation - 1
                    context = (self.model.no_sync() if self.world_size > 1 and not synchronizing
                               else nullcontext())
                    with context:
                        if self.stage_b and self.phase == 'router_only':
                            paired = self.model(
                                batch,
                                paired=True,
                                paired_loss_weight=float(cfg.training.paired_loss_weight),
                                temporal_loss_weight=float(cfg.training.temporal_loss_weight),
                            )
                            local_budget_sum_V = paired.get('_budget_expected_sum')
                            local_budget_count_V = paired.get('_budget_observation_count')
                            if local_budget_sum_V is not None:
                                has_budget_stats = True
                                self._budget_stats_V_2[:, 0].add_(local_budget_sum_V.detach().double())
                                self._budget_stats_V_2[:, 1].add_(local_budget_count_V.detach().double())
                            self._globalize_elu_budget(paired)
                            self._accumulate_view_metrics(
                                view_metric_accumulator,
                                paired['_view_metric_sums'], paired['_view_metric_counts'])
                            if temporal_view_valid_V is not None:
                                temporal_view_valid_V.add_(paired['_temporal_view_valid_count'].detach().double())
                                temporal_view_matched_V.add_(paired['_temporal_view_matched_count'].detach().double())
                            paired_batch_count = len(batch['action'])
                            for key, value in paired.items():
                                if key.startswith('_') or key in ('loss', 'total_loss', 'routing'):
                                    continue
                                if torch.is_tensor(value) and value.numel() == 1:
                                    scale = 1.0 if key.endswith('_count') else paired_batch_count
                                    paired_scalar_sums[key] = paired_scalar_sums.get(key, 0.0) + float(
                                        value.detach().item()) * scale
                            paired_scalar_count += paired_batch_count
                            micro_loss_1 = paired['total_loss']
                        else:
                            paired = None
                            micro_loss_1 = self.model(batch)
                        if log_step:
                            # pi0.5 has T=1; sample weighting covers every accumulated window.
                            batch_size = len(batch['action'])
                            for key, value in self._foveation_metrics().items():
                                visual_sums[key] = visual_sums.get(key, 0.0) + value * batch_size
                            visual_count += batch_size
                        loss_1.add_(micro_loss_1.detach() / self.sampler.accumulation)
                        (micro_loss_1 / self.sampler.accumulation).backward()
                if self.world_size > 1:
                    dist.all_reduce(loss_1)
                    loss_1.div_(self.world_size)
                if not torch.isfinite(loss_1):
                    raise FloatingPointError('Nonfinite pi0.5 training loss')
                diagnostics = None
                if self.rank == 0 and cfg.training.diagnose_first_update and self.global_step == 0:
                    diagnostics = begin_update_diagnostics(self.policy, self.optimizer, self.ema)
                clip_gradients(self.policy.parameters(), cfg.training.gradient_clip)
                lr = self.optimizer.param_groups[0]['lr']
                self.optimizer.step()
                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.policy, self.optimizer)
                self.global_step += 1
                self.sampler.advance()
                if self.stage_b:
                    self._has_budget_stats = has_budget_stats
                    self._update_stage_b_state()
                if diagnostics is not None:
                    report = finish_update_diagnostics(self.policy, self.optimizer, self.ema, *diagnostics)
                    (self.output / 'first_update_diagnostics.json').write_text(
                        json.dumps(report, indent=2, allow_nan=False) + '\n')
                # Release gradients before validation/checkpointing; save only at update boundaries.
                self.optimizer.zero_grad(set_to_none=True)
                step_log = {}
                if log_step:
                    train_metrics = {
                        'diffusion_loss': loss_1.item(),
                        'budget_loss': 0.0,
                        'total_loss': loss_1.item(),
                    }
                    if paired is not None:
                        scalar_keys = sorted(paired_scalar_sums)
                        scalar_values = torch.tensor(
                            [paired_scalar_sums[key] for key in scalar_keys],
                            dtype=torch.float64, device=self.device)
                        scalar_count = torch.tensor(
                            float(paired_scalar_count), dtype=torch.float64, device=self.device)
                        scalar_values = self._reduce_sum(scalar_values)
                        scalar_count = self._reduce_sum(scalar_count)
                        train_metrics.update({key: float(value.item()) / max(1.0, scalar_count.item())
                                              for key, value in zip(scalar_keys, scalar_values)})
                        for key in ('temporal_valid_pair_count', 'temporal_matched_pair_count'):
                            if key in paired_scalar_sums:
                                value = self._reduce_sum(torch.tensor(
                                    paired_scalar_sums[key], dtype=torch.float64, device=self.device))
                                train_metrics[key] = float(value.item())
                        train_metrics['total_loss'] = loss_1.item()
                    if self.stage_b:
                        train_metrics['downstream_finetune_active'] = float(self.phase == 'downstream')
                    if visual_sums:
                        # All ranks use the same camera metrics; reduce sums and count together.
                        metric_keys = sorted(visual_sums)
                        visual_stats_M = torch.tensor(
                            [visual_sums[key] for key in metric_keys] + [visual_count],
                            dtype=torch.float64, device=self.device)
                        if self.world_size > 1:
                            dist.all_reduce(visual_stats_M)
                        visual_means = (visual_stats_M[:-1] / visual_stats_M[-1]).tolist()
                        train_metrics.update(zip(metric_keys, visual_means))
                    if view_metric_accumulator:
                        train_metrics.update(self._global_view_metrics(view_metric_accumulator))
                    if temporal_view_valid_V is not None and paired is not None:
                        valid_V = self._reduce_sum(temporal_view_valid_V)
                        matched_V = self._reduce_sum(temporal_view_matched_V)
                        for index, key in enumerate(self.policy.foveation_net.view_keys):
                            step_log_key = f'views/{key}'
                            train_metrics[f'{step_log_key}/temporal_valid_pair_count'] = float(valid_V[index].item())
                            train_metrics[f'{step_log_key}/temporal_matched_pair_count'] = float(matched_V[index].item())
                    step_log.update({'epoch': self.sampler.epoch,
                                     'global_step': self.global_step,
                                     'train/lr': lr})
                    step_log.update({f'train/{key}': value for key, value in train_metrics.items()})
                    if getattr(self, '_last_budget_metrics', None):
                        step_log.update(self._last_budget_metrics)
                    self._log(dict(event='train', **step_log))
                final = self.global_step == cfg.training.max_steps
                validate = self.global_step % cfg.validation.every == 0 or final
                # An extra rollout uses the absolute update count, including after resume.
                rollout = cfg.rollout.enabled and (
                    self.global_step == self.first_rollout_step
                    or self.global_step % cfg.rollout.every == 0 or final)
                if self.sample_enabled and self.global_step % self.sample_every == 0:
                    step_log.update(self.evaluation.call(self._sample))
                if validate:
                    step_log.update(self.evaluation.call(self._validate))
                if rollout:
                    step_log.update(self.evaluation.call(self._rollout))
                offline_video = self.offline_video_enabled and (
                    self.global_step == self.first_rollout_step
                    or self.global_step % cfg.rollout.every == 0 or final)
                if offline_video:
                    step_log.update(self.evaluation.call(self._offline_video))
                if validate or rollout or final:
                    self.evaluation.call(lambda: self._export(step_log))
                self.evaluation.call(lambda: log_wandb(self.wandb_run, step_log, self.global_step)
                                     if self.rank == 0 and step_log else None)
                if cfg.checkpoint.save_full and (self.global_step % cfg.checkpoint.full_every == 0 or final):
                    self.save_checkpoint()
                if self.sampler.update == 0 and not final:
                    self.dataset.set_pairing_epoch(self.sampler.epoch)
                    self.iterator = iter(self.loader)
        finally:
            if getattr(self, 'wandb_run', None) is not None:
                self.wandb_run.finish()
            if hasattr(self, 'dataset'):
                self.dataset.close()
            if hasattr(self, 'validation_dataset'):
                self.validation_dataset.close()
            if getattr(self, 'evaluation', None) is not None:
                self.evaluation.close()
            if dist.is_initialized():
                dist.destroy_process_group()
