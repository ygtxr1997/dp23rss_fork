r"""Benchmark PushT with synthetic inputs or a trained policy and real images.

Examples:
    python exp/pusht_compare_infer_speed.py --ratios 0 0.5 1
    python exp/pusht_compare_infer_speed.py --log-dir 2026-09-19-23-09-10
    python exp/pusht_compare_infer_speed.py --log-dir outputs/2026-09-20/12-00-00 \
        --output-csv /tmp/pusht_latency.csv
    # Optional: choose a checkpoint/live weights and replace saved dataset paths.
    python exp/pusht_compare_infer_speed.py --log-dir outputs/2026-09-20/12-00-00 \
        --checkpoint checkpoints/epoch=1200-test_mean_score=0.900.ckpt \
        --weights model --dataset-paths data/pusht/pusht_256_size15.zarr
    python exp/pusht_compare_infer_speed.py \
        --log-dir outputs/2026-09-19/23-09-10 \
        --device cuda:0 \
        --batch-size 128 \
        --warmup 300 \
        --repeats 800

Each batch runs predict_action once, collecting nested CUDA timings along the way.
Data loading and GPU transfer are excluded; instrumentation adds a small overhead.
CPU threads default to 1; real inputs prefetch one CPU batch in the background.
Use --cpu-threads N to tune CPU work or --no-prefetch for a serial comparison.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from functools import wraps
from pathlib import Path

# Small per-frame CPU operations suffer from excessive thread scheduling overhead.
# Set library defaults before numerical imports, preserving explicit environment settings.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import dill
import hydra
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.model.common.normalizer import EmptyNormalizer, LinearNormalizer

DEFAULT_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
POLICY_TYPES = {
    "diffusion_policy.policy.foveation_dp_transformer_policy."
    "FoveationTransformerHybridImagePolicy": "foveation",
    "diffusion_policy.policy.diffusion_transformer_hybrid_image_policy."
    "DiffusionTransformerHybridImagePolicy": "standard",
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="Synthetic-mode config (default: PushT foveation).")
    source.add_argument("--log-dir", type=Path,
                        help="Training directory or YYYY-MM-DD-HH-MM-SS under outputs/. "
                             "A timestamp directory name is expanded to YYYY-MM-DD/HH-MM-SS.")
    parser.add_argument("--checkpoint", type=Path, help="Absolute path or path relative to log-dir; default checkpoints/latest.ckpt.")
    parser.add_argument("--weights", choices=("auto", "ema", "model"), default="auto")
    parser.add_argument("--dataset-paths", type=Path, nargs="+", help="Override saved Zarr paths.")
    parser.add_argument("--ratios", type=float, nargs="+", help="Synthetic-mode fixed split ratios.")
    parser.add_argument("--include-router", action="store_true",
                        help="Synthetic mode: run router before overriding its topology.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="PyTorch CPU threads (default: 1, overrides its environment default).")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Prepare real inputs serially instead of prefetching one CPU batch.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path,
                        help="Summary CSV; also writes a sibling *_details.csv.")
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error("cpu-threads must be positive.")
    if args.batch_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("batch-size/repeats must be positive; warmup cannot be negative.")
    if args.log_dir is not None:
        if args.ratios is not None or args.include_router:
            parser.error("Checkpoint mode preserves routing: do not pass ratios/include-router.")
    elif args.checkpoint is not None or args.dataset_paths is not None or args.weights != "auto":
        parser.error("checkpoint, dataset-paths and weights require --log-dir.")
    if args.ratios is not None and any(
        not math.isfinite(ratio) or not 0 <= ratio <= 1 for ratio in args.ratios
    ):
        parser.error("Every split ratio must be in [0, 1].")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("Run this CUDA event benchmark on a GPU machine with --device cuda:N.")
    return args


def _root_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT_DIR / path).resolve()


def _resolved_subtree(cfg):
    # Logging's Hydra ${now:...} fields are unrelated to policy/dataset construction.
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _log_dir_path(path):
    path = Path(path).expanduser()
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})", path.name)
    if match:
        # Preserve explicit roots; a bare timestamp is shorthand for outputs/.
        parent = Path("outputs") if path.parent == Path(".") else path.parent
        path = parent / match[1] / match[2]
    return _root_path(path)


def _load_source(args):
    if args.log_dir is None:
        config_path = _root_path(args.config or "configs/pusht256_foveation_dp.yaml")
        return OmegaConf.load(config_path), None, {
            "config": str(config_path), "checkpoint": "", "weights": "random",
        }
    checkpoint = (args.checkpoint or Path("checkpoints/latest.ckpt")).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = _log_dir_path(args.log_dir) / checkpoint
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    weights = args.weights
    if weights == "auto":
        weights = "ema" if cfg.training.get("use_ema", False) else "model"
    state = payload["state_dicts"]["ema_model" if weights == "ema" else "model"].copy()
    if weights == "ema":
        # Frozen shared encoders are saved only once, in the live model state.
        for name, weight_X in payload["state_dicts"]["model"].items():
            if name.startswith("foveation_net.global_encoder."):
                state.setdefault(name, weight_X)
    return cfg, state, {
        "config": "checkpoint:cfg", "checkpoint": str(checkpoint.resolve()), "weights": weights,
    }


def _make_dataset(cfg, args):
    from torchvision import transforms

    dataset_cfg = _resolved_subtree(cfg.task.dataset)
    target = dataset_cfg._target_.rsplit(".", 1)[-1]
    if target not in ("PushTImageDataset", "MergedPushTImageDataset"):
        raise ValueError(f"Unsupported PushT dataset: {dataset_cfg._target_}")
    paths = args.dataset_paths
    if paths is None:
        paths = dataset_cfg.zarr_paths if target == "MergedPushTImageDataset" else [dataset_cfg.zarr_path]
    paths = [str(_root_path(path)) for path in paths]
    if target == "MergedPushTImageDataset":
        dataset_cfg.zarr_paths = paths
    else:
        if len(paths) != 1:
            raise ValueError("PushTImageDataset requires exactly one dataset path.")
        dataset_cfg.zarr_path = paths[0]
    dataset_cfg.max_train_episodes = None
    dataset_cfg.val_ratio = 0.0
    dataset_cfg.transform_color_jitter = False
    dataset = hydra.utils.instantiate(dataset_cfg)
    if not len(dataset):
        raise ValueError("No observation windows available in the dataset.")
    image_hw = tuple(int(dim) for dim in dataset_cfg.shape_meta.obs.image.shape[1:])
    frame_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize(image_hw), transforms.ToTensor(),
    ])

    def transform_video(video_T_H_W_C):
        # Dataset _sample_to_data retains image/state scaling; only augmentation changes.
        return torch.stack([frame_transform(frame_H_W_C) for frame_H_W_C in video_T_H_W_C])

    for child in getattr(dataset, "datasets", [dataset]):
        child.obs_image_transform = transform_video
    return dataset, paths


def _make_policy(policy_cfg, state, args, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = hydra.utils.instantiate(policy_cfg)
    if state is None:
        normalizer = LinearNormalizer()
        for key in (*policy_cfg.shape_meta.obs, "action"):
            normalizer[key] = EmptyNormalizer.create_identity()
        policy.set_normalizer(normalizer)
    else:
        policy.load_state_dict(state, strict=True)
        # Match existing evaluation's explicit restoration of dynamic normalizer entries.
        normalizer_state = {name[len("normalizer."):]: weight_X for name, weight_X in state.items()
                            if name.startswith("normalizer.")}
        policy.normalizer.load_state_dict(normalizer_state)
    return policy.to(device).eval()


def _input_batches(policy, shape_meta, dataset, args):
    if dataset is None:
        obs_dict = {
            key: torch.zeros(
                (args.batch_size, policy.n_obs_steps, *attr.shape),
                dtype=(torch.uint8 if attr.get("type") == "rgb" and hasattr(policy, "foveation_net")
                       else policy.dtype), device=policy.device,
            ) for key, attr in shape_meta.obs.items()
        }
        torch.cuda.synchronize(policy.device)
        for _ in range(args.warmup + args.repeats):
            yield obs_dict, [], 0.0, 0.0
        return
    # A local RNG keeps window selection independent of model initialization/noise.
    rng = random.Random(args.seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)
    n_obs_steps = policy.n_obs_steps
    obs_keys = tuple(shape_meta.obs)
    total = args.warmup + args.repeats

    def prepare_batch(run):
        # This worker touches only CPU data, with no model calls or RNG changes.
        started = time.perf_counter()
        cursor = run * args.batch_size
        indices = [order[(cursor + offset) % len(order)] for offset in range(args.batch_size)]
        samples = [dataset[index]["obs"] for index in indices]
        obs_dict = {
            key: torch.stack([sample[key][:n_obs_steps] for sample in samples])
            for key in obs_keys
        }
        return obs_dict, indices, (time.perf_counter() - started) * 1000

    with ExitStack() as stack:
        executor = None if args.no_prefetch else stack.enter_context(ThreadPoolExecutor(max_workers=1))
        future = executor.submit(prepare_batch, 0) if executor is not None else None
        for run in range(total):
            started = time.perf_counter()
            obs_dict, indices, prepare_ms = (
                future.result() if executor is not None else prepare_batch(run)
            )
            wait_ms = (time.perf_counter() - started) * 1000
            # Exactly one next batch can be outstanding while the caller runs CUDA.
            if executor is not None:
                future = executor.submit(prepare_batch, run + 1) if run + 1 < total else None
            yield obs_dict, indices, prepare_ms, wait_ms


def _install_fixed_random_topology(
    foveation_net: torch.nn.Module,
    ratio: float,
    batch_size: int,
    device: torch.device,
    include_router: bool,
) -> None:
    # Generate the legal tree once so timed calls measure routing and packing,
    # not the sequential random-tree construction used by fixed baselines.
    num_splits = int(ratio * foveation_net.num_internal_nodes + 0.5)
    num_splits_B_1 = torch.full(
        (batch_size, 1),
        num_splits,
        dtype=torch.long,
        device=device,
    )
    fixed_split_hard_B_1_M = foveation_net._sample_random_split_mask(
        num_splits_B_1
    )
    if include_router:
        sample_router_decisions = foveation_net._sample_split_decisions

        def sample_fixed_topology(split_logits_B_T_M: torch.Tensor):
            split_soft_B_T_M, _ = sample_router_decisions(split_logits_B_T_M)
            fixed_split_hard_B_T_M = fixed_split_hard_B_1_M.expand(
                -1, split_logits_B_T_M.shape[1], -1
            )
            return split_soft_B_T_M, fixed_split_hard_B_T_M

        foveation_net._sample_split_decisions = sample_fixed_topology
    else:
        def sample_cached_topology(num_splits_B_T: torch.Tensor):
            # Fixed-ratio inference reuses one random tree across time.
            return fixed_split_hard_B_1_M.expand(
                -1, num_splits_B_T.shape[1], -1
            )

        foveation_net._sample_random_split_mask = sample_cached_topology

class _CudaTimings:
    """Collect nested events from one real policy invocation without intermediate syncs."""

    def __init__(self, policy):
        self.policy = policy
        self.events = defaultdict(list)
        self.counts = defaultdict(int)
        self.num_patches_B_T = None
        self.stack = ExitStack()

    def _wrap(self, obj, attribute, metric):
        original = getattr(obj, attribute)
        had_local = attribute in vars(obj)

        def restore():
            if had_local:
                setattr(obj, attribute, original)
            else:
                delattr(obj, attribute)
        self.stack.callback(restore)

        @wraps(original)
        def timed(*args, **kwargs):
            index = self.counts[metric]
            self.counts[metric] += 1
            if index == len(self.events[metric]):
                self.events[metric].append((torch.cuda.Event(enable_timing=True),
                                            torch.cuda.Event(enable_timing=True)))
            start, end = self.events[metric][index]
            start.record()
            result = original(*args, **kwargs)
            end.record()
            if metric == "foveation":
                # Retain only the small (B,T) hard counts; read on CPU after inference.
                self.num_patches_B_T = result["num_patches_B_T"].detach()
            return result

        setattr(obj, attribute, timed)

    def __enter__(self):
        try:
            self._wrap(self.policy, "predict_action", "predict_action")
            self._wrap(self.policy.obs_encoder, "forward", "visual_encoder")
            self._wrap(self.policy.model, "forward", "denoiser_step")
            self._wrap(self.policy.noise_scheduler, "step", "scheduler_step")
            net = getattr(self.policy, "foveation_net", None)
            if net is not None:
                self._wrap(net, "forward", "foveation")
                self._wrap(net.router, "forward", "router")
                # FPN calls forward_stages directly, bypassing encoder.forward hooks.
                for method in ("_encode_global_images", "_encode_global_stages", "_encode_quadtree_features"):
                    self._wrap(net, method, "global_encoder")
        except Exception:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def _reset(self):
        self.counts.clear()
        self.num_patches_B_T = None

    def _elapsed(self):
        metrics = {}
        for metric, count in self.counts.items():
            values = [start.elapsed_time(end) for start, end in self.events[metric][:count]]
            if metric.endswith("_step"):
                metrics[metric.replace("_step", "_total")] = sum(values)
                metrics.update({f"{metric}/{index:03d}": value for index, value in enumerate(values)})
            else:
                metrics[metric] = sum(values)
        return metrics


def _statistics(values):
    elapsed_ms_R = torch.tensor(values, dtype=torch.float64)
    return {
        "mean_ms": float(elapsed_ms_R.mean()),
        "std_ms": float(elapsed_ms_R.std(unbiased=False)),
        "p50_ms": float(torch.quantile(elapsed_ms_R, 0.50)),
        "p95_ms": float(torch.quantile(elapsed_ms_R, 0.95)),
        "min_ms": float(elapsed_ms_R.min()),
        "max_ms": float(elapsed_ms_R.max()),
    }


def _benchmark(policy, shape_meta, dataset, args, device, metadata):
    observations = _input_batches(policy, shape_meta, dataset, args)
    timings = defaultdict(list)
    details = []
    patch_counts = []
    router_executed = False
    with (
        torch.inference_mode(), closing(observations), _CudaTimings(policy) as recorder,
        tqdm(total=args.warmup + args.repeats, desc="benchmark", unit="batch") as progress,
    ):
        for run, (input_obs_dict, indices, prepare_ms, wait_ms) in enumerate(observations):
            h2d_ms = 0.0
            obs_dict = input_obs_dict
            if dataset is not None:
                started = time.perf_counter()
                obs_dict = {
                    key: obs_B_T_X.to(device=device, dtype=policy.dtype)
                    for key, obs_B_T_X in input_obs_dict.items()
                }
                # Transfer completion belongs to input timing, not model latency.
                torch.cuda.synchronize(device)
                h2d_ms = (time.perf_counter() - started) * 1000
            policy.reset()
            recorder._reset()
            # Nested module boundaries never synchronize; wait only after prediction.
            started = time.perf_counter()
            policy.predict_action(obs_dict)
            torch.cuda.synchronize(device)
            wall_ms = (time.perf_counter() - started) * 1000
            progress.set_postfix({
                "phase": "warmup" if run < args.warmup else "measure",
                "infer": f"{wall_ms:.1f}",
                "prepare": f"{prepare_ms:.1f}", "wait": f"{wait_ms:.1f}",
                "h2d": f"{h2d_ms:.1f}",
            }, refresh=False)
            progress.update()
            if run < args.warmup:
                continue
            metrics = recorder._elapsed()
            metrics["predict_action_wall"] = wall_ms
            metrics.update({
                "input_prepare_wall": prepare_ms,
                "input_wait_wall": wait_ms,
                "input_h2d_wall": h2d_ms,
            })
            router_executed = "router" in metrics
            counts_B_T = recorder.num_patches_B_T
            counts = [] if counts_B_T is None else counts_B_T.cpu().tolist()
            flattened = [value for frame_counts in counts for value in frame_counts]
            patch_counts.extend(flattened)
            batch_info = {
                **metadata, "run": run - args.warmup,
                "sample_indices": json.dumps(indices), "num_patches_B_T": json.dumps(counts),
                "patches_mean": sum(flattened) / len(flattened) if flattened else "",
                "patches_min": min(flattened) if flattened else "",
                "patches_max": max(flattened) if flattened else "",
                "router_executed": router_executed,
            }
            for metric, elapsed in metrics.items():
                timings[metric].append(elapsed)
                details.append({**batch_info, "metric": metric, "elapsed_ms": elapsed})
    patch_stats = {
        "patches_mean": sum(patch_counts) / len(patch_counts) if patch_counts else "",
        "patches_min": min(patch_counts) if patch_counts else "",
        "patches_max": max(patch_counts) if patch_counts else "",
    }
    rows = []
    for metric, values in timings.items():
        stats = _statistics(values)
        rows.append({
            **metadata, **patch_stats, "router_executed": router_executed,
            "metric": metric, "calls": len(values), **stats,
            "samples_per_second": args.batch_size * 1000 / stats["mean_ms"]
            if metric in ("predict_action", "predict_action_wall") else "",
        })
    return rows, details


def _print_results(rows):
    print("\nNested timings: foveation includes global_encoder/router; predict_action includes all modules.")
    print("Do not sum nested rows. Wall time includes instrumentation and final GPU synchronization.")
    print("Throughput counts observation windows, not individual frames or executed actions.")
    print("Input preparation can overlap inference; input_wait_wall is the exposed wait.")
    header = (f"{'ratio':>7} {'metric':>25} {'mean':>10} {'std':>10} {'p50':>10} "
              f"{'p95':>10} {'min':>10} {'max':>10} {'samples/s':>11}")
    print("All latency columns are in milliseconds.")
    print(header)
    previous_ratio = None
    # Keep input pipeline diagnostics separate from model-module timing rows.
    ordered_rows = [row for row in rows if not row["metric"].startswith("input_")]
    ordered_rows.extend(row for row in rows if row["metric"].startswith("input_"))
    showing_inputs = False
    for row in ordered_rows:
        if row["metric"].startswith("input_") and not showing_inputs:
            print("\nInput pipeline wall times (overlapping intervals; do not add to model latency):")
            print(header)
            showing_inputs = True
            previous_ratio = None
        if row["ratio"] != previous_ratio:
            print(f"patches mean/min/max: {row['patches_mean']} / {row['patches_min']} / "
                  f"{row['patches_max']}; router executed: {row['router_executed']}")
            previous_ratio = row["ratio"]
        ratio = "-" if row["ratio"] == "" else f"{row['ratio']:.3f}"
        throughput = row["samples_per_second"]
        throughput_text = "-" if throughput == "" else f"{throughput:.2f}"
        stats = " ".join(f"{row[key + '_ms']:10.3f}" for key in ("mean", "std", "p50", "p95", "min", "max"))
        print(f"{ratio:>7} {row['metric']:>25} {stats} {throughput_text:>11}")


def _write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run single-pass module profiling on synthetic or checkpoint-backed PushT inputs."""
    args = _parse_args()
    torch.set_num_threads(args.cpu_threads)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg, state, source = _load_source(args)
    policy_cfg = _resolved_subtree(cfg.policy)
    policy_type = POLICY_TYPES.get(str(policy_cfg._target_))
    if policy_type is None:
        raise ValueError(f"Unsupported policy target: {policy_cfg._target_}")
    if policy_type == "standard" and (args.ratios is not None or args.include_router):
        raise ValueError("ratios/include-router are only valid for foveation policies.")
    if not policy_cfg.obs_as_cond:
        raise ValueError("The PushT benchmark expects obs_as_cond=true.")
    if policy_type == "foveation" and sum(
        attr.get("type") == "rgb" for attr in policy_cfg.shape_meta.obs.values()
    ) != 1:
        raise ValueError("The PushT benchmark expects one RGB stream.")
    dataset, paths = _make_dataset(cfg, args) if args.log_dir is not None else (None, [])
    metadata = {
        **source, "policy_type": policy_type, "dataset_paths": json.dumps(paths),
        "dataset_windows": len(dataset) if dataset is not None else 0,
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size, "n_obs_steps": int(policy_cfg.n_obs_steps),
        "inference_steps": policy_cfg.get("num_inference_steps"), "precision": "float32",
        "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
        "cpu_threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
        "prefetch": dataset is not None and not args.no_prefetch,
    }
    print(json.dumps(metadata, indent=2))
    print("Inputs are prepared on GPU before timing. Dataset augmentation is disabled.")
    if args.warmup == 0:
        print("No warmup: first-call initialization and CUDA event setup affect measured latency.")
    ratios = (args.ratios or DEFAULT_RATIOS) if state is None and policy_type == "foveation" else (None,)
    rows, details = [], []
    for ratio in ratios:
        current_cfg = copy.deepcopy(policy_cfg)
        if ratio is not None:
            current_cfg.foveation_net.fix_split_mode = None if args.include_router else f"random:{ratio}"
        policy = _make_policy(current_cfg, state, args, device)
        if ratio is not None:
            _install_fixed_random_topology(policy.foveation_net, ratio, args.batch_size,
                                           device, args.include_router)
        current_metadata = {
            **metadata, "ratio": "" if ratio is None else ratio,
            "inference_steps": policy.num_inference_steps,
            "fix_split_mode": current_cfg.foveation_net.get("fix_split_mode")
            if policy_type == "foveation" else "",
        }
        run_rows, run_details = _benchmark(policy, current_cfg.shape_meta, dataset,
                                           args, device, current_metadata)
        rows.extend(run_rows)
        details.extend(run_details)
        del policy
    _print_results(rows)
    if args.output_csv is not None:
        details_path = args.output_csv.with_name(args.output_csv.stem + "_details.csv")
        _write_csv(rows, args.output_csv)
        _write_csv(details, details_path)
        print(f"saved CSV: {args.output_csv}; {details_path}")


if __name__ == "__main__":
    main()
