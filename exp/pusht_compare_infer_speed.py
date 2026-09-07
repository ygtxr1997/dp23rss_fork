"""Benchmark PushT inference latency for standard and foveated policies."""

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List

import hydra
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.model.common.normalizer import (
    EmptyNormalizer,
    LinearNormalizer,
)


DEFAULT_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
POLICY_TYPES = {
    "diffusion_policy.policy.foveation_dp_transformer_policy."
    "FoveationTransformerHybridImagePolicy": "foveation",
    "diffusion_policy.policy.diffusion_transformer_hybrid_image_policy."
    "DiffusionTransformerHybridImagePolicy": "standard",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA inference latency for PushT policies."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pusht256_foveation_dp.yaml"),
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=None,
        help="Fixed quadtree split ratios for a foveation policy.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-router",
        action="store_true",
        help=(
            "Execute the global encoder and router, then override its hard "
            "decisions with a fixed-ratio topology."
        ),
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def _validate_args(
    args: argparse.Namespace,
    policy_type: str,
) -> tuple[float, ...]:
    if args.batch_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError(
            "batch-size and repeats must be positive; warmup cannot be negative."
        )
    if policy_type == "foveation":
        ratios = tuple(DEFAULT_RATIOS if args.ratios is None else args.ratios)
        if any(
            not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0
            for ratio in ratios
        ):
            raise ValueError("Every random split ratio must be in [0, 1].")
    else:
        if args.ratios is not None:
            raise ValueError("--ratios is only valid for a foveation policy.")
        if args.include_router:
            raise ValueError("--include-router is only valid for a foveation policy.")
        ratios = ()
    if not str(args.device).startswith("cuda"):
        raise ValueError("This benchmark requires a CUDA device for event-based timing.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this benchmark on a GPU machine.")
    return ratios


def _get_policy_type(policy_cfg) -> str:
    target = str(policy_cfg.get("_target_", ""))
    if target not in POLICY_TYPES:
        raise ValueError(f"Unsupported policy target: {target or '<missing>'}.")
    return POLICY_TYPES[target]


def _make_synthetic_obs(
    policy,
    obs_shape_meta,
    batch_size: int,
    rgb_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    obs_dict = {}
    for key, attr in obs_shape_meta.items():
        shape = tuple(int(dim) for dim in attr["shape"])
        obs_dict[key] = torch.zeros(
            (batch_size, policy.n_obs_steps, *shape),
            dtype=(
                rgb_dtype
                if attr.get("type", "low_dim") == "rgb"
                else policy.dtype
            ),
            device=policy.device,
        )
    return obs_dict


def _make_identity_normalizer(shape_meta) -> LinearNormalizer:
    normalizer = LinearNormalizer()
    for key in (*shape_meta["obs"], "action"):
        normalizer[key] = EmptyNormalizer.create_identity()
    return normalizer


def _instantiate_policy(policy_cfg, args, device, normalizer=None):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = hydra.utils.instantiate(policy_cfg)
    if normalizer is not None:
        policy.set_normalizer(normalizer)
    return policy.to(device).eval()


def _run_foveation(
    policy: torch.nn.Module,
    obs_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    # PushT has one RGB stream; preserve the policy's uint8 normalization path.
    rgb_key = policy.rgb_keys[0]
    image_B_T_C_H_W = policy._prepare_rgb(rgb_key, obs_dict[rgb_key])
    image_B_C_T_H_W = image_B_T_C_H_W.permute(0, 2, 1, 3, 4).contiguous()
    return policy.foveation_net(image_B_C_T_H_W, cond_B_T_D=None)


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


def _measure_cuda(
    fn: Callable[[], object],
    warmup: int,
    repeats: int,
    device: torch.device,
    description: str,
) -> Dict[str, float]:
    elapsed_ms_values: List[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with tqdm(total=warmup + repeats, desc=description, unit="run") as progress:
        for _ in range(warmup):
            fn()
            progress.update()
        torch.cuda.synchronize(device)

        for _ in range(repeats):
            start_event.record()
            fn()
            end_event.record()
            end_event.synchronize()
            elapsed_ms_values.append(float(start_event.elapsed_time(end_event)))
            progress.update()

    elapsed_ms_R = torch.tensor(elapsed_ms_values, dtype=torch.float64)
    return {
        "mean_ms": float(elapsed_ms_R.mean()),
        "std_ms": float(elapsed_ms_R.std(unbiased=False)),
        "p50_ms": float(torch.quantile(elapsed_ms_R, 0.50)),
        "p95_ms": float(torch.quantile(elapsed_ms_R, 0.95)),
    }


def _make_common_metric_fns(
    policy,
    obs_dict,
    visual_encoder: Callable[[], object],
    encode_obs: Callable[[], torch.Tensor],
    batch_size: int,
) -> Dict[str, Callable[[], object]]:
    cond_B_To_D = encode_obs()
    trajectory_B_H_Da = torch.randn(
        (batch_size, policy.horizon, policy.action_dim),
        dtype=policy.dtype,
        device=policy.device,
    )
    policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
    timestep = policy.noise_scheduler.timesteps[0]
    return {
        "visual_encoder": visual_encoder,
        "denoiser": lambda: policy.model(
            trajectory_B_H_Da, timestep, cond_B_To_D
        ),
        "encode_plus_one_denoiser": lambda: policy.model(
            trajectory_B_H_Da, timestep, encode_obs()
        ),
        "predict_action": lambda: policy.predict_action(obs_dict),
    }


def _benchmark_metrics(
    metric_fns: Dict[str, Callable[[], object]],
    args: argparse.Namespace,
    device: torch.device,
    ratio="",
    num_tokens="",
    router_executed="",
) -> List[Dict[str, object]]:
    rows = []
    for metric_name, metric_fn in metric_fns.items():
        ratio_prefix = "" if ratio == "" else f"ratio={ratio:.3f} "
        stats = _measure_cuda(
            metric_fn,
            args.warmup,
            args.repeats,
            device,
            description=f"{ratio_prefix}{metric_name}",
        )
        throughput = (
            args.batch_size * 1000.0 / stats["mean_ms"]
            if metric_name == "predict_action"
            else ""
        )
        rows.append(
            {
                "ratio": ratio,
                "num_tokens": num_tokens,
                "batch_size": args.batch_size,
                "router_executed": router_executed,
                "metric": metric_name,
                **stats,
                "samples_per_second": throughput,
            }
        )
    return rows


def _benchmark_ratio(
    policy_cfg,
    ratio: float,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, object]]:
    ratio_policy_cfg = copy.deepcopy(policy_cfg)
    ratio_policy_cfg.foveation_net.fix_split_mode = (
        None if args.include_router else f"random:{ratio}"
    )
    # Re-seeding keeps dense model weights identical across ratios.
    policy = _instantiate_policy(ratio_policy_cfg, args, device)
    if len(policy.rgb_keys) != 1 or not policy.obs_as_cond:
        raise ValueError(
            "The PushT benchmark expects one RGB key and obs_as_cond=true."
        )

    obs_dict = _make_synthetic_obs(
        policy=policy,
        obs_shape_meta=policy.obs_shape_meta,
        batch_size=args.batch_size,
        rgb_dtype=torch.uint8,
    )
    _install_fixed_random_topology(
        policy.foveation_net,
        ratio=ratio,
        batch_size=args.batch_size,
        device=device,
        include_router=args.include_router,
    )
    with torch.inference_mode():
        cached_foveation_out = _run_foveation(policy, obs_dict)
        num_splits = int(
            ratio * policy.foveation_net.num_internal_nodes + 0.5
        )
        num_tokens = policy.foveation_net.min_num_patches + 3 * num_splits
        encode_obs = lambda: policy._encode_obs_batch(
            obs_dict, policy.n_obs_steps
        )[0]
        metric_fns = {
            "foveation": lambda: _run_foveation(policy, obs_dict)
        }
        metric_fns.update(
            _make_common_metric_fns(
                policy,
                obs_dict,
                lambda: policy.obs_encoder(cached_foveation_out),
                encode_obs,
                args.batch_size,
            )
        )
        return _benchmark_metrics(
            metric_fns,
            args,
            device,
            ratio=ratio,
            num_tokens=num_tokens,
            router_executed=args.include_router,
        )


def _prepare_standard_encoder_obs(
    policy: torch.nn.Module,
    obs_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    normalized_obs_dict = policy.normalizer.normalize(obs_dict)
    return {
        key: obs_B_T_X[:, : policy.n_obs_steps].reshape(
            -1, *obs_B_T_X.shape[2:]
        )
        for key, obs_B_T_X in normalized_obs_dict.items()
    }


def _benchmark_standard(
    policy_cfg,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, object]]:
    # The standalone benchmark does not instantiate a dataset, whose normalizer
    # is identity for the current PushT configuration.
    shape_meta = policy_cfg.shape_meta
    policy = _instantiate_policy(
        policy_cfg, args, device, _make_identity_normalizer(shape_meta)
    )
    if not policy.obs_as_cond:
        raise ValueError("The PushT benchmark expects obs_as_cond=true.")

    obs_dict = _make_synthetic_obs(
        policy=policy,
        obs_shape_meta=shape_meta["obs"],
        batch_size=args.batch_size,
        rgb_dtype=policy.dtype,
    )
    with torch.inference_mode():
        cached_encoder_obs_dict = _prepare_standard_encoder_obs(policy, obs_dict)

        def encode_obs():
            encoder_obs_dict = _prepare_standard_encoder_obs(policy, obs_dict)
            features_BTo_D = policy.obs_encoder(encoder_obs_dict)
            return features_BTo_D.reshape(
                args.batch_size, policy.n_obs_steps, -1
            )

        metric_fns = _make_common_metric_fns(
            policy,
            obs_dict,
            lambda: policy.obs_encoder(cached_encoder_obs_dict),
            encode_obs,
            args.batch_size,
        )
        return _benchmark_metrics(metric_fns, args, device)


def _print_results(rows: List[Dict[str, object]]) -> None:
    header = (
        f"{'ratio':>7} {'tokens':>7} {'router':>7} {'metric':>28} "
        f"{'mean(ms)':>10} {'std(ms)':>10} {'p50(ms)':>10} {'p95(ms)':>10} "
        f"{'samples/s':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ratio = row["ratio"]
        ratio_text = "-" if ratio == "" else f"{ratio:.3f}"
        num_tokens = row["num_tokens"]
        num_tokens_text = "-" if num_tokens == "" else str(num_tokens)
        router_executed = row["router_executed"]
        router_text = "-" if router_executed == "" else str(router_executed)
        throughput = row["samples_per_second"]
        throughput_text = "-" if throughput == "" else f"{throughput:.2f}"
        print(
            f"{ratio_text:>7} {num_tokens_text:>7} {router_text:>7} "
            f"{row['metric']:>28} {row['mean_ms']:10.3f} "
            f"{row['std_ms']:10.3f} {row['p50_ms']:10.3f} "
            f"{row['p95_ms']:10.3f} {throughput_text:>11}"
        )


def _write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the PushT inference benchmark."""
    args = _parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT_DIR / args.config
    cfg = OmegaConf.load(config_path)
    # Resolve only the policy subtree; unrelated Hydra `${now:...}` logging
    # fields do not need to be available in this standalone script.
    policy_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.policy, resolve=True)
    )
    policy_type = _get_policy_type(policy_cfg)
    ratios = _validate_args(args, policy_type)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print(f"config: {config_path}")
    print(f"policy_type: {policy_type}")
    print(f"device: {torch.cuda.get_device_name(device)} ({device})")
    print(f"ratios: {list(ratios) if ratios else 'n/a'}")
    print(f"include_router: {args.include_router if ratios else 'n/a'}")
    print(
        f"batch_size={args.batch_size}, warmup={args.warmup}, "
        f"repeats={args.repeats}, inference_steps={policy_cfg.num_inference_steps}"
    )

    if policy_type == "foveation":
        rows = [
            row
            for ratio in ratios
            for row in _benchmark_ratio(policy_cfg, ratio, args, device)
        ]
    else:
        rows = _benchmark_standard(policy_cfg, args, device)
    _print_results(rows)
    if args.output_csv is not None:
        _write_csv(rows, args.output_csv)
        print(f"saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
