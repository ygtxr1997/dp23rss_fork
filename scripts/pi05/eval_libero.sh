#!/usr/bin/env bash
# Usage example: search outputs recursively for a unique run, then load checkpoints/<type>_ema.pt.
# checkpoint_type=best (default) or latest selects EMA weights, not latest_full.ckpt.
# Matches <date>/<time> or a directory name containing <date>-<time>; multiple matches are an error.
# Also accepts log_time=2026-10-12/02-04-29 (e.g. outputs/pi05_stageA/2026-10-12/02-04-29).
tmp="
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts/pi05/eval_libero.sh \
  log_time=2026-10-12-02-04-29 checkpoint_type=best \
  all_tasks=true runner.n_test=50 runner.n_envs=10 \
  runner.conda_env=libero logging.mode=online logging.project=eval_libero
"

# One process per explicitly selected GPU; otherwise keep the single-device entry.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PI05_EVAL_TIME="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  export PI05_EVAL_OUTPUT_DIR="${OUTPUT_DIR}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a PI05_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  exec torchrun --standalone --nnodes=1 --nproc-per-node="${#PI05_GPUS[@]}" \
    -m diffusion_policy.common.pi05_eval_libero "$@"
else
  exec python -m diffusion_policy.common.pi05_eval_libero "$@"
fi
