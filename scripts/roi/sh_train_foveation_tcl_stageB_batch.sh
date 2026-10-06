#!/usr/bin/env bash
# Train Stage B sequentially from Stage A log times (YYYY-MM-DD-HH-MM-SS).
# Fill LOG_TIMES; keep architecture settings as in A.
# Environment variables override defaults; trailing Hydra arguments reach every run.
# Stage A's full task.dataset replaces task.dataset.* overrides, including p_cond_drop.
# Example: CUDA_VISIBLE_DEVICES=6,7,8,9 bash scripts/roi/sh_train_foveation_tcl_stageB_batch.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SEED="${SEED:-42}"
export DATA_SEED_FROM_CKPT="${DATA_SEED_FROM_CKPT:-true}"
export PRETRAINED_CKPT_MODE="${PRETRAINED_CKPT_MODE:-best}"
export HORIZON="${HORIZON:-32}"
export VIS_ENC_LAYERS="${VIS_ENC_LAYERS:-4}"

RESUME_OUTPUT_ROOT="${RESUME_OUTPUT_ROOT:-outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
LOG_TIMES=(
  2026-10-05-18-11-37
)
if (( ${#LOG_TIMES[@]} == 0 )); then
  echo "[ERROR] Fill LOG_TIMES with the Stage A run times (YYYY-MM-DD-HH-MM-SS)." >&2
  exit 1
fi

for LOG_TIME in "${LOG_TIMES[@]}"; do
  if [[ ! "$LOG_TIME" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "[ERROR] Invalid log time: ${LOG_TIME}; expected YYYY-MM-DD-HH-MM-SS." >&2
    exit 1
  fi
  PRETRAINED_RUN_DIR="${RESUME_OUTPUT_ROOT}/${LOG_TIME:0:10}/${LOG_TIME:11}"
  RUN_DIR="${OUTPUT_ROOT}/$(date +%Y-%m-%d/%H-%M-%S)"
  echo "[Stage B] Pretrained run: ${PRETRAINED_RUN_DIR} -> ${RUN_DIR}"
  PRETRAINED_CKPT="${PRETRAINED_RUN_DIR}" OUTPUT_DIR="${RUN_DIR}" \
    bash scripts/roi/sh_train_foveation_tcl_stageB.sh \
      reuse_pretrained_dataset_config=true "$@"
done
