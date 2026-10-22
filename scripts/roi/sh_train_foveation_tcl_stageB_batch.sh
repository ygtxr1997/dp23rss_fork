#!/usr/bin/env bash
# Requires new per-camera Stage A weights with the same ordered RGB keys.
# Paired samples one camera per window/pair; ELU budgets and lambdas are per camera.
# Train Stage B sequentially from Stage A log times (YYYY-MM-DD-HH-MM-SS).
# Fill LOG_TIMES; keep architecture settings as in A.
# Environment variables override defaults; trailing Hydra arguments reach every run.
# Stage A's full task.dataset replaces task.dataset.* overrides, including p_cond_drop.
# Example: CUDA_VISIBLE_DEVICES=6,7,8,9 bash scripts/roi/sh_train_foveation_tcl_stageB_batch.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export HORIZON="${HORIZON:-32}"
export SEED="${SEED:-43}"
export DATA_SEED_FROM_CKPT="${DATA_SEED_FROM_CKPT:-true}"
# Select by Stage A top-k filename score; PRETRAINED_CKPT_MODE=latest uses the last save.
export PRETRAINED_CKPT_MODE="${PRETRAINED_CKPT_MODE:-best}"
export VIS_ENC_LAYERS="${VIS_ENC_LAYERS:-4}"
export POOLING_TYPE="${POOLING_TYPE:-cls_token}"
export PAIRED_NODE_SAMPLING="${PAIRED_NODE_SAMPLING:-reachable}"
export PAIRED_LOSS_WEIGHT="${PAIRED_LOSS_WEIGHT:-1.0}"
export TEMPORAL_ADAPTIVE="${TEMPORAL_ADAPTIVE:-true}"
export TEMPORAL_LOSS_WEIGHT="${TEMPORAL_LOSS_WEIGHT:-1.0}"
export BUDGET_LOSS_WEIGHT="${BUDGET_LOSS_WEIGHT:-0.1}"
export TARGET_NUM_PATCHES="${TARGET_NUM_PATCHES:-0.1}"
export BUDGET_LOSS_TYPE="${BUDGET_LOSS_TYPE:-upper_bound_lagrangian_elu}"
export BUDGET_BASE_COST_WEIGHT="${BUDGET_BASE_COST_WEIGHT:-0.0}"
export BUDGET_LAMBDA_INIT="${BUDGET_LAMBDA_INIT:-0.0}"
export BUDGET_LAMBDA_LR="${BUDGET_LAMBDA_LR:-0.01}"
export NUM_EPOCHS="${NUM_EPOCHS:-1201}"
export TARGET_NUM_PATCHES_DECAY_EPOCHS="${TARGET_NUM_PATCHES_DECAY_EPOCHS:-600}"
export DOWNSTREAM_FINETUNE_START_EPOCH="${DOWNSTREAM_FINETUNE_START_EPOCH:-600}"

RESUME_OUTPUT_ROOT="${RESUME_OUTPUT_ROOT:-outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
LOG_TIMES=(
  2026-10-06-14-10-43
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
