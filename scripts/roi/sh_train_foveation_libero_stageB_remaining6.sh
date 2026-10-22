#!/usr/bin/env bash
# Requires new per-camera Stage A weights with the same ordered RGB keys.
# Paired samples one camera per window/pair; ELU budgets and lambdas are per camera.
# Run Stage B sequentially from Stage A log times, inheriting each task.dataset.
# Fill LOG_TIMES below, then run: bash scripts/roi/sh_train_foveation_libero_stageB_remaining6.sh
# Environment variables override defaults; trailing Hydra arguments reach every run.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export HORIZON="${HORIZON:-10}"
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
LOG_TIMES=(
#  2026-10-01-20-29-09
#  2026-10-01-23-03-30
#  2026-10-02-02-42-10
#  2026-10-02-05-49-07
#  2026-10-02-08-35-42
#  2026-10-02-11-47-59
#  2026-10-02-15-29-17
#  2026-10-02-20-00-29
#  2026-10-03-00-16-16
  2026-10-03-03-41-11
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
  RUN_DIR="${RESUME_OUTPUT_ROOT}/${LOG_TIME:0:10}/${LOG_TIME:11}"
  echo "[Stage B] Pretrained run: ${RUN_DIR}"
  # Let Stage B create a new timestamped output directory for each run.
  PRETRAINED_CKPT="${RUN_DIR}" \
    bash scripts/roi/sh_train_foveation_libero_stageB.sh \
      task.env_runner.conda_env=libero \
      libero_root=/home/geyuan/code/LIBERO-FT \
      reuse_pretrained_dataset_config=true \
      "$@"
done
