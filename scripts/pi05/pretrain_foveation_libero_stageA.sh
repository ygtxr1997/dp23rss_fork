#!/usr/bin/env bash
# One shared LIBERO-10 DP model with independent image/gripper routing branches.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the training GPUs}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPUS[@]}"
LAST_GPU="${GPUS[$((NUM_GPUS - 1))]}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-1234${LAST_GPU: -1}}"
LAUNCH_ARGS=(--num_processes="${NUM_GPUS}" --num_machines=1 --mixed_precision=no)
if (( NUM_GPUS > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT}")
fi

# Example:
tmp="
GPU_RESERVE_GB=50 \
OUTPUT_DIR=outputs/dp_for_pi05_stageA/2026-10-13/20-14-19/ \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/pi05/pretrain_foveation_libero_stageA.sh
"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dp_for_pi05_stageA/$(date +%Y-%m-%d/%H-%M-%S)}"
FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-[\"40%level:2\",\"60%random:0.05-0.99\"]}"
# Preserve Hydra lists; quote comma-containing scalar topology specifications.
SPLIT_MODE_OVERRIDE="policy.foveation_net.fix_split_mode=${FIX_SPLIT_MODE}"
if [[ "$FIX_SPLIT_MODE" != \[* && "$FIX_SPLIT_MODE" != \'* && "$FIX_SPLIT_MODE" != \"* && "$FIX_SPLIT_MODE" != null ]]; then
  SPLIT_MODE_OVERRIDE="policy.foveation_net.fix_split_mode='${FIX_SPLIT_MODE}'"
fi

# GPU_RESERVE_GB=8 holds an extra 8 GiB per GPU for the process lifetime; 0 disables.
accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name=liberoft_foveation_dp_for_pi05 \
  "hydra.run.dir=${OUTPUT_DIR}" \
  training.reserve_gpu_memory_gb="${GPU_RESERVE_GB:-0}" \
  training.seed="${SEED:-43}" \
  task.dataset.seed="${DATASET_SPLIT_SEED:-${SEED:-43}}" \
  horizon="${HORIZON:-32}" \
  training.num_epochs="${NUM_EPOCHS:-901}" \
  training.router_only=false \
  "${SPLIT_MODE_OVERRIDE}" \
  policy.foveation_net.random_topology_prob=0.0 \
  policy.foveation_net.budget_loss_weight=0.0 \
  policy.visual_encoder_layers="${VIS_ENC_LAYERS:-4}" \
  policy.pooling_type="${POOLING_TYPE:-cls_token}" \
  policy.patch_embedding_type="${PATCH_EMBEDDING_TYPE:-conv}" \
  policy.patch_level_adapter_channels="${PATCH_LEVEL_ADAPTER_CHANNELS:-16}" \
  "$@"
