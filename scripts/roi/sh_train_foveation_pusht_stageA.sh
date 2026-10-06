#!/usr/bin/env bash
# Weighted Stage A example (keep the other required launch variables):
# FIX_SPLIT_MODE='["20%level:0","25%level:1","35%level:2","20%random:0.25,0.5,0.75"]' \
# RANDOM_TOPOLOGY_PROB=0.0 bash scripts/roi/sh_train_foveation_pusht_stageA.sh

# Usage Example:
tmp="
CUDA_VISIBLE_DEVICES=1,3,5,7 \
  FIX_SPLIT_MODE=\'random:0.0,0.25,0.5,0.75,1.0\' \
  RANDOM_TOPOLOGY_PROB=0.0 \
  VIS_ENC_LAYERS=4 \
  POOLING_TYPE=cls_token \
  PATCH_EMBEDDING_TYPE=conv \
  PATCH_LEVEL_ADAPTER_CHANNELS=16 \
  SEED=42 \
  bash scripts/roi/sh_train_foveation_pusht.sh policy.foveation_net.budget_loss_weight=0.0
"

set -euo pipefail

CONFIG_DIR="configs"
CONFIG_NAME="pusht256_foveation_dp.yaml"
SEED="${SEED:-42}"
PATCH_EMBEDDING_TYPE="${PATCH_EMBEDDING_TYPE:-linear}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] CUDA_VISIBLE_DEVICES is empty."
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPUS[@]}"
LAST_GPU="${GPUS[$((NUM_GPUS - 1))]}"
PORT="1234${LAST_GPU: -1}"

# Preserve Hydra lists and already-quoted legacy values; quote bare scalar commas.
SPLIT_MODE_OVERRIDE="policy.foveation_net.fix_split_mode=${FIX_SPLIT_MODE}"
if [[ "$FIX_SPLIT_MODE" != \[* && "$FIX_SPLIT_MODE" != \'* && "$FIX_SPLIT_MODE" != \"* && "$FIX_SPLIT_MODE" != null ]]; then
  SPLIT_MODE_OVERRIDE="policy.foveation_net.fix_split_mode='${FIX_SPLIT_MODE}'"
fi

accelerate launch \
  --multi_gpu \
  --num_processes="${NUM_GPUS}" \
  --main_process_port="${PORT}" \
  train.py \
  --config-dir="${CONFIG_DIR}" \
  --config-name="${CONFIG_NAME}" \
  training.seed="${SEED}"  \
  "${SPLIT_MODE_OVERRIDE}" \
  policy.visual_encoder_layers="${VIS_ENC_LAYERS}" \
  policy.pooling_type="${POOLING_TYPE}" \
  policy.patch_embedding_type="${PATCH_EMBEDDING_TYPE}" \
  policy.patch_level_adapter_channels=${PATCH_LEVEL_ADAPTER_CHANNELS} \
  policy.foveation_net.random_topology_prob=${RANDOM_TOPOLOGY_PROB} \
  "$@"
