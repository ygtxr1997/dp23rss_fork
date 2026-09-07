#!/usr/bin/env bash

# Match the architecture and data/diffusion settings of the stage A checkpoint.
# Example (stage A trained with Conv patches and level adapters):
tmp="
CUDA_VISIBLE_DEVICES=1,3,5,7 \
 VIS_ENC_LAYERS=4 POOLING_TYPE=cls_token \
 PATCH_EMBEDDING_TYPE=conv PATCH_LEVEL_ADAPTER_CHANNELS=16 \
 SEED=42 bash scripts/roi/sh_train_foveation_pusht_stageB0.sh
"

PRETRAINED_CKPT="/home/geyuan/code/dp23rss_fork/outputs/2026-09-05/23-54-36/checkpoints/epoch=1800-test_mean_score=0.613.ckpt"

set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the training GPUs.}"
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the stage A checkpoint.}"

CONFIG_DIR="configs"
CONFIG_NAME="pusht256_foveation_dp.yaml"
SEED="${SEED:-42}"
VIS_ENC_LAYERS="${VIS_ENC_LAYERS:-4}"
POOLING_TYPE="${POOLING_TYPE:-cls_token}"
PATCH_EMBEDDING_TYPE="${PATCH_EMBEDDING_TYPE:-conv}"
PATCH_LEVEL_ADAPTER_CHANNELS="${PATCH_LEVEL_ADAPTER_CHANNELS:-null}"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPUS[@]}"
LAST_GPU="${GPUS[$((NUM_GPUS - 1))]}"
PORT="1234${LAST_GPU: -1}"

accelerate launch \
  --multi_gpu \
  --num_processes="${NUM_GPUS}" \
  --main_process_port="${PORT}" \
  train.py \
  --config-dir="${CONFIG_DIR}" \
  --config-name="${CONFIG_NAME}" \
  training.seed="${SEED}" \
  training.router_only=true \
  training.resume=false \
  training.use_ema=true \
  "pretrained_ckpt='${PRETRAINED_CKPT}'" \
  policy.visual_encoder_layers="${VIS_ENC_LAYERS}" \
  policy.pooling_type="${POOLING_TYPE}" \
  policy.patch_embedding_type="${PATCH_EMBEDDING_TYPE}" \
  policy.patch_level_adapter_channels="${PATCH_LEVEL_ADAPTER_CHANNELS}" \
  policy.foveation_net.fix_split_mode=null \
  policy.foveation_net.freeze_global_encoder=false \
  policy.foveation_net.freeze_router=false \
  policy.foveation_net.random_topology_prob=0.0 \
  policy.foveation_net.budget_loss_weight=0.0 \
  "$@"
