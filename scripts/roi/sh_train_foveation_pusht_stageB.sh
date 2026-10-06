#!/usr/bin/env bash

# Stage B: paired stop/split loss with optional differentiable budget loss.
# Match the architecture and data/diffusion settings of the stage A checkpoint.
# Enable the global upper-bound budget below; omitting BUDGET_LOSS_TYPE keeps
# the original per_observation_mse. batch_mean_mse keeps its per-GPU mean.
# A trailing Hydra override also works:
# policy.foveation_net.budget_loss_type=upper_bound_lagrangian
# Experimental saturating savings: set BUDGET_LOSS_TYPE=upper_bound_lagrangian_elu
# and BUDGET_ELU_TAU=0.025 (6 patches at a 16-256 legal range).
# ELU mode uses only lambda above target; BASE_COST_WEIGHT controls reward at/below it.
# Only reachable is accepted; single_level was removed (override old checkpoint configs).
# Temporal matches full-subtree deltas using random anchors; failures keep Spatial/budget.
# Active matching changes Spatial's candidate distribution, even at Temporal weight 0.
# Original Spatial only: one window, uniform reachable sampling.
tmp="
CUDA_VISIBLE_DEVICES=1,3,5,7 \
 VIS_ENC_LAYERS=4 POOLING_TYPE=cls_token \
 PATCH_EMBEDDING_TYPE=conv PATCH_LEVEL_ADAPTER_CHANNELS=16 \
 PAIRED_NODE_SAMPLING=reachable PAIRED_LOSS_WEIGHT=1.0 \
 TEMPORAL_ADAPTIVE=false \
 BUDGET_LOSS_WEIGHT=0.1 TARGET_NUM_PATCHES=0.1 \
 BUDGET_LOSS_TYPE=upper_bound_lagrangian_elu BUDGET_BASE_COST_WEIGHT=0.0 \
 BUDGET_LAMBDA_INIT=0.0 BUDGET_LAMBDA_LR=0.01 \
 DOWNSTREAM_FINETUNE_START_EPOCH=1200 SEED=42 \
 bash scripts/roi/sh_train_foveation_pusht_stageB.sh
"

# Spatial + Temporal: two windows, active matching, Temporal loss.
# For the matched double-window ablation, change only TEMPORAL_LOSS_WEIGHT to 0.0.
# Both examples use the same Stage A checkpoint below, architecture, seed and budget.
tmp="
CUDA_VISIBLE_DEVICES=1,3,5,7 \
 VIS_ENC_LAYERS=4 POOLING_TYPE=cls_token \
 PATCH_EMBEDDING_TYPE=conv PATCH_LEVEL_ADAPTER_CHANNELS=16 \
 PAIRED_NODE_SAMPLING=reachable PAIRED_LOSS_WEIGHT=1.0 \
 TEMPORAL_ADAPTIVE=true TEMPORAL_LOSS_WEIGHT=1.0 \
 BUDGET_LOSS_WEIGHT=0.1 TARGET_NUM_PATCHES=0.1 \
 BUDGET_LOSS_TYPE=upper_bound_lagrangian_elu BUDGET_BASE_COST_WEIGHT=0.0 \
 BUDGET_LAMBDA_INIT=0.0 BUDGET_LAMBDA_LR=0.01 \
 DOWNSTREAM_FINETUNE_START_EPOCH=1200 SEED=42 \
 bash scripts/roi/sh_train_foveation_pusht_stageB.sh
"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/geyuan/code/dp23rss_fork/outputs/2026-09-05/23-54-36/checkpoints/epoch=1800-test_mean_score=0.613.ckpt}"

set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the training GPUs.}"
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the stage A checkpoint.}"

CONFIG_DIR="configs"
CONFIG_NAME="pusht256_foveation_dp.yaml"
SEED="${SEED:-42}"
PAIRED_LOSS_WEIGHT="${PAIRED_LOSS_WEIGHT:-1.0}"
TEMPORAL_ADAPTIVE="${TEMPORAL_ADAPTIVE:-false}"
TEMPORAL_LOSS_WEIGHT="${TEMPORAL_LOSS_WEIGHT:-1.0}"
PAIRED_NODE_SAMPLING="${PAIRED_NODE_SAMPLING:-reachable}"
BUDGET_LOSS_WEIGHT="${BUDGET_LOSS_WEIGHT:-0.1}"
# Original per-observation MSE; set batch_mean_mse for this forward's per-GPU mean.
BUDGET_LOSS_TYPE="${BUDGET_LOSS_TYPE:-per_observation_mse}"
# Both upper-bound modes: relative prices inside the single BUDGET_LOSS_WEIGHT.
# ELU mode: zero disables the below-target reward, leaving only excess cost.
BUDGET_BASE_COST_WEIGHT="${BUDGET_BASE_COST_WEIGHT:-0.01}"
# ELU mode only; normalized saturation width, independent of the base price.
BUDGET_ELU_TAU="${BUDGET_ELU_TAU:-0.025}"
BUDGET_LAMBDA_INIT="${BUDGET_LAMBDA_INIT:-0.0}"
BUDGET_LAMBDA_LR="${BUDGET_LAMBDA_LR:-0.01}"
TARGET_NUM_PATCHES="${TARGET_NUM_PATCHES:-0.5}"
VIS_ENC_LAYERS="${VIS_ENC_LAYERS:-4}"
POOLING_TYPE="${POOLING_TYPE:-cls_token}"
PATCH_EMBEDDING_TYPE="${PATCH_EMBEDDING_TYPE:-conv}"
PATCH_LEVEL_ADAPTER_CHANNELS="${PATCH_LEVEL_ADAPTER_CHANNELS:-null}"
# Capture the same test episodes as sim_video, then render their quadtrees.
SAVE_FOVEATION_VIDEO="${SAVE_FOVEATION_VIDEO:-true}"
# Optional final downstream-only phase; e.g. 1600 for the last 400/2000 epochs.
DOWNSTREAM_FINETUNE_START_EPOCH="${DOWNSTREAM_FINETUNE_START_EPOCH:-null}"

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
  training.budget_lambda_init="${BUDGET_LAMBDA_INIT}" \
  training.budget_lambda_lr="${BUDGET_LAMBDA_LR}" \
  training.downstream_finetune_start_epoch="${DOWNSTREAM_FINETUNE_START_EPOCH}" \
  training.resume=false \
  training.use_ema=true \
  task.env_runner.save_foveation_video="${SAVE_FOVEATION_VIDEO}" \
  "pretrained_ckpt='${PRETRAINED_CKPT}'" \
  policy.router_loss_type=paired \
  policy.paired_loss_weight="${PAIRED_LOSS_WEIGHT}" \
  policy.temporal_adaptive="${TEMPORAL_ADAPTIVE}" \
  policy.temporal_loss_weight="${TEMPORAL_LOSS_WEIGHT}" \
  policy.paired_node_sampling="${PAIRED_NODE_SAMPLING}" \
  policy.visual_encoder_layers="${VIS_ENC_LAYERS}" \
  policy.pooling_type="${POOLING_TYPE}" \
  policy.patch_embedding_type="${PATCH_EMBEDDING_TYPE}" \
  policy.patch_level_adapter_channels="${PATCH_LEVEL_ADAPTER_CHANNELS}" \
  policy.foveation_net.fix_split_mode=null \
  policy.foveation_net.freeze_global_encoder=false \
  policy.foveation_net.freeze_router=false \
  policy.foveation_net.random_topology_prob=0.0 \
  policy.foveation_net.budget_loss_weight="${BUDGET_LOSS_WEIGHT}" \
  policy.foveation_net.budget_loss_type="${BUDGET_LOSS_TYPE}" \
  policy.foveation_net.budget_base_cost_weight="${BUDGET_BASE_COST_WEIGHT}" \
  policy.foveation_net.budget_elu_tau="${BUDGET_ELU_TAU}" \
  policy.foveation_net.target_num_patches="${TARGET_NUM_PATCHES}" \
  "$@"
