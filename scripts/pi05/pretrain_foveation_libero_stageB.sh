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

# Example
tmp="
GPU_RESERVE_GB=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
PRETRAINED_CKPT=outputs/dp_for_pi05_stageA/2026-10-13/20-14-19/checkpoints/latest.ckpt \
OUTPUT_DIR=outputs/dp_for_pi05_stageB/2026-10-14/22-48-42/ \
bash scripts/pi05/pretrain_foveation_libero_stageB.sh \
  training.resume=true
"

# PRETRAINED_CKPT must come from the dedicated dual-camera, ten-task DP StageA.
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the DP StageA checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dp_for_pi05_stageB/$(date +%Y-%m-%d/%H-%M-%S)}"
# Relative export paths are run-relative; pass an absolute path to export elsewhere.
FOVEATION_EXPORT_PATH="${FOVEATION_EXPORT_PATH:-foveation_pretrained.pt}"

# GPU_RESERVE_GB=8 holds an extra 8 GiB per GPU for the process lifetime; 0 disables.
accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name=liberoft_foveation_dp_for_pi05 \
  "hydra.run.dir=${OUTPUT_DIR}" \
  training.reserve_gpu_memory_gb="${GPU_RESERVE_GB:-0}" \
  "pretrained_ckpt='${PRETRAINED_CKPT}'" \
  pretrained_ckpt_mode="${PRETRAINED_CKPT_MODE:-null}" \
  "foveation_export_path='${FOVEATION_EXPORT_PATH}'" \
  data_seed_from_ckpt=true \
  training.seed="${SEED:-43}" \
  horizon="${HORIZON:-32}" \
  training.num_epochs="${NUM_EPOCHS:-1201}" \
  training.router_only=true \
  training.downstream_finetune_start_epoch="${DOWNSTREAM_FINETUNE_START_EPOCH:-600}" \
  training.target_num_patches_start="${TARGET_NUM_PATCHES_START:-0.75}" \
  training.target_num_patches_decay_epochs="${TARGET_NUM_PATCHES_DECAY_EPOCHS:-600}" \
  training.budget_lambda_init="${BUDGET_LAMBDA_INIT:-0.0}" \
  training.budget_lambda_lr="${BUDGET_LAMBDA_LR:-0.01}" \
  policy.router_loss_type=paired \
  policy.paired_node_sampling="${PAIRED_NODE_SAMPLING:-reachable}" \
  policy.paired_loss_weight="${PAIRED_LOSS_WEIGHT:-1.0}" \
  policy.temporal_adaptive="${TEMPORAL_ADAPTIVE:-true}" \
  policy.temporal_loss_weight="${TEMPORAL_LOSS_WEIGHT:-1.0}" \
  policy.visual_encoder_layers="${VIS_ENC_LAYERS:-4}" \
  policy.pooling_type="${POOLING_TYPE:-cls_token}" \
  policy.patch_embedding_type="${PATCH_EMBEDDING_TYPE:-conv}" \
  policy.patch_level_adapter_channels="${PATCH_LEVEL_ADAPTER_CHANNELS:-16}" \
  policy.foveation_net.fix_split_mode=null \
  policy.foveation_net.random_topology_prob=0.0 \
  policy.foveation_net.target_num_patches="${TARGET_NUM_PATCHES:-0.1}" \
  policy.foveation_net.budget_loss_type=upper_bound_lagrangian_elu \
  policy.foveation_net.budget_loss_weight="${BUDGET_LOSS_WEIGHT:-0.1}" \
  policy.foveation_net.budget_base_cost_weight=0.0 \
  task.env_runner.log_patch_counts=true \
  task.env_runner.save_foveation_video="${SAVE_FOVEATION_VIDEO:-true}" \
  "$@"
