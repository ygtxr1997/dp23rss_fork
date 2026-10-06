#!/usr/bin/env bash
# Stage A EMA -> paired routing with frozen visual/action weights.
# CUDA_VISIBLE_DEVICES=0,1 PRETRAINED_CKPT=/path/to/stageA/checkpoints/latest.ckpt \
#   bash scripts/roi/sh_train_foveation_tcl_stageB.sh
# Supply the same DATA_ROOT/H5_PATH and architecture overrides as Stage A.
# DATA_SEED_FROM_CKPT restores seed, val_ratio and max_train_episodes.
# Checkpoint episode lists and full-dataset scaling are reused in either case.
# PRETRAINED_CKPT_MODE=best/latest accepts a run directory; null loads the exact file.
# TEMPORAL_ADAPTIVE=true enables same-episode temporal pairing; weight=0 is its ablation.
# DOWNSTREAM_FINETUNE_START_EPOCH=null keeps routing-only training for all epochs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the training GPUs}"
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the Stage A checkpoint or run directory}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPUS[@]}"
LAST_GPU="${GPUS[$((NUM_GPUS - 1))]}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-1234${LAST_GPU: -1}}"
LAUNCH_ARGS=(--num_processes="${NUM_GPUS}" --num_machines=1 --mixed_precision=no)
if (( NUM_GPUS > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT}")
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$(date +%Y-%m-%d/%H-%M-%S)}"
DATA_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then DATA_ARGS+=("task.dataset.data_root='${DATA_ROOT}'"); fi
if [[ -n "${H5_PATH:-}" ]]; then DATA_ARGS+=("task.dataset.h5_path='${H5_PATH}'"); fi

accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name=tcl_force_foveation_dp.yaml \
  "hydra.run.dir=${OUTPUT_DIR}" \
  "pretrained_ckpt='${PRETRAINED_CKPT}'" \
  pretrained_ckpt_mode="${PRETRAINED_CKPT_MODE:-null}" \
  training.seed="${SEED:-42}" \
  task.dataset.seed="${DATASET_SPLIT_SEED:-42}" \
  data_seed_from_ckpt="${DATA_SEED_FROM_CKPT:-true}" \
  horizon="${HORIZON:-32}" \
  training.num_epochs="${NUM_EPOCHS:-300}" \
  training.router_only=true \
  training.budget_lambda_init="${BUDGET_LAMBDA_INIT:-0.0}" \
  training.budget_lambda_lr="${BUDGET_LAMBDA_LR:-0.01}" \
  training.downstream_finetune_start_epoch="${DOWNSTREAM_FINETUNE_START_EPOCH:-null}" \
  training.target_num_patches_start="${TARGET_NUM_PATCHES_START:-0.75}" \
  training.target_num_patches_decay_epochs="${TARGET_NUM_PATCHES_DECAY_EPOCHS:-200}" \
  policy.router_loss_type=paired \
  policy.paired_loss_weight="${PAIRED_LOSS_WEIGHT:-1.0}" \
  policy.paired_node_sampling="${PAIRED_NODE_SAMPLING:-reachable}" \
  policy.temporal_adaptive="${TEMPORAL_ADAPTIVE:-false}" \
  policy.temporal_loss_weight="${TEMPORAL_LOSS_WEIGHT:-1.0}" \
  policy.visual_encoder_layers="${VIS_ENC_LAYERS:-4}" \
  policy.pooling_type="${POOLING_TYPE:-cls_token}" \
  policy.patch_embedding_type="${PATCH_EMBEDDING_TYPE:-conv}" \
  policy.patch_level_adapter_channels="${PATCH_LEVEL_ADAPTER_CHANNELS:-16}" \
  policy.foveation_net.fix_split_mode=null \
  policy.foveation_net.random_topology_prob=0.0 \
  policy.foveation_net.target_num_patches="${TARGET_NUM_PATCHES:-0.1}" \
  policy.foveation_net.budget_loss_type="${BUDGET_LOSS_TYPE:-upper_bound_lagrangian_elu}" \
  policy.foveation_net.budget_loss_weight="${BUDGET_LOSS_WEIGHT:-0.1}" \
  policy.foveation_net.budget_base_cost_weight="${BUDGET_BASE_COST_WEIGHT:-0.0}" \
  task.dataset.p_cond_drop="${P_COND_DROP:-0.0}" \
  task.env_runner.save_foveation_video="${SAVE_FOVEATION_VIDEO:-true}" \
  task.env_runner.max_episodes="${VIDEO_MAX_EPISODES:-null}" \
  training.topology_samples_per_task="${TOPOLOGY_SAMPLES_PER_TASK:-64}" \
  "${DATA_ARGS[@]}" "$@"
