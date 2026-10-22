#!/usr/bin/env bash
# Pi0.5 LIBERO Stage B: Stage A EMA initialisation, DP-artifact routing and
# downstream phase using the Pi0.5-compatible node-aligned FPN.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the training GPUs}"
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to a Stage A EMA checkpoint or run directory}"
: "${FOVEATION_PRETRAINED_PATH:?Set FOVEATION_PRETRAINED_PATH to a DP routing artifact}"

# Example
tmp="
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PRETRAINED_CKPT=outputs/pi05_stageA/2026-10-16/12-13-09 \
FOVEATION_PRETRAINED_PATH=outputs/dp_for_pi05_stageB/2026-10-14/22-48-42/foveation_pretrained.pt \
bash scripts/pi05/train_libero_stageB.sh
"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPUS[@]}"
LAST_GPU="${GPUS[$((NUM_GPUS - 1))]}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-1234${LAST_GPU: -1}}"
LAUNCH_ARGS=(
  --num_processes="${NUM_GPUS}"
  --num_machines=1
  --machine_rank=0
  --mixed_precision=no
  --dynamo_backend=no
)
if (( NUM_GPUS > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT}")
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_stageB/$(date +%Y-%m-%d/%H-%M-%S)}"
SEED="${SEED:-42}"
MICRO_BATCH="${MICRO_BATCH:-16}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
MAX_STEPS="${MAX_STEPS:-6001}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-200}"
N_TEST="${N_TEST:-4}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"

accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name=pi05_libero_sft.yaml \
  "hydra.run.dir=${OUTPUT_DIR}" \
  training.stage_b=true \
  training.seed="${SEED}" \
  training.micro_batch="${MICRO_BATCH}" \
  training.global_batch="${GLOBAL_BATCH}" \
  training.max_steps="${MAX_STEPS}" \
  training.temporal_adaptive="${TEMPORAL_ADAPTIVE:-true}" \
  training.temporal_loss_weight="${TEMPORAL_LOSS_WEIGHT:-1.0}" \
  training.paired_loss_weight="${PAIRED_LOSS_WEIGHT:-1.0}" \
  training.paired_node_sampling="${PAIRED_NODE_SAMPLING:-reachable}" \
  training.pairing_seed="${PAIRING_SEED:-${SEED}}" \
  training.target_num_patches_start="${TARGET_NUM_PATCHES_START:-0.75}" \
  training.target_num_patches_decay_steps="${TARGET_NUM_PATCHES_DECAY_STEPS:-3000}" \
  training.budget_lambda_init="${BUDGET_LAMBDA_INIT:-0.0}" \
  training.budget_lambda_lr="${BUDGET_LAMBDA_LR:-0.01}" \
  training.downstream_finetune_start_step="${DOWNSTREAM_FINETUNE_START_STEP:-3000}" \
  pretrained_ckpt="${PRETRAINED_CKPT}" \
  pretrained_ckpt_mode="${PRETRAINED_CKPT_MODE:-latest}" \
  foveation_pretrained_path="${FOVEATION_PRETRAINED_PATH}" \
  model.visual_input_mode=mixed_foveation \
  model.foveation_split_mode=null \
  model.foveation_router_type=node_aligned_fpn \
  model.foveation_global_encoder_type=resnet10 \
  model.foveation_global_feature_dim=512 \
  model.foveation_router_hidden_dim=128 \
  model.foveation_global_encoder_resize=64 \
  model.foveation_patch_level_adapter=true \
  model.foveation_apt_position_encoding=true \
  model.foveation_level_embedding=true \
  model.foveation_random_topology_prob=0.0 \
  model.foveation_torch_home="${TORCH_HOME:-${FOVEATION_TORCH_HOME:-null}}" \
  model.foveation_freeze_global_encoder="${FREEZE_GLOBAL_ENCODER:-true}" \
  model.foveation_router_temperature=1.0 \
  model.foveation_target_num_patches="${TARGET_NUM_PATCHES:-0.1}" \
  model.foveation_budget_loss_weight="${BUDGET_LOSS_WEIGHT:-0.1}" \
  model.foveation_budget_loss_type="${BUDGET_LOSS_TYPE:-upper_bound_lagrangian_elu}" \
  model.foveation_budget_base_cost_weight="${BUDGET_BASE_COST_WEIGHT:-0.0}" \
  rollout.every="${ROLLOUT_EVERY}" \
  runner.n_test="${N_TEST}" \
  runner.log_patch_counts=true \
  runner.save_foveation_video=true \
  dataloader.num_workers="${DATALOADER_WORKERS}" \
  "$@"
