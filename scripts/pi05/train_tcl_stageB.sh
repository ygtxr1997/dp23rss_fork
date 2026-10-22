#!/usr/bin/env bash
# Pi0.5 TCL Stage B: Stage A EMA plus a DP routing artifact.
# ASSET_DIR selects assets; a trailing asset_dir=... argument takes precedence.
set -euo pipefail

ASSET_DIR="${ASSET_DIR:-./outputs/pi05_data/tcl_foveation_seed42}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the training GPUs}"
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to a Pi0.5 TCL Stage A EMA or run directory}"
: "${FOVEATION_PRETRAINED_PATH:?Set FOVEATION_PRETRAINED_PATH to a DP TCL routing artifact}"

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

OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_tcl_stageB/$(date +%Y-%m-%d/%H-%M-%S)}"
# Keep each dataset list as one Hydra argument; prefer *_OVERRIDE for new runs.
DATA_ROOTS_OVERRIDE="${DATA_ROOTS_OVERRIDE:-${DATA_ROOTS:-[]}}"
H5_PATHS_OVERRIDE="${H5_PATHS_OVERRIDE:-${H5_PATHS:-[]}}"
CONFIG_NAME="${CONFIG_NAME:-pi05_tcl_sft}"
if [[ "${DATA_ROOTS_OVERRIDE}" == '[]' || "${H5_PATHS_OVERRIDE}" == '[]' ]]; then
  echo "[ERROR] Set non-empty DATA_ROOTS_OVERRIDE and H5_PATHS_OVERRIDE Hydra lists." >&2
  exit 1
fi

accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name="${CONFIG_NAME}" \
  "hydra.run.dir=${OUTPUT_DIR}" \
  "data_roots=${DATA_ROOTS_OVERRIDE}" \
  "h5_paths=${H5_PATHS_OVERRIDE}" \
  task_mapping_path="${TASK_MAPPING_PATH:-meta/tcl_tasks_10.yaml}" \
  pretrained_ckpt="${PRETRAINED_CKPT}" \
  pretrained_ckpt_mode="${PRETRAINED_CKPT_MODE:-latest}" \
  foveation_pretrained_path="${FOVEATION_PRETRAINED_PATH}" \
  training.stage_b=true \
  training.seed="${SEED:-42}" \
  dataset_seed="${DATASET_SEED:-${SEED:-42}}" \
  stats_seed="${STATS_SEED:-${SEED:-42}}" \
  state_dim="${STATE_DIM:-6}" \
  training.global_batch="${GLOBAL_BATCH:-128}" \
  training.micro_batch="${MICRO_BATCH:-16}" \
  training.max_steps="${MAX_STEPS:-6001}" \
  training.temporal_adaptive="${TEMPORAL_ADAPTIVE:-true}" \
  training.temporal_loss_weight="${TEMPORAL_LOSS_WEIGHT:-1.0}" \
  training.paired_loss_weight="${PAIRED_LOSS_WEIGHT:-1.0}" \
  training.paired_node_sampling=reachable \
  training.pairing_seed="${PAIRING_SEED:-${SEED:-42}}" \
  training.target_num_patches_start="${TARGET_NUM_PATCHES_START:-0.75}" \
  training.target_num_patches_decay_steps="${TARGET_NUM_PATCHES_DECAY_STEPS:-3000}" \
  training.budget_lambda_init="${BUDGET_LAMBDA_INIT:-0.0}" \
  training.budget_lambda_lr="${BUDGET_LAMBDA_LR:-0.01}" \
  training.downstream_finetune_start_step="${DOWNSTREAM_FINETUNE_START_STEP:-3000}" \
  model.action_horizon=32 \
  model.n_action_steps="${N_ACTION_STEPS:-24}" \
  model.visual_input_mode=mixed_foveation \
  model.foveation_split_mode=null \
  model.foveation_router_type=node_aligned_fpn \
  model.foveation_global_encoder_type=resnet10 \
  model.foveation_global_feature_dim=512 \
  model.foveation_global_encoder_resize=64 \
  model.foveation_router_hidden_dim=128 \
  model.foveation_router_cond_dim=512 \
  model.foveation_patch_level_adapter=true \
  model.foveation_apt_position_encoding=true \
  model.foveation_level_embedding=true \
  model.foveation_random_topology_prob=0.0 \
  model.foveation_freeze_global_encoder=true \
  model.foveation_router_temperature=1.0 \
  model.foveation_target_num_patches="${TARGET_NUM_PATCHES:-0.1}" \
  model.foveation_budget_loss_weight="${BUDGET_LOSS_WEIGHT:-0.1}" \
  model.foveation_budget_loss_type="${BUDGET_LOSS_TYPE:-upper_bound_lagrangian_elu}" \
  model.foveation_budget_base_cost_weight="${BUDGET_BASE_COST_WEIGHT:-0.0}" \
  rollout.enabled=false \
  checkpoint.monitor=val_loss \
  checkpoint.save_full=true \
  dataloader.num_workers="${DATALOADER_WORKERS:-0}" \
  "asset_dir=${ASSET_DIR}" \
  "$@"
