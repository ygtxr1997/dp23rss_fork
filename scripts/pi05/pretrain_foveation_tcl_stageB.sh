#!/usr/bin/env bash
# DP TCL Stage B: router-only training and Pi0.5 routing artifact export.
# ASSET_DIR selects assets; a trailing asset_dir=... argument takes precedence.
set -euo pipefail

# Example
tmp="
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/pi05/pretrain_foveation_tcl_stageB.sh
"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the training GPUs}"
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the DP TCL Stage A checkpoint}"

# Pass one complete Hydra list argument. *_OVERRIDE is the preferred interface;
# DATA_ROOTS/H5_PATHS remain accepted for compatibility with older commands.
DATA_ROOTS_OVERRIDE="${DATA_ROOTS_OVERRIDE:-${DATA_ROOTS:-}}"
H5_PATHS_OVERRIDE="${H5_PATHS_OVERRIDE:-${H5_PATHS:-}}"
: "${DATA_ROOTS_OVERRIDE:?Set DATA_ROOTS_OVERRIDE to a Hydra list of TCL task directories}"
: "${H5_PATHS_OVERRIDE:?Set H5_PATHS_OVERRIDE to a Hydra list of TCL HDF5 files}"
if [[ "${DATA_ROOTS_OVERRIDE}" == '[]' || "${H5_PATHS_OVERRIDE}" == '[]' ]]; then
  echo "[ERROR] DATA_ROOTS_OVERRIDE and H5_PATHS_OVERRIDE must not be empty lists." >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPUS[@]}"
LAST_GPU="${GPUS[$((NUM_GPUS - 1))]}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-1234${LAST_GPU: -1}}"
LAUNCH_ARGS=(--num_processes="${NUM_GPUS}" --num_machines=1 --machine_rank=0 --mixed_precision=no)
if (( NUM_GPUS > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT}")
fi

ASSET_DIR="${ASSET_DIR:-./outputs/pi05_data/tcl_foveation_seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dp_tcl_for_pi05/$(date +%Y-%m-%d/%H-%M-%S)}"
FOVEATION_EXPORT_PATH="${FOVEATION_EXPORT_PATH:-foveation_pretrained.pt}"

# GPU_RESERVE_GB holds an extra fixed GiB per GPU for the process lifetime; 0 disables it.
accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name=tcl_foveation_dp_for_pi05 \
  "hydra.run.dir=${OUTPUT_DIR}" \
  training.reserve_gpu_memory_gb="${GPU_RESERVE_GB:-0}" \
  "task.dataset.data_roots=${DATA_ROOTS_OVERRIDE}" \
  "task.dataset.h5_paths=${H5_PATHS_OVERRIDE}" \
  task.dataset.task_mapping_path="${TASK_MAPPING_PATH:-meta/tcl_tasks_10.yaml}" \
  "pretrained_ckpt=${PRETRAINED_CKPT}" \
  pretrained_ckpt_mode="${PRETRAINED_CKPT_MODE:-latest}" \
  "foveation_export_path=${FOVEATION_EXPORT_PATH}" \
  data_seed_from_ckpt=true \
  training.seed="${SEED:-42}" \
  training.num_epochs="${NUM_EPOCHS:-900}" \
  training.router_only=true \
  training.downstream_finetune_start_epoch=null \
  training.target_num_patches_start="${TARGET_NUM_PATCHES_START:-0.75}" \
  training.target_num_patches_decay_epochs="${TARGET_NUM_PATCHES_DECAY_EPOCHS:-600}" \
  training.budget_lambda_init="${BUDGET_LAMBDA_INIT:-0.0}" \
  training.budget_lambda_lr="${BUDGET_LAMBDA_LR:-0.01}" \
  policy.router_loss_type=paired \
  policy.temporal_adaptive=false \
  policy.temporal_loss_weight="${TEMPORAL_LOSS_WEIGHT:-1.0}" \
  policy.foveation_net.fix_split_mode=null \
  policy.foveation_net.random_topology_prob=0.0 \
  policy.foveation_net.freeze_global_encoder=false \
  policy.foveation_net.freeze_router=false \
  policy.foveation_net.target_num_patches="${TARGET_NUM_PATCHES:-0.1}" \
  policy.foveation_net.budget_loss_weight="${BUDGET_LOSS_WEIGHT:-0.1}" \
  horizon="${HORIZON:-32}" \
  "asset_dir=${ASSET_DIR}" \
  "$@"
