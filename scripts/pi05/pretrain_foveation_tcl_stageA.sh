#!/usr/bin/env bash
# DP TCL Stage A: fixed mixed-resolution topology and frozen routing branches.
# ASSET_DIR selects assets; a trailing asset_dir=... argument takes precedence.
set -euo pipefail

# Example: reuse weights with new data/statistics. ASSET_DIR (or asset_dir=...)
# must point to assets prepared for the current DATA_ROOTS/H5_PATHS.
tmp="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/pi05/pretrain_foveation_tcl_stageA.sh \
  pretrained_ckpt=/home/geyuan/code/dp23rss_fork/outputs/dp_tcl_for_pi05/2026-10-19/21-08-48/checkpoints/latest.ckpt \
  pretrained_weights_only=true training.resume=false
"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the training GPUs}"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPUS[@]}"
LAST_GPU="${GPUS[$((NUM_GPUS - 1))]}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-1234${LAST_GPU: -1}}"
LAUNCH_ARGS=(--num_processes="${NUM_GPUS}" --num_machines=1 --machine_rank=0 --mixed_precision=no)
if (( NUM_GPUS > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT}")
fi

DATA_ROOTS=(
  /home/geyuan/datasets/TCL/1014_lh_tower_three_blocks/
  /home/geyuan/datasets/TCL/260514_ethernet/
  /home/geyuan/datasets/TCL/1014_insert_two_batteries/
  /home/geyuan/datasets/reverse/0417_brush_reversed/
)
H5_PATHS=(
  /home/geyuan/datasets/TCL/hdf5/1014_lh_tower_three_blocks_256x320.h5
  /home/geyuan/datasets/TCL/hdf5/260514_ethernet_256x320.h5
  /home/geyuan/datasets/TCL/hdf5/1014_insert_two_batteries_240p.h5
  /home/geyuan/datasets/reverse/hdf5/0417_brush_reversed_240p.h5
)

# Serialize the editable Bash arrays into one Hydra/OmegaConf list argument.
# Quoting each path keeps commas, spaces, and JSON-significant characters inside
# one list item instead of letting the shell or Hydra split the value.
to_hydra_list() {
  local value escaped output='[' separator=''
  for value in "$@"; do
    escaped="${value//\\/\\\\}"
    escaped="${escaped//\"/\\\"}"
    output+="${separator}\"${escaped}\""
    separator=','
  done
  printf '%s]' "${output}"
}

if (( ${#DATA_ROOTS[@]} == 0 || ${#DATA_ROOTS[@]} != ${#H5_PATHS[@]} )); then
  echo "[ERROR] DATA_ROOTS and H5_PATHS must be non-empty arrays with matching lengths." >&2
  exit 1
fi
DATA_ROOTS_HYDRA="$(to_hydra_list "${DATA_ROOTS[@]}")"
H5_PATHS_HYDRA="$(to_hydra_list "${H5_PATHS[@]}")"
DATA_ROOTS_OVERRIDE="${DATA_ROOTS_OVERRIDE:-${DATA_ROOTS_HYDRA}}"
H5_PATHS_OVERRIDE="${H5_PATHS_OVERRIDE:-${H5_PATHS_HYDRA}}"
if [[ "${DATA_ROOTS_OVERRIDE}" == '[]' || "${H5_PATHS_OVERRIDE}" == '[]' ]]; then
  echo "[ERROR] DATA_ROOTS_OVERRIDE and H5_PATHS_OVERRIDE must not be empty lists." >&2
  exit 1
fi

ASSET_DIR="${ASSET_DIR:-./outputs/pi05_data/tcl_foveation_seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dp_tcl_for_pi05/$(date +%Y-%m-%d/%H-%M-%S)}"
# GPU_RESERVE_GB holds an extra fixed GiB per GPU for the process lifetime; 0 disables it.
FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-[\"40%level:2\",\"60%random:0.05-0.99\"]}"
SPLIT_MODE_OVERRIDE="policy.foveation_net.fix_split_mode=${FIX_SPLIT_MODE}"

accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name=tcl_foveation_dp_for_pi05 \
  "hydra.run.dir=${OUTPUT_DIR}" \
  training.reserve_gpu_memory_gb="${GPU_RESERVE_GB:-0}" \
  "task.dataset.data_roots=${DATA_ROOTS_OVERRIDE}" \
  "task.dataset.h5_paths=${H5_PATHS_OVERRIDE}" \
  task.dataset.task_mapping_path="${TASK_MAPPING_PATH:-meta/tcl_tasks_10.yaml}" \
  training.seed="${SEED:-42}" \
  task.dataset.seed="${DATASET_SEED:-${SEED:-42}}" \
  training.num_epochs="${NUM_EPOCHS:-1200}" \
  training.router_only=false \
  policy.router_loss_type=st \
  "${SPLIT_MODE_OVERRIDE}" \
  policy.foveation_net.random_topology_prob=0.0 \
  policy.foveation_net.freeze_global_encoder=true \
  policy.foveation_net.freeze_router=true \
  policy.foveation_net.budget_loss_weight=0.0 \
  horizon="${HORIZON:-32}" \
  "asset_dir=${ASSET_DIR}" \
  "$@"
