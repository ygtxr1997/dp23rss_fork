#!/usr/bin/env bash
# ASSET_DIR selects assets; a trailing asset_dir=... argument takes precedence.
set -euo pipefail

ASSET_DIR="${ASSET_DIR:-./outputs/pi05_data/tcl_foveation_seed42}"

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

OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_tcl_stageA/$(date +%Y-%m-%d/%H-%M-%S)}"
# These values must be Hydra list strings, e.g. '["/path/task_a","/path/task_b"]'.
# DATA_ROOTS_OVERRIDE/H5_PATHS_OVERRIDE take precedence over the legacy names.
CONFIG_NAME="${CONFIG_NAME:-pi05_tcl_sft}"
DATA_ROOTS_OVERRIDE="${DATA_ROOTS_OVERRIDE:-${DATA_ROOTS:-[]}}"
H5_PATHS_OVERRIDE="${H5_PATHS_OVERRIDE:-${H5_PATHS:-[]}}"
if [[ "${DATA_ROOTS_OVERRIDE}" == '[]' || "${H5_PATHS_OVERRIDE}" == '[]' ]]; then
  echo "[ERROR] Set non-empty DATA_ROOTS_OVERRIDE and H5_PATHS_OVERRIDE Hydra lists." >&2
  exit 1
fi

accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name="${CONFIG_NAME}" \
  "hydra.run.dir=${OUTPUT_DIR}" \
  "data_roots=${DATA_ROOTS_OVERRIDE}" \
  "h5_paths=${H5_PATHS_OVERRIDE}" \
  training.seed="${SEED:-42}" \
  dataset_seed="${DATASET_SEED:-${SEED:-42}}" \
  stats_seed="${STATS_SEED:-${SEED:-42}}" \
  state_dim="${STATE_DIM:-6}" \
  training.global_batch="${GLOBAL_BATCH:-128}" \
  training.micro_batch="${MICRO_BATCH:-16}" \
  training.max_steps="${MAX_STEPS:-4001}" \
  model.action_horizon=32 \
  model.n_action_steps="${N_ACTION_STEPS:-24}" \
  model.visual_input_mode=mixed_foveation \
  model.foveation_router_type=none \
  model.foveation_patch_level_adapter=true \
  model.foveation_apt_position_encoding=true \
  model.foveation_level_embedding=true \
  rollout.enabled=false \
  checkpoint.monitor=val_loss \
  "asset_dir=${ASSET_DIR}" \
  "$@"
