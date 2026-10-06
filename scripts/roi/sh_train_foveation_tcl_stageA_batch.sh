#!/usr/bin/env bash
# Train TCL datasets sequentially. Add matching entries to DATA_ROOTS and H5_PATHS.
# Environment variables override defaults; trailing Hydra arguments reach every run.
# Example: CUDA_VISIBLE_DEVICES=0,1 bash scripts/roi/sh_train_foveation_tcl_stageA_batch.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,6}"
export SEED="${SEED:-42}"
export DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
export HORIZON="${HORIZON:-32}"
export VIS_ENC_LAYERS="${VIS_ENC_LAYERS:-4}"
export FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-[\"40%level:2\",\"60%random:0.05-0.99\"]}"

DATA_ROOTS=(
  /home/geyuan/local_soft/TCL/260514_ethernet/
)
H5_PATHS=(
  /home/geyuan/local_soft/TCL/hdf5/260514_ethernet_256x320.h5
)
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
if (( ${#DATA_ROOTS[@]} == 0 || ${#DATA_ROOTS[@]} != ${#H5_PATHS[@]} )); then
  echo "[ERROR] Fill DATA_ROOTS and H5_PATHS with matching dataset entries." >&2
  exit 1
fi

for i in "${!DATA_ROOTS[@]}"; do
  RUN_DIR="${OUTPUT_ROOT}/$(date +%Y-%m-%d/%H-%M-%S)"
  echo "[Stage A] Dataset: ${DATA_ROOTS[$i]} -> ${RUN_DIR}"
  DATA_ROOT="${DATA_ROOTS[$i]}" H5_PATH="${H5_PATHS[$i]}" OUTPUT_DIR="${RUN_DIR}" \
    bash scripts/roi/sh_train_foveation_tcl_stageA.sh "$@"
done
