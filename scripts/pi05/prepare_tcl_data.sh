#!/usr/bin/env bash
# ASSET_DIR selects assets; a trailing asset_dir=... argument takes precedence.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Example
tmp="
ASSET_DIR='outputs/pi05_data/tcl_foveation_seed42_new' \
  bash scripts/pi05/prepare_tcl_data.sh
"

# Keep the editable source lists in the same format as the DP TCL scripts.
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

# Serialize Bash arrays into one Hydra/OmegaConf list argument.
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
TASK_MAPPING_PATH="${TASK_MAPPING_PATH:-meta/tcl_tasks_10.yaml}"
# Writes dataset_metadata.json only (q01/q99, min/max, mean/std); existing assets are never overwritten.

exec python -m diffusion_policy.common.prepare_data --config-name=pi05_tcl_data \
  "data_roots=${DATA_ROOTS_OVERRIDE}" \
  "h5_paths=${H5_PATHS_OVERRIDE}" \
  "task_mapping_path=${TASK_MAPPING_PATH}" \
  "asset_dir=${ASSET_DIR}" \
  "$@"
