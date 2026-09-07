
## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:$PYTHONPATH
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/reverse/sh_train_cfgrl_baseline.sh
"

CONFIG_DIR="./configs"
CONFIG_NAME="tcl_force_cfgrl.yaml"


#DATA_ROOT="/home/geyuan/datasets/reverse/0209_tower_boby_easy_reversed/"
#H5_PATH="/home/geyuan/datasets/reverse/hdf5/0209_tower_boby_easy_reversed_240p.h5"

#DATA_ROOT="/home/geyuan/datasets/reverse/0417_put_mouse_reversed/"
#H5_PATH="/home/geyuan/datasets/reverse/hdf5/0417_put_mouse_reversed_240p.h5"

#DATA_ROOT="/home/geyuan/datasets/reverse/0417_test_tube_reversed/"
#H5_PATH="/home/geyuan/datasets/reverse/hdf5/0417_test_tube_reversed_240p.h5"

#DATA_ROOT="/home/geyuan/datasets/reverse/0417_brush_reversed/"
#H5_PATH="/home/geyuan/datasets/reverse/hdf5/0417_brush_reversed_240p.h5"

DATA_ROOT="/home/geyuan/datasets/TCL/260514_ethernet/"
H5_PATH="/home/geyuan/datasets/TCL/hdf5/260514_ethernet_240p.h5"


DEVICE="cuda"

export HYDRA_FULL_ERROR=1

set -e
set -x

wandb online

### DDP training with accelerate
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] CUDA_VISIBLE_DEVICES is empty. Please set it before running."
  exit 1
fi

IFS=',' read -r -a _raw_gpus <<< "${CUDA_VISIBLE_DEVICES}"
_gpus=()
for _gpu in "${_raw_gpus[@]}"; do
  _gpu="${_gpu//[[:space:]]/}"
  if [[ -n "${_gpu}" ]]; then
    _gpus+=("${_gpu}")
  fi
done

if [[ ${#_gpus[@]} -eq 0 ]]; then
  echo "[ERROR] No valid GPU id parsed from CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'."
  exit 1
fi

NUM_GPUS=${#_gpus[@]}
_last_gpu="${_gpus[$((NUM_GPUS-1))]}"
_last_gpu_last_digit="${_last_gpu: -1}"
MAIN_PORT="1234${_last_gpu_last_digit}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}, MAIN_PORT=${MAIN_PORT}"

accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  --main_process_port=${MAIN_PORT} \
  train.py --config-dir=${CONFIG_DIR} --config-name=${CONFIG_NAME} training.seed=42 \
  training.device=${DEVICE} \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
  task.dataset.data_root="${DATA_ROOT}" \
  task.dataset.h5_path="${H5_PATH}"