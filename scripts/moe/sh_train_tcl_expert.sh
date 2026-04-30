## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:$PYTHONPATH
CUDA_VISIBLE_DEVICES=4,5,6,7 TRAIN_PRESET=tcl_force_table bash scripts/moe/sh_train_tcl_expert.sh
"

### Merge MoE ###
CONFIG_DIR="configs/"
CONFIG_NAME="tcl_force_dp.yaml"

# ===== Expert presets (domain_shift, task_name, ffn_expand_factor) =====
TRAIN_PRESET="${TRAIN_PRESET:-tcl_force}"
EXPERT_TRIPLETS=(
  "tcl_force         1201_screw_bulb_turn_off       4"
  "tcl_force_table   1201_screw_bulb_turn_off_table 6"
)
PRETRAINED_TIME_LOG="2026.01.17-00.06.29"  # 1201_screw_bulb_turn_off, original


domain_shift=""
task_name=""
ffn_expand_factor=""
for triple in "${EXPERT_TRIPLETS[@]}"; do
  read -r shift path expand <<< "${triple}"
  if [[ "${shift}" == "${TRAIN_PRESET}" ]]; then
    domain_shift="${shift}"
    task_name="${path}"
    ffn_expand_factor="${expand}"
    break
  fi
done

if [[ -z "${domain_shift}" ]]; then
  echo "[ERROR] TRAIN_PRESET='${TRAIN_PRESET}' not found."
  echo "[ERROR] Available presets:"
  for triple in "${EXPERT_TRIPLETS[@]}"; do
    read -r shift _ <<< "${triple}"
    echo "  - ${shift}"
  done
  exit 1
fi

to_path() {
  local s="$1"
  printf '%s\n' "${s/-/\/}"   # 只替换第一个 '-'
}

data_root="/home/geyuan/datasets/TCL/${task_name}/"
hdf5_path="/home/geyuan/datasets/TCL/hdf5/${task_name}_240p.h5"
PRETRAINED_CKPT="data/outputs/$(to_path "$PRETRAINED_TIME_LOG")_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"

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
  policy.ffn_expand_factor=${ffn_expand_factor} \
  task.dataset.data_root="${data_root}" \
  task.dataset.h5_path=${hdf5_path} \
  pretrained_ckpt="${PRETRAINED_CKPT}"

