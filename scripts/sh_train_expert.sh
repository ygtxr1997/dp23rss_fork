## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:$PYTHONPATH
CUDA_VISIBLE_DEVICES=6,7 TRAIN_PRESET=block bash scripts/sh_train_expert.sh
"

### Merge MoE ###
CONFIG_DIR="configs/"
CONFIG_NAME="pusht256_dp.yaml"

# ===== Expert presets (domain_shift, zarr_path, ffn_expand_factor) =====
TRAIN_PRESET="${TRAIN_PRESET:-goal}"
EXPERT_TRIPLETS=(
  "none data/pusht/pusht_256.zarr 2"
  "block data/pusht/pusht_256_block.zarr 2"
  "light data/pusht/pusht_256_light.zarr 4"
  "goal data/pusht/pusht_256_goal.zarr 6"
)

domain_shift=""
zarr_path=""
ffn_expand_factor=""
for triple in "${EXPERT_TRIPLETS[@]}"; do
  read -r shift path expand <<< "${triple}"
  if [[ "${shift}" == "${TRAIN_PRESET}" ]]; then
    domain_shift="${shift}"
    zarr_path="${path}"
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

zarr_paths=("${zarr_path}")
zarr_paths_override="[$(IFS=,; echo "${zarr_paths[*]}")]"

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
  task.env_runner.domain_shift=${domain_shift} \
  task.dataset.zarr_paths=${zarr_paths_override}
