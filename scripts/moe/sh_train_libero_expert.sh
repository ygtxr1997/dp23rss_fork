## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:$PYTHONPATH
CUDA_VISIBLE_DEVICES=1,4,5,7 TRAIN_PRESET=libero_force_view_slice bash scripts/moe/sh_train_libero_expert.sh
"

### Merge MoE ###
CONFIG_DIR="configs/"
CONFIG_NAME="libero_force_dp.yaml"

# ===== Expert presets (domain_shift, hdf5_path, ffn_expand_factor) =====
HDF5_PATH="KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench.hdf5"
#HDF5_PATH="STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo_wrench.hdf5"
#HDF5_PATH="KITCHEN_SCENE6_close_the_microwave_demo_wrench.hdf5"
TRAIN_PRESET="${TRAIN_PRESET:-libero_force}"
EXPERT_TRIPLETS=(
  "libero_force         ${HDF5_PATH}  4"
  "libero_force_color_slice   ${HDF5_PATH}  2"
  "libero_force_view_slice    ${HDF5_PATH}  10"
  "libero_force_texture_slice ${HDF5_PATH}  6"
  "libero_force_darken  ${HDF5_PATH}  8"
)
PRETRAINED_TIME_LOG="2026.04.27-01.29.21"  # libero_force, original, k1_open_top_put_bowl


domain_shift=""
hdf5_path=""
ffn_expand_factor=""
for triple in "${EXPERT_TRIPLETS[@]}"; do
  read -r shift path expand <<< "${triple}"
  if [[ "${shift}" == "${TRAIN_PRESET}" ]]; then
    domain_shift="${shift}"
    hdf5_path="${path}"
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

hdf5_paths=("${hdf5_path}")
hdf5_paths_override="[$(IFS=,; echo "${hdf5_paths[*]}")]"

to_path() {
  local s="$1"
  printf '%s\n' "${s/-/\/}"   # 只替换第一个 '-'
}
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
  task.dataset.dataset_subname="${domain_shift}" \
  task.dataset.hdf5_fns=${hdf5_paths_override} \
  pretrained_ckpt="${PRETRAINED_CKPT}" \
  task.dataset.transform_color_jitter="False"
