## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:$PYTHONPATH
CUDA_VISIBLE_DEVICES=3,4,5,6 bash scripts/moe/sh_train_tcl_moe.sh
"

### Merge MoE ###
CONFIG_DIR="configs/"
CONFIG_NAME="tcl_force_moe.yaml"

task_names=(
#  "1201_screw_bulb_turn_off_table"
#  "1201_screw_bulb_turn_off_object"
#  "1201_screw_bulb_turn_off_light"
  "0209_tower_boby_hard"
  "0417_plastic_egg"
  "0417_put_mouse_hard"
)
ffn_expand_factor=(6 4 2)
teacher_ckpts=(
#  "data/outputs/2026.04.28/21.28.06_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"
#  "data/outputs/2026.04.30/22.33.39_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"
#  "data/outputs/2026.05.01/17.54.53_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"
  "data/outputs/2026.09.09/11.12.55_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"
  "data/outputs/2026.09.09/11.13.27_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"
  "data/outputs/2026.09.12/01.41.29_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"
)

DATASET_ROOT="${DATASET_ROOT:-/home/geyuan/datasets/TCL}"
master_data_root="${DATASET_ROOT}/${task_names[0]}/"
master_h5_path="${DATASET_ROOT}/hdf5/${task_names[0]}_240p.h5"

slave_data_roots=()
slave_h5_paths=()
for task_name in "${task_names[@]:1}"; do
  slave_data_roots+=("${DATASET_ROOT}/${task_name}/")
  slave_h5_paths+=("${DATASET_ROOT}/hdf5/${task_name}_240p.h5")
done

slave_data_roots_override="[$(IFS=,; echo "${slave_data_roots[*]}")]"
slave_h5_paths_override="[$(IFS=,; echo "${slave_h5_paths[*]}")]"
ffn_expand_factor_override="[$(IFS=,; echo "${ffn_expand_factor[*]}")]"
teacher_ckpts_override="[$(IFS=,; echo "${teacher_ckpts[*]}")]"

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
  logging.project="moemerge" \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
  task.dataset.data_root="${master_data_root}" \
  task.dataset.h5_path="${master_h5_path}" \
  task.dataset.slave_data_roots="${slave_data_roots_override}" \
  task.dataset.slave_h5_paths="${slave_h5_paths_override}" \
  policy.ffn_expand_factor=${ffn_expand_factor_override} \
  policy.teacher_ckpts=${teacher_ckpts_override} \
  policy.en_freeze_obs_encoder=${EN_FREEZE_OBS_ENCODER:-false}
