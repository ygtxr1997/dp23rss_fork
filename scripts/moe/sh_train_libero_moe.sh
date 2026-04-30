## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:$PYTHONPATH
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/moe/sh_train_libero_moe.sh
"

### Merge MoE ###
CONFIG_DIR="configs/"
CONFIG_NAME="libero_force_moe.yaml"

dataset_subnames=(
  "libero_force_color_slice"  # shift: color
  "libero_force_view_slice"  # shift: view
  "libero_force_texture_slice"
)
hdf5_fns=(
  "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench.hdf5"
  "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench.hdf5"
  "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench.hdf5"
)
ffn_expand_factor=(6 10 2)
teacher_ckpts=(
  "data/outputs/2026.04.30/11.31.13_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"
  "data/outputs/2026.04.30/13.28.53_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"
  "data/outputs/2026.04.30/10.49.10_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"
)

dataset_subnames_override="[$(IFS=,; echo "${dataset_subnames[*]}")]"
hdf5_fns_override="[$(IFS=,; echo "${hdf5_fns[*]}")]"
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
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
  task.dataset.dataset_subname="${dataset_subnames_override}" \
  task.dataset.hdf5_fns=${hdf5_fns_override} \
  policy.ffn_expand_factor=${ffn_expand_factor_override} \
  policy.teacher_ckpts=${teacher_ckpts_override} \
  policy.en_freeze_obs_encoder=${EN_FREEZE_OBS_ENCODER:-false}
