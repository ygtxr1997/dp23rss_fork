## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:$PYTHONPATH
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/sh_train_visual_encoder.sh
"

### Merge MoE ###
CONFIG_DIR="configs/"
CONFIG_NAME="pusht256_visual_encoder.yaml"

zarr_paths=(
  "data/pusht/pusht_256.zarr"
#  "data/pusht/pusht_256_texture.zarr"
#  "data/pusht/pusht_256_goal.zarr"
#  "data/pusht/pusht_256_light.zarr"
#  "data/pusht/pusht_256_block.zarr"
)

zarr_paths_override="[$(IFS=,; echo "${zarr_paths[*]}")]"

DEVICE="cuda"

export HYDRA_FULL_ERROR=1

set -e
set -x

wandb online

### DDP training with accelerate
NUM_GPUS=4
MAIN_PORT=12347

accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  --main_process_port=${MAIN_PORT} \
  train.py --config-dir=${CONFIG_DIR} --config-name=${CONFIG_NAME} training.seed=42 \
  training.device=${DEVICE} \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
  task.dataset.zarr_paths=${zarr_paths_override}
