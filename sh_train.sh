#CONFIG_NAME="image_pusht_diffusion_policy_transformer.yaml"
#DEVICE="cuda:2"
#ZARR_PATH="data/pusht/pusht_cchi_v7_replay.zarr"
##ZARR_PATH="data/pusht/pusht_orange.zarr"
##ZARR_PATH="data/pusht/pusht_orange_random.zarr"
#
#export HYDRA_FULL_ERROR=1
#
#set -e
#set -x
#
#wandb online
#
#python train.py --config-dir=. --config-name=${CONFIG_NAME} training.seed=42  \
#  training.device=${DEVICE}  \
#  task.dataset.zarr_path=${ZARR_PATH}  \
#  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,4,5 bash sh_train.sh
"

CONFIG_DIR="./"
#CONFIG_NAME="tcl_dp_transformer.yaml"
#CONFIG_NAME="tcl_hdfree_shovel.yaml"
#CONFIG_NAME="tcl_hdfree_dp.yaml"
CONFIG_NAME="tcl_dp_force.yaml"
#CONFIG_NAME="libero_force_dp.yaml"

### Reverse Collect ###
#CONFIG_NAME="reverse_dp_force.yaml"

### Merge MoE ###
#CONFIG_DIR="configs/"
#CONFIG_NAME="pusht256_dp.yaml"

DEVICE="cuda"

export HYDRA_FULL_ERROR=1

set -e
set -x

wandb online

### DDP training with accelerate
NUM_GPUS=4
MAIN_PORT=29504

# python train.py --config-dir=${CONFIG_DIR} --config-name=${CONFIG_NAME} training.seed=42  \
#   training.device=${DEVICE}  \
#   hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  --main_process_port=${MAIN_PORT} \
  train.py --config-dir=${CONFIG_DIR} --config-name=${CONFIG_NAME} training.seed=42 \
  training.device=${DEVICE} \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
