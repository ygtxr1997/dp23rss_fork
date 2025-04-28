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


CONFIG_NAME="tcl_dp_transformer.yaml"
DEVICE="cuda:0"

export HYDRA_FULL_ERROR=1

set -e
set -x

wandb online

python train.py --config-dir=. --config-name=${CONFIG_NAME} training.seed=42  \
  training.device=${DEVICE}  \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'s
