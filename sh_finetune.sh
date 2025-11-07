#CONFIG_NAME="tcl_dp_transformer.yaml"
CONFIG_NAME="tcl_dp_finetune.yaml"
#CONFIG_NAME="tcl_hdfree_shovel.yaml"
#CONFIG_NAME="tcl_hdfree_dp.yaml"
DEVICE="cuda"

export HYDRA_FULL_ERROR=1

set -e
set -x

wandb online

python train.py --config-dir=. --config-name=${CONFIG_NAME} training.seed=42  \
  training.device=${DEVICE}  \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'s