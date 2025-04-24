CONFIG_NAME="image_pusht_diffusion_policy_transformer.yaml"
DEVICES="4,5,6"
ZARR_PATH="data/pusht/pusht_cchi_v7_replay.zarr"
#ZARR_PATH="data/pusht/pusht_orange.zarr"
#ZARR_PATH="data/pusht/pusht_orange_random.zarr"

export CUDA_VISIBLE_DEVICES=${DEVICES}
ray start --head --num-gpus=3  --port 1234

export HYDRA_FULL_ERROR=1

set -e
set -x

python ray_train_multirun.py --config-dir=. --config-name=${CONFIG_NAME} --seeds=42,43,44  \
  --monitor_key=test/mean_score  \
  task.dataset.zarr_path=${ZARR_PATH}  \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
