## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:\$PYTHONPATH
CUDA_VISIBLE_DEVICES=3,4,6,7 IMAGE_H=120 SEED=42 ROI_PRESET=medium bash scripts/roi/sh_train_roi_tcl.sh
"

CONFIG_DIR="configs/"
CONFIG_NAME="tcl_roi_dp.yaml"
DEVICE="cuda"

IMAGE_H="${IMAGE_H:-120}"
IMAGE_W=$((IMAGE_H * 4 / 3))
IMAGE_SHAPE="[3, ${IMAGE_H}, ${IMAGE_W}]"
ROI_SIZE="[${IMAGE_H}, ${IMAGE_W}]"
SEED="${SEED:-42}"
ROI_PRESET="${ROI_PRESET:-small}"

export HYDRA_FULL_ERROR=1

set -e
set -x

wandb online

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

echo "[INFO] CONFIG=${CONFIG_DIR}${CONFIG_NAME}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}, MAIN_PORT=${MAIN_PORT}"
echo "[INFO] IMAGE_SHAPE=${IMAGE_SHAPE}, ROI_SIZE=${ROI_SIZE}"

accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  --main_process_port=${MAIN_PORT} \
  train.py --config-dir=${CONFIG_DIR} --config-name=${CONFIG_NAME} training.seed=${SEED} \
  training.device=${DEVICE} \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
  image_shape="${IMAGE_SHAPE}" \
  shape_meta.obs.image.shape="${IMAGE_SHAPE}" \
  shape_meta.obs.image_roi.shape="${IMAGE_SHAPE}" \
  policy.roi_net.roi_size="${ROI_SIZE}"  \
  policy.roi_net.preset="${ROI_PRESET}"  \
  policy.roi_net.locator_type="resnet"  \
  policy.roi_net.predict_type="logit_meanstd"  \
  policy.roi_center_diversity_loss_weight=1.0  \
  policy.roi_area_loss_mode="identity"  \
  policy.roi_area_loss_weight=0.01  \
  policy.roi_scale_diversity_loss_weight=1.0  \
  policy.roi_scale_diversity_mode="pairwise_hinge"  \
  policy.roi_scale_diversity_value=0.125
