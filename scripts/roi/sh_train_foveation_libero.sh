#!/usr/bin/env bash

set -euo pipefail

CONFIG_DIR="configs"
CONFIG_NAME="liberoft_foveation_dp.yaml"
SEED="${SEED:-42}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] CUDA_VISIBLE_DEVICES is empty."
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPUS[@]}"
LAST_GPU="${GPUS[$((NUM_GPUS - 1))]}"
PORT="1234${LAST_GPU: -1}"

accelerate launch \
  --multi_gpu \
  --num_processes="${NUM_GPUS}" \
  --main_process_port="${PORT}" \
  train.py \
  --config-dir="${CONFIG_DIR}" \
  --config-name="${CONFIG_NAME}" \
  training.seed="${SEED}"
