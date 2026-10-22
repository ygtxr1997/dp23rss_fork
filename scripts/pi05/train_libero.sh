#!/usr/bin/env bash
# Dense pi0.5 full-parameter SFT. Copy the command below to start a run.
# tmp only records the example; it is not evaluated by this script.
tmp="
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MICRO_BATCH=16 GLOBAL_BATCH=128 MAX_STEPS=30001 SEED=42 \
ROLLOUT_EVERY=1500 N_TEST=3 \
PI05_IMAGE_SIZE=56 \
bash scripts/pi05/train_libero.sh \
  runner.conda_env=libero logging.mode=online
"

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the training GPUs}"
IFS=',' read -r -a PI05_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#PI05_GPUS[@]}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
# The policy owns selective bf16/FP32; Accelerate only launches processes.
LAUNCH_ARGS=(--num_processes="${NUM_GPUS}" --num_machines=1 --machine_rank=0
             --mixed_precision=no --dynamo_backend=no --gpu_ids=all)
if (( NUM_GPUS > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT}")
fi

# Resolve the timestamp once so every rank uses the same output directory.
export PI05_OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_sft/$(date +%Y-%m-%d/%H-%M-%S)}"

exec accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name=pi05_libero_sft \
  training.seed="${SEED:-42}" \
  training.micro_batch="${MICRO_BATCH:-1}" \
  training.global_batch="${GLOBAL_BATCH:-256}" \
  training.max_steps="${MAX_STEPS:-30000}" \
  model.image_size="${PI05_IMAGE_SIZE:-224}" \
  rollout.every="${ROLLOUT_EVERY:-1000}" \
  runner.n_test="${N_TEST:-3}" \
  "$@"
