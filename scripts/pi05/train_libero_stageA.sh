#!/usr/bin/env bash
# Pi0.5 LIBERO Stage A: fixed 224px canvas with mixed-resolution SigLIP patches.
# Weighted split example (keep the value as one shell argument):
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/pi05/train_libero_stageA.sh
# Append Hydra overrides to change dataset, logging, rollout, or runner settings.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG_DIR="configs"
CONFIG_NAME="pi05_libero_sft.yaml"
SEED="${SEED:-42}"
MICRO_BATCH="${MICRO_BATCH:-16}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
MAX_STEPS="${MAX_STEPS:-4001}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-400}"
FIRST_ROLLOUT_STEP="${FIRST_ROLLOUT_STEP:-50}"
N_TEST="${N_TEST:-4}"

## Op.1 All-Fine; Opt.2 All-Coarse; Opt.3 StageA Mixed-Resolution;
#FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-[\"100%level:2\"]}"
#FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-[\"100%level:0\"]}"
FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-[\"40%level:2\",\"60%random:0.05-0.99\"]}"


if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] CUDA_VISIBLE_DEVICES is empty."
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPUS[@]}"
LAST_GPU="${GPUS[$((NUM_GPUS - 1))]}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-1234${LAST_GPU: -1}}"

LAUNCH_ARGS=(
  --num_processes="${NUM_GPUS}"
  --num_machines=1
  --machine_rank=0
  --mixed_precision=no
  --dynamo_backend=no
)
if (( NUM_GPUS > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT}")
fi

# Stage A always uses a 224px canvas; PI05_IMAGE_SIZE is intentionally ignored.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_stageA/$(date +%Y-%m-%d/%H-%M-%S)}"

# Preserve Hydra lists and already-quoted values; quote bare comma-containing values.
SPLIT_MODE_OVERRIDE="model.foveation_split_mode=${FIX_SPLIT_MODE}"
if [[ "${FIX_SPLIT_MODE}" != \[* && \
      "${FIX_SPLIT_MODE}" != \'* && \
      "${FIX_SPLIT_MODE}" != \"* && \
      "${FIX_SPLIT_MODE}" != null ]]; then
  SPLIT_MODE_OVERRIDE="model.foveation_split_mode='${FIX_SPLIT_MODE}'"
fi

DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"

accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir="${CONFIG_DIR}" \
  --config-name="${CONFIG_NAME}" \
  "hydra.run.dir=${OUTPUT_DIR}" \
  training.seed="${SEED}" \
  training.stage_b=false \
  training.micro_batch="${MICRO_BATCH}" \
  training.global_batch="${GLOBAL_BATCH}" \
  training.max_steps="${MAX_STEPS}" \
  model.image_size=224 \
  model.visual_input_mode=mixed_foveation \
  model.foveation_router_type=none \
  model.foveation_patch_level_adapter=true \
  model.foveation_apt_position_encoding=true \
  model.foveation_level_embedding=true \
  model.foveation_random_topology_prob=0.0 \
  "${SPLIT_MODE_OVERRIDE}" \
  rollout.every="${ROLLOUT_EVERY}" \
  rollout.first_step="${FIRST_ROLLOUT_STEP}" \
  runner.n_test="${N_TEST}" \
  runner.save_foveation_video=true \
  dataloader.num_workers="${DATALOADER_WORKERS}" \
  "$@"
