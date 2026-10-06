#!/usr/bin/env bash
# Weighted Stage A example (keep the other required launch variables):
# FIX_SPLIT_MODE='["10%level:0","25%level:1","50%level:2","15%random:0.25,0.5,0.75"]' \
# RANDOM_TOPOLOGY_PROB=0.0 bash scripts/roi/sh_train_foveation_libero_stageA.sh
# Two-by-two split: train on A_train, validate on B_val.
# DATASET_SPLIT_SEED defaults to SEED; Stage B inherits it from the checkpoint.
# DATA_SEED_FROM_CKPT=true restores the saved data seed when loading a checkpoint.
# evaluate each task for 50 episodes and report per-task and overall success.
# Override dataset/evaluation settings with trailing Hydra arguments.
# Example: CUDA_VISIBLE_DEVICES=0,1 bash scripts/roi/sh_train_foveation_libero_stageA.sh \
#   task.env_runner.conda_env=libero libero_root=/home/geyuan/code/LIBERO-FT
# Low-frequency fine baseline (128 -> 32 -> 128; 64 fine patches per frame):
# CUDA_VISIBLE_DEVICES=0,1 FIX_SPLIT_MODE='random:1.0' \
#   bash scripts/roi/sh_train_foveation_libero_stageA.sh \
#   policy.foveation_net.debug_lowpass_size=32
# Keep all other environment variables and Hydra overrides identical to the fine baseline.

tmp="
CUDA_VISIBLE_DEVICES=3,4,5,6 \
FIX_SPLIT_MODE='["100%random:1.0"]' \
HORIZON=10 SEED=43 \
VIS_ENC_LAYERS=4 \
bash scripts/roi/sh_train_foveation_libero_stageA.sh \
  task.env_runner.conda_env=libero libero_root=/home/geyuan/code/LIBERO-FT
"

#PRETRAINED_CKPT="/home/geyuan/code/dp23rss_fork/outputs/2026-09-24/20-01-11/checkpoints/latest.ckpt"

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the training GPUs}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPUS[@]}"
LAST_GPU="${GPUS[$((NUM_GPUS - 1))]}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-1234${LAST_GPU: -1}}"
LAUNCH_ARGS=(--num_processes="${NUM_GPUS}" --num_machines=1 --mixed_precision=no)
if (( NUM_GPUS > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT}")
fi

FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-random:0.0,0.25,0.5,0.75,1.0}"
HORIZON="${HORIZON:-32}"
# Resolve the timestamp once; independent Hydra processes can start in different seconds.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$(date +%Y-%m-%d/%H-%M-%S)}"

# Topology diagnostics use this many fixed windows per task, processed serially.

# Preserve Hydra lists and already-quoted legacy values; quote bare scalar commas.
SPLIT_MODE_OVERRIDE="policy.foveation_net.fix_split_mode=${FIX_SPLIT_MODE}"
if [[ "$FIX_SPLIT_MODE" != \[* && "$FIX_SPLIT_MODE" != \'* && "$FIX_SPLIT_MODE" != \"* && "$FIX_SPLIT_MODE" != null ]]; then
  SPLIT_MODE_OVERRIDE="policy.foveation_net.fix_split_mode='${FIX_SPLIT_MODE}'"
fi

accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name=liberoft_foveation_dp.yaml \
  "hydra.run.dir=${OUTPUT_DIR}" \
  training.seed="${SEED:-43}" \
  task.dataset.seed="${DATASET_SPLIT_SEED:-${SEED:-43}}" \
  data_seed_from_ckpt="${DATA_SEED_FROM_CKPT:-false}" \
  task.dataset.split_mode=legacy \
  "${SPLIT_MODE_OVERRIDE}" \
  policy.foveation_net.random_topology_prob="${RANDOM_TOPOLOGY_PROB:-0.0}" \
  policy.foveation_net.budget_loss_weight=0.0 \
  policy.visual_encoder_layers="${VIS_ENC_LAYERS:-4}" \
  policy.pooling_type="${POOLING_TYPE:-cls_token}" \
  policy.patch_embedding_type="${PATCH_EMBEDDING_TYPE:-conv}" \
  policy.patch_level_adapter_channels="${PATCH_LEVEL_ADAPTER_CHANNELS:-16}" \
  horizon="${HORIZON}" \
  "$@"
