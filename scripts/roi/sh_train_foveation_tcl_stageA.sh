#!/usr/bin/env bash
# Each camera independently samples mode/ratio/tree; window frames share the tree.
# Stage A: random/legal trees train the visual encoder and diffusion policy.
# CUDA_VISIBLE_DEVICES=0,1 DATA_ROOT=/path/to/recordings H5_PATH=/path/to/data.h5 \
#   bash scripts/roi/sh_train_foveation_tcl_stageA.sh
# DATASET_SPLIT_SEED is independent of SEED and all training/augmentation RNG.
# FIX_SPLIT_MODE accepts a scalar or Hydra list, as in the LIBERO Stage A script.
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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$(date +%Y-%m-%d/%H-%M-%S)}"
DATA_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then DATA_ARGS+=("task.dataset.data_root='${DATA_ROOT}'"); fi
if [[ -n "${H5_PATH:-}" ]]; then DATA_ARGS+=("task.dataset.h5_path='${H5_PATH}'"); fi
FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-random:0.0,0.25,0.5,0.75,1.0}"
SPLIT_MODE_OVERRIDE="policy.foveation_net.fix_split_mode=${FIX_SPLIT_MODE}"
if [[ "$FIX_SPLIT_MODE" != \[* && "$FIX_SPLIT_MODE" != \'* && "$FIX_SPLIT_MODE" != \"* && "$FIX_SPLIT_MODE" != null ]]; then
  SPLIT_MODE_OVERRIDE="policy.foveation_net.fix_split_mode='${FIX_SPLIT_MODE}'"
fi

accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name=tcl_force_foveation_dp.yaml \
  "hydra.run.dir=${OUTPUT_DIR}" \
  training.seed="${SEED:-42}" \
  task.dataset.seed="${DATASET_SPLIT_SEED:-42}" \
  data_seed_from_ckpt="${DATA_SEED_FROM_CKPT:-false}" \
  training.num_epochs="${NUM_EPOCHS:-1500}" \
  training.router_only=false \
  policy.router_loss_type=st \
  "${SPLIT_MODE_OVERRIDE}" \
  policy.foveation_net.random_topology_prob="${RANDOM_TOPOLOGY_PROB:-0.0}" \
  policy.foveation_net.budget_loss_weight=0.0 \
  policy.visual_encoder_layers="${VIS_ENC_LAYERS:-4}" \
  policy.pooling_type="${POOLING_TYPE:-cls_token}" \
  policy.patch_embedding_type="${PATCH_EMBEDDING_TYPE:-conv}" \
  policy.patch_level_adapter_channels="${PATCH_LEVEL_ADAPTER_CHANNELS:-16}" \
  horizon="${HORIZON:-32}" \
  task.dataset.p_cond_drop="${P_COND_DROP:-0.3}" \
  task.env_runner.save_foveation_video="${SAVE_FOVEATION_VIDEO:-true}" \
  task.env_runner.max_episodes="${VIDEO_MAX_EPISODES:-null}" \
  training.topology_samples_per_task="${TOPOLOGY_SAMPLES_PER_TASK:-64}" \
  "${DATA_ARGS[@]}" "$@"
