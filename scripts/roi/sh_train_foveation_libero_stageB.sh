#!/usr/bin/env bash
# Requires new per-camera Stage A weights with the same ordered RGB keys.
# Paired samples one camera per window/pair; ELU budgets and lambdas are per camera.

# Stage A EMA -> Spatial reachable paired routing with a frozen downstream policy.
# Match the architecture and data/diffusion settings of the Stage A checkpoint.
# Two-by-two split: train on B_train, validate on A_val.
# DATA_SEED_FROM_CKPT=true overrides DATASET_SPLIT_SEED with the saved split seed.
# SEED can vary across runs; disabling inheritance requires a matching data seed.
# Default: 300 router-only epochs; null disables downstream adaptation.
# Cosine budget target: 0.75 -> 0.1 over 200 epochs, then held constant.
# At 128x128, patch16 and 3 levels: 49 -> 10 soft target patches/view, not hard counts.
# ELU upper-bound budget uses lambda above target; base cost 0 disables savings reward.
# Evaluation logs actual online patch counts; task.env_runner.n_test_vis controls videos.
# Temporal adds an episode-local window and matches full-subtree patch deltas.
# Batch size is unchanged: batch=64 processes 128 windows when Temporal is enabled.
# Experiment variables follow PushT Stage B; other settings use trailing Hydra arguments.
tmp="
CUDA_VISIBLE_DEVICES=2,3,4,6 \
 HORIZON=10 SEED=43 DATA_SEED_FROM_CKPT=true \
 VIS_ENC_LAYERS=4 POOLING_TYPE=cls_token \
 PAIRED_NODE_SAMPLING=reachable PAIRED_LOSS_WEIGHT=1.0 \
 TEMPORAL_ADAPTIVE=true TEMPORAL_LOSS_WEIGHT=1.0 \
 BUDGET_LOSS_WEIGHT=0.1 TARGET_NUM_PATCHES=0.1 \
 BUDGET_LOSS_TYPE=upper_bound_lagrangian_elu BUDGET_BASE_COST_WEIGHT=0.0 \
 BUDGET_LAMBDA_INIT=0.0 BUDGET_LAMBDA_LR=0.01 \
 NUM_EPOCHS=901 TARGET_NUM_PATCHES_DECAY_EPOCHS=600 \
 DOWNSTREAM_FINETUNE_START_EPOCH=600 \
 bash scripts/roi/sh_train_foveation_libero_stageB.sh \
 libero_root=/home/geyuan/code/LIBERO-FT
"

# Keep the checkpoint, seed and budget above fixed across these three variants:
# Spatial only:       TEMPORAL_ADAPTIVE=false
# Matching ablation:  TEMPORAL_ADAPTIVE=true TEMPORAL_LOSS_WEIGHT=0.0
# Spatial + Temporal: TEMPORAL_ADAPTIVE=true TEMPORAL_LOSS_WEIGHT=1.0
# Weight 0 retains both windows and active matching; failed matches keep Spatial/budget.
# Override dataloader.batch_size through trailing Hydra arguments if needed.

# best/latest selects from a Stage A run directory or a checkpoint's run; unset loads the exact file.
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/geyuan/code/dp23rss_fork/outputs/2026-09-28/20-36-09/checkpoints/latest.ckpt}"

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
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the Stage A checkpoint}"
# Resolve the timestamp once; independent Hydra processes can start in different seconds.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$(date +%Y-%m-%d/%H-%M-%S)}"

# Topology diagnostics process windows per task serially; training.sample_resample controls resampling.

accelerate launch "${LAUNCH_ARGS[@]}" train.py \
  --config-dir=configs --config-name=liberoft_foveation_dp.yaml \
  "hydra.run.dir=${OUTPUT_DIR}" \
  "pretrained_ckpt='${PRETRAINED_CKPT}'" \
  pretrained_ckpt_mode="${PRETRAINED_CKPT_MODE:-null}" \
  training.seed="${SEED:-43}" \
  data_seed_from_ckpt="${DATA_SEED_FROM_CKPT:-true}" \
  task.dataset.split_mode=legacy \
  horizon="${HORIZON:-10}" \
  training.num_epochs="${NUM_EPOCHS:-300}" \
  training.router_only="true" \
  training.budget_lambda_init="${BUDGET_LAMBDA_INIT:-0.0}" \
  training.budget_lambda_lr="${BUDGET_LAMBDA_LR:-0.01}" \
  training.downstream_finetune_start_epoch="${DOWNSTREAM_FINETUNE_START_EPOCH:-null}" \
  training.target_num_patches_start="${TARGET_NUM_PATCHES_START:-0.75}" \
  training.target_num_patches_decay_epochs="${TARGET_NUM_PATCHES_DECAY_EPOCHS:-200}" \
  policy.router_loss_type="paired" \
  policy.paired_loss_weight="${PAIRED_LOSS_WEIGHT:-1.0}" \
  policy.paired_node_sampling="${PAIRED_NODE_SAMPLING:-reachable}" \
  policy.temporal_adaptive="${TEMPORAL_ADAPTIVE:-false}" \
  policy.temporal_loss_weight="${TEMPORAL_LOSS_WEIGHT:-1.0}" \
  policy.visual_encoder_layers="${VIS_ENC_LAYERS:-4}" \
  policy.pooling_type="${POOLING_TYPE:-cls_token}" \
  policy.patch_embedding_type="${PATCH_EMBEDDING_TYPE:-conv}" \
  policy.patch_level_adapter_channels="${PATCH_LEVEL_ADAPTER_CHANNELS:-16}" \
  policy.foveation_net.fix_split_mode="null" \
  policy.foveation_net.random_topology_prob=0.0 \
  policy.foveation_net.target_num_patches="${TARGET_NUM_PATCHES:-0.1}" \
  policy.foveation_net.budget_loss_type="${BUDGET_LOSS_TYPE:-upper_bound_lagrangian_elu}" \
  policy.foveation_net.budget_loss_weight="${BUDGET_LOSS_WEIGHT:-0.1}" \
  policy.foveation_net.budget_base_cost_weight="${BUDGET_BASE_COST_WEIGHT:-0.0}" \
  task.env_runner.log_patch_counts="true" \
  task.env_runner.save_foveation_video="${SAVE_FOVEATION_VIDEO:-true}" \
  training.topology_samples_per_task="${TOPOLOGY_SAMPLES_PER_TASK:-64}" \
  "$@"
