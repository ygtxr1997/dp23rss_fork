#!/usr/bin/env bash
# Each camera independently samples mode/ratio/tree; window frames share the tree.
# Train the remaining six LIBERO-10 tasks sequentially, one task per Stage A run.
# Example: bash scripts/roi/sh_train_foveation_libero_stageA_remaining6.sh
# Environment variables override defaults; trailing Hydra arguments reach every run.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,2,3,1}"
# Keep 15% mixed trees, with continuous ratios covering low-budget transitions.
#export FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-[\"100%level:0\"]}"
export FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-[\"40%level:2\",\"60%random:0.05-0.99\"]}"
#export FIX_SPLIT_MODE="${FIX_SPLIT_MODE:-[\"80%level:2\",\"20%random:0.05-0.99\"]}"
export HORIZON="${HORIZON:-10}"
export SEED="${SEED:-42}"
export VIS_ENC_LAYERS="${VIS_ENC_LAYERS:-4}"

HDF5_FNS=(
#  KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo_wrench.hdf5
#  KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo_wrench.hdf5
#  KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo_wrench.hdf5
#  STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo_wrench.hdf5
#  LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo_wrench.hdf5
#  LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo_wrench.hdf5
#  LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo_wrench.hdf5
#  LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo_wrench.hdf5
#  LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo_wrench.hdf5
  KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo_wrench.hdf5
)

for HDF5_FN in "${HDF5_FNS[@]}"; do
  TASK_NAME="${HDF5_FN%_demo_wrench.hdf5}"
  echo "[Stage A] Training ${TASK_NAME}"
  bash scripts/roi/sh_train_foveation_libero_stageA.sh \
    task.env_runner.conda_env=libero \
    libero_root=/home/geyuan/code/LIBERO-FT \
    "$@" \
    "task.dataset.hdf5_fns=[${HDF5_FN}]"
done
