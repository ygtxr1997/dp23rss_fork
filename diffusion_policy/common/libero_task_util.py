"""Stable LIBERO-10 identities, independent of experiment subsets and task order."""

from pathlib import Path


# Official libero_suite_task_map.py order, before benchmark task-order permutations.
LIBERO_TASK_NAMES = (
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
)
LIBERO_TASK_IDS = {name: index for index, name in enumerate(LIBERO_TASK_NAMES)}


def libero_task_id(source: str, fallback=None) -> int:
    """Resolve an official task name or replay filename to its fixed ID."""
    name = Path(source).stem.removesuffix("_demo_wrench").removesuffix("_demo")
    if name in LIBERO_TASK_IDS:
        return LIBERO_TASK_IDS[name]
    if fallback is not None:
        return fallback
    raise ValueError(f"Unknown LIBERO-10 task for one-hot conditioning: {source}")
