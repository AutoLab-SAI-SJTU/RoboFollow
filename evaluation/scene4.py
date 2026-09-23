from __future__ import annotations
import copy
import hashlib
import json
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from . import common as base
import robofollow.tasks.scene4 as scene4_env_mod
from robofollow.tasks.scene4 import SCENES as SCENE4_BASE_SCENES, TASKS as SCENE4_BASE_TASKS
from robofollow.tasks.scene4 import SLOT_XY, TaskDef as Scene4TaskDef

LEVEL_ORDER = ("L0", "L1", "L2", "L3")

SCENE_UNDERSCORE_TO_DOT = {
    "scene4_1": "scene4.1",
    "scene4_2": "scene4.2",
    "scene4_3": "scene4.3",
}

@dataclass
class Scene4EvalTask:
    name: str
    scene: str
    source: str
    arm: str
    action: str
    instruction: str
    level_tag: str
    base_task_type: str
    instruction_variant: str
    scenario_tag: str
    target: str | None = None
    place_slot: int | None = None
    second_source: str | None = None
    second_arm: str | None = None

@dataclass
class PhaseResult:
    phase_name: str
    intent_score: float
    intent_reason: str
    exec_score: float
    exec_reason: str
    max_score: float

@dataclass
class Scene4Trajectory:
    ee: dict[str, list[np.ndarray]]
    grip: dict[str, list[float]]
    obj_pos: dict[str, list[np.ndarray]]
    bowls: set[str]
    n_steps: int
    table_height: float
    half_heights: dict[str, float]

TASKS: dict[str, Scene4EvalTask] = {}

LEVELS: dict[str, list[str]] = {}

SCENE_VARIANTS: dict[str, object] = {}

SCENE_VARIANT_NOTES: dict[str, str] = {}

BASE_TASK_TYPES: list[str] = []

_EVAL_INSTRUCTION_TYPES: dict[str, str] = {}

def _stable_hash32(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)

def _load_collect_scene4_templates():
    return json.loads((Path(__file__).resolve().parents[1] / "tasks" / "scene4_instructions.json").read_text())

INSTRUCTION_TEMPLATES = _load_collect_scene4_templates()

def _register_task(task: Scene4EvalTask) -> None:
    TASKS[task.name] = task
    LEVELS.setdefault(task.level_tag, []).append(task.name)

def _add_task(
    level_tag: str,
    name_suffix: str,
    scene: str,
    action: str,
    source: str,
    arm: str,
    instruction: str,
    base_task_type: str,
    scenario_tag: str,
    instruction_variant: str = "manual_v1",
    target: str | None = None,
    place_slot: int | None = None,
    second_source: str | None = None,
    second_arm: str | None = None,
) -> None:
    _register_task(
        Scene4EvalTask(
            name=f"{level_tag}_{name_suffix}",
            scene=scene,
            source=source,
            arm=arm,
            action=action,
            instruction=instruction,
            level_tag=level_tag,
            base_task_type=base_task_type,
            instruction_variant=instruction_variant,
            scenario_tag=scenario_tag,
            target=target,
            place_slot=place_slot,
            second_source=second_source,
            second_arm=second_arm,
        )
    )

def _find_object_def(cfg, name: str):
    for obj in cfg.objects:
        if obj.name == name:
            return obj
    return None

def _register_scene_variants() -> None:
    SCENE_VARIANTS.clear()
    SCENE_VARIANT_NOTES.clear()

    for base_name in ("scene4_1", "scene4_2", "scene4_3"):
        cfg = copy.deepcopy(SCENE4_BASE_SCENES[base_name])
        SCENE_VARIANTS[base_name] = cfg
        SCENE_VARIANT_NOTES[base_name] = "training layout"

        dot_name = SCENE_UNDERSCORE_TO_DOT[base_name]
        cfg_dot = copy.deepcopy(cfg)
        cfg_dot.name = dot_name
        SCENE_VARIANTS[dot_name] = cfg_dot
        SCENE_VARIANT_NOTES[dot_name] = f"alias of {base_name}"

    # Scene 4.4: remove the cube from the bowl.
    cfg44 = copy.deepcopy(SCENE4_BASE_SCENES["scene4_1"])
    cfg44.name = "scene4.4"
    cfg44.objects = [obj for obj in cfg44.objects if obj.name != "red_cube_in_bowl"]
    SCENE_VARIANTS["scene4.4"] = cfg44
    SCENE_VARIANT_NOTES["scene4.4"] = "scene4.1 without red_cube_in_bowl"

    # Scene 4.5: swap the red and blue spheres.
    cfg45 = copy.deepcopy(SCENE4_BASE_SCENES["scene4_1"])
    cfg45.name = "scene4.5"
    blue = _find_object_def(cfg45, "blue_sphere")
    red = _find_object_def(cfg45, "red_sphere")
    if blue is not None and red is not None:
        blue.slot, red.slot = red.slot, blue.slot
    SCENE_VARIANTS["scene4.5"] = cfg45
    SCENE_VARIANT_NOTES["scene4.5"] = "scene4.1 with red/blue sphere swapped"

    # Scene 4.6: move the green cylinder from the right cube to the yellow cylinder.
    cfg46 = copy.deepcopy(SCENE4_BASE_SCENES["scene4_1"])
    cfg46.name = "scene4.6"
    gtop = _find_object_def(cfg46, "green_cylinder_top")
    if gtop is not None:
        gtop.on_top_of = "yellow_cylinder"
        gtop.slot = 2
    SCENE_VARIANTS["scene4.6"] = cfg46
    SCENE_VARIANT_NOTES["scene4.6"] = "scene4.1 with green_cylinder_top moved onto yellow_cylinder"

    # Scene 4.7: swap the yellow cylinder with the right cube and its stacked cylinder.
    cfg47 = copy.deepcopy(SCENE4_BASE_SCENES["scene4_1"])
    cfg47.name = "scene4.7"
    yellow = _find_object_def(cfg47, "yellow_cylinder")
    blue_cube = _find_object_def(cfg47, "blue_cube_right")
    gtop = _find_object_def(cfg47, "green_cylinder_top")
    if yellow is not None and blue_cube is not None:
        yellow.slot, blue_cube.slot = blue_cube.slot, yellow.slot
    if gtop is not None and blue_cube is not None:
        gtop.on_top_of = "blue_cube_right"
        gtop.slot = blue_cube.slot
    SCENE_VARIANTS["scene4.7"] = cfg47
    SCENE_VARIANT_NOTES["scene4.7"] = "scene4.1 with yellow cylinder swapped with right cube stack"

    # Scene 4.8: move the red sphere to slot 6.
    cfg48 = copy.deepcopy(SCENE4_BASE_SCENES["scene4_1"])
    cfg48.name = "scene4.8"
    red = _find_object_def(cfg48, "red_sphere")
    if red is not None:
        red.slot = 6
    SCENE_VARIANTS["scene4.8"] = cfg48
    SCENE_VARIANT_NOTES["scene4.8"] = "scene4.1 with red_sphere moved to slot 6"

def _register_l0_tasks() -> None:
    for task_type, td in SCENE4_BASE_TASKS.items():
        scene = SCENE_UNDERSCORE_TO_DOT.get(td.scene, td.scene)
        templates = INSTRUCTION_TEMPLATES.get(task_type, [])
        instruction = templates[0] if templates else f"Execute task: {task_type}."
        _add_task(
            "L0",
            task_type,
            scene=scene,
            action=td.action,
            source=td.source,
            arm=td.arm,
            target=td.target,
            place_slot=td.place_slot,
            second_source=td.second_source,
            second_arm=td.second_arm,
            instruction=instruction,
            base_task_type=task_type,
            scenario_tag=scene,
            instruction_variant="template_0",
        )

def _register_l1_tasks() -> None:
    specs = [
        # Scene 4.4.
        dict(
            id=1,
            scenario="scene4.4",
            action="stack_on_target",
            source="green_cylinder_top",
            arm="right",
            target="yellow_cylinder",
            instruction="If there is an object in the bowl, use your left arm to pick up the blue sphere; otherwise use your right arm to place the cylinder from the right cube on the yellow cylinder.",
            base="l1_s44_if_bowl_nonempty_then_l_pick_blue_else_r_stack",
        ),
        dict(
            id=2,
            scenario="scene4.4",
            action="pickup",
            source="blue_sphere",
            arm="left",
            instruction="If there is an object in the bowl, use your right arm to place the cylinder from the right cube on the yellow cylinder; otherwise use your left arm to pick up the blue sphere.",
            base="l1_s44_if_bowl_nonempty_then_r_stack_else_l_pick_blue",
        ),
        dict(
            id=3,
            scenario="scene4.4",
            action="pickup",
            source="green_cylinder_top",
            arm="right",
            instruction="Use your right arm to pick up the object that is not touching the table.",
            base="l1_s44_r_pick_object_not_touching_table",
        ),
        # Scene 4.5.
        dict(
            id=4,
            scenario="scene4.5",
            action="pickup",
            source="blue_sphere",
            arm="left",
            instruction="If the blue sphere is to the right of the red sphere, use your left arm to pick up the blue sphere; otherwise use your right arm to place the cylinder from the right cube on the yellow cylinder.",
            base="l1_s45_if_blue_right_of_red_then_l_pick_blue_else_r_stack",
        ),
        dict(
            id=5,
            scenario="scene4.5",
            action="stack_on_target",
            source="green_cylinder_top",
            arm="right",
            target="yellow_cylinder",
            instruction="If the blue sphere is to the right of the red sphere, use your right arm to place the cylinder from the right cube on the yellow cylinder; otherwise use your left arm to pick up the blue sphere.",
            base="l1_s45_if_blue_right_of_red_then_r_stack_else_l_pick_blue",
        ),
        # Scene 4.6.
        dict(
            id=6,
            scenario="scene4.6",
            action="pickup",
            source="blue_sphere",
            arm="left",
            instruction="If there is an object on top of the right cube, use your right arm to place the cylinder from the right cube on the yellow cylinder; otherwise use your left arm to pick up the blue sphere.",
            base="l1_s46_if_object_on_right_cube_then_r_stack_else_l_pick_blue",
        ),
        dict(
            id=8,
            scenario="scene4.6",
            action="sequence_pickups",
            source="red_cube_in_bowl",
            arm="left",
            second_source="green_cylinder_top",
            second_arm="right",
            instruction="First use your left arm to pick up the cube inside the bowl, then use your right arm to pick up the green cylinder.",
            base="l1_s46_seq_l_bowl_then_r_green",
        ),
        dict(
            id=9,
            scenario="scene4.6",
            action="sequence_pickups",
            source="green_cylinder_top",
            arm="right",
            second_source="red_cube_in_bowl",
            second_arm="left",
            instruction="First use your right arm to pick up the green cylinder, then use your left arm to pick up the cube inside the bowl.",
            base="l1_s46_seq_r_green_then_l_bowl",
        ),
        dict(
            id=10,
            scenario="scene4.6",
            action="sequence_pickups",
            source="red_sphere",
            arm="left",
            second_source="green_cylinder_top",
            second_arm="right",
            instruction="First use your left arm to pick up the red sphere, then use your right arm to pick up the green cylinder.",
            base="l1_s46_seq_l_red_then_r_green",
        ),
        dict(
            id=11,
            scenario="scene4.6",
            action="sequence_pickups",
            source="green_cylinder_top",
            arm="right",
            second_source="red_sphere",
            second_arm="left",
            instruction="First use your right arm to pick up the green cylinder, then use your left arm to pick up the red sphere.",
            base="l1_s46_seq_r_green_then_l_red",
        ),
        # Scene 4.7.
        dict(
            id=12,
            scenario="scene4.7",
            action="stack_on_target",
            source="green_cylinder_top",
            arm="right",
            target="yellow_cylinder",
            instruction="If the yellow cylinder is to the right of the green cylinder, use your right arm to place the cylinder from the right cube on the yellow cylinder; otherwise use your left arm to pick up the blue sphere.",
            base="l1_s47_if_yellow_right_of_green_then_r_stack_else_l_pick_blue",
        ),
        dict(
            id=13,
            scenario="scene4.7",
            action="pickup",
            source="blue_sphere",
            arm="left",
            instruction="If the yellow cylinder is to the right of the green cylinder, use your left arm to pick up the blue sphere; otherwise use your right arm to place the cylinder from the right cube on the yellow cylinder.",
            base="l1_s47_if_yellow_right_of_green_then_l_pick_blue_else_r_stack",
        ),
        # Scene 4.8.
        dict(
            id=14,
            scenario="scene4.8",
            action="pickup",
            source="blue_sphere",
            arm="left",
            instruction="If there is an object in the bowl, use your left arm to pick up the blue sphere; otherwise use your right arm to pick up the red sphere.",
            base="l1_s48_if_bowl_nonempty_then_l_pick_blue_else_r_pick_red",
        ),
        dict(
            id=15,
            scenario="scene4.8",
            action="pickup",
            source="red_sphere",
            arm="right",
            instruction="If there is an object in the bowl, use your right arm to pick up the red sphere; otherwise use your left arm to pick up the blue sphere.",
            base="l1_s48_if_bowl_nonempty_then_r_pick_red_else_l_pick_blue",
        ),
    ]

    for s in specs:
        suffix = f"{s['scenario'].replace('.', '_')}_{s['id']:02d}_{s['base']}"
        _add_task(
            "L1",
            suffix,
            scene=s["scenario"],
            action=s["action"],
            source=s["source"],
            arm=s["arm"],
            target=s.get("target"),
            place_slot=s.get("place_slot"),
            second_source=s.get("second_source"),
            second_arm=s.get("second_arm"),
            instruction=s["instruction"],
            base_task_type=s["base"],
            scenario_tag=s["scenario"],
        )

def _register_l2_tasks() -> None:
    specs = [
        # Scene 4.1.
        dict(scenario="scene4.1", action="pickup", source="blue_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not red.", base="l2_s41_l_pick_ball_not_red"),
        dict(scenario="scene4.1", action="pickup", source="blue_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not the rightmost.", base="l2_s41_l_pick_ball_not_rightmost"),
        dict(scenario="scene4.1", action="pickup", source="red_cube_in_bowl", arm="left", instruction="Use your left arm to pick up the cube that is not touching the table.", base="l2_s41_l_pick_cube_not_touching_table"),
        dict(scenario="scene4.1", action="pickup", source="red_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not blue.", base="l2_s41_l_pick_ball_not_blue"),
        dict(scenario="scene4.1", action="pickup", source="yellow_cylinder", arm="left", instruction="Use your left arm to pick up the cylinder that is not green.", base="l2_s41_l_pick_cylinder_not_green"),
        dict(scenario="scene4.1", action="sequence_pickups", source="blue_sphere", arm="left", second_source="green_cylinder_top", second_arm="right", instruction="First use your left arm to pick up the blue sphere, then use your right arm to pick up the green cylinder.", base="l2_s41_seq_l_blue_then_r_green"),
        dict(scenario="scene4.1", action="sequence_pickups", source="green_cylinder_top", arm="right", second_source="blue_sphere", second_arm="left", instruction="First use your right arm to pick up the green cylinder, then use your left arm to pick up the blue sphere.", base="l2_s41_seq_r_green_then_l_blue"),
        dict(scenario="scene4.1", action="pickup", source="green_cylinder_top", arm="right", instruction="If the red sphere is to the right of the blue sphere, use your right arm to pick up the green cylinder; otherwise use your left arm to pick up the blue sphere.", base="l2_s41_if_red_right_of_blue_then_r_pick_green_else_l_pick_blue"),
        dict(scenario="scene4.1", action="pickup", source="blue_sphere", arm="left", instruction="If the yellow cylinder is to the left of the bowl, use your right arm to pick up the green cylinder; otherwise use your left arm to pick up the blue sphere.", base="l2_s41_if_yellow_left_of_bowl_then_r_pick_green_else_l_pick_blue"),
        dict(scenario="scene4.1", action="pickup", source="green_cylinder_top", arm="right", instruction="If the right cube is under the green cylinder, use your right arm to pick up the green cylinder; otherwise use your left arm to pick up the blue sphere.", base="l2_s41_if_right_cube_under_green_then_r_pick_green_else_l_pick_blue"),
        # Scene 4.2.
        dict(scenario="scene4.2", action="pickup", source="blue_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not red.", base="l2_s42_l_pick_ball_not_red"),
        dict(scenario="scene4.2", action="pickup", source="red_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not the rightmost.", base="l2_s42_l_pick_ball_not_rightmost"),
        dict(scenario="scene4.2", action="pickup", source="green_cylinder_top", arm="left", instruction="Use your left arm to pick up the object that is not touching the table.", base="l2_s42_l_pick_object_not_touching_table"),
        dict(scenario="scene4.2", action="pickup", source="red_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not blue.", base="l2_s42_l_pick_ball_not_blue"),
        dict(scenario="scene4.2", action="pickup", source="yellow_cylinder", arm="left", instruction="Use your left arm to pick up the cylinder that is not green.", base="l2_s42_l_pick_cylinder_not_green"),
        dict(scenario="scene4.2", action="sequence_pickups", source="blue_sphere", arm="left", second_source="green_cylinder_top", second_arm="right", instruction="First use your left arm to pick up the blue sphere, then use your right arm to pick up the green cylinder.", base="l2_s42_seq_l_blue_then_r_green"),
        dict(scenario="scene4.2", action="sequence_pickups", source="green_cylinder_top", arm="right", second_source="blue_sphere", second_arm="left", instruction="First use your right arm to pick up the green cylinder, then use your left arm to pick up the blue sphere.", base="l2_s42_seq_r_green_then_l_blue"),
        dict(scenario="scene4.2", action="pickup", source="blue_sphere", arm="left", instruction="If the red sphere is to the right of the blue sphere, use your right arm to pick up the green cylinder; otherwise use your left arm to pick up the blue sphere.", base="l2_s42_if_red_right_of_blue_then_r_pick_green_else_l_pick_blue"),
        dict(scenario="scene4.2", action="pickup", source="blue_sphere", arm="left", instruction="If the yellow cylinder is to the left of the bowl, use your right arm to pick up the green cylinder; otherwise use your left arm to pick up the blue sphere.", base="l2_s42_if_yellow_left_of_bowl_then_r_pick_green_else_l_pick_blue"),
        dict(scenario="scene4.2", action="pickup", source="green_cylinder_top", arm="right", instruction="If the right cube is under the green cylinder, use your right arm to pick up the green cylinder; otherwise use your left arm to pick up the blue sphere.", base="l2_s42_if_right_cube_under_green_then_r_pick_green_else_l_pick_blue"),
        # Scene 4.3.
        dict(scenario="scene4.3", action="pickup", source="blue_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not red.", base="l2_s43_l_pick_ball_not_red"),
        dict(scenario="scene4.3", action="pickup", source="blue_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not the rightmost.", base="l2_s43_l_pick_ball_not_rightmost"),
        dict(scenario="scene4.3", action="pickup", source="green_cylinder_right", arm="right", instruction="Use your right arm to pick up the cylinder that is not yellow.", base="l2_s43_r_pick_cylinder_not_yellow"),
        dict(scenario="scene4.3", action="pickup", source="blue_sphere", arm="left", instruction="If there is no cube on top of the yellow cylinder, use your right arm to pick up the object that is not touching the table; otherwise use your left arm to pick up the blue sphere.", base="l2_s43_if_no_cube_on_yellow_then_r_pick_not_touch_else_l_pick_blue"),
        dict(scenario="scene4.3", action="pickup", source="blue_sphere", arm="left", instruction="If the yellow cylinder is below the blue cube, use your left arm to pick up the blue sphere; otherwise use your right arm to pick up the object that is not touching the table.", base="l2_s43_if_yellow_below_blue_cube_then_l_pick_blue_else_r_pick_not_touch"),
    ]

    for idx, s in enumerate(specs, start=1):
        suffix = f"{s['scenario'].replace('.', '_')}_{idx:02d}_{s['base']}"
        _add_task(
            "L2",
            suffix,
            scene=s["scenario"],
            action=s["action"],
            source=s["source"],
            arm=s["arm"],
            target=s.get("target"),
            place_slot=s.get("place_slot"),
            second_source=s.get("second_source"),
            second_arm=s.get("second_arm"),
            instruction=s["instruction"],
            base_task_type=s["base"],
            scenario_tag=s["scenario"],
        )

def _register_l3_tasks() -> None:
    specs = [
        # Scene 4.4.
        dict(scenario="scene4.4", action="pickup", source="blue_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not red.", base="l3_s44_l_pick_ball_not_red"),
        dict(scenario="scene4.4", action="pickup", source="blue_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not the rightmost.", base="l3_s44_l_pick_ball_not_rightmost"),
        dict(scenario="scene4.4", action="pickup", source="red_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not blue.", base="l3_s44_l_pick_ball_not_blue"),
        dict(scenario="scene4.4", action="pickup", source="yellow_cylinder", arm="left", instruction="Use your left arm to pick up the cylinder that is not green.", base="l3_s44_l_pick_cylinder_not_green"),
        dict(scenario="scene4.4", action="pickup", source="red_sphere", arm="left", instruction="If there is an object in the bowl, use your left arm to pick up a non-red ball; otherwise use your left arm to pick up the red ball.", base="l3_s44_if_bowl_nonempty_then_pick_not_red_else_pick_red"),
        dict(scenario="scene4.4", action="pickup", source="blue_sphere", arm="left", instruction="If there is an object in the bowl, use your left arm to pick up the red ball; otherwise use your left arm to pick up a non-red ball.", base="l3_s44_if_bowl_nonempty_then_pick_red_else_pick_not_red"),
        # Scene 4.5.
        dict(scenario="scene4.5", action="pickup", source="blue_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not red.", base="l3_s45_l_pick_ball_not_red"),
        dict(scenario="scene4.5", action="pickup", source="red_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not the rightmost.", base="l3_s45_l_pick_ball_not_rightmost"),
        dict(scenario="scene4.5", action="pickup", source="red_cube_in_bowl", arm="left", instruction="Use your left arm to pick up the cube that is not touching the table.", base="l3_s45_l_pick_cube_not_touching_table"),
        dict(scenario="scene4.5", action="pickup", source="red_sphere", arm="left", instruction="Use your left arm to pick up the ball that is not blue.", base="l3_s45_l_pick_ball_not_blue"),
        dict(scenario="scene4.5", action="pickup", source="yellow_cylinder", arm="left", instruction="Use your left arm to pick up the cylinder that is not green.", base="l3_s45_l_pick_cylinder_not_green"),
        dict(scenario="scene4.5", action="pickup", source="green_cylinder_top", arm="right", instruction="If the red sphere is to the right of the blue sphere, use your left arm to pick up a non-blue ball; otherwise use your right arm to pick up the cylinder that is not touching the table.", base="l3_s45_if_red_right_of_blue_then_l_pick_not_blue_else_r_pick_not_touch_cyl"),
        dict(scenario="scene4.5", action="pickup", source="red_sphere", arm="left", instruction="If the red sphere is to the right of the blue sphere, use your right arm to pick up the cylinder that is not touching the table; otherwise use your left arm to pick up a non-blue ball.", base="l3_s45_if_red_right_of_blue_then_r_pick_not_touch_cyl_else_l_pick_not_blue"),
        dict(scenario="scene4.5", action="sequence_pickups", source="red_sphere", arm="left", second_source="blue_sphere", second_arm="right", instruction="First use your left arm to pick up the left ball, then use your right arm to pick up the right ball.", base="l3_s45_seq_l_left_ball_then_r_right_ball"),
        dict(scenario="scene4.5", action="sequence_pickups", source="blue_sphere", arm="right", second_source="red_sphere", second_arm="left", instruction="First use your right arm to pick up the right ball, then use your left arm to pick up the left ball.", base="l3_s45_seq_r_right_ball_then_l_left_ball"),
        # Scene 4.6.
        dict(scenario="scene4.6", action="pickup", source="green_cylinder_top", arm="left", instruction="Use your left arm to pick up the cylinder that is not yellow.", base="l3_s46_l_pick_cylinder_not_yellow"),
        dict(scenario="scene4.6", action="pickup", source="red_sphere", arm="left", instruction="If there is an object on top of the right cube, use your left arm to pick up the left ball; otherwise pick up the ball that is not on the left side.", base="l3_s46_if_object_on_right_cube_then_l_pick_left_ball_else_pick_not_left_ball"),
        dict(scenario="scene4.6", action="pickup", source="blue_sphere", arm="left", instruction="If there is an object on top of the right cube, pick up the ball that is not on the left side; otherwise use your left arm to pick up the left ball.", base="l3_s46_if_object_on_right_cube_then_pick_not_left_ball_else_l_pick_left_ball"),
        dict(scenario="scene4.6", action="sequence_pickups", source="green_cylinder_top", arm="right", second_source="red_cube_in_bowl", second_arm="left", instruction="First use your right arm to pick up the cylinder that is not touching the table, then use your left arm to pick up the cube that is not touching the table.", base="l3_s46_seq_r_pick_not_touch_cyl_then_l_pick_not_touch_cube"),
        dict(scenario="scene4.6", action="sequence_pickups", source="red_cube_in_bowl", arm="left", second_source="green_cylinder_top", second_arm="right", instruction="First use your left arm to pick up the cube that is not touching the table, then use your right arm to pick up the cylinder that is not touching the table.", base="l3_s46_seq_l_pick_not_touch_cube_then_r_pick_not_touch_cyl"),
        # Scene 4.7.
        dict(scenario="scene4.7", action="pickup", source="blue_sphere", arm="left", instruction="If the yellow cylinder is to the right of the green cylinder, use your left arm to pick up a non-red ball; otherwise use your right arm to pick up a non-green ball.", base="l3_s47_if_yellow_right_of_green_then_l_pick_not_red_else_r_pick_not_green"),
        dict(scenario="scene4.7", action="pickup", source="red_sphere", arm="right", instruction="If the yellow cylinder is to the right of the green cylinder, use your right arm to pick up a non-blue ball; otherwise use your left arm to pick up a non-red ball.", base="l3_s47_if_yellow_right_of_green_then_r_pick_not_blue_else_l_pick_not_red"),
        dict(scenario="scene4.7", action="sequence_pickups", source="red_sphere", arm="right", second_source="blue_sphere", second_arm="left", instruction="First use your right arm to pick up the right ball, then use your left arm to pick up the left ball.", base="l3_s47_seq_r_right_ball_then_l_left_ball"),
        dict(scenario="scene4.7", action="sequence_pickups", source="blue_sphere", arm="left", second_source="red_sphere", second_arm="right", instruction="First use your left arm to pick up the left ball, then use your right arm to pick up the right ball.", base="l3_s47_seq_l_left_ball_then_r_right_ball"),
    ]

    for idx, s in enumerate(specs, start=1):
        suffix = f"{s['scenario'].replace('.', '_')}_{idx:02d}_{s['base']}"
        _add_task(
            "L3",
            suffix,
            scene=s["scenario"],
            action=s["action"],
            source=s["source"],
            arm=s["arm"],
            target=s.get("target"),
            place_slot=s.get("place_slot"),
            second_source=s.get("second_source"),
            second_arm=s.get("second_arm"),
            instruction=s["instruction"],
            base_task_type=s["base"],
            scenario_tag=s["scenario"],
        )

def _validate_registry_counts() -> None:
    expected = {"L0": 27, "L1": 14, "L2": 25, "L3": 24}
    for lvl, exp in expected.items():
        got = len(LEVELS.get(lvl, []))
        if got != exp:
            sys.exit(f"Registry count mismatch for {lvl}: expected {exp}, got {got}")

def init_registry() -> None:
    global BASE_TASK_TYPES
    TASKS.clear()
    LEVELS.clear()
    _register_scene_variants()
    _register_l0_tasks()
    _register_l1_tasks()
    _register_l2_tasks()
    _register_l3_tasks()
    _validate_registry_counts()
    BASE_TASK_TYPES = sorted({task.base_task_type for task in TASKS.values()})

def _first_close_event(
    grips: list[float],
    close_threshold: float = 0.35,
    start_t: int = 0,
    end_t: int | None = None,
) -> int | None:
    if not grips:
        return None
    if end_t is None:
        end_t = len(grips) - 1
    start_t = max(0, start_t)
    end_t = min(end_t, len(grips) - 1)
    if start_t > end_t:
        return None

    for t in range(max(start_t, 1), end_t + 1):
        if grips[t - 1] > close_threshold and grips[t] <= close_threshold:
            return t

    for t in range(start_t, end_t + 1):
        if grips[t] <= close_threshold:
            return t
    return None

def _first_open_event_after(
    grips: list[float],
    start_t: int,
    open_threshold: float = 0.7,
    end_t: int | None = None,
) -> int | None:
    if not grips:
        return None
    if end_t is None:
        end_t = len(grips) - 1
    start_t = max(1, min(start_t + 1, len(grips) - 1))
    end_t = min(end_t, len(grips) - 1)
    if start_t > end_t:
        return None

    for t in range(start_t, end_t + 1):
        if grips[t - 1] < open_threshold and grips[t] >= open_threshold:
            return t
    for t in range(start_t, end_t + 1):
        if grips[t] >= open_threshold:
            return t
    return None

def _close_events(
    grips: list[float],
    close_threshold: float = 0.35,
    start_t: int = 0,
    end_t: int | None = None,
) -> list[int]:
    if not grips:
        return []
    if end_t is None:
        end_t = len(grips) - 1
    start_t = max(0, start_t)
    end_t = min(end_t, len(grips) - 1)
    if start_t > end_t:
        return []

    events = []
    for t in range(max(1, start_t), end_t + 1):
        if grips[t - 1] > close_threshold and grips[t] <= close_threshold:
            events.append(t)

    if start_t == 0 and grips[0] <= close_threshold:
        events.insert(0, 0)

    return events

def _dist_xy(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2])))

def _is_conditional_task(task: Scene4EvalTask) -> bool:
    base_name = (task.base_task_type or "").lower()
    if "_if_" in base_name or base_name.startswith("if_"):
        return True
    return (task.instruction or "").strip().lower().startswith("if ")

class Scene4TrajectoryRecorder:
    def __init__(self):
        self.traj = Scene4Trajectory(
            ee={"left": [], "right": []},
            grip={"left": [], "right": []},
            obj_pos={},
            bowls=set(),
            n_steps=0,
            table_height=0.741,
            half_heights={},
        )

    def record(self, env, obs):
        for arm in ("left", "right"):
            self.traj.ee[arm].append(base.get_ee_pos(env, arm))

        sv = obs["joint_action"]["vector"]
        self.traj.grip["left"].append(float(sv[6]))
        self.traj.grip["right"].append(float(sv[13]))

        if hasattr(env, "object_dict") and isinstance(env.object_dict, dict):
            for name, ent in env.object_dict.items():
                pose = ent.get_pose()
                self.traj.obj_pos.setdefault(name, []).append(np.asarray(pose.p, dtype=np.float32))
        if hasattr(env, "_actor_defs") and isinstance(env._actor_defs, dict):
            self.traj.bowls.update(env._actor_defs.keys())
        elif hasattr(env, "actors") and isinstance(env.actors, dict):
            self.traj.bowls.update(env.actors.keys())

        if hasattr(env, "_table_height"):
            self.traj.table_height = float(env._table_height)

        if hasattr(env, "_object_defs") and isinstance(env._object_defs, dict):
            for name, obj_def in env._object_defs.items():
                if obj_def.shape == "cube":
                    h = float(obj_def.size[0])
                elif obj_def.shape == "sphere":
                    h = float(obj_def.size[0])
                elif obj_def.shape == "cylinder":
                    h = float(obj_def.size[1])
                else:
                    continue
                self.traj.half_heights[name] = h

        self.traj.n_steps = len(self.traj.ee["left"])

    def get_trajectory(self) -> Scene4Trajectory:
        return self.traj

class Scene4ThreeStageEvaluator:
    """Score source selection, stacking or unstacking, and finish behavior."""

    def __init__(
        self,
        task: Scene4EvalTask,
        traj: Scene4Trajectory,
    ):
        self.task = task
        self.traj = traj
        self.other_arm = "right" if task.arm == "left" else "left"

        # Source selection thresholds.
        self.pick_xy_threshold = 0.07
        self.lift_threshold = 0.01
        self.stacked_xy_tol = 0.02
        self.stacked_min_z_gap = 0.005

        # Stacking and unstacking thresholds.
        self.stack_intent_xy = 0.07
        self.stack_intent_z = 0.06
        self.stack_exec_xy = 0.05
        self.stack_exec_z = 0.05
        self.stack_gap = 0.002

        self.slot_intent_xy = 0.07
        self.slot_intent_z = 0.07
        self.slot_exec_xy = 0.05
        self.slot_exec_z = 0.05
        self.table_clearance = 0.001

        # Finish thresholds.
        self.reset_dist_threshold = 0.10
        self.home_stable_steps = 20
        self.return_steps_limit = 180
        self.leave_home_threshold = 0.14
        self.grip_close_threshold = 0.35

    def _source_object_names(self) -> list[str]:
        return [name for name in self.traj.obj_pos if name not in self.traj.bowls]

    def _stack_support_under_source(self, source: str, t: int) -> str | None:
        if source not in self.traj.obj_pos:
            return None
        src = self.traj.obj_pos[source][t]
        support_name = None
        min_z_gap = float("inf")
        for name in self._source_object_names():
            if name == source:
                continue
            obj = self.traj.obj_pos[name][t]
            xy = _dist_xy(src, obj)
            z_gap = float(src[2]) - float(obj[2])
            if xy <= self.stacked_xy_tol and z_gap >= self.stacked_min_z_gap and z_gap < min_z_gap:
                min_z_gap = z_gap
                support_name = name
        return support_name

    def _nearest_source_candidates(self, source: str, t: int) -> list[str]:
        names = self._source_object_names()
        support_name = self._stack_support_under_source(source, t)
        if support_name is not None:
            names = [name for name in names if name != support_name]
        return names

    def _object_half_height(self, name: str) -> float:
        return float(self.traj.half_heights.get(name, 0.025))

    def _evaluate_stage1(self, stage1_max: float = 0.4) -> tuple[PhaseResult, dict]:
        ctx = {
            "close_t_expected": None,
            "open_t_expected": None,
            "first_close_arm": None,
            "wrong_arm_closed": False,
            "stage1_intent_ok": False,
            "source_lifted": False,
            "stage2_intent_t": None,
            "pickup_completion_t": None,
            "support_name_at_close": None,
            "support_max_lift_after_close": 0.0,
        }

        if self.traj.n_steps <= 0 or self.task.source not in self.traj.obj_pos:
            reason = "trajectory/source missing"
            return (
                PhaseResult("stage1_source_select", 0.0, reason, 0.0, reason, stage1_max),
                ctx,
            )

        close_expected = _first_close_event(self.traj.grip[self.task.arm])
        close_other = _first_close_event(self.traj.grip[self.other_arm])
        ctx["close_t_expected"] = close_expected
        if self.task.action == "pickup":
            # Count early wrong-arm closes here; the finish stage checks later closes.
            ctx["wrong_arm_closed"] = (
                close_other is not None
                and close_expected is not None
                and close_other <= close_expected
            )
        else:
            ctx["wrong_arm_closed"] = close_other is not None

        if close_expected is None and close_other is None:
            first_close_arm = None
        elif close_other is None or (close_expected is not None and close_expected <= close_other):
            first_close_arm = self.task.arm
        else:
            first_close_arm = self.other_arm
        ctx["first_close_arm"] = first_close_arm

        if close_expected is None:
            reason = "operating arm never closed gripper"
            return (
                PhaseResult("stage1_source_select", 0.0, reason, 0.0, reason, stage1_max),
                ctx,
            )

        if first_close_arm != self.task.arm or ctx["wrong_arm_closed"]:
            reason = "wrong arm participated in grasp stage"
            return (
                PhaseResult("stage1_source_select", 0.0, reason, 0.0, reason, stage1_max),
                ctx,
            )

        t = close_expected
        ee = self.traj.ee[self.task.arm][t]
        src = self.traj.obj_pos[self.task.source][t]
        src_xy = _dist_xy(ee, src)
        support_name = self._stack_support_under_source(self.task.source, t)
        ctx["support_name_at_close"] = support_name
        if support_name is not None and support_name in self.traj.obj_pos:
            support_z0 = float(self.traj.obj_pos[support_name][0][2])
            max_support_lift = 0.0
            for tt in range(t, self.traj.n_steps):
                support_lift = float(self.traj.obj_pos[support_name][tt][2]) - support_z0
                if support_lift > max_support_lift:
                    max_support_lift = support_lift
            ctx["support_max_lift_after_close"] = max_support_lift

        min_name = None
        min_dist = None
        for name in self._nearest_source_candidates(self.task.source, t):
            d = _dist_xy(ee, self.traj.obj_pos[name][t])
            if min_dist is None or d < min_dist:
                min_dist = d
                min_name = name
        closest_ok = min_name == self.task.source

        intent_ok = closest_ok and (src_xy < self.pick_xy_threshold)
        ctx["stage1_intent_ok"] = intent_ok

        z0 = float(self.traj.obj_pos[self.task.source][0][2])
        lift_t = None
        max_lift = 0.0
        for tt in range(t, self.traj.n_steps):
            lift = float(self.traj.obj_pos[self.task.source][tt][2]) - z0
            if lift > max_lift:
                max_lift = lift
            if lift_t is None and lift >= self.lift_threshold:
                lift_t = tt
        source_lifted = lift_t is not None
        ctx["source_lifted"] = source_lifted
        ctx["pickup_completion_t"] = lift_t if lift_t is not None else t

        exec_ok = intent_ok and source_lifted

        intent_reason = "ok" if intent_ok else (
            f"at first close: closest={min_name}, source_xy={src_xy*100:.1f}cm"
        )
        exec_reason = "ok" if exec_ok else (
            f"source lift={max_lift*100:.1f}cm (need >= {self.lift_threshold*100:.1f}cm)"
            if intent_ok
            else "intent not correct at first close"
        )

        return (
            PhaseResult(
                "stage1_source_select",
                stage1_max if intent_ok else 0.0,
                intent_reason,
                stage1_max if exec_ok else 0.0,
                exec_reason,
                stage1_max,
            ),
            ctx,
        )

    def _evaluate_stage2_stack(self, ctx: dict) -> PhaseResult:
        close_t = ctx["close_t_expected"]
        open_t = None if close_t is None else _first_open_event_after(self.traj.grip[self.task.arm], close_t)
        ctx["open_t_expected"] = open_t

        if self.task.target is None:
            return PhaseResult("stage2_stack", 0.0, "target missing", 0.0, "target missing", 0.4)
        if self.task.source not in self.traj.obj_pos or self.task.target not in self.traj.obj_pos:
            reason = "source/target trajectory missing"
            return PhaseResult("stage2_stack", 0.0, reason, 0.0, reason, 0.4)
        if ctx["wrong_arm_closed"] or ctx["first_close_arm"] != self.task.arm:
            reason = "wrong arm participated in stage1/2"
            return PhaseResult("stage2_stack", 0.0, reason, 0.0, reason, 0.4)
        if close_t is None:
            reason = "no grasp close event"
            return PhaseResult("stage2_stack", 0.0, reason, 0.0, reason, 0.4)

        src_h = self._object_half_height(self.task.source)
        tgt_h = self._object_half_height(self.task.target)

        intent_ok = False
        best_xy = float("inf")
        best_z = float("inf")

        for t in range(close_t + 1, self.traj.n_steps):
            src = self.traj.obj_pos[self.task.source][t]
            tgt = self.traj.obj_pos[self.task.target][t]
            exp_z = float(tgt[2] + src_h + tgt_h + self.stack_gap)
            xy = _dist_xy(src, tgt)
            z_err = abs(float(src[2]) - exp_z)
            best_xy = min(best_xy, xy)
            best_z = min(best_z, z_err)

            if xy <= self.stack_intent_xy and z_err <= self.stack_intent_z:
                intent_ok = True
                ctx["stage2_intent_t"] = t
                break

        exec_ok = False
        exec_reason = "no gripper-open event after close"
        if intent_ok and open_t is not None:
            src_o = self.traj.obj_pos[self.task.source][open_t]
            tgt_o = self.traj.obj_pos[self.task.target][open_t]
            exp_open = float(tgt_o[2] + src_h + tgt_h + self.stack_gap)
            open_xy = _dist_xy(src_o, tgt_o)
            open_z = abs(float(src_o[2]) - exp_open)

            src_f = self.traj.obj_pos[self.task.source][self.traj.n_steps - 1]
            tgt_f = self.traj.obj_pos[self.task.target][self.traj.n_steps - 1]
            exp_fin = float(tgt_f[2] + src_h + tgt_h + self.stack_gap)
            fin_xy = _dist_xy(src_f, tgt_f)
            fin_z = abs(float(src_f[2]) - exp_fin)

            exec_ok = (
                open_xy <= self.stack_exec_xy
                and open_z <= self.stack_exec_z
                and fin_xy <= self.stack_exec_xy
                and fin_z <= self.stack_exec_z
            )
            if exec_ok:
                exec_reason = "ok"
            else:
                exec_reason = (
                    f"open(xy={open_xy*100:.1f}cm,z={open_z*100:.1f}cm), "
                    f"final(xy={fin_xy*100:.1f}cm,z={fin_z*100:.1f}cm)"
                )
        elif intent_ok:
            exec_reason = "no gripper-open event after close"

        intent_reason = "ok" if intent_ok else (
            f"best stack pose xy={best_xy*100:.1f}cm z_err={best_z*100:.1f}cm"
        )

        return PhaseResult(
            "stage2_stack",
            0.4 if intent_ok else 0.0,
            intent_reason,
            0.4 if exec_ok else 0.0,
            exec_reason,
            0.4,
        )

    def _evaluate_stage2_unstack(self, ctx: dict) -> PhaseResult:
        close_t = ctx["close_t_expected"]
        open_t = None if close_t is None else _first_open_event_after(self.traj.grip[self.task.arm], close_t)
        ctx["open_t_expected"] = open_t

        if self.task.place_slot is None:
            return PhaseResult("stage2_unstack_slot", 0.0, "place_slot missing", 0.0, "place_slot missing", 0.4)
        if self.task.source not in self.traj.obj_pos:
            return PhaseResult("stage2_unstack_slot", 0.0, "source missing", 0.0, "source missing", 0.4)
        if ctx["wrong_arm_closed"] or ctx["first_close_arm"] != self.task.arm:
            reason = "wrong arm participated in stage1/2"
            return PhaseResult("stage2_unstack_slot", 0.0, reason, 0.0, reason, 0.4)
        if close_t is None:
            reason = "no grasp close event"
            return PhaseResult("stage2_unstack_slot", 0.0, reason, 0.0, reason, 0.4)

        slot_xy = SLOT_XY[self.task.place_slot]
        src_h = self._object_half_height(self.task.source)
        exp_z = float(self.traj.table_height + src_h + self.table_clearance)

        intent_ok = False
        best_xy = float("inf")
        best_z = float("inf")

        for t in range(close_t + 1, self.traj.n_steps):
            src = self.traj.obj_pos[self.task.source][t]
            xy = float(np.linalg.norm(np.asarray(src[:2]) - np.asarray(slot_xy)))
            z_err = abs(float(src[2]) - exp_z)
            best_xy = min(best_xy, xy)
            best_z = min(best_z, z_err)
            if xy <= self.slot_intent_xy and z_err <= self.slot_intent_z:
                intent_ok = True
                ctx["stage2_intent_t"] = t
                break

        exec_ok = False
        exec_reason = "no gripper-open event after close"
        if intent_ok and open_t is not None:
            src_o = self.traj.obj_pos[self.task.source][open_t]
            open_xy = float(np.linalg.norm(np.asarray(src_o[:2]) - np.asarray(slot_xy)))
            open_z = abs(float(src_o[2]) - exp_z)

            src_f = self.traj.obj_pos[self.task.source][self.traj.n_steps - 1]
            fin_xy = float(np.linalg.norm(np.asarray(src_f[:2]) - np.asarray(slot_xy)))
            fin_z = abs(float(src_f[2]) - exp_z)

            exec_ok = (
                open_xy <= self.slot_exec_xy
                and open_z <= self.slot_exec_z
                and fin_xy <= self.slot_exec_xy
                and fin_z <= self.slot_exec_z
            )
            if exec_ok:
                exec_reason = "ok"
            else:
                exec_reason = (
                    f"open(xy={open_xy*100:.1f}cm,z={open_z*100:.1f}cm), "
                    f"final(xy={fin_xy*100:.1f}cm,z={fin_z*100:.1f}cm)"
                )
        elif intent_ok:
            exec_reason = "no gripper-open event after close"

        intent_reason = "ok" if intent_ok else (
            f"best slot pose xy={best_xy*100:.1f}cm z_err={best_z*100:.1f}cm"
        )

        return PhaseResult(
            "stage2_unstack_slot",
            0.4 if intent_ok else 0.0,
            intent_reason,
            0.4 if exec_ok else 0.0,
            exec_reason,
            0.4,
        )

    def _completion_t(self, ctx: dict) -> int:
        if ctx.get("open_t_expected") is not None:
            return int(ctx["open_t_expected"])
        if ctx.get("stage2_intent_t") is not None:
            return int(ctx["stage2_intent_t"])
        if ctx.get("pickup_completion_t") is not None:
            return int(ctx["pickup_completion_t"])
        if ctx.get("close_t_expected") is not None:
            return int(ctx["close_t_expected"])
        return max(0, self.traj.n_steps - 1)

    def _evaluate_stage3(self, ctx: dict) -> PhaseResult:
        if self.traj.n_steps <= 0:
            return PhaseResult("stage3_finish", 0.0, "empty trajectory", 0.0, "empty trajectory", 0.2)

        completion_t = self._completion_t(ctx)
        problems = []

        if any(base.in_operation_area(p) for p in self.traj.ee[self.other_arm]):
            problems.append("non-operating arm entered operation area")

        home = np.asarray(self.traj.ee[self.task.arm][0])
        arm_traj = self.traj.ee[self.task.arm]
        n = len(arm_traj)
        stable_steps = min(self.home_stable_steps, max(1, n - completion_t))
        first_home_t = None

        for t in range(completion_t, n):
            end = min(n, t + stable_steps)
            dists = [
                float(np.linalg.norm(np.asarray(arm_traj[s]) - home))
                for s in range(t, end)
            ]
            if dists and all(d <= self.reset_dist_threshold for d in dists):
                first_home_t = t
                break

        if first_home_t is None:
            final_dist = float(np.linalg.norm(np.asarray(arm_traj[-1]) - home))
            problems.append(
                f"{self.task.arm} arm did not return home stably (final dist={final_dist*100:.1f}cm)"
            )
        else:
            if (first_home_t - completion_t) > self.return_steps_limit:
                problems.append(
                    f"{self.task.arm} arm returned too late ({first_home_t - completion_t} steps)"
                )
            for t in range(first_home_t + 1, n):
                d = float(np.linalg.norm(np.asarray(arm_traj[t]) - home))
                if d > self.leave_home_threshold:
                    problems.append(f"{self.task.arm} arm left home again (dist={d*100:.1f}cm)")
                    break

        grips = self.traj.grip[self.task.arm]
        for t in range(max(1, completion_t + 1), len(grips)):
            if grips[t - 1] > self.grip_close_threshold and grips[t] <= self.grip_close_threshold:
                problems.append("operating gripper re-closed after completion")
                break

        if problems:
            reason = "; ".join(problems)
            return PhaseResult("stage3_finish", 0.0, reason, 0.0, reason, 0.2)
        return PhaseResult("stage3_finish", 0.2, "ok", 0.2, "ok", 0.2)

    def _apply_gates(self, stage1: PhaseResult, stage2: PhaseResult, stage3: PhaseResult) -> list[PhaseResult]:
        eps = 1e-6
        s1 = copy.deepcopy(stage1)
        s2 = copy.deepcopy(stage2)
        s3 = copy.deepcopy(stage3)

        if s1.intent_score + eps < s1.max_score:
            s2.intent_score = 0.0
            s2.intent_reason = "gated: stage1 intent not full"
        if s1.intent_score + eps < s1.max_score or s2.intent_score + eps < s2.max_score:
            s3.intent_score = 0.0
            s3.intent_reason = "gated: prior intent not full"

        if not (
            s1.intent_score + eps >= s1.max_score
            and s1.exec_score + eps >= s1.max_score
            and s2.intent_score + eps >= s2.max_score
        ):
            s2.exec_score = 0.0
            s2.exec_reason = "gated: stage1 intent/exec or stage2 intent incomplete"

        if not (
            s1.intent_score + eps >= s1.max_score
            and s1.exec_score + eps >= s1.max_score
            and s2.intent_score + eps >= s2.max_score
            and s2.exec_score + eps >= s2.max_score
            and s3.intent_score + eps >= s3.max_score
        ):
            s3.exec_score = 0.0
            s3.exec_reason = "gated: prior intent/exec not full"
        return [s1, s2, s3]

    def _apply_gates_pickup(self, stage1: PhaseResult, stage3: PhaseResult) -> list[PhaseResult]:
        eps = 1e-6
        s1 = copy.deepcopy(stage1)
        s3 = copy.deepcopy(stage3)

        if s1.intent_score + eps < s1.max_score:
            s3.intent_score = 0.0
            s3.intent_reason = "gated: stage1 intent not full"

        if not (
            s1.intent_score + eps >= s1.max_score
            and s1.exec_score + eps >= s1.max_score
            and s3.intent_score + eps >= s3.max_score
        ):
            s3.exec_score = 0.0
            s3.exec_reason = "gated: stage1 intent/exec or stage3 intent incomplete"

        return [s1, s3]

    def _apply_pickup_support_lift_penalty(self, phases: list[PhaseResult], ctx: dict) -> list[PhaseResult]:
        if self.task.action != "pickup":
            return phases
        support_name = ctx.get("support_name_at_close")
        support_max_lift = float(ctx.get("support_max_lift_after_close", 0.0))
        penalty_threshold = 0.01
        penalty = 0.3
        if support_name is None or support_max_lift < penalty_threshold:
            return phases

        scored = [copy.deepcopy(p) for p in phases]
        stage1 = scored[0]
        stage1.exec_score = max(0.0, stage1.exec_score - penalty)
        penalty_msg = (
            f"support-lift penalty: support={support_name}, "
            f"lift={support_max_lift*100:.1f}cm >= {penalty_threshold*100:.1f}cm, -{penalty:.2f}"
        )
        stage1.exec_reason = penalty_msg if stage1.exec_reason == "ok" else f"{stage1.exec_reason}; {penalty_msg}"
        return scored

    def evaluate(self) -> list[PhaseResult]:
        if self.task.action == "pickup":
            stage1, ctx = self._evaluate_stage1(stage1_max=0.8)
            stage3 = self._evaluate_stage3(ctx)
            phases = self._apply_gates_pickup(stage1, stage3)
            phases = self._apply_pickup_support_lift_penalty(phases, ctx)
            return phases

        stage1, ctx = self._evaluate_stage1(stage1_max=0.4)
        if self.task.action == "stack_on_target":
            stage2 = self._evaluate_stage2_stack(ctx)
        elif self.task.action == "unstack_to_slot":
            stage2 = self._evaluate_stage2_unstack(ctx)
        else:
            stage2 = PhaseResult("stage2_action", 0.0, "unsupported action", 0.0, "unsupported action", 0.4)
        stage3 = self._evaluate_stage3(ctx)
        phases = self._apply_gates(stage1, stage2, stage3)
        return phases

class Scene4SequenceEvaluator:
    """Score ordered pickups and finish behavior with weights 0.4/0.4/0.2."""

    def __init__(self, task: Scene4EvalTask, traj: Scene4Trajectory):
        self.task = task
        self.traj = traj
        self.pick_xy_threshold = 0.07
        self.lift_threshold = 0.01
        self.stacked_xy_tol = 0.02
        self.stacked_min_z_gap = 0.005

        self.reset_dist_threshold = 0.10
        self.home_stable_steps = 20
        self.return_steps_limit = 180
        self.leave_home_threshold = 0.14
        self.grip_close_threshold = 0.35

    def _source_object_names(self) -> list[str]:
        return [name for name in self.traj.obj_pos if name not in self.traj.bowls]

    def _stack_support_under_source(self, source: str, t: int) -> str | None:
        if source not in self.traj.obj_pos:
            return None
        src = self.traj.obj_pos[source][t]
        support_name = None
        min_z_gap = float("inf")
        for name in self._source_object_names():
            if name == source:
                continue
            obj = self.traj.obj_pos[name][t]
            xy = _dist_xy(src, obj)
            z_gap = float(src[2]) - float(obj[2])
            if xy <= self.stacked_xy_tol and z_gap >= self.stacked_min_z_gap and z_gap < min_z_gap:
                min_z_gap = z_gap
                support_name = name
        return support_name

    def _closest_object(self, arm: str, t: int, source: str | None = None) -> tuple[str | None, float]:
        ee = self.traj.ee[arm][t]
        candidate_names = self._source_object_names()
        if source is not None:
            support_name = self._stack_support_under_source(source, t)
            if support_name is not None:
                candidate_names = [name for name in candidate_names if name != support_name]

        best_name = None
        best_dist = float("inf")
        for name in candidate_names:
            d = _dist_xy(ee, self.traj.obj_pos[name][t])
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name, best_dist

    def _find_matching_start(self, arm: str, source: str, start_t: int = 0) -> int | None:
        events = _close_events(self.traj.grip[arm], start_t=start_t)
        for t in events:
            closest, d = self._closest_object(arm, t, source=source)
            if closest == source and d <= self.pick_xy_threshold:
                return t
        return None

    def _pickup_raw_from_start(
        self,
        arm: str,
        source: str,
        start_t: int,
        end_t: int | None = None,
    ) -> tuple[float, float, int | None, str, str]:
        if source not in self.traj.obj_pos:
            return 0.0, 0.0, None, "source missing", "source missing"
        if end_t is None:
            end_t = self.traj.n_steps - 1

        closest, d = self._closest_object(arm, start_t, source=source)
        intent_ok = closest == source and d <= self.pick_xy_threshold

        z0 = float(self.traj.obj_pos[source][0][2])
        completion_t = None
        max_lift = 0.0
        for t in range(start_t, min(end_t + 1, self.traj.n_steps)):
            lift = float(self.traj.obj_pos[source][t][2]) - z0
            max_lift = max(max_lift, lift)
            if completion_t is None and lift >= self.lift_threshold:
                completion_t = t
        exec_ok = intent_ok and completion_t is not None

        intent_reason = "ok" if intent_ok else f"closest={closest}, dist={d*100:.1f}cm"
        exec_reason = "ok" if exec_ok else (
            f"lift={max_lift*100:.1f}cm (<{self.lift_threshold*100:.1f}cm)" if intent_ok else "intent not satisfied"
        )
        return (
            1.0 if intent_ok else 0.0,
            1.0 if exec_ok else 0.0,
            completion_t,
            intent_reason,
            exec_reason,
        )

    def _evaluate_finish(self, completion_t: int | None) -> PhaseResult:
        if self.traj.n_steps <= 0:
            return PhaseResult("stage3_finish", 0.0, "empty trajectory", 0.0, "empty trajectory", 0.2)

        if completion_t is None:
            completion_t = max(0, self.traj.n_steps - 1)

        problems = []
        for arm in ("left", "right"):
            home = np.asarray(self.traj.ee[arm][0])
            arm_traj = self.traj.ee[arm]
            n = len(arm_traj)
            stable_steps = min(self.home_stable_steps, max(1, n - completion_t))
            first_home_t = None

            for t in range(completion_t, n):
                end = min(n, t + stable_steps)
                dists = [
                    float(np.linalg.norm(np.asarray(arm_traj[s]) - home))
                    for s in range(t, end)
                ]
                if dists and all(d <= self.reset_dist_threshold for d in dists):
                    first_home_t = t
                    break

            if first_home_t is None:
                final_dist = float(np.linalg.norm(np.asarray(arm_traj[-1]) - home))
                problems.append(f"{arm} arm not home (final={final_dist*100:.1f}cm)")
                continue

            if (first_home_t - completion_t) > self.return_steps_limit:
                problems.append(f"{arm} arm home too late ({first_home_t - completion_t} steps)")

            for t in range(first_home_t + 1, n):
                d = float(np.linalg.norm(np.asarray(arm_traj[t]) - home))
                if d > self.leave_home_threshold:
                    problems.append(f"{arm} arm left home again ({d*100:.1f}cm)")
                    break

            grips = self.traj.grip[arm]
            for t in range(max(1, completion_t + 1), len(grips)):
                if grips[t - 1] > self.grip_close_threshold and grips[t] <= self.grip_close_threshold:
                    problems.append(f"{arm} gripper re-closed after completion")
                    break

        if problems:
            reason = "; ".join(problems)
            return PhaseResult("stage3_finish", 0.0, reason, 0.0, reason, 0.2)
        return PhaseResult("stage3_finish", 0.2, "ok", 0.2, "ok", 0.2)

    def evaluate(self) -> list[PhaseResult]:
        # Starting B before A invalidates the entire sequence.
        a_start = self._find_matching_start(self.task.arm, self.task.source, start_t=0)
        b_start_any = self._find_matching_start(self.task.second_arm, self.task.second_source, start_t=0)

        if a_start is not None and b_start_any is not None and b_start_any < a_start:
            reason = f"reverse order detected: B@{b_start_any} < A@{a_start}"
            return [
                PhaseResult("stage1_action_a", 0.0, reason, 0.0, reason, 0.4),
                PhaseResult("stage2_action_b", 0.0, reason, 0.0, reason, 0.4),
                PhaseResult("stage3_finish", 0.0, reason, 0.0, reason, 0.2),
            ]

        if a_start is None:
            reason = "A action never started"
            return [
                PhaseResult("stage1_action_a", 0.0, reason, 0.0, reason, 0.4),
                PhaseResult("stage2_action_b", 0.0, "gated: A score is zero", 0.0, "gated: A score is zero", 0.4),
                PhaseResult("stage3_finish", 0.0, "gated: A score is zero", 0.0, "gated: A score is zero", 0.2),
            ]

        a_i_raw, a_e_raw, a_completion_t, a_i_reason, a_e_reason = self._pickup_raw_from_start(
            self.task.arm,
            self.task.source,
            start_t=a_start,
        )
        phase1 = PhaseResult(
            "stage1_action_a",
            0.4 * a_i_raw,
            a_i_reason,
            0.4 * a_e_raw,
            a_e_reason,
            0.4,
        )

        # Zero credit for A invalidates the entire sequence.
        if a_i_raw <= 0.0:
            reason = "A score is zero"
            return [
                PhaseResult("stage1_action_a", 0.0, reason, 0.0, reason, 0.4),
                PhaseResult("stage2_action_b", 0.0, "gated: A score is zero", 0.0, "gated: A score is zero", 0.4),
                PhaseResult("stage3_finish", 0.0, "gated: A score is zero", 0.0, "gated: A score is zero", 0.2),
            ]

        b_start = self._find_matching_start(
            self.task.second_arm,
            self.task.second_source,
            start_t=a_completion_t if a_completion_t is not None else a_start,
        )

        if b_start is None:
            phase2 = PhaseResult(
                "stage2_action_b",
                0.0,
                "B action never started after A completion",
                0.0,
                "B action never started after A completion",
                0.4,
            )
            b_completion_t = a_completion_t if a_completion_t is not None else a_start
        else:
            b_i_raw, b_e_raw, b_completion_t, b_i_reason, b_e_reason = self._pickup_raw_from_start(
                self.task.second_arm,
                self.task.second_source,
                start_t=b_start,
            )
            phase2 = PhaseResult(
                "stage2_action_b",
                0.4 * b_i_raw,
                b_i_reason,
                0.4 * b_e_raw,
                b_e_reason,
                0.4,
            )

        finish_anchor = b_completion_t
        if finish_anchor is None:
            finish_anchor = a_completion_t if a_completion_t is not None else a_start
        phase3 = self._evaluate_finish(finish_anchor)

        return [phase1, phase2, phase3]

def run_once(
    env,
    policy,
    task: Scene4EvalTask,
    max_steps: int = 10,
    actions_per_step: int = 50,
    direct_step: bool = False,
    sim_steps: int = 15,
) -> dict:
    """Run one evaluation episode and return scores and video frames."""
    recorder = Scene4TrajectoryRecorder()
    policy.reset()
    policy.set_instruction(task.instruction)

    frames = []
    env._update_render()
    obs = env.get_obs()
    frames.append(obs["observation"]["head_camera"]["rgb"])
    recorder.record(env, obs)

    for step in range(max_steps):
        try:
            actions = policy.predict(obs)[:actions_per_step]
            for action in actions:
                if direct_step:
                    base._step_action_direct(env, action, n_sim_steps=sim_steps)
                else:
                    env.take_action(action)
                obs = env.get_obs()
                frames.append(obs["observation"]["head_camera"]["rgb"])
                recorder.record(env, obs)
            env._update_render()
        except Exception as exc:
            raise RuntimeError(f"Rollout failed at policy step {step}") from exc

    traj = recorder.get_trajectory()

    if task.action == "sequence_pickups" and task.second_source and task.second_arm:
        evaluator = Scene4SequenceEvaluator(task, traj)
    else:
        evaluator = Scene4ThreeStageEvaluator(task, traj)
    phases = evaluator.evaluate()

    intent_score = sum(p.intent_score for p in phases)
    exec_score = sum(p.exec_score for p in phases)
    max_total = sum(p.max_score for p in phases)
    intent_ok = intent_score >= max_total * 0.99
    exec_ok = exec_score >= max_total * 0.99

    return dict(
        task=task.name,
        instruction=task.instruction,
        intent_ok=intent_ok,
        intent_msg="ok" if intent_ok else f"intent_score={intent_score:.2f}/{max_total:.2f}",
        exec_ok=exec_ok,
        completion_ok=base.is_complete(phases, intent_score, exec_score, action=task.action),
        conditional_full_success=(base.is_full_success(phases, intent_score, exec_score)
                                  if _is_conditional_task(task) else None),
        exec_msg="ok" if exec_ok else f"exec_score={exec_score:.2f}/{max_total:.2f}",
        intent_score=float(intent_score),
        exec_score=float(exec_score),
        phases=[
            {
                "phase": p.phase_name,
                "intent_score": p.intent_score,
                "intent_reason": p.intent_reason,
                "exec_score": p.exec_score,
                "exec_reason": p.exec_reason,
                "max_score": p.max_score,
            }
            for p in phases
        ],
        frames=frames,
    )
