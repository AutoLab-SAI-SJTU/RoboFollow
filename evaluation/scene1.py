from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from . import common as base
from robofollow.tasks.scene1 import TASKS as SCENE1_TRAIN_TASKS

TIP_FORWARD_BIAS = 0.04
WRONG_ARM_INTERVENTION_PENALTY = 0.2

LEVEL_ORDER = ("L0", "L1", "L2", "L3")

RELATION_ACTIONS = ("place_behind", "place_in_front", "place_left_of", "place_beside")

@dataclass
class Scene1EvalTask:
    name: str
    scene: str
    source: str
    target: str | None
    arm: str
    action: str
    instruction: str
    level_tag: str
    base_task_type: str
    instruction_variant: str
    scenario_tag: str

@dataclass
class PhaseResult:
    phase_name: str
    intent_score: float
    intent_reason: str
    exec_score: float
    exec_reason: str
    max_score: float

TASKS: dict[str, Scene1EvalTask] = {}

LEVELS: dict[str, list[str]] = {}

BASE_TASK_TYPES: list[str] = []

SCENE_VARIANTS: dict[str, object] = {}

def _stable_hash32(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)

def _register_task(task: Scene1EvalTask) -> None:
    TASKS[task.name] = task
    LEVELS.setdefault(task.level_tag, []).append(task.name)

def _load_collect_scene1_templates():
    return json.loads((Path(__file__).resolve().parents[1] / "tasks" / "scene1_instructions.json").read_text())

INSTRUCTION_TEMPLATES = _load_collect_scene1_templates()

def _scene1_perturb_rules() -> dict[str, object]:
    return {
        "red_sphere": base.PerturbRule(x_range=0.015, y_range=0.015, y_direction="down"),
        "yellow_block_1": base.PerturbRule(x_range=0.015, y_range=0.015, y_direction="down"),
        "red_cylinder": base.PerturbRule(x_range=0.015, y_range=0.015, y_direction="down"),
        "yellow_block_2": base.PerturbRule(x_range=0.015, y_range=0.015, y_direction="up"),
        "green_sphere": base.PerturbRule(x_range=0.015, y_range=0.015, y_direction="up"),
        "yellow_block_3": base.PerturbRule(x_range=0.015, y_range=0.015, y_direction="up"),
    }

def _register_scene_variants() -> None:
    SCENE_VARIANTS.clear()
    # Default layout.
    SCENE_VARIANTS["default"] = base.SceneConfig(
        name="default",
        objects=[
            base.ObjectDef("red_sphere", "sphere", (0.02,), (1.0, 0.0, 0.0), (-0.14, -0.04)),
            base.ObjectDef("yellow_block_1", "cube", (0.02,), (1.0, 1.0, 0.0), (-0.02, -0.04)),
            base.ObjectDef("red_cylinder", "cylinder", (0.025, 0.025), (1.0, 0.0, 0.0), (0.10, -0.04)),
            base.ObjectDef("yellow_block_2", "cube", (0.02,), (1.0, 1.0, 0.0), (-0.14, -0.18)),
            base.ObjectDef("green_sphere", "sphere", (0.02,), (0.0, 1.0, 0.0), (-0.02, -0.18)),
            base.ObjectDef("yellow_block_3", "cube", (0.02,), (1.0, 1.0, 0.0), (0.10, -0.18)),
        ],
        actors=[],
        perturb_rules=_scene1_perturb_rules(),
        default_perturb=base.PerturbRule(x_range=0.01, y_range=0.01, y_direction="both"),
    )

    # Scene 1.1.
    SCENE_VARIANTS["scene1_1"] = base.SceneConfig(
        name="scene1_1",
        objects=[
            base.ObjectDef("red_sphere", "sphere", (0.02,), (1.0, 0.0, 0.0), (-0.02, -0.18)),
            base.ObjectDef("yellow_block_1", "cube", (0.02,), (1.0, 1.0, 0.0), (-0.02, -0.04)),
            base.ObjectDef("red_cylinder", "cylinder", (0.025, 0.025), (1.0, 0.0, 0.0), (-0.14, -0.04)),
            base.ObjectDef("yellow_block_2", "cube", (0.02,), (1.0, 1.0, 0.0), (-0.14, -0.18)),
            base.ObjectDef("green_sphere", "sphere", (0.02,), (0.0, 1.0, 0.0), (0.10, -0.04)),
            base.ObjectDef("yellow_block_3", "cube", (0.02,), (1.0, 1.0, 0.0), (0.10, -0.18)),
        ],
        actors=[],
        perturb_rules=_scene1_perturb_rules(),
        default_perturb=base.PerturbRule(x_range=0.01, y_range=0.01, y_direction="both"),
    )

    # Scene 1.2.
    SCENE_VARIANTS["scene1_2"] = base.SceneConfig(
        name="scene1_2",
        objects=[
            base.ObjectDef("red_sphere", "sphere", (0.02,), (1.0, 0.0, 0.0), (-0.02, -0.18)),
            base.ObjectDef("yellow_block_1", "cube", (0.02,), (1.0, 1.0, 0.0), (-0.02, -0.04)),
            base.ObjectDef("red_cylinder", "cylinder", (0.025, 0.025), (1.0, 0.0, 0.0), (0.10, -0.04)),
            base.ObjectDef("yellow_block_2", "cube", (0.02,), (1.0, 1.0, 0.0), (-0.14, -0.18)),
            base.ObjectDef("green_sphere", "sphere", (0.02,), (0.0, 1.0, 0.0), (-0.14, -0.04)),
            base.ObjectDef("yellow_block_3", "cube", (0.02,), (1.0, 1.0, 0.0), (0.10, -0.18)),
        ],
        actors=[],
        perturb_rules=_scene1_perturb_rules(),
        default_perturb=base.PerturbRule(x_range=0.01, y_range=0.01, y_direction="both"),
    )

    # Scene 1.3.
    SCENE_VARIANTS["scene1_3"] = base.SceneConfig(
        name="scene1_3",
        objects=[
            base.ObjectDef("red_sphere", "sphere", (0.02,), (1.0, 0.0, 0.0), (-0.14, -0.04)),
            base.ObjectDef("yellow_block_1", "cube", (0.02,), (1.0, 1.0, 0.0), (-0.02, -0.04)),
            base.ObjectDef("red_cylinder", "cylinder", (0.025, 0.025), (1.0, 0.0, 0.0), (-0.02, -0.18)),
            base.ObjectDef("yellow_block_2", "cube", (0.02,), (1.0, 1.0, 0.0), (-0.14, -0.18)),
            base.ObjectDef("green_sphere", "sphere", (0.02,), (0.0, 1.0, 0.0), (0.10, -0.04)),
            base.ObjectDef("yellow_block_3", "cube", (0.02,), (1.0, 1.0, 0.0), (0.10, -0.18)),
        ],
        actors=[],
        perturb_rules=_scene1_perturb_rules(),
        default_perturb=base.PerturbRule(x_range=0.01, y_range=0.01, y_direction="both"),
    )

def _add_task(
    level_tag: str,
    base_task_type: str,
    scene: str,
    source: str,
    target: str | None,
    arm: str,
    action: str,
    instruction: str,
    scenario_tag: str,
) -> None:
    _register_task(
        Scene1EvalTask(
            name=f"{level_tag}_{base_task_type}",
            scene=scene,
            source=source,
            target=target,
            arm=arm,
            action=action,
            instruction=instruction,
            level_tag=level_tag,
            base_task_type=base_task_type,
            instruction_variant="manual_v1",
            scenario_tag=scenario_tag,
        )
    )

def _register_l0_tasks() -> None:
    for task_type, task_def in SCENE1_TRAIN_TASKS.items():
        templates = INSTRUCTION_TEMPLATES.get(task_type, [])
        if templates:
            instruction = templates[0]
        else:
            instruction = (
                f"Use your {task_def.arm} arm to perform {task_def.action} on "
                f"{task_def.source}."
            )

        _add_task(
            "L0",
            task_type,
            task_def.scene,
            task_def.source,
            task_def.target,
            task_def.arm,
            task_def.action,
            instruction,
            "default",
        )

def _register_l1_tasks() -> None:
    # Scene 1.1.
    _add_task(
        "L1",
        "s11_pick_left_of_green",
        "scene1_1",
        "yellow_block_1",
        None,
        "left",
        "pickup",
        "Use your left arm to pick up the block to the left of the green ball.",
        "scene1.1",
    )
    _add_task(
        "L1",
        "s11_pick_right_of_red",
        "scene1_1",
        "yellow_block_3",
        None,
        "right",
        "pickup",
        "Use your right arm to pick up the block to the right of the red ball.",
        "scene1.1",
    )

    # Scene 1.2.
    _add_task(
        "L1",
        "s12_pick_right_of_red",
        "scene1_2",
        "yellow_block_3",
        None,
        "right",
        "pickup",
        "Use your right arm to pick up the block to the right of the red ball.",
        "scene1.2",
    )
    _add_task(
        "L1",
        "s12_place_right_of_red_behind_red_cylinder",
        "scene1_2",
        "yellow_block_3",
        "red_cylinder",
        "right",
        "place_behind",
        "Use your right arm to pick up the block to the right of the red ball and place it behind the red cylinder.",
        "scene1.2",
    )
    _add_task(
        "L1",
        "s12_place_right_of_red_right_of_red_cylinder",
        "scene1_2",
        "yellow_block_3",
        "red_cylinder",
        "right",
        "place_beside",
        "Use your right arm to pick up the block to the right of the red ball and place it to the right of the red cylinder.",
        "scene1.2",
    )

    # Scene 1.3.
    _add_task(
        "L1",
        "s13_pick_left_of_green",
        "scene1_3",
        "yellow_block_1",
        None,
        "left",
        "pickup",
        "Use your left arm to pick up the block to the left of the green ball.",
        "scene1.3",
    )
    _add_task(
        "L1",
        "s13_place_left_of_green_behind_red_sphere",
        "scene1_3",
        "yellow_block_1",
        "red_sphere",
        "left",
        "place_behind",
        "Use your left arm to pick up the block to the left of the green ball and place it behind the red ball.",
        "scene1.3",
    )
    _add_task(
        "L1",
        "s13_place_left_of_green_left_of_red_sphere",
        "scene1_3",
        "yellow_block_1",
        "red_sphere",
        "left",
        "place_left_of",
        "Use your left arm to pick up the block to the left of the green ball and place it to the left of the red ball.",
        "scene1.3",
    )

def _register_l2_tasks() -> None:
    _add_task(
        "L2",
        "pick_block_left_of_red_cylinder_left",
        "default",
        "yellow_block_1",
        None,
        "left",
        "pickup",
        "Use your left arm to pick up the block to the left of the red cylinder.",
        "default",
    )
    _add_task(
        "L2",
        "pick_block_left_of_red_cylinder_right",
        "default",
        "yellow_block_1",
        None,
        "right",
        "pickup",
        "Use your right arm to pick up the block to the left of the red cylinder.",
        "default",
    )
    _add_task(
        "L2",
        "pick_block_in_front_of_red_sphere_left",
        "default",
        "yellow_block_2",
        None,
        "left",
        "pickup",
        "Use your left arm to pick up the block in front of the red ball.",
        "default",
    )
    _add_task(
        "L2",
        "pick_block_right_of_green_sphere_right",
        "default",
        "yellow_block_3",
        None,
        "right",
        "pickup",
        "Use your right arm to pick up the block to the right of the green ball.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_left_of_red_cylinder_behind_red_cylinder",
        "default",
        "yellow_block_1",
        "red_cylinder",
        "right",
        "place_behind",
        "Use your right arm to pick up the block to the left of the red cylinder and place it behind the red cylinder.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_left_of_red_cylinder_right_of_red_cylinder",
        "default",
        "yellow_block_1",
        "red_cylinder",
        "right",
        "place_beside",
        "Use your right arm to pick up the block to the left of the red cylinder and place it to the right of the red cylinder.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_left_of_red_cylinder_behind_red_sphere",
        "default",
        "yellow_block_1",
        "red_sphere",
        "left",
        "place_behind",
        "Use your left arm to pick up the block to the left of the red cylinder and place it behind the red ball.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_left_of_red_cylinder_left_of_red_sphere",
        "default",
        "yellow_block_1",
        "red_sphere",
        "left",
        "place_left_of",
        "Use your left arm to pick up the block to the left of the red cylinder and place it to the left of the red ball.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_in_front_of_red_sphere_behind_green_sphere",
        "default",
        "yellow_block_2",
        "green_sphere",
        "left",
        "place_behind",
        "Use your left arm to pick up the block in front of the red ball and place it behind the green ball.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_in_front_of_red_sphere_in_front_of_green_sphere",
        "default",
        "yellow_block_2",
        "green_sphere",
        "left",
        "place_in_front",
        "Use your left arm to pick up the block in front of the red ball and place it in front of the green ball.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_in_front_of_red_sphere_behind_red_sphere",
        "default",
        "yellow_block_2",
        "red_sphere",
        "left",
        "place_behind",
        "Use your left arm to pick up the block in front of the red ball and place it behind the red ball.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_in_front_of_red_sphere_left_of_red_sphere",
        "default",
        "yellow_block_2",
        "red_sphere",
        "left",
        "place_left_of",
        "Use your left arm to pick up the block in front of the red ball and place it to the left of the red ball.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_right_of_green_sphere_behind_red_cylinder",
        "default",
        "yellow_block_3",
        "red_cylinder",
        "right",
        "place_behind",
        "Use your right arm to pick up the block to the right of the green ball and place it behind the red cylinder.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_right_of_green_sphere_right_of_red_cylinder",
        "default",
        "yellow_block_3",
        "red_cylinder",
        "right",
        "place_beside",
        "Use your right arm to pick up the block to the right of the green ball and place it to the right of the red cylinder.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_right_of_green_sphere_behind_green_sphere",
        "default",
        "yellow_block_3",
        "green_sphere",
        "right",
        "place_behind",
        "Use your right arm to pick up the block to the right of the green ball and place it behind the green ball.",
        "default",
    )
    _add_task(
        "L2",
        "place_block_right_of_green_sphere_in_front_of_green_sphere",
        "default",
        "yellow_block_3",
        "green_sphere",
        "right",
        "place_in_front",
        "Use your right arm to pick up the block to the right of the green ball and place it in front of the green ball.",
        "default",
    )

def _register_l3_tasks() -> None:
    target_phrases = {
        "red_sphere": "red ball",
        "green_sphere": "green ball",
        "red_cylinder": "red cylinder",
    }
    relation_phrases = {
        "place_behind": "behind",
        "place_in_front": "in front of",
        "place_left_of": "to the left of",
        "place_beside": "to the right of",
    }

    def _render_l3_instruction(action: str, arm: str, source_phrase: str, target: str | None) -> str:
        if action == "pickup":
            return f"Use your {arm} arm to pick up {source_phrase}."
        if target is None:
            raise ValueError(f"target must not be None for action={action}")
        relation = relation_phrases.get(action)
        if relation is None:
            raise ValueError(f"unsupported L3 relation action: {action}")
        target_phrase = target_phrases.get(target, target.replace("_", " "))
        return f"Use your {arm} arm to pick up {source_phrase} and place it {relation} the {target_phrase}."

    l3_source_variants = {
        "scene1.1": {
            "yellow_block_1": [
                ("right_of_red_cylinder", "the block to the right of the red cylinder"),
                ("behind_red_sphere", "the block behind the red ball"),
            ],
            "yellow_block_2": [
                ("left_of_red_sphere", "the block to the left of the red ball"),
            ],
            "yellow_block_3": [
                ("in_front_of_green_sphere", "the block in front of the green ball"),
            ],
        },
        "scene1.2": {
            "yellow_block_1": [
                ("right_of_green_sphere", "the block to the right of the green ball"),
                ("behind_red_sphere", "the block behind the red ball"),
                ("left_of_red_cylinder", "the block to the left of the red cylinder"),
            ],
            "yellow_block_2": [
                ("in_front_of_green_sphere", "the block in front of the green ball"),
                ("left_of_red_sphere", "the block to the left of the red ball"),
            ],
        },
        "scene1.3": {
            "yellow_block_1": [
                ("behind_red_cylinder", "the block behind the red cylinder"),
            ],
            "yellow_block_2": [
                ("in_front_of_red_sphere", "the block in front of the red ball"),
                ("left_of_red_cylinder", "the block to the left of the red cylinder"),
            ],
            "yellow_block_3": [
                ("in_front_of_green_sphere", "the block in front of the green ball"),
                ("right_of_red_cylinder", "the block to the right of the red cylinder"),
            ],
        },
    }

    l3_scene_specs = [
        {
            "scene_code": "s11",
            "scene": "scene1_1",
            "scenario_tag": "scene1.1",
            "actions": [
                ("yellow_block_1", "left", "pickup", None),
                ("yellow_block_1", "right", "pickup", None),
                ("yellow_block_2", "left", "pickup", None),
                ("yellow_block_3", "right", "pickup", None),
                ("yellow_block_1", "right", "place_behind", "green_sphere"),
                ("yellow_block_1", "right", "place_beside", "green_sphere"),
                ("yellow_block_1", "left", "place_behind", "red_cylinder"),
                ("yellow_block_1", "left", "place_left_of", "red_cylinder"),
                ("yellow_block_2", "left", "place_behind", "red_sphere"),
                ("yellow_block_2", "left", "place_in_front", "red_sphere"),
                ("yellow_block_2", "left", "place_behind", "red_cylinder"),
                ("yellow_block_2", "left", "place_left_of", "red_cylinder"),
                ("yellow_block_3", "right", "place_behind", "green_sphere"),
                ("yellow_block_3", "right", "place_beside", "green_sphere"),
                ("yellow_block_3", "right", "place_behind", "red_sphere"),
                ("yellow_block_3", "right", "place_in_front", "red_sphere"),
            ],
        },
        {
            "scene_code": "s12",
            "scene": "scene1_2",
            "scenario_tag": "scene1.2",
            "actions": [
                ("yellow_block_1", "left", "pickup", None),
                ("yellow_block_1", "right", "pickup", None),
                ("yellow_block_2", "left", "pickup", None),
                ("yellow_block_1", "right", "place_behind", "red_cylinder"),
                ("yellow_block_1", "right", "place_beside", "red_cylinder"),
                ("yellow_block_1", "left", "place_behind", "green_sphere"),
                ("yellow_block_1", "left", "place_left_of", "green_sphere"),
                ("yellow_block_2", "left", "place_behind", "red_sphere"),
                ("yellow_block_2", "left", "place_in_front", "red_sphere"),
                ("yellow_block_2", "left", "place_behind", "green_sphere"),
                ("yellow_block_2", "left", "place_left_of", "green_sphere"),
            ],
        },
        {
            "scene_code": "s13",
            "scene": "scene1_3",
            "scenario_tag": "scene1.3",
            "actions": [
                ("yellow_block_1", "left", "pickup", None),
                ("yellow_block_1", "right", "pickup", None),
                ("yellow_block_2", "left", "pickup", None),
                ("yellow_block_3", "right", "pickup", None),
                ("yellow_block_1", "right", "place_behind", "green_sphere"),
                ("yellow_block_1", "right", "place_beside", "green_sphere"),
                ("yellow_block_1", "left", "place_behind", "red_sphere"),
                ("yellow_block_1", "left", "place_left_of", "red_sphere"),
                ("yellow_block_2", "left", "place_behind", "red_cylinder"),
                ("yellow_block_2", "left", "place_in_front", "red_cylinder"),
                ("yellow_block_2", "left", "place_behind", "red_sphere"),
                ("yellow_block_2", "left", "place_left_of", "red_sphere"),
                ("yellow_block_3", "right", "place_behind", "green_sphere"),
                ("yellow_block_3", "right", "place_beside", "green_sphere"),
                ("yellow_block_3", "right", "place_behind", "red_cylinder"),
                ("yellow_block_3", "right", "place_in_front", "red_cylinder"),
            ],
        },
    ]

    for scene_spec in l3_scene_specs:
        scene_code = scene_spec["scene_code"]
        scene_name = scene_spec["scene"]
        scenario_tag = scene_spec["scenario_tag"]
        source_variants = l3_source_variants[scenario_tag]
        for index, (source, arm, action, target) in enumerate(scene_spec["actions"], start=1):
            op_slug = action if target is None else f"{action}_{target}"
            for src_slug, source_phrase in source_variants[source]:
                base_task_type = f"{scene_code}_i{index:02d}_{op_slug}_{arm}__src_{src_slug}"
                instruction = _render_l3_instruction(action, arm, source_phrase, target)
                _add_task(
                    "L3",
                    base_task_type,
                    scene_name,
                    source,
                    target,
                    arm,
                    action,
                    instruction,
                    scenario_tag,
                )

def init_registry() -> None:
    TASKS.clear()
    LEVELS.clear()
    _register_scene_variants()
    _register_l0_tasks()
    _register_l1_tasks()
    _register_l2_tasks()
    _register_l3_tasks()

    global BASE_TASK_TYPES
    BASE_TASK_TYPES = sorted({task.base_task_type for task in TASKS.values()})

def _first_close_event(grips: list[float], close_threshold: float = 0.35) -> int | None:
    if not grips:
        return None
    # Prefer a closing transition; otherwise use the first closed frame.
    for t in range(1, len(grips)):
        if grips[t - 1] > close_threshold and grips[t] <= close_threshold:
            return t
    for t in range(len(grips)):
        if grips[t] <= close_threshold:
            return t
    return None

def _first_open_event_after(
    grips: list[float],
    start_t: int,
    open_threshold: float = 0.7,
) -> int | None:
    if not grips:
        return None
    start_t = max(1, min(start_t + 1, len(grips) - 1))
    # Prefer an opening transition; otherwise use the first open frame.
    for t in range(start_t, len(grips)):
        if grips[t - 1] < open_threshold and grips[t] >= open_threshold:
            return t
    for t in range(start_t, len(grips)):
        if grips[t] >= open_threshold:
            return t
    return None

def _dist_xy(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2])))

def _cos_similarity_2d(v1, v2) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def _relation_vector_from_action(action: str) -> np.ndarray | None:
    if action == "place_left_of":
        return np.array([-1.0, 0.0], dtype=np.float64)
    if action == "place_beside":
        return np.array([1.0, 0.0], dtype=np.float64)
    if action == "place_behind":
        return np.array([0.0, 1.0], dtype=np.float64)
    if action == "place_in_front":
        return np.array([0.0, -1.0], dtype=np.float64)
    return None

def _fingertip_center_from_tcp(env, arm: str, tip_forward_bias: float = TIP_FORWARD_BIAS) -> np.ndarray:
    import transforms3d as t3d
    # Offset the TCP along its local +X axis to approximate the fingertips.
    tcp_pose = env.robot.get_left_tcp_pose() if arm == "left" else env.robot.get_right_tcp_pose()
    tcp_pos = np.asarray(tcp_pose[:3], dtype=np.float64)
    if tip_forward_bias <= 1e-9:
        return tcp_pos

    tcp_quat = np.asarray(tcp_pose[3:7], dtype=np.float64)
    q_norm = float(np.linalg.norm(tcp_quat))
    if q_norm < 1e-8:
        return tcp_pos
    tcp_quat = tcp_quat / q_norm

    rot = t3d.quaternions.quat2mat(tcp_quat)
    return tcp_pos + rot @ np.array([tip_forward_bias, 0.0, 0.0], dtype=np.float64)

class Scene1TrajectoryRecorder(base.TrajectoryRecorder):
    """Record trajectories using a fingertip position proxy."""
    def __init__(self, tip_forward_bias: float = TIP_FORWARD_BIAS):
        super().__init__()
        self.tip_forward_bias = float(tip_forward_bias)

    def record(self, env, obs):
        for arm in ("left", "right"):
            self.traj.ee[arm].append(_fingertip_center_from_tcp(env, arm, self.tip_forward_bias))

        sv = obs["joint_action"]["vector"]
        self.traj.grip["left"].append(sv[6])
        self.traj.grip["right"].append(sv[13])

        for name in list(env.object_dict) + list(env.label_dict) + list(env.bowl_dict):
            pos = env.get_object_position(name)
            if pos is not None:
                self.traj.obj_pos.setdefault(name, []).append(pos.copy())

        self.traj.labels = set(env.label_dict.keys())
        self.traj.bowls = set(env.bowl_dict.keys())
        self.traj.n_steps = len(self.traj.ee["left"])

class Scene1ThreeStageEvaluator:
    """Score source selection, relative placement, and finish behavior."""

    def __init__(
        self,
        task: Scene1EvalTask,
        traj: base.Trajectory,
    ):
        self.task = task
        self.traj = traj
        self.other_arm = "right" if task.arm == "left" else "left"

        # Measure grasp proximity from the fingertip proxy.
        self.pick_xy_threshold = 0.07
        self.return_table_tol = 0.02
        self.lift_threshold = 0.01
        self.place_dist_threshold = 0.12
        self.relation_cos_threshold = float(np.cos(np.deg2rad(45.0)))
        # Finish checks prioritize returning home and stopping.
        self.reset_dist_threshold = 0.12
        self.home_stable_steps = 20
        self.return_steps_limit = 180
        self.leave_home_threshold = 0.16
        self.grip_close_threshold = 0.35
        self.wrong_arm_intervention_penalty = WRONG_ARM_INTERVENTION_PENALTY
        self._wrong_arm_intervened = False

    def wrong_arm_penalty(self) -> float:
        return self.wrong_arm_intervention_penalty if self._wrong_arm_intervened else 0.0

    def _mark_wrong_arm_participation(
        self,
        ctx: dict,
        close_other: int | None,
        first_close_arm: str | None,
    ) -> bool:
        wrong_arm_participated = close_other is not None or first_close_arm != self.task.arm
        ctx["wrong_arm_participated"] = wrong_arm_participated
        self._wrong_arm_intervened = wrong_arm_participated
        return wrong_arm_participated

    def evaluate(self) -> list[PhaseResult]:
        self._wrong_arm_intervened = False
        # Pickup weights: 0.8/0.2; placement weights: 0.4/0.4/0.2.
        pickup_only = self.task.action == "pickup"
        stage1_max = 0.8 if pickup_only else 0.4
        stage1, ctx = self._evaluate_stage1(stage1_max=stage1_max)

        if pickup_only:
            stage3 = self._evaluate_stage3(ctx)
            return self._apply_gates_pickup(stage1, stage3)

        stage2 = self._evaluate_stage2(ctx)
        stage3 = self._evaluate_stage3(ctx)
        return self._apply_gates(stage1, stage2, stage3)

    def _evaluate_stage1(self, stage1_max: float = 0.4) -> tuple[PhaseResult, dict]:
        ctx = {
            "close_t_expected": None,
            "open_t_expected": None,
            "first_close_arm": None,
            "wrong_arm_closed": False,
            "wrong_arm_participated": False,
            "stage1_intent_ok": False,
            "source_lifted": False,
            "stage2_intent_t": None,
        }
        n = self.traj.n_steps
        if n <= 0 or self.task.source not in self.traj.obj_pos:
            return (
                PhaseResult(
                    "stage1_source_select",
                    0.0,
                    "trajectory/source missing",
                    0.0,
                    "trajectory/source missing",
                    stage1_max,
                ),
                ctx,
            )

        close_expected = _first_close_event(self.traj.grip[self.task.arm])
        close_other = _first_close_event(self.traj.grip[self.other_arm])
        ctx["close_t_expected"] = close_expected
        ctx["wrong_arm_closed"] = close_other is not None

        first_close_arm = None
        if close_expected is None and close_other is None:
            first_close_arm = None
        elif close_other is None or (
            close_expected is not None and close_expected <= close_other
        ):
            first_close_arm = self.task.arm
        else:
            first_close_arm = self.other_arm
        ctx["first_close_arm"] = first_close_arm
        wrong_arm_participated = self._mark_wrong_arm_participation(
            ctx,
            close_other,
            first_close_arm,
        )

        if close_expected is None:
            return (
                PhaseResult(
                    "stage1_source_select",
                    0.0,
                    "operating arm never closed gripper",
                    0.0,
                    "operating arm never closed gripper",
                    stage1_max,
                ),
                ctx,
            )

        t = close_expected
        # Judge source selection at the first close of the designated gripper.
        ee = self.traj.ee[self.task.arm][t]
        src = self.traj.obj_pos[self.task.source][t]
        src_xy = _dist_xy(ee, src)

        names = [
            name
            for name in self.traj.obj_pos
            if name not in self.traj.labels and name not in self.traj.bowls
        ]
        min_name = None
        min_dist = None
        for name in names:
            d = _dist_xy(ee, self.traj.obj_pos[name][t])
            if min_dist is None or d < min_dist:
                min_dist = d
                min_name = name
        closest_ok = min_name == self.task.source

        intent_ok = closest_ok and (src_xy < self.pick_xy_threshold)
        ctx["stage1_intent_ok"] = intent_ok

        z0 = float(self.traj.obj_pos[self.task.source][0][2])
        max_lift = max(float(p[2]) for p in self.traj.obj_pos[self.task.source]) - z0
        source_lifted = max_lift >= self.lift_threshold
        ctx["source_lifted"] = source_lifted

        exec_ok = intent_ok and source_lifted

        intent_reason = "ok" if intent_ok else (
            f"at first close: closest={min_name}, source_xy={src_xy*100:.1f}cm"
        )
        exec_reason = "ok" if exec_ok else (
            f"source lift={max_lift*100:.1f}cm (need >= {self.lift_threshold*100:.1f}cm)"
            if intent_ok
            else "intent not correct at first close"
        )
        if wrong_arm_participated:
            intent_reason = f"{intent_reason}; wrong arm participated"
            exec_reason = f"{exec_reason}; wrong arm participated"

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

    def _relation_ok_at(self, t: int) -> tuple[bool, str]:
        if self.task.target is None:
            return False, "target missing"
        if self.task.source not in self.traj.obj_pos or self.task.target not in self.traj.obj_pos:
            return False, "source/target trajectory missing"

        desired = _relation_vector_from_action(self.task.action)
        if desired is None:
            return False, f"unsupported relation action={self.task.action}"

        source = self.traj.obj_pos[self.task.source][t]
        target = self.traj.obj_pos[self.task.target][t]
        vec = source[:2] - target[:2]
        dist = float(np.linalg.norm(vec))
        if dist > self.place_dist_threshold:
            return False, f"distance={dist*100:.1f}cm > {self.place_dist_threshold*100:.0f}cm"
        if dist < 1e-6:
            return False, "source/target overlap"

        cos = _cos_similarity_2d(vec, desired)
        if cos < self.relation_cos_threshold:
            return False, f"relation angle mismatch (cos={cos:.3f})"
        return True, "ok"

    def _on_table(self, t: int) -> bool:
        # Use the initial source height as the table reference.
        z0 = float(self.traj.obj_pos[self.task.source][0][2])
        z = float(self.traj.obj_pos[self.task.source][t][2])
        return abs(z - z0) <= self.return_table_tol

    def _evaluate_stage2(self, ctx: dict) -> PhaseResult:
        close_t = ctx["close_t_expected"]
        open_t = None if close_t is None else _first_open_event_after(
            self.traj.grip[self.task.arm], close_t
        )
        ctx["open_t_expected"] = open_t

        if self.task.target is None:
            return PhaseResult(
                "stage2_relative_place",
                0.4,
                "ok (n/a: no target)",
                0.4,
                "ok (n/a: no target)",
                0.4,
            )

        if close_t is None:
            return PhaseResult(
                "stage2_relative_place",
                0.0,
                "no grasp close event",
                0.0,
                "no grasp close event",
                0.4,
            )

        lifted = False
        intent_ok = False
        intent_reason = "source never returned to table in correct relative pose"
        # Require a lift before accepting a valid placement.
        for t in range(close_t + 1, self.traj.n_steps):
            z0 = float(self.traj.obj_pos[self.task.source][0][2])
            z = float(self.traj.obj_pos[self.task.source][t][2])
            if z - z0 >= self.lift_threshold:
                lifted = True
            if not lifted:
                continue
            if not self._on_table(t):
                continue
            rel_ok, rel_reason = self._relation_ok_at(t)
            if rel_ok:
                intent_ok = True
                ctx["stage2_intent_t"] = t
                intent_reason = "ok"
                break
            intent_reason = rel_reason

        exec_ok = False
        exec_reason = "gripper never opened in correct relative pose"
        if intent_ok and open_t is not None:
            # Execution requires correct placement at release.
            lifted_before_open = False
            z0 = float(self.traj.obj_pos[self.task.source][0][2])
            for t in range(close_t, min(open_t + 1, self.traj.n_steps)):
                if float(self.traj.obj_pos[self.task.source][t][2]) - z0 >= self.lift_threshold:
                    lifted_before_open = True
                    break
            rel_ok, rel_reason = self._relation_ok_at(open_t)
            if lifted_before_open and self._on_table(open_t) and rel_ok:
                exec_ok = True
                exec_reason = "ok"
            else:
                exec_reason = (
                    rel_reason
                    if not rel_ok
                    else "source not on table when gripper opened"
                )
        elif intent_ok and open_t is None:
            exec_reason = "no gripper-open event after close"

        return PhaseResult(
            "stage2_relative_place",
            0.4 if intent_ok else 0.0,
            intent_reason,
            0.4 if exec_ok else 0.0,
            exec_reason,
            0.4,
        )

    def _completion_t(self, ctx: dict) -> int:
        if self.task.target is not None and self.task.action in RELATION_ACTIONS:
            if ctx.get("open_t_expected") is not None:
                return int(ctx["open_t_expected"])
            if ctx.get("stage2_intent_t") is not None:
                return int(ctx["stage2_intent_t"])
        z0 = float(self.traj.obj_pos[self.task.source][0][2])
        start = ctx["close_t_expected"] if ctx["close_t_expected"] is not None else 0
        for t in range(start, self.traj.n_steps):
            if float(self.traj.obj_pos[self.task.source][t][2]) - z0 >= self.lift_threshold:
                return t
        return max(0, self.traj.n_steps - 1)

    def _evaluate_stage3(self, ctx: dict) -> PhaseResult:
        if self.traj.n_steps <= 0:
            return PhaseResult(
                "stage3_finish",
                0.0,
                "empty trajectory",
                0.0,
                "empty trajectory",
                0.2,
            )

        completion_t = self._completion_t(ctx)
        problems = []

        if any(base.in_operation_area(p) for p in self.traj.ee[self.other_arm]):
            problems.append("non-operating arm entered operation area")

        home = np.asarray(self.traj.ee[self.task.arm][0])
        arm_traj = self.traj.ee[self.task.arm]
        n = len(arm_traj)
        # Require consecutive frames near the initial end-effector position.
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
                    problems.append(
                        f"{self.task.arm} arm left home again (dist={d*100:.1f}cm)"
                    )
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

    def _apply_gates(
        self,
        stage1: PhaseResult,
        stage2: PhaseResult,
        stage3: PhaseResult,
    ) -> list[PhaseResult]:
        eps = 1e-6
        s1 = copy.deepcopy(stage1)
        s2 = copy.deepcopy(stage2)
        s3 = copy.deepcopy(stage3)

        # Later stages require full credit on their prerequisites.
        if s1.intent_score + eps < s1.max_score:
            s2.intent_score = 0.0
            s2.intent_reason = "gated: stage1 intent not full"
        if s1.intent_score + eps < s1.max_score or s2.intent_score + eps < s2.max_score:
            s3.intent_score = 0.0
            s3.intent_reason = "gated: prior intent not full"

        # Execution also requires intent credit.
        if not (
            s1.intent_score + eps >= s1.max_score and
            s1.exec_score + eps >= s1.max_score and
            s2.intent_score + eps >= s2.max_score
        ):
            s2.exec_score = 0.0
            s2.exec_reason = "gated: stage1 intent/exec or stage2 intent incomplete"

        if not (
            s1.intent_score + eps >= s1.max_score and
            s1.exec_score + eps >= s1.max_score and
            s2.intent_score + eps >= s2.max_score and
            s2.exec_score + eps >= s2.max_score and
            s3.intent_score + eps >= s3.max_score
        ):
            s3.exec_score = 0.0
            s3.exec_reason = "gated: prior intent/exec not full"
        return [s1, s2, s3]

    def _apply_gates_pickup(
        self,
        stage1: PhaseResult,
        stage3: PhaseResult,
    ) -> list[PhaseResult]:
        eps = 1e-6
        s1 = copy.deepcopy(stage1)
        s3 = copy.deepcopy(stage3)

        if s1.intent_score + eps < s1.max_score:
            s3.intent_score = 0.0
            s3.intent_reason = "gated: stage1 intent not full"

        if not (
            s1.intent_score + eps >= s1.max_score and
            s1.exec_score + eps >= s1.max_score and
            s3.intent_score + eps >= s3.max_score
        ):
            s3.exec_score = 0.0
            s3.exec_reason = "gated: stage1 intent/exec or stage3 intent incomplete"

        return [s1, s3]

def run_once(
    env,
    policy,
    task: Scene1EvalTask,
    max_steps: int = 10,
    actions_per_step: int = 50,
    direct_step: bool = False,
    sim_steps: int = 15,
) -> dict:
    """Run one evaluation episode and return scores and video frames."""
    recorder = Scene1TrajectoryRecorder(tip_forward_bias=TIP_FORWARD_BIAS)
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
    evaluator = Scene1ThreeStageEvaluator(task, traj)
    phases = evaluator.evaluate()

    intent_score = sum(p.intent_score for p in phases)
    exec_score = sum(p.exec_score for p in phases)
    max_total = sum(p.max_score for p in phases)

    # Give partial credit for a stable final placement when stage scores are low.
    final_target_ok = False
    final_target_reason = "n/a (no relation target)"
    final_target_window = 10
    if (
        traj.n_steps > 0
        and task.target is not None
        and task.action in RELATION_ACTIONS
    ):
        if traj.n_steps < final_target_window:
            final_target_reason = (
                f"trajectory too short for final {final_target_window}-frame check"
            )
        else:
            start_t = traj.n_steps - final_target_window
            fail_reason = None
            fail_t = None
            for t in range(start_t, traj.n_steps):
                rel_ok, rel_reason = evaluator._relation_ok_at(t)
                if not rel_ok:
                    fail_t = t
                    fail_reason = rel_reason
                    break
                if not evaluator._on_table(t):
                    fail_t = t
                    fail_reason = "source not on table"
                    break

            final_target_ok = fail_reason is None
            if final_target_ok:
                final_target_reason = f"ok (last {final_target_window} frames)"
            else:
                final_target_reason = f"final-window failed at t={fail_t}: {fail_reason}"

    intent_bonus = 0.0
    exec_bonus = 0.0
    if final_target_ok:
        if intent_score < 0.5:
            intent_bonus = 0.3
            intent_score += intent_bonus
        if exec_score < 0.5:
            exec_bonus = 0.3
            exec_score += exec_bonus

    wrong_arm_intervened = evaluator.wrong_arm_penalty() > 0.0
    wrong_arm_intent_penalty = 0.0
    wrong_arm_exec_penalty = 0.0
    if wrong_arm_intervened and phases:
        tail = phases[-1]
        eps = 1e-6
        # Apply the wrong-arm penalty only when the finish stage has full credit.
        if tail.intent_score + eps >= tail.max_score:
            wrong_arm_intent_penalty = evaluator.wrong_arm_penalty()
            intent_score = max(0.0, intent_score - wrong_arm_intent_penalty)
        if tail.exec_score + eps >= tail.max_score:
            wrong_arm_exec_penalty = evaluator.wrong_arm_penalty()
            exec_score = max(0.0, exec_score - wrong_arm_exec_penalty)

    # Allow a 1% tolerance for full-credit classification.
    intent_ok = intent_score >= max_total * 0.99
    exec_ok = exec_score >= max_total * 0.99
    intent_msg = "ok" if intent_ok else f"intent_score={intent_score:.2f}/{max_total:.2f}"
    exec_msg = "ok" if exec_ok else f"exec_score={exec_score:.2f}/{max_total:.2f}"
    if wrong_arm_intent_penalty > 0.0:
        intent_msg += f" (-{wrong_arm_intent_penalty:.1f} wrong-arm penalty)"
    if wrong_arm_exec_penalty > 0.0:
        exec_msg += f" (-{wrong_arm_exec_penalty:.1f} wrong-arm penalty)"

    return dict(
        task=task.name,
        instruction=task.instruction,
        intent_ok=intent_ok,
        intent_msg=intent_msg,
        exec_ok=exec_ok,
        completion_ok=base.is_complete(phases, intent_score, exec_score,
                                       action=task.action, final_target_ok=final_target_ok),
        exec_msg=exec_msg,
        intent_score=float(intent_score),
        exec_score=float(exec_score),
        final_target_ok=final_target_ok,
        final_target_reason=final_target_reason,
        final_target_window=final_target_window,
        intent_bonus=float(intent_bonus),
        exec_bonus=float(exec_bonus),
        wrong_arm_intervened=wrong_arm_intervened,
        wrong_arm_intent_penalty=float(wrong_arm_intent_penalty),
        wrong_arm_exec_penalty=float(wrong_arm_exec_penalty),
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
