from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from . import common as base
from robofollow.tasks.scene2 import SCENES as SCENE2_SCENES, TASKS as SCENE2_TASKS

WRONG_ARM_INTERVENTION_PENALTY = 0.2

LEVEL_ORDER = ("L0", "L1", "L2", "L3")

@dataclass
class Scene2EvalTask:
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

TASKS: dict[str, Scene2EvalTask] = {}

LEVELS: dict[str, list[str]] = {}

SCENE_VARIANTS: dict[str, object] = {}

SCENE_VARIANT_NOTES: dict[str, str] = {}

def _stable_hash32(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)

def _load_collect_scene2_templates():
    return json.loads((Path(__file__).resolve().parents[1] / "tasks" / "scene2_instructions.json").read_text())

INSTRUCTION_TEMPLATES = _load_collect_scene2_templates()

BASE_TASK_TYPES = list(SCENE2_TASKS.keys())

TRAIN_REFERENCES = {
    "large_red_cylinder": "larger red object",
    "large_blue_cube": "leftmost cube",
    "small_red_cube": "rightmost cube",
    "small_blue_cylinder": "smaller blue object",
    "green_bowl": "green bowl",
    "yellow_bowl": "yellow bowl",
}

L2_REFERENCES: dict[str, list[tuple[str, str]]] = {
    "large_red_cylinder": [
        ("left_cylinder", "left cylinder"),
        ("larger_cylinder", "larger cylinder"),
    ],
    "large_blue_cube": [
        ("larger_blue_object", "larger blue object"),
        ("blue_cube", "blue cube"),
        ("larger_cube", "larger cube"),
    ],
    "small_red_cube": [
        ("smaller_red_object", "smaller red object"),
        ("red_cube", "red cube"),
        ("smaller_cube", "smaller cube"),
    ],
    "small_blue_cylinder": [
        ("right_cylinder", "right cylinder"),
        ("smaller_cylinder", "smaller cylinder"),
    ],
    "green_bowl": [
        ("left_bowl", "left bowl"),
    ],
    "yellow_bowl": [
        ("right_bowl", "right bowl"),
    ],
}

def _register_task(task: Scene2EvalTask) -> None:
    TASKS[task.name] = task
    LEVELS.setdefault(task.level_tag, []).append(task.name)

def _with_the(phrase: str | None) -> str:
    if phrase is None:
        return ""
    p = phrase.strip()
    if p.lower().startswith("the "):
        return p
    return f"the {p}"

def _render_instruction(action: str, arm: str, source_phrase: str, target_phrase: str | None) -> str:
    src = _with_the(source_phrase)
    tgt = _with_the(target_phrase)
    if action == "pickup":
        return f"Use your {arm} arm to pick up {src}."
    if action == "stack":
        return f"Use your {arm} arm to stack {src} onto {tgt}."
    if action == "push":
        return f"Use your {arm} arm to push {src} towards {tgt}."
    if action == "place_bowl":
        return f"Use your {arm} arm to pick up {src} and place it in {tgt}."
    return f"Use your {arm} arm to perform {action} with {src}."

def _source_phrase(obj_name: str) -> str:
    return TRAIN_REFERENCES.get(obj_name, obj_name)

def _target_phrase(obj_name: str | None) -> str | None:
    if obj_name is None:
        return None
    return TRAIN_REFERENCES.get(obj_name, obj_name)

def _swap_positions(items: list[object], name_a: str, name_b: str) -> None:
    by_name = {item.name: item for item in items}
    if name_a not in by_name or name_b not in by_name:
        return
    pos_a = by_name[name_a].position
    pos_b = by_name[name_b].position
    by_name[name_a].position = pos_b
    by_name[name_b].position = pos_a

def _register_scene_variants() -> None:
    SCENE_VARIANTS.clear()
    SCENE_VARIANT_NOTES.clear()

    base_cfg = copy.deepcopy(SCENE2_SCENES["default"])
    base_cfg.name = "default"
    SCENE_VARIANTS["default"] = base_cfg
    SCENE_VARIANT_NOTES["default"] = "training layout"

    def add_variant(name: str, object_swaps: list[tuple[str, str]], actor_swaps: list[tuple[str, str]], note: str):
        cfg = copy.deepcopy(SCENE2_SCENES["default"])
        cfg.name = name
        for a, b in object_swaps:
            _swap_positions(cfg.objects, a, b)
        for a, b in actor_swaps:
            _swap_positions(cfg.actors, a, b)
        SCENE_VARIANTS[name] = cfg
        SCENE_VARIANT_NOTES[name] = note

    add_variant(
        "scene2.1",
        object_swaps=[("large_red_cylinder", "large_blue_cube")],
        actor_swaps=[],
        note="swap large_red_cylinder <-> large_blue_cube",
    )
    add_variant(
        "scene2.2",
        object_swaps=[("large_blue_cube", "small_blue_cylinder")],
        actor_swaps=[],
        note="swap large_blue_cube <-> small_blue_cylinder",
    )
    add_variant(
        "scene2.3",
        object_swaps=[("small_red_cube", "small_blue_cylinder")],
        actor_swaps=[],
        note="swap small_red_cube <-> small_blue_cylinder",
    )
    add_variant(
        "scene2.4",
        object_swaps=[("large_red_cylinder", "small_red_cube")],
        actor_swaps=[],
        note="swap large_red_cylinder <-> small_red_cube",
    )
    add_variant(
        "scene2.5",
        object_swaps=[],
        actor_swaps=[("green_bowl", "yellow_bowl")],
        note="swap green_bowl <-> yellow_bowl",
    )
    add_variant(
        "scene2.6",
        object_swaps=[("small_red_cube", "small_blue_cylinder")],
        actor_swaps=[("green_bowl", "yellow_bowl")],
        note="swap small_red_cube <-> small_blue_cylinder and green_bowl <-> yellow_bowl",
    )

def _add_task(
    level_tag: str,
    name_suffix: str,
    base_task_type: str,
    scene: str,
    instruction: str,
    instruction_variant: str,
    scenario_tag: str,
) -> None:
    task_def = SCENE2_TASKS[base_task_type]
    _register_task(
        Scene2EvalTask(
            name=f"{level_tag}_{name_suffix}",
            scene=scene,
            source=task_def.source,
            target=task_def.target,
            arm=task_def.arm,
            action=task_def.action,
            instruction=instruction,
            level_tag=level_tag,
            base_task_type=base_task_type,
            instruction_variant=instruction_variant,
            scenario_tag=scenario_tag,
        )
    )

def _register_l0_tasks() -> None:
    for task_type, task_def in SCENE2_TASKS.items():
        templates = INSTRUCTION_TEMPLATES.get(task_type, [])
        if templates:
            instruction = templates[0]
            variant = "template_0"
        else:
            instruction = _render_instruction(
                task_def.action,
                task_def.arm,
                _source_phrase(task_def.source),
                _target_phrase(task_def.target),
            )
            variant = "fallback_template_0"

        _add_task(
            "L0",
            task_type,
            task_type,
            "default",
            instruction,
            variant,
            "default",
        )

def _register_l1_tasks() -> None:
    specs = [
        # Scene 2.1.
        ("scene2.1", "pickup_red_cylinder", "larger_red_object", "larger red object", None),
        ("scene2.1", "pickup_red_cylinder", "red_cylinder", "red cylinder", None),
        ("scene2.1", "pickup_left_cube", "leftmost_cube", "leftmost cube", None),
        (
            "scene2.1",
            "push_left_cube_to_red_cylinder",
            "leftmost_cube_to_larger_red_object",
            "leftmost cube",
            "larger red object",
        ),
        (
            "scene2.1",
            "push_red_cylinder_to_left_cube",
            "red_cylinder_to_leftmost_cube",
            "red cylinder",
            "leftmost cube",
        ),
        # Scene 2.2.
        ("scene2.2", "pickup_blue_cylinder", "smaller_blue_object", "smaller blue object", None),
        ("scene2.2", "pickup_blue_cylinder", "blue_cylinder", "blue cylinder", None),
        # Scene 2.3.
        ("scene2.3", "pickup_right_cube", "rightmost_cube", "rightmost cube", None),
        ("scene2.3", "pickup_blue_cylinder", "smaller_blue_object", "smaller blue object", None),
        ("scene2.3", "pickup_blue_cylinder", "blue_cylinder", "blue cylinder", None),
        (
            "scene2.3",
            "place_right_cube_in_green_bowl",
            "rightmost_cube_to_green_bowl",
            "rightmost cube",
            "green bowl",
        ),
        (
            "scene2.3",
            "place_right_cube_in_yellow_bowl",
            "rightmost_cube_to_yellow_bowl",
            "rightmost cube",
            "yellow bowl",
        ),
        (
            "scene2.3",
            "place_blue_cylinder_in_green_bowl",
            "smaller_blue_object_to_green_bowl",
            "smaller blue object",
            "green bowl",
        ),
        (
            "scene2.3",
            "place_blue_cylinder_in_yellow_bowl",
            "smaller_blue_object_to_yellow_bowl",
            "smaller blue object",
            "yellow bowl",
        ),
        # Scene 2.4.
        ("scene2.4", "pickup_red_cylinder", "larger_red_object", "larger red object", None),
        ("scene2.4", "pickup_red_cylinder", "red_cylinder", "red cylinder", None),
        # Scene 2.5.
        (
            "scene2.5",
            "place_right_cube_in_green_bowl",
            "rightmost_cube_to_green_bowl",
            "rightmost cube",
            "green bowl",
        ),
        (
            "scene2.5",
            "place_right_cube_in_yellow_bowl",
            "rightmost_cube_to_yellow_bowl",
            "rightmost cube",
            "yellow bowl",
        ),
        (
            "scene2.5",
            "place_blue_cylinder_in_green_bowl",
            "smaller_blue_object_to_green_bowl",
            "smaller blue object",
            "green bowl",
        ),
        (
            "scene2.5",
            "place_blue_cylinder_in_yellow_bowl",
            "smaller_blue_object_to_yellow_bowl",
            "smaller blue object",
            "yellow bowl",
        ),
        # Scene 2.6.
        (
            "scene2.6",
            "place_right_cube_in_green_bowl",
            "rightmost_cube_to_green_bowl",
            "rightmost cube",
            "green bowl",
        ),
        (
            "scene2.6",
            "place_right_cube_in_yellow_bowl",
            "rightmost_cube_to_yellow_bowl",
            "rightmost cube",
            "yellow bowl",
        ),
        (
            "scene2.6",
            "place_blue_cylinder_in_green_bowl",
            "smaller_blue_object_to_green_bowl",
            "smaller blue object",
            "green bowl",
        ),
        (
            "scene2.6",
            "place_blue_cylinder_in_yellow_bowl",
            "smaller_blue_object_to_yellow_bowl",
            "smaller blue object",
            "yellow bowl",
        ),
    ]

    per_scene_idx: dict[str, int] = {}
    for scenario_tag, task_type, variant, src_phrase, tgt_phrase in specs:
        task_def = SCENE2_TASKS[task_type]
        instruction = _render_instruction(task_def.action, task_def.arm, src_phrase, tgt_phrase)
        if task_type == "pickup_blue_cylinder" and variant == "smaller_blue_object":
            instruction = INSTRUCTION_TEMPLATES[task_type][1]
        per_scene_idx[scenario_tag] = per_scene_idx.get(scenario_tag, 0) + 1
        index = per_scene_idx[scenario_tag]
        suffix = f"{scenario_tag.replace('.', '_')}_{index:02d}_{task_type}_{variant}"
        _add_task("L1", suffix, task_type, scenario_tag, instruction, variant, scenario_tag)

def _register_l2_tasks() -> None:
    for task_type, task_def in SCENE2_TASKS.items():
        source_variants = L2_REFERENCES.get(task_def.source)
        if not source_variants:
            fallback_source = _source_phrase(task_def.source)
            source_variants = [(fallback_source.replace(" ", "_"), fallback_source)]

        if task_def.target is None:
            target_variants: list[tuple[str | None, str | None]] = [(None, None)]
        else:
            target_pool = L2_REFERENCES.get(task_def.target)
            if target_pool:
                target_variants = [(slug, phrase) for slug, phrase in target_pool]
            else:
                fallback_target = _target_phrase(task_def.target) or task_def.target
                target_variants = [(fallback_target.replace(" ", "_"), fallback_target)]

        for src_slug, src_phrase in source_variants:
            for tgt_slug, tgt_phrase in target_variants:
                instruction = _render_instruction(task_def.action, task_def.arm, src_phrase, tgt_phrase)
                variant = f"src_{src_slug}"
                if tgt_slug is not None:
                    variant += f"__tgt_{tgt_slug}"
                suffix = f"{task_type}__{variant}"
                _add_task("L2", suffix, task_type, "default", instruction, variant, "default")

def _pick_l3_reference(task_name: str, obj_name: str) -> tuple[str, str]:
    variants = L2_REFERENCES.get(obj_name)
    if not variants:
        fallback = TRAIN_REFERENCES.get(obj_name, obj_name)
        return fallback.replace(" ", "_"), fallback
    idx = _stable_hash32(f"{task_name}:{obj_name}:l3_ref") % len(variants)
    return variants[idx]

def _register_l3_tasks() -> None:
    for l1_task_name in LEVELS.get("L1", []):
        l1_task = TASKS[l1_task_name]

        src_slug, src_phrase = _pick_l3_reference(l1_task_name, l1_task.source)
        tgt_slug = None
        tgt_phrase = None
        if l1_task.target is not None:
            tgt_slug, tgt_phrase = _pick_l3_reference(l1_task_name, l1_task.target)

        instruction = _render_instruction(
            l1_task.action,
            l1_task.arm,
            src_phrase,
            tgt_phrase,
        )

        variant = f"stable_src_{src_slug}"
        if tgt_slug is not None:
            variant += f"__tgt_{tgt_slug}"

        suffix = l1_task_name[len("L1_") :] if l1_task_name.startswith("L1_") else l1_task_name
        _register_task(
            Scene2EvalTask(
                name=f"L3_{suffix}",
                scene=l1_task.scene,
                source=l1_task.source,
                target=l1_task.target,
                arm=l1_task.arm,
                action=l1_task.action,
                instruction=instruction,
                level_tag="L3",
                base_task_type=l1_task.base_task_type,
                instruction_variant=variant,
                scenario_tag=l1_task.scenario_tag,
            )
        )

def init_registry(_l3_template_index: int) -> None:
    TASKS.clear()
    LEVELS.clear()
    _register_scene_variants()
    _register_l0_tasks()
    _register_l1_tasks()
    _register_l2_tasks()
    _register_l3_tasks()

def _first_close_event(grips: list[float], close_threshold: float = 0.35) -> int | None:
    if not grips:
        return None
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

class Scene2ThreeStageEvaluator:
    """Score source selection, action completion, and finish behavior."""

    def __init__(
        self,
        task: Scene2EvalTask,
        traj: base.Trajectory,
    ):
        self.task = task
        self.traj = traj
        self.other_arm = "right" if task.arm == "left" else "left"
        # The gripper contact point is about 0.12 m below the end-effector center.
        self.ee_contact_z_offset = 0.12

        # Shared thresholds.
        self.pick_xy_threshold = 0.07
        self.lift_threshold = 0.01
        self.pick_exec_lift_threshold = 0.03
        self.return_table_tol = 0.02

        # Stacking thresholds.
        self.stack_ee_src_xy_threshold = 0.07
        self.stack_src_tgt_xy_threshold = 0.03
        self.stack_above_margin = 0.005
        self.stack_drop_margin = 0.002

        # Pushing thresholds.
        self.push_close_threshold = 0.25
        self.push_intent_xy_threshold = 0.07
        self.push_exec_xy_threshold = 0.08
        self.push_ee_near_table_tol = 0.12
        self.push_stage1_min_move = 0.005
        self.push_source_table_z_tol = self.return_table_tol
        self.push_dir_cos_threshold = float(np.cos(np.deg2rad(45.0)))
        self.push_intent_move = 0.01
        self.push_exec_move = 0.02
        self.push_bonus_move_threshold = self.push_stage1_min_move
        # Require contact on the back side of the source, with no extra margin.
        self.push_backside_proj_threshold = 0.0

        # Bowl placement thresholds.
        self.place_descend_height_threshold = 0.06
        self.place_xy_threshold = 0.04
        self.place_bowl_z_cap = 0.06

        # Retry bonus.
        self.retry_bonus_score_threshold = 0.5
        self.retry_bonus_delta = 0.3

        # Finish thresholds.
        self.reset_dist_threshold = 0.10
        self.home_stable_steps = 20
        self.return_steps_limit = 180
        self.leave_home_threshold = 0.14
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
        pickup_only = self.task.action == "pickup"
        stage1_max = 0.8 if pickup_only else 0.4
        stage1, ctx = self._evaluate_stage1(stage1_max=stage1_max)

        if pickup_only:
            stage3 = self._evaluate_stage3(ctx)
            return self._apply_gates_pickup(stage1, stage3)

        stage2 = self._evaluate_stage2(ctx)
        stage3 = self._evaluate_stage3(ctx)
        return self._apply_gates(stage1, stage2, stage3)

    def _source_object_names(self) -> list[str]:
        return [
            name
            for name in self.traj.obj_pos
            if name not in self.traj.labels and name not in self.traj.bowls
        ]

    def _evaluate_stage1(self, stage1_max: float = 0.4) -> tuple[PhaseResult, dict]:
        if self.task.action == "push":
            return self._evaluate_stage1_push(stage1_max)
        return self._evaluate_stage1_grasp(stage1_max)

    def _evaluate_stage1_grasp(self, stage1_max: float = 0.4) -> tuple[PhaseResult, dict]:
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

        if self.traj.n_steps <= 0 or self.task.source not in self.traj.obj_pos:
            reason = "trajectory/source missing"
            return (
                PhaseResult("stage1_source_select", 0.0, reason, 0.0, reason, stage1_max),
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
            reason = "operating arm never closed gripper"
            return (
                PhaseResult("stage1_source_select", 0.0, reason, 0.0, reason, stage1_max),
                ctx,
            )

        t = close_expected
        ee = self.traj.ee[self.task.arm][t]
        src = self.traj.obj_pos[self.task.source][t]
        src_xy = _dist_xy(ee, src)

        min_name = None
        min_dist = None
        for name in self._source_object_names():
            d = _dist_xy(ee, self.traj.obj_pos[name][t])
            if min_dist is None or d < min_dist:
                min_dist = d
                min_name = name
        closest_ok = min_name == self.task.source

        intent_ok = closest_ok and (src_xy < self.pick_xy_threshold)
        ctx["stage1_intent_ok"] = intent_ok

        z0 = float(self.traj.obj_pos[self.task.source][0][2])
        max_lift = max(float(p[2]) for p in self.traj.obj_pos[self.task.source]) - z0
        required_lift = (
            self.pick_exec_lift_threshold
            if self.task.action == "pickup"
            else self.lift_threshold
        )
        source_lifted = max_lift >= required_lift
        ctx["source_lifted"] = source_lifted

        exec_ok = intent_ok and source_lifted

        intent_reason = "ok" if intent_ok else (
            f"at first close: closest={min_name}, source_xy={src_xy*100:.1f}cm"
        )
        exec_reason = "ok" if exec_ok else (
            f"source lift={max_lift*100:.1f}cm (need >= {required_lift*100:.1f}cm)"
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

    def _expected_push_contact_z(self) -> float:
        # Keep the contact point near the initial source height during pushing.
        src_z0 = float(self.traj.obj_pos[self.task.source][0][2])
        return src_z0

    def _evaluate_stage1_push(self, stage1_max: float = 0.4) -> tuple[PhaseResult, dict]:
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

        if self.traj.n_steps <= 0 or self.task.source not in self.traj.obj_pos:
            reason = "trajectory/source missing"
            return (
                PhaseResult("stage1_push_setup", 0.0, reason, 0.0, reason, stage1_max),
                ctx,
            )

        close_expected = _first_close_event(
            self.traj.grip[self.task.arm],
            close_threshold=self.push_close_threshold,
        )
        close_other = _first_close_event(
            self.traj.grip[self.other_arm],
            close_threshold=self.push_close_threshold,
        )
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
            reason = "operating arm never fully closed gripper"
            return (
                PhaseResult("stage1_push_setup", 0.0, reason, 0.0, reason, stage1_max),
                ctx,
            )

        target_push_z = self._expected_push_contact_z()
        src_z0 = float(self.traj.obj_pos[self.task.source][0][2])
        s_ref = np.asarray(self.traj.obj_pos[self.task.source][close_expected][:2])
        source_names = self._source_object_names()

        pushed_on_table = False
        max_on_table_move = 0.0
        max_leave_table = 0.0
        best_push_xy_closest = float("inf")
        best_push_xy_exec_closest = float("inf")
        move_eval_t = None
        move_eval_z_offset = None
        near_table_at_move = None

        intent_ok = False
        exec_ok = False

        for t in range(close_expected + 1, self.traj.n_steps):
            src = self.traj.obj_pos[self.task.source][t]
            ee = self.traj.ee[self.task.arm][t]

            move_xy = float(np.linalg.norm(np.asarray(src[:2]) - s_ref))
            src_lift = max(0.0, float(src[2]) - src_z0)
            max_leave_table = max(max_leave_table, src_lift)

            on_table = src_lift <= self.push_source_table_z_tol
            if on_table:
                max_on_table_move = max(max_on_table_move, move_xy)

            if (not on_table) or (move_xy < self.push_stage1_min_move):
                continue

            pushed_on_table = True

            src_xy = _dist_xy(ee, src)
            contact_z = float(ee[2]) - self.ee_contact_z_offset
            z_offset = abs(contact_z - target_push_z)
            if move_eval_t is None:
                move_eval_t = t
                move_eval_z_offset = z_offset
                near_table_at_move = z_offset <= self.push_ee_near_table_tol

            min_name = None
            min_dist = None
            for name in source_names:
                d = _dist_xy(ee, self.traj.obj_pos[name][t])
                if min_dist is None or d < min_dist:
                    min_dist = d
                    min_name = name
            closest_ok = min_name == self.task.source

            if closest_ok and src_xy < best_push_xy_closest:
                best_push_xy_closest = src_xy
            if closest_ok and src_xy < best_push_xy_exec_closest:
                best_push_xy_exec_closest = src_xy

            if closest_ok and src_xy < self.push_intent_xy_threshold:
                intent_ok = True
            if (
                near_table_at_move is True
                and closest_ok
                and src_xy < self.push_exec_xy_threshold
            ):
                exec_ok = True

        ctx["stage1_intent_ok"] = intent_ok

        if intent_ok:
            intent_reason = "ok"
        elif not pushed_on_table:
            intent_reason = (
                "source not pushed on-table "
                f"(on_table_move={max_on_table_move*100:.1f}cm, max_lift={max_leave_table*100:.1f}cm)"
            )
        elif np.isfinite(best_push_xy_closest):
            intent_reason = (
                f"on-table push detected with source-nearest frame, best source_xy={best_push_xy_closest*100:.1f}cm "
                f"(need < {self.push_intent_xy_threshold*100:.0f}cm)"
            )
        else:
            intent_reason = "on-table push detected but source was never nearest to gripper"

        if exec_ok:
            exec_reason = "ok"
        elif not pushed_on_table:
            exec_reason = (
                "source not pushed on-table "
                f"(on_table_move={max_on_table_move*100:.1f}cm, max_lift={max_leave_table*100:.1f}cm)"
            )
        elif move_eval_t is None or move_eval_z_offset is None or near_table_at_move is None:
            exec_reason = "no valid 0.5cm push-move event found"
        elif not near_table_at_move:
            exec_reason = (
                "at first 0.5cm source move, contact point not near table enough: "
                f"offset={move_eval_z_offset*100:.1f}cm "
                f"(need < {self.push_ee_near_table_tol*100:.0f}cm)"
            )
        elif np.isfinite(best_push_xy_exec_closest):
            exec_reason = (
                f"need source_xy<{self.push_exec_xy_threshold*100:.0f}cm after 0.5cm move event; got "
                f"{best_push_xy_exec_closest*100:.1f}cm"
            )
        else:
            exec_reason = "on-table push detected but source was never nearest to gripper"
        if wrong_arm_participated:
            intent_reason = f"{intent_reason}; wrong arm participated"
            exec_reason = f"{exec_reason}; wrong arm participated"

        return (
            PhaseResult(
                "stage1_push_setup",
                stage1_max if intent_ok else 0.0,
                intent_reason,
                stage1_max if exec_ok else 0.0,
                exec_reason,
                stage1_max,
            ),
            ctx,
        )

    def _stack_metrics(self, t: int) -> tuple[float, float]:
        src = self.traj.obj_pos[self.task.source][t]
        tgt = self.traj.obj_pos[self.task.target][t]
        return _dist_xy(src, tgt), float(src[2] - tgt[2])

    def _push_backside_projection(self, t: int, desired_u) -> float:
        src_xy = np.asarray(self.traj.obj_pos[self.task.source][t][:2], dtype=float)
        ee_xy = np.asarray(self.traj.ee[self.task.arm][t][:2], dtype=float)
        return float(np.dot(ee_xy - src_xy, desired_u))

    def _is_stacked_on_target_at(self, t: int, xy_thresh: float, z_margin: float) -> tuple[bool, float, float]:
        xy, z_diff = self._stack_metrics(t)
        return (xy <= xy_thresh and z_diff > z_margin), xy, z_diff

    def _is_in_bowl_at(self, t: int) -> tuple[bool, float]:
        if self.task.target is None:
            return False, float("inf")
        src = self.traj.obj_pos[self.task.source][t]
        bowl = self.traj.obj_pos[self.task.target][t]
        xy = _dist_xy(src, bowl)
        inside = xy <= self.place_xy_threshold and float(src[2]) <= float(bowl[2]) + self.place_bowl_z_cap
        return inside, xy

    def _is_target_bowl_nearest(self, t: int) -> bool:
        if self.task.target is None:
            return False
        bowl_names = [n for n in self.traj.bowls if n in self.traj.obj_pos]
        if self.task.target not in bowl_names:
            return False
        src = self.traj.obj_pos[self.task.source][t]
        best_name = None
        best_dist = None
        for name in bowl_names:
            d = _dist_xy(src, self.traj.obj_pos[name][t])
            if best_dist is None or d < best_dist:
                best_dist = d
                best_name = name
        return best_name == self.task.target

    def _evaluate_stage2(self, ctx: dict) -> PhaseResult:
        if self.task.action == "stack":
            return self._evaluate_stage2_stack(ctx)
        if self.task.action == "push":
            return self._evaluate_stage2_push(ctx)
        if self.task.action == "place_bowl":
            return self._evaluate_stage2_place(ctx)
        return PhaseResult("stage2_action", 0.0, "unsupported action", 0.0, "unsupported action", 0.4)

    def _evaluate_stage2_stack(self, ctx: dict) -> PhaseResult:
        close_t = ctx["close_t_expected"]
        open_t = None if close_t is None else _first_open_event_after(
            self.traj.grip[self.task.arm],
            close_t,
        )
        ctx["open_t_expected"] = open_t

        if self.task.target is None:
            return PhaseResult("stage2_stack", 0.0, "target missing", 0.0, "target missing", 0.4)
        if (
            self.task.source not in self.traj.obj_pos
            or self.task.target not in self.traj.obj_pos
        ):
            return PhaseResult(
                "stage2_stack",
                0.0,
                "source/target trajectory missing",
                0.0,
                "source/target trajectory missing",
                0.4,
            )
        if close_t is None:
            return PhaseResult("stage2_stack", 0.0, "no grasp close event", 0.0, "no grasp close event", 0.4)

        z0 = float(self.traj.obj_pos[self.task.source][0][2])
        lifted = False
        intent_ok = False
        intent_reason = "source never reached valid stack intent pose"

        for t in range(close_t + 1, self.traj.n_steps):
            src_z = float(self.traj.obj_pos[self.task.source][t][2])
            tgt_z = float(self.traj.obj_pos[self.task.target][t][2])
            if src_z - z0 >= self.lift_threshold:
                lifted = True
            if not lifted:
                continue
            ee = self.traj.ee[self.task.arm][t]
            src = self.traj.obj_pos[self.task.source][t]
            ee_src_xy = _dist_xy(ee, src)
            if src_z > tgt_z + self.stack_above_margin and ee_src_xy <= self.stack_ee_src_xy_threshold:
                intent_ok = True
                ctx["stage2_intent_t"] = t
                intent_reason = "ok"
                break
            intent_reason = (
                "need source above target and "
                f"ee-source<={self.stack_ee_src_xy_threshold*100:.0f}cm; got "
                f"z_diff={(src_z-tgt_z)*100:.1f}cm ee_source={ee_src_xy*100:.1f}cm"
            )

        exec_ok = False
        exec_reason = "gripper never opened with stable stacked result"
        if intent_ok and open_t is not None:
            open_ok, open_xy, open_z_diff = self._is_stacked_on_target_at(
                open_t,
                self.stack_src_tgt_xy_threshold,
                self.stack_above_margin,
            )
            final_ok, final_xy, final_z_diff = self._is_stacked_on_target_at(
                self.traj.n_steps - 1,
                self.stack_src_tgt_xy_threshold,
                self.stack_above_margin,
            )
            dropped = any(
                float(self.traj.obj_pos[self.task.source][t][2])
                <= float(self.traj.obj_pos[self.task.target][t][2]) + self.stack_drop_margin
                for t in range(open_t, self.traj.n_steps)
            )
            exec_ok = open_ok and final_ok and not dropped
            if exec_ok:
                exec_reason = "ok"
            else:
                exec_reason = (
                    f"open(xy={open_xy*100:.1f}cm,z_diff={open_z_diff*100:.1f}cm), "
                    f"final(xy={final_xy*100:.1f}cm,z_diff={final_z_diff*100:.1f}cm), "
                    f"dropped={dropped}"
                )
        elif intent_ok and open_t is None:
            exec_reason = "no gripper-open event after close"

        return PhaseResult(
            "stage2_stack",
            0.4 if intent_ok else 0.0,
            intent_reason,
            0.4 if exec_ok else 0.0,
            exec_reason,
            0.4,
        )

    def _evaluate_stage2_push(self, ctx: dict) -> PhaseResult:
        close_t = ctx["close_t_expected"]
        ctx["open_t_expected"] = None

        if self.task.target is None:
            return PhaseResult("stage2_push", 0.0, "target missing", 0.0, "target missing", 0.4)
        if (
            self.task.source not in self.traj.obj_pos
            or self.task.target not in self.traj.obj_pos
        ):
            return PhaseResult(
                "stage2_push",
                0.0,
                "source/target trajectory missing",
                0.0,
                "source/target trajectory missing",
                0.4,
            )
        if close_t is None:
            return PhaseResult("stage2_push", 0.0, "no push-setup close event", 0.0, "no push-setup close event", 0.4)

        s_ref = self.traj.obj_pos[self.task.source][close_t][:2]
        s0 = self.traj.obj_pos[self.task.source][0][:2]
        t0 = self.traj.obj_pos[self.task.target][0][:2]
        desired = t0 - s0
        dn = float(np.linalg.norm(desired))
        if dn < 1e-8:
            reason = "target/source overlap in initial layout"
            return PhaseResult("stage2_push", 0.0, reason, 0.0, reason, 0.4)
        desired_u = desired / dn

        best_projected = 0.0
        best_cos = -1.0
        best_t = None
        best_backside_proj = float("inf")
        has_directional_motion = False

        for t in range(close_t + 1, self.traj.n_steps):
            st = self.traj.obj_pos[self.task.source][t][:2]
            disp = st - s_ref
            move = float(np.linalg.norm(disp))
            if move < 1e-6:
                continue
            cos = _cos_similarity_2d(disp, desired)
            if cos < self.push_dir_cos_threshold:
                continue
            has_directional_motion = True

            backside_proj = self._push_backside_projection(t, desired_u)
            if backside_proj < best_backside_proj:
                best_backside_proj = backside_proj
            if backside_proj > -self.push_backside_proj_threshold:
                continue

            projected = float(np.dot(disp, desired_u))
            if projected > best_projected:
                best_projected = projected
                best_cos = cos
                best_t = t

        intent_ok = best_projected >= self.push_intent_move
        exec_ok = best_projected >= self.push_exec_move
        if intent_ok and best_t is not None:
            ctx["stage2_intent_t"] = best_t

        if best_t is None:
            if not has_directional_motion:
                intent_reason = (
                    f"never moved in target direction (<=45deg); need >= {self.push_intent_move*100:.0f}cm"
                )
                exec_reason = (
                    f"never moved in target direction (<=45deg); need >= {self.push_exec_move*100:.0f}cm"
                )
            else:
                need_text = (
                    "(need <= 0.0cm, i.e., on source back-side)"
                    if self.push_backside_proj_threshold <= 0.0
                    else f"(need <= -{self.push_backside_proj_threshold*100:.1f}cm)"
                )
                intent_reason = (
                    "directional push observed but gripper not on source back-side: "
                    f"best backside_proj={best_backside_proj*100:.1f}cm {need_text}"
                )
                exec_reason = intent_reason
        else:
            best_angle = float(np.rad2deg(np.arccos(np.clip(best_cos, -1.0, 1.0))))
            intent_reason = "ok" if intent_ok else (
                f"direction ok (angle={best_angle:.1f}deg) but moved {best_projected*100:.1f}cm < {self.push_intent_move*100:.0f}cm"
            )
            exec_reason = "ok" if exec_ok else (
                f"direction ok (angle={best_angle:.1f}deg) but moved {best_projected*100:.1f}cm < {self.push_exec_move*100:.0f}cm"
            )

        return PhaseResult(
            "stage2_push",
            0.4 if intent_ok else 0.0,
            intent_reason,
            0.4 if exec_ok else 0.0,
            exec_reason,
            0.4,
        )

    def _evaluate_stage2_place(self, ctx: dict) -> PhaseResult:
        close_t = ctx["close_t_expected"]
        open_t = None if close_t is None else _first_open_event_after(
            self.traj.grip[self.task.arm],
            close_t,
        )
        ctx["open_t_expected"] = open_t

        if self.task.target is None:
            return PhaseResult("stage2_place_bowl", 0.0, "target missing", 0.0, "target missing", 0.4)
        if (
            self.task.source not in self.traj.obj_pos
            or self.task.target not in self.traj.obj_pos
        ):
            return PhaseResult(
                "stage2_place_bowl",
                0.0,
                "source/target trajectory missing",
                0.0,
                "source/target trajectory missing",
                0.4,
            )
        if close_t is None:
            return PhaseResult(
                "stage2_place_bowl",
                0.0,
                "no grasp close event",
                0.0,
                "no grasp close event",
                0.4,
            )

        src_z0 = float(self.traj.obj_pos[self.task.source][0][2])
        intent_ok = False
        intent_reason = "no valid low-height in-bowl approach near target bowl"
        best_xy = float("inf")
        best_low_h = float("inf")

        for t in range(close_t + 1, self.traj.n_steps):
            src = self.traj.obj_pos[self.task.source][t]
            low_h = float(src[2] - src_z0)
            if low_h >= self.place_descend_height_threshold:
                continue

            in_bowl, xy = self._is_in_bowl_at(t)
            nearest_ok = self._is_target_bowl_nearest(t)
            best_xy = min(best_xy, xy)
            best_low_h = min(best_low_h, low_h)

            if nearest_ok and xy <= self.place_xy_threshold and in_bowl:
                intent_ok = True
                ctx["stage2_intent_t"] = t
                intent_reason = "ok"
                break

        exec_ok = False
        exec_reason = "gripper never opened with object placed in target bowl"
        if intent_ok and open_t is not None:
            in_bowl_open, xy_open = self._is_in_bowl_at(open_t)
            in_bowl_final, xy_final = self._is_in_bowl_at(self.traj.n_steps - 1)
            exec_ok = (
                in_bowl_open
                and xy_open <= self.place_xy_threshold
                and in_bowl_final
                and xy_final <= self.place_xy_threshold
            )
            if exec_ok:
                exec_reason = "ok"
            else:
                exec_reason = (
                    f"open(xy={xy_open*100:.1f}cm,in_bowl={in_bowl_open}), "
                    f"final(xy={xy_final*100:.1f}cm,in_bowl={in_bowl_final})"
                )
        elif intent_ok and open_t is None:
            exec_reason = "no gripper-open event after close"

        if not intent_ok:
            if np.isfinite(best_xy) and np.isfinite(best_low_h):
                intent_reason = (
                    f"best low-height approach: h={best_low_h*100:.1f}cm, "
                    f"xy={best_xy*100:.1f}cm (need <={self.place_xy_threshold*100:.0f}cm and in target bowl)"
                )

        return PhaseResult(
            "stage2_place_bowl",
            0.4 if intent_ok else 0.0,
            intent_reason,
            0.4 if exec_ok else 0.0,
            exec_reason,
            0.4,
        )

    def _completion_t(self, ctx: dict) -> int:
        if self.task.action in ("stack", "place_bowl"):
            if ctx.get("open_t_expected") is not None:
                return int(ctx["open_t_expected"])
            if ctx.get("stage2_intent_t") is not None:
                return int(ctx["stage2_intent_t"])

        if self.task.action == "push":
            if ctx.get("stage2_intent_t") is not None:
                return int(ctx["stage2_intent_t"])
            if ctx.get("close_t_expected") is not None:
                return int(ctx["close_t_expected"])

        z0 = float(self.traj.obj_pos[self.task.source][0][2])
        start = ctx["close_t_expected"] if ctx["close_t_expected"] is not None else 0
        for t in range(start, self.traj.n_steps):
            if float(self.traj.obj_pos[self.task.source][t][2]) - z0 >= self.lift_threshold:
                return t
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

    def _final_target_ok_for_bonus(self) -> tuple[bool, str]:
        if self.traj.n_steps <= 0:
            return False, "empty trajectory"
        if self.task.source not in self.traj.obj_pos:
            return False, "source trajectory missing"

        last_t = self.traj.n_steps - 1

        if self.task.action == "push":
            if self.task.target is None or self.task.target not in self.traj.obj_pos:
                return False, "push target missing"
            s0 = np.asarray(self.traj.obj_pos[self.task.source][0][:2], dtype=float)
            t0 = np.asarray(self.traj.obj_pos[self.task.target][0][:2], dtype=float)
            desired = t0 - s0
            dn = float(np.linalg.norm(desired))
            if dn < 1e-8:
                return False, "target/source overlap in initial layout"
            desired_u = desired / dn
            best_projected = 0.0
            for t in range(self.traj.n_steps):
                st = np.asarray(self.traj.obj_pos[self.task.source][t][:2], dtype=float)
                projected = float(np.dot(st - s0, desired_u))
                if projected > best_projected:
                    best_projected = projected
            ok = best_projected >= self.push_bonus_move_threshold
            return ok, (
                "ok" if ok else
                f"best projected move={best_projected*100:.1f}cm (need >= {self.push_bonus_move_threshold*100:.1f}cm)"
            )

        if self.task.action == "stack":
            if self.task.target is None or self.task.target not in self.traj.obj_pos:
                return False, "stack target missing"
            ok, xy, z_diff = self._is_stacked_on_target_at(
                last_t,
                self.stack_src_tgt_xy_threshold,
                self.stack_above_margin,
            )
            return ok, (
                "ok" if ok else
                f"final stack mismatch: xy={xy*100:.1f}cm, z_diff={z_diff*100:.1f}cm"
            )

        if self.task.action == "place_bowl":
            if self.task.target is None or self.task.target not in self.traj.obj_pos:
                return False, "place target missing"
            in_bowl, xy = self._is_in_bowl_at(last_t)
            nearest_ok = self._is_target_bowl_nearest(last_t)
            ok = in_bowl and nearest_ok
            return ok, (
                "ok" if ok else
                f"final place mismatch: in_bowl={in_bowl}, nearest={nearest_ok}, xy={xy*100:.1f}cm"
            )

        return False, "action has no target-based bonus"

    def compute_retry_bonus(
        self,
        intent_score: float,
        exec_score: float,
        max_total: float,
    ) -> dict:
        final_ok, final_reason = self._final_target_ok_for_bonus()
        intent_bonus = 0.0
        exec_bonus = 0.0

        if final_ok:
            if intent_score < self.retry_bonus_score_threshold:
                intent_bonus = self.retry_bonus_delta
            if exec_score < self.retry_bonus_score_threshold:
                exec_bonus = self.retry_bonus_delta

        new_intent = min(max_total, float(intent_score + intent_bonus))
        new_exec = min(max_total, float(exec_score + exec_bonus))

        return {
            "final_target_ok": bool(final_ok),
            "final_target_reason": final_reason,
            "intent_bonus": float(intent_bonus),
            "exec_bonus": float(exec_bonus),
            "intent_score_after_bonus": float(new_intent),
            "exec_score_after_bonus": float(new_exec),
        }

def run_once(
    env,
    policy,
    task: Scene2EvalTask,
    max_steps: int = 10,
    actions_per_step: int = 50,
    direct_step: bool = False,
    sim_steps: int = 15,
) -> dict:
    """Run one evaluation episode and return scores and video frames."""
    recorder = base.TrajectoryRecorder()
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
    evaluator = Scene2ThreeStageEvaluator(task, traj)
    phases = evaluator.evaluate()

    intent_score = sum(p.intent_score for p in phases)
    exec_score = sum(p.exec_score for p in phases)
    max_total = sum(p.max_score for p in phases)
    retry_bonus = evaluator.compute_retry_bonus(intent_score, exec_score, max_total)
    intent_score = retry_bonus["intent_score_after_bonus"]
    exec_score = retry_bonus["exec_score_after_bonus"]
    wrong_arm_intervened = evaluator.wrong_arm_penalty() > 0.0
    wrong_arm_intent_penalty = 0.0
    wrong_arm_exec_penalty = 0.0
    if wrong_arm_intervened and phases:
        tail = phases[-1]
        eps = 1e-6
        if tail.intent_score + eps >= tail.max_score:
            wrong_arm_intent_penalty = evaluator.wrong_arm_penalty()
            intent_score = max(0.0, intent_score - wrong_arm_intent_penalty)
        if tail.exec_score + eps >= tail.max_score:
            wrong_arm_exec_penalty = evaluator.wrong_arm_penalty()
            exec_score = max(0.0, exec_score - wrong_arm_exec_penalty)
    intent_ok = intent_score >= max_total * 0.99
    exec_ok = exec_score >= max_total * 0.99

    intent_msg = "ok" if intent_ok else f"intent_score={intent_score:.2f}/{max_total:.2f}"
    exec_msg = "ok" if exec_ok else f"exec_score={exec_score:.2f}/{max_total:.2f}"
    if retry_bonus["intent_bonus"] > 0.0:
        intent_msg += f" (+{retry_bonus['intent_bonus']:.1f} retry bonus)"
    if retry_bonus["exec_bonus"] > 0.0:
        exec_msg += f" (+{retry_bonus['exec_bonus']:.1f} retry bonus)"
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
        completion_ok=base.is_complete(
            phases, intent_score, exec_score, action=task.action,
            retry_final_target_ok=retry_bonus["final_target_ok"],
        ),
        exec_msg=exec_msg,
        intent_score=float(intent_score),
        exec_score=float(exec_score),
        retry_bonus=retry_bonus,
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
