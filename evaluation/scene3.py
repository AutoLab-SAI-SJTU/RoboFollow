from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from . import common as base
import robofollow.tasks.scene3 as scene3_env_mod
from robofollow.tasks.scene3 import SCENES as SCENE3_BASE_SCENES, TASKS as SCENE3_BASE_TASKS

LEVEL_ORDER = ("L0", "L1", "L2", "L3")

WRONG_ARM_INTERVENTION_PENALTY = 0.2

@dataclass
class Scene3EvalTask:
    name: str
    scene: str
    source: str
    via: str
    target: str
    arm: str
    action: str
    route_mode: str
    place_mode: str
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

@dataclass
class Scene3Trajectory:
    ee: dict[str, list[np.ndarray]]
    grip: dict[str, list[float]]
    obj_pos: dict[str, list[np.ndarray]]
    obj_quat: dict[str, list[np.ndarray]]
    labels: set[str]
    label_half_size: dict[str, np.ndarray]
    n_steps: int = 0

TASKS: dict[str, Scene3EvalTask] = {}

LEVELS: dict[str, list[str]] = {}

SCENE_VARIANTS: dict[str, object] = {}

SCENE_VARIANT_NOTES: dict[str, str] = {}

BASE_TASK_TYPES = list(SCENE3_BASE_TASKS.keys())

OBJECT_PHRASE = {
    "red_bar_2": "red slab",
    "green_bar_3": "green slab",
    "yellow_bar_4": "yellow slab",
    "blue_bar_5": "blue slab",
}

LABEL_PHRASE = {
    "green_label_1": "green square label",
    "blue_label_6": "blue square label",
}

PAIR_OBJECT = {
    "red_bar_2": "green_bar_3",
    "green_bar_3": "red_bar_2",
    "yellow_bar_4": "blue_bar_5",
    "blue_bar_5": "yellow_bar_4",
}

ORIENTATION_EQUIV = {
    "short_front": "long_right",
    "long_front": "short_right",
    "short_right": "long_front",
    "long_right": "short_front",
}

ORIENTATION_TARGET_YAW = {
    "short_front": 0.0,
    "long_right": 0.0,
    "long_front": np.pi / 2,
    "short_right": np.pi / 2,
}

L1_SCENE_33_TASKS = [
    "r_pick3_via5_to6_short_front",
    "r_pick3_via5_to6_long_front",
    "l_pick5_around3_to1_short_right",
    "l_pick5_around3_to1_long_right",
]

L1_SCENE_34_TASKS = [
    "r_pick2_via4_to6_short_front",
    "r_pick2_via4_to6_long_front",
    "l_pick4_around2_to1_short_right",
    "l_pick4_around2_to1_long_right",
]

L3_SCENARIOS = {"scene3.1", "scene3.3", "scene3.4"}

def _stable_hash32(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)

def _load_collect_scene3_templates():
    return json.loads((Path(__file__).resolve().parents[1] / "tasks" / "scene3_instructions.json").read_text())

INSTRUCTION_TEMPLATES = _load_collect_scene3_templates()

def _register_task(task: Scene3EvalTask) -> None:
    TASKS[task.name] = task
    LEVELS.setdefault(task.level_tag, []).append(task.name)

def _with_the(phrase: str) -> str:
    p = phrase.strip()
    return p if p.lower().startswith("the ") else f"the {p}"

def _swap_positions(items: list[object], name_a: str, name_b: str) -> None:
    by_name = {item.name: item for item in items}
    if name_a not in by_name or name_b not in by_name:
        return
    pa = by_name[name_a].position
    pb = by_name[name_b].position
    by_name[name_a].position = pb
    by_name[name_b].position = pa

def _register_scene_variants() -> None:
    SCENE_VARIANTS.clear()
    SCENE_VARIANT_NOTES.clear()

    base_cfg = copy.deepcopy(SCENE3_BASE_SCENES["default"])
    base_cfg.name = "default"
    SCENE_VARIANTS["default"] = base_cfg
    SCENE_VARIANT_NOTES["default"] = "training layout"

    def add_variant(name: str, swap_pairs: list[tuple[str, str]], note: str):
        cfg = copy.deepcopy(SCENE3_BASE_SCENES["default"])
        cfg.name = name
        for a, b in swap_pairs:
            _swap_positions(cfg.objects, a, b)
        SCENE_VARIANTS[name] = cfg
        SCENE_VARIANT_NOTES[name] = note

    add_variant("scene3.1", [("red_bar_2", "green_bar_3")], "swap red_bar_2 <-> green_bar_3")
    add_variant("scene3.2", [("blue_bar_5", "yellow_bar_4")], "swap blue_bar_5 <-> yellow_bar_4")
    add_variant("scene3.3", [("red_bar_2", "yellow_bar_4")], "swap red_bar_2 <-> yellow_bar_4")
    add_variant("scene3.4", [("green_bar_3", "blue_bar_5")], "swap green_bar_3 <-> blue_bar_5")

def _canonical_instruction(task_type: str) -> str:
    templates = INSTRUCTION_TEMPLATES.get(task_type, [])
    if templates:
        return templates[0]

    td = SCENE3_BASE_TASKS[task_type]
    src = _with_the(OBJECT_PHRASE.get(td.source, td.source.replace("_", " ")))
    via = _with_the(OBJECT_PHRASE.get(td.via, td.via.replace("_", " ")))
    tgt = _with_the(LABEL_PHRASE.get(td.target, td.target.replace("_", " ")))
    if td.route_mode == "via":
        route_clause = f"pass over {via}"
    else:
        route_clause = f"avoid passing over {via}"

    ori = {
        "short_front": "short side facing front",
        "long_front": "long side facing front",
        "short_right": "short side facing right",
        "long_right": "long side facing right",
    }.get(td.place_mode, td.place_mode)
    return f"Use your {td.arm} arm to pick up {src}, {route_clause}, and place it on {tgt} with the {ori}."

def _orientation_phrase(place_mode: str, style: int) -> str:
    if style == 0:
        table = {
            "short_front": "short side facing front",
            "long_front": "long side facing front",
            "short_right": "short side facing right",
            "long_right": "long side facing right",
        }
    else:
        table = {
            "short_front": "short face pointing to the front",
            "long_front": "long face pointing to the front",
            "short_right": "short face pointing to the right",
            "long_right": "long face pointing to the right",
        }
    return table[place_mode]

def _route_clause(route_mode: str, via_name: str, style: int) -> str:
    via_phrase = _with_the(OBJECT_PHRASE.get(via_name, via_name.replace("_", " ")))
    if route_mode == "via":
        if style == 0:
            return f"pass over {via_phrase}"
        return f"move above {via_phrase} on the way"

    if style == 0:
        return f"avoid passing over {via_phrase} by detouring around it"
    return f"go around {via_phrase} instead of passing over it"

def _render_instruction(
    task_def,
    route_mode: str,
    via_name: str,
    place_mode: str,
    route_style: int,
    orientation_style: int,
) -> str:
    src = _with_the(OBJECT_PHRASE.get(task_def.source, task_def.source.replace("_", " ")))
    tgt = _with_the(LABEL_PHRASE.get(task_def.target, task_def.target.replace("_", " ")))
    route = _route_clause(route_mode, via_name, route_style)
    ori = _orientation_phrase(place_mode, orientation_style)
    return f"Use your {task_def.arm} arm to pick up {src}, {route}, and place it on {tgt} with the {ori}."

def _rewrite_route(task_def) -> tuple[str, str]:
    if task_def.route_mode == "via":
        return "around", PAIR_OBJECT[task_def.via]
    return "via", PAIR_OBJECT[task_def.via]

def _rewrite_place_mode(task_def) -> str:
    return ORIENTATION_EQUIV[task_def.place_mode]

def _add_task(
    level_tag: str,
    name_suffix: str,
    base_task_type: str,
    scene: str,
    instruction: str,
    instruction_variant: str,
    scenario_tag: str,
) -> None:
    td = SCENE3_BASE_TASKS[base_task_type]
    _register_task(
        Scene3EvalTask(
            name=f"{level_tag}_{name_suffix}",
            scene=scene,
            source=td.source,
            via=td.via,
            target=td.target,
            arm=td.arm,
            action=td.action,
            route_mode=td.route_mode,
            place_mode=td.place_mode,
            instruction=instruction,
            level_tag=level_tag,
            base_task_type=base_task_type,
            instruction_variant=instruction_variant,
            scenario_tag=scenario_tag,
        )
    )

def _register_l0_tasks() -> None:
    for task_type in BASE_TASK_TYPES:
        _add_task(
            "L0",
            task_type,
            task_type,
            "default",
            _canonical_instruction(task_type),
            "template_0",
            "default",
        )

def _register_l1_tasks() -> None:
    for task_type in BASE_TASK_TYPES:
        _add_task(
            "L1",
            f"scene3_1__{task_type}",
            task_type,
            "scene3.1",
            _canonical_instruction(task_type),
            "template_0",
            "scene3.1",
        )
    for task_type in BASE_TASK_TYPES:
        _add_task(
            "L1",
            f"scene3_2__{task_type}",
            task_type,
            "scene3.2",
            _canonical_instruction(task_type),
            "template_0",
            "scene3.2",
        )

    for task_type in L1_SCENE_33_TASKS:
        _add_task(
            "L1",
            f"scene3_3__{task_type}",
            task_type,
            "scene3.3",
            _canonical_instruction(task_type),
            "template_0",
            "scene3.3",
        )

    for task_type in L1_SCENE_34_TASKS:
        _add_task(
            "L1",
            f"scene3_4__{task_type}",
            task_type,
            "scene3.4",
            _canonical_instruction(task_type),
            "template_0",
            "scene3.4",
        )

def _register_l2_tasks() -> None:
    for task_type in BASE_TASK_TYPES:
        td = SCENE3_BASE_TASKS[task_type]
        route_mode_rw, via_rw = _rewrite_route(td)
        place_mode_rw = _rewrite_place_mode(td)
        for route_style in (0, 1):
            for ori_style in (0, 1):
                variant = f"rw_route{route_style}_ori{ori_style}"
                instruction = _render_instruction(
                    td,
                    route_mode_rw,
                    via_rw,
                    place_mode_rw,
                    route_style,
                    ori_style,
                )
                _add_task(
                    "L2",
                    f"{task_type}__{variant}",
                    task_type,
                    "default",
                    instruction,
                    variant,
                    "default",
                )

def _register_l3_tasks() -> None:
    for l1_name in LEVELS.get("L1", []):
        l1_task = TASKS[l1_name]
        if l1_task.scenario_tag not in L3_SCENARIOS:
            continue

        td = SCENE3_BASE_TASKS[l1_task.base_task_type]
        route_mode_rw, via_rw = _rewrite_route(td)
        place_mode_rw = _rewrite_place_mode(td)

        suffix = l1_name[len("L1_") :] if l1_name.startswith("L1_") else l1_name
        for route_style in (0, 1):
            for ori_style in (0, 1):
                variant = f"rw_route{route_style}_ori{ori_style}"
                instruction = _render_instruction(
                    td,
                    route_mode_rw,
                    via_rw,
                    place_mode_rw,
                    route_style,
                    ori_style,
                )
                _register_task(
                    Scene3EvalTask(
                        name=f"L3_{suffix}__{variant}",
                        scene=l1_task.scene,
                        source=l1_task.source,
                        via=l1_task.via,
                        target=l1_task.target,
                        arm=l1_task.arm,
                        action=l1_task.action,
                        route_mode=l1_task.route_mode,
                        place_mode=l1_task.place_mode,
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

class Scene3TrajectoryRecorder:
    def __init__(self):
        self.traj = Scene3Trajectory(
            ee={"left": [], "right": []},
            grip={"left": [], "right": []},
            obj_pos={},
            obj_quat={},
            labels=set(),
            label_half_size={},
        )

    def record(self, env, obs):
        for arm in ("left", "right"):
            self.traj.ee[arm].append(base.get_ee_pos(env, arm))

        sv = obs["joint_action"]["vector"]
        self.traj.grip["left"].append(float(sv[6]))
        self.traj.grip["right"].append(float(sv[13]))

        for name, ent in env.object_dict.items():
            pose = ent.get_pose()
            self.traj.obj_pos.setdefault(name, []).append(np.asarray(pose.p, dtype=np.float32))
            self.traj.obj_quat.setdefault(name, []).append(np.asarray(pose.q, dtype=np.float32))

        if hasattr(env, "labels") and isinstance(env.labels, dict):
            self.traj.labels = set(env.labels.keys())
        if (not self.traj.label_half_size) and hasattr(env, "_label_defs") and isinstance(env._label_defs, dict):
            for name, label_def in env._label_defs.items():
                half_size = getattr(label_def, "half_size", None)
                if half_size is not None:
                    self.traj.label_half_size[name] = np.asarray(half_size, dtype=np.float32)
        self.traj.n_steps = len(self.traj.ee["left"])

    def get_trajectory(self) -> Scene3Trajectory:
        return self.traj

class Scene3ThreeStageEvaluator:
    """Score source selection, route, and final pose."""

    def __init__(
        self,
        task: Scene3EvalTask,
        traj: Scene3Trajectory,
    ):
        self.task = task
        self.traj = traj
        self.other_arm = "right" if task.arm == "left" else "left"

        # Stage weights.
        self.stage1_max = 0.3
        self.stage2_max = 0.3
        self.stage3_max = 0.4

        # Source selection thresholds.
        self.pick_xy_threshold = 0.07
        self.lift_threshold = 0.01

        # Route thresholds.
        self.stage2_x_align = 0.01
        self.stage2_y_correct = 0.06
        self.stage2_y_wrong = 0.04

        # Orientation and placement thresholds.
        self.yaw_intent = np.deg2rad(40.0)
        self.yaw_exec = np.deg2rad(20.0)
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

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _quat_to_yaw(q) -> float:
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _axis_yaw_error(self, yaw: float, target: float) -> float:
        # The bar axis is unchanged by a 180-degree rotation.
        candidates = [target, target + np.pi, target - np.pi]
        return min(abs(self._wrap_to_pi(yaw - c)) for c in candidates)

    def _source_object_names(self) -> list[str]:
        return [name for name in self.traj.obj_pos if name not in self.traj.labels]

    def _target_label_edge_length(self) -> float | None:
        half_size = self.traj.label_half_size.get(self.task.target)
        if half_size is None:
            scene_cfg = SCENE_VARIANTS.get(self.task.scene)
            labels = getattr(scene_cfg, "labels", []) if scene_cfg is not None else []
            for label_def in labels:
                if getattr(label_def, "name", None) == self.task.target:
                    half_size = label_def.half_size
                    break
        if half_size is None:
            return None
        return 2.0 * float(max(half_size[0], half_size[1]))

    def _evaluate_stage1(self) -> tuple[PhaseResult, dict]:
        ctx = {
            "close_t_expected": None,
            "open_t_expected": None,
            "first_close_arm": None,
            "wrong_arm_closed": False,
            "wrong_arm_participated": False,
            "stage1_intent_ok": False,
            "source_lifted": False,
            "stage2_intent_t": None,
            "stage3_intent_t": None,
        }

        if self.traj.n_steps <= 0 or self.task.source not in self.traj.obj_pos:
            reason = "trajectory/source missing"
            return (
                PhaseResult("stage1_source_select", 0.0, reason, 0.0, reason, self.stage1_max),
                ctx,
            )

        close_expected = _first_close_event(self.traj.grip[self.task.arm])
        close_other = _first_close_event(self.traj.grip[self.other_arm])
        ctx["close_t_expected"] = close_expected
        ctx["wrong_arm_closed"] = close_other is not None

        if close_expected is None and close_other is None:
            first_close_arm = None
        elif close_other is None or (close_expected is not None and close_expected <= close_other):
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
                PhaseResult("stage1_source_select", 0.0, reason, 0.0, reason, self.stage1_max),
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
                self.stage1_max if intent_ok else 0.0,
                intent_reason,
                self.stage1_max if exec_ok else 0.0,
                exec_reason,
                self.stage1_max,
            ),
            ctx,
        )

    def _route_objects_for_stage2(self) -> tuple[str, str]:
        if self.task.route_mode == "via":
            correct = self.task.via
            wrong = PAIR_OBJECT.get(self.task.via, self.task.via)
        else:
            correct = PAIR_OBJECT.get(self.task.via, self.task.via)
            wrong = self.task.via
        return correct, wrong

    def _evaluate_stage2(self, ctx: dict) -> PhaseResult:
        close_t = ctx["close_t_expected"]
        if close_t is None:
            reason = "no grasp close event"
            return PhaseResult("stage2_route", 0.0, reason, 0.0, reason, self.stage2_max)

        correct, wrong = self._route_objects_for_stage2()
        if correct not in self.traj.obj_pos or wrong not in self.traj.obj_pos:
            reason = f"missing route objects: correct={correct} wrong={wrong}"
            return PhaseResult("stage2_route", 0.0, reason, 0.0, reason, self.stage2_max)

        z0 = float(self.traj.obj_pos[self.task.source][0][2])
        lifted = False
        intent_ok = False
        best = None

        for t in range(close_t + 1, self.traj.n_steps):
            src = self.traj.obj_pos[self.task.source][t]
            c = self.traj.obj_pos[correct][t]
            w = self.traj.obj_pos[wrong][t]

            if float(src[2]) - z0 >= self.lift_threshold:
                lifted = True
            if not lifted:
                continue

            x_diff = abs(float(src[0]) - float(c[0]))
            y_diff_c = abs(float(src[1]) - float(c[1]))
            y_diff_w = abs(float(src[1]) - float(w[1]))
            cond = (
                x_diff < self.stage2_x_align
                and y_diff_c < self.stage2_y_correct
                and y_diff_w > self.stage2_y_wrong
                and y_diff_c < y_diff_w
            )
            if cond:
                intent_ok = True
                ctx["stage2_intent_t"] = t
                best = (x_diff, y_diff_c, y_diff_w)
                break

            if best is None:
                best = (x_diff, y_diff_c, y_diff_w)
            else:
                bx, byc, byw = best
                # Prefer alignment with the correct waypoint and separation from the wrong one.
                if (x_diff + y_diff_c - y_diff_w) < (bx + byc - byw):
                    best = (x_diff, y_diff_c, y_diff_w)

        if intent_ok:
            x_diff, y_diff_c, y_diff_w = best
            reason = (
                f"ok: |x-src-correct|={x_diff*100:.1f}cm, "
                f"|y-src-correct|={y_diff_c*100:.1f}cm, "
                f"|y-src-wrong|={y_diff_w*100:.1f}cm"
            )
            return PhaseResult(
                "stage2_route",
                self.stage2_max,
                reason,
                self.stage2_max,
                reason,
                self.stage2_max,
            )

        if best is None:
            reason = "no post-grasp lifted trajectory"
        else:
            x_diff, y_diff_c, y_diff_w = best
            reason = (
                f"need |x-src-correct|<1.0cm, |y-src-correct|<6.0cm, "
                f"|y-src-wrong|>4.0cm, and |y-src-correct|<|y-src-wrong|; "
                f"best={x_diff*100:.1f}/{y_diff_c*100:.1f}/{y_diff_w*100:.1f}cm"
            )
        return PhaseResult("stage2_route", 0.0, reason, 0.0, reason, self.stage2_max)

    def _target_yaw(self) -> float:
        return float(ORIENTATION_TARGET_YAW[self.task.place_mode])

    def _evaluate_stage3(self, ctx: dict) -> PhaseResult:
        close_t = ctx["close_t_expected"]
        if close_t is None:
            reason = "no grasp close event"
            return PhaseResult("stage3_orientation", 0.0, reason, 0.0, reason, self.stage3_max)

        if self.task.target not in self.traj.obj_pos:
            reason = "target label trajectory missing"
            return PhaseResult("stage3_orientation", 0.0, reason, 0.0, reason, self.stage3_max)
        if self.task.source not in self.traj.obj_pos:
            reason = "source trajectory missing"
            return PhaseResult("stage3_orientation", 0.0, reason, 0.0, reason, self.stage3_max)

        label_edge = self._target_label_edge_length()
        if label_edge is None:
            reason = "target label size missing"
            return PhaseResult("stage3_orientation", 0.0, reason, 0.0, reason, self.stage3_max)

        final_t = self.traj.n_steps - 1
        final_quat = self.traj.obj_quat[self.task.source][final_t]
        final_yaw = self._quat_to_yaw(final_quat)
        final_yaw_err = self._axis_yaw_error(final_yaw, self._target_yaw())
        final_src = self.traj.obj_pos[self.task.source][final_t]
        final_tgt = self.traj.obj_pos[self.task.target][final_t]
        final_xy_dist = _dist_xy(final_src, final_tgt)
        place_ok = final_xy_dist < label_edge

        intent_yaw_ok = final_yaw_err <= self.yaw_intent
        exec_yaw_ok = final_yaw_err <= self.yaw_exec
        intent_ok = intent_yaw_ok and place_ok
        exec_ok = exec_yaw_ok and place_ok

        if intent_ok:
            intent_reason = "ok"
            ctx["stage3_intent_t"] = final_t
        else:
            reasons = []
            if not intent_yaw_ok:
                reasons.append(
                    f"final yaw error={np.rad2deg(final_yaw_err):.1f}deg "
                    f"(need <= {np.rad2deg(self.yaw_intent):.1f}deg)"
                )
            if not place_ok:
                reasons.append(
                    f"final center distance={final_xy_dist*100:.1f}cm "
                    f"(need < {label_edge*100:.1f}cm)"
                )
            intent_reason = "; ".join(reasons)

        if exec_ok:
            exec_reason = "ok"
        else:
            reasons = []
            if not exec_yaw_ok:
                reasons.append(
                    f"final yaw error={np.rad2deg(final_yaw_err):.1f}deg "
                    f"(need <= {np.rad2deg(self.yaw_exec):.1f}deg)"
                )
            if not place_ok:
                reasons.append(
                    f"final center distance={final_xy_dist*100:.1f}cm "
                    f"(need < {label_edge*100:.1f}cm)"
                )
            exec_reason = "; ".join(reasons)

        return PhaseResult(
            "stage3_orientation",
            self.stage3_max if intent_ok else 0.0,
            intent_reason,
            self.stage3_max if exec_ok else 0.0,
            exec_reason,
            self.stage3_max,
        )

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

    def evaluate(self) -> list[PhaseResult]:
        self._wrong_arm_intervened = False
        stage1, ctx = self._evaluate_stage1()
        stage2 = self._evaluate_stage2(ctx)
        stage3 = self._evaluate_stage3(ctx)
        return self._apply_gates(stage1, stage2, stage3)

Scene3FourStageEvaluator = Scene3ThreeStageEvaluator

def run_once(
    env,
    policy,
    task: Scene3EvalTask,
    max_steps: int = 10,
    actions_per_step: int = 50,
    direct_step: bool = False,
    sim_steps: int = 15,
) -> dict:
    """Run one evaluation episode and return scores and video frames."""
    recorder = Scene3TrajectoryRecorder()
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
    evaluator = Scene3ThreeStageEvaluator(task, traj)
    phases = evaluator.evaluate()

    intent_score = sum(p.intent_score for p in phases)
    exec_score = sum(p.exec_score for p in phases)
    max_total = sum(p.max_score for p in phases)
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
        completion_ok=base.is_complete(phases, intent_score, exec_score, action=task.action),
        exec_msg=exec_msg,
        intent_score=float(intent_score),
        exec_score=float(exec_score),
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
