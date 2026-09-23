from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from robofollow.tasks.common import *
_DIRECT_STEP_PATCHED = False
OPERATION_AREA = dict(x_min=-0.20, x_max=0.15, y_min=-0.15, y_max=0.15, z_min=0.74, z_max=1.0)

def is_complete(phases, intent_score, exec_score, *, action,
                final_target_ok=None, retry_final_target_ok=None,
                pickup_threshold=0.8):
    """Classify episode completion from final scores and task-specific signals.

    Pickup-like actions require both final scores to reach the threshold.
    Other actions prefer explicit final-layout signals, including False;
    only absent signals fall back to the designated phase execution credit.
    """
    action = (action or "").strip().lower()
    if action in {"pickup", "sequence_pickups"} or action.startswith("pickup"):
        return intent_score >= pickup_threshold and exec_score >= pickup_threshold

    if final_target_ok is not None:
        return bool(final_target_ok)
    if retry_final_target_ok is not None:
        return bool(retry_final_target_ok)

    phase_exec = {p.phase_name: p.exec_score for p in phases}
    if "stage3_orientation" in phase_exec:
        return phase_exec["stage3_orientation"] > 0.0
    for name in ("stage2_stack", "stage2_unstack_slot", "stage2_place_bowl", "stage2_push"):
        if name in phase_exec:
            return phase_exec[name] > 0.0
    return False


def is_full_success(phases, intent_score, exec_score):
    """Strict diagnostic requiring every stage, including finish, and both totals."""
    eps = 1e-6
    total = sum(p.max_score for p in phases)
    return bool(phases) and all(
        p.intent_score + eps >= p.max_score and p.exec_score + eps >= p.max_score
        for p in phases
    ) and intent_score + eps >= total and exec_score + eps >= total

class _StubPlanner:
    """Planner stub for direct-step action execution."""

    def plan_grippers(self, now_val, target_val):
        num_step = 200
        vals = np.linspace(now_val, target_val, num_step)
        return {
            "num_step": num_step,
            "per_step": (target_val - now_val) / num_step,
            "result": vals,
        }

    def plan_batch(self, *args, **kwargs):
        return {"status": "skip", "position": []}

    def plan_path(self, *args, **kwargs):
        return {"status": "skip", "position": []}

    def update_point_cloud(self, *args, **kwargs):
        return None

def _patch_robot_for_direct_step() -> None:
    """
    Replace Robot planner hooks with stubs for direct-step execution.
    Arm targets are applied directly without cuRobo/MPLIB trajectory planning.
    """
    global _DIRECT_STEP_PATCHED
    if _DIRECT_STEP_PATCHED:
        return

    from envs.robot.robot import Robot as _Robot

    def _noop_set_planner(self, scene=None):
        self.communication_flag = False
        self.left_planner = _StubPlanner()
        self.right_planner = _StubPlanner()

    def _patched_reset(self, scene, need_topp=False, **kwargs):
        self._init_robot_(scene, need_topp, **kwargs)
        self.communication_flag = False
        self.left_planner = _StubPlanner()
        self.right_planner = _StubPlanner()
        self.init_joints()

    _Robot.set_planner = _noop_set_planner
    _Robot.reset = _patched_reset
    _DIRECT_STEP_PATCHED = True
    print("[DirectStep] Planner hooks configured for direct action execution.")

def _step_action_direct(env: Base_Task, action: np.ndarray, n_sim_steps: int = 15) -> None:
    """
    Set arm and gripper targets directly, then advance physics.
    """
    left_arm = action[:6]
    left_gripper = float(action[6])
    right_arm = action[7:13]
    right_gripper = float(action[13])

    env.robot.set_arm_joints(left_arm, [0.0] * 6, "left")
    env.robot.set_arm_joints(right_arm, [0.0] * 6, "right")
    env.robot.set_gripper(left_gripper, "left")
    env.robot.set_gripper(right_gripper, "right")

    for _ in range(n_sim_steps):
        env.scene.step()
    env._update_render()

@dataclass
class Trajectory:
    """Complete recorded trajectory."""
    ee: dict[str, list[np.ndarray]]        # arm -> [pos_t0, pos_t1, ...]
    grip: dict[str, list[float]]           # arm -> [grip_t0, grip_t1, ...]
    obj_pos: dict[str, list[np.ndarray]]   # obj_name -> [pos_t0, pos_t1, ...]
    labels: set[str]                       # set of label names (static, excluded from disturbance check)
    bowls: set[str]                        # set of bowl names
    n_steps: int = 0

    def source_z(self, source: str, t: int) -> float:
        return self.obj_pos[source][t][2]

    def initial_z(self, name: str) -> float:
        return self.obj_pos[name][0][2]

def dist_xy(a, b):  return np.hypot(a[0] - b[0], a[1] - b[1]) # 2D distance

def dist_3d(a, b):  return np.linalg.norm(np.asarray(a) - np.asarray(b)) # 3D distance

def in_operation_area(pos):
    a = OPERATION_AREA
    return (a["x_min"] <= pos[0] <= a["x_max"] and
            a["y_min"] <= pos[1] <= a["y_max"] and
            a["z_min"] <= pos[2] <= a["z_max"])

def get_ee_pos(env, arm="left"):
    if arm == "left":
        return np.array(env.robot.get_left_tcp_pose()[:3])
    else:
        return np.array(env.robot.get_right_tcp_pose()[:3])

def perturb_position(center_x, center_y, rule: PerturbRule):
    x = center_x + np.random.uniform(-rule.x_range, rule.x_range)
    if rule.y_direction == "up":
        y = center_y + np.random.uniform(0, rule.y_range)
    elif rule.y_direction == "down":
        y = center_y - np.random.uniform(0, rule.y_range)
    else:
        y = center_y + np.random.uniform(-rule.y_range, rule.y_range)
    return x, y

class TrajectoryRecorder:
    def __init__(self):
        self.traj = Trajectory(
            ee={"left": [], "right": []},
            grip={"left": [], "right": []},
            obj_pos={},
            labels=set(),
            bowls=set(),
        )

    def record(self, env, obs):
        for arm in ("left", "right"):
            self.traj.ee[arm].append(get_ee_pos(env, arm)) 

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

    def get_trajectory(self) -> Trajectory:
        return self.traj
