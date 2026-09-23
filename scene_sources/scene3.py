from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
import os
from dataclasses import dataclass, field
from copy import deepcopy

from robofollow.routes import check_training_route

from envs.utils.action import Action
from robofollow.tasks.scene3 import *
from robofollow.geometry import create_box, create_sphere, create_standing_cylinder, create_actor


class scene3(Base_Task):
    # Top-down pose (no in-plane rotation), short-edge-front baseline.
    GRASP_QUAT_SHORT_FRONT = [-0.5, 0.5, -0.5, -0.5]
    # Top-down + 90deg CCW around world z.
    GRASP_QUAT_LONG_FRONT = [0.0, 0.70710678, 0.0, -0.70710678]
    GRIPPER_OFFSET = 0.12
    TRANSFER_SPEEDUP = 1.0
    AROUND_OVER_MAP = {
        "red_bar_2": "green_bar_3",
        "green_bar_3": "red_bar_2",
        "yellow_bar_4": "blue_bar_5",
        "blue_bar_5": "yellow_bar_4",
    }

    def setup_demo(self, **kwargs):
        self.instruction_type = kwargs.get(
            "instruction_type",
            os.environ.get("ROBOTWIN_INSTRUCTION_TYPE", "r_pick2_via4_to6_short_front"),
        )
        self._scene_override = kwargs.get("scene_name", None)
        print(f"Instruction type: {self.instruction_type}")
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        task_def = TASKS[self.instruction_type]
        scene_name = getattr(self, "_scene_override", None) or task_def.scene
        scene_cfg = SCENES[scene_name]
        table_height = 0.74 + self.table_z_bias

        self._scene_cfg = scene_cfg
        self._task_def = task_def
        self._table_height = table_height

        self.objects = []
        self.labels = {}
        self.object_dict = {}
        self._object_defs: dict[str, ObjectDef] = {}
        self._label_defs: dict[str, LabelDef] = {}

        for obj_def in scene_cfg.objects:
            x = obj_def.position[0] + np.random.uniform(-scene_cfg.layout_perturb_x, scene_cfg.layout_perturb_x)
            y = obj_def.position[1] + np.random.uniform(-scene_cfg.layout_perturb_y, scene_cfg.layout_perturb_y)
            hx, hy, hz = obj_def.half_size
            z = table_height + hz + 0.001
            quat = yaw_to_quat(obj_def.yaw)

            obj = create_box(
                scene=self,
                pose=sapien.Pose([x, y, z], quat),
                half_size=(hx, hy, hz),
                color=obj_def.color,
                name=obj_def.name,
                friction=obj_def.friction,
                mass=obj_def.mass,
                boxtype=obj_def.boxtype,
            )

            print(f"create long box: {obj_def.name} at ({x:.3f}, {y:.3f}, {z:.3f})")
            self.objects.append(obj)
            self.object_dict[obj_def.name] = obj
            self._object_defs[obj_def.name] = obj_def
            self.add_prohibit_area(obj, padding=0.10)

        for label_def in scene_cfg.labels:
            x = label_def.position[0] + np.random.uniform(-scene_cfg.layout_perturb_x, scene_cfg.layout_perturb_x)
            y = label_def.position[1] + np.random.uniform(-scene_cfg.layout_perturb_y, scene_cfg.layout_perturb_y)
            lhx, lhy, lhz = label_def.half_size
            z = table_height + lhz

            label = create_box(
                scene=self,
                pose=sapien.Pose([x, y, z]),
                half_size=(lhx, lhy, lhz),
                color=label_def.color,
                name=label_def.name,
                is_static=True,
                boxtype="default",
            )

            print(f"create label: {label_def.name} at ({x:.3f}, {y:.3f}, {z:.3f})")
            self.labels[label_def.name] = label
            self.object_dict[label_def.name] = label
            self._label_defs[label_def.name] = label_def

        # Keep grippers open before grasping, consistent with other tasks.
        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(3):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        self.action_type = task_def.action
        self.source_name = task_def.source
        self.route_name = task_def.via
        self.via_name = self.route_name
        self.target_name = task_def.target
        self.source_obj = self.object_dict[self.source_name]
        self.route_obj = self.object_dict[self.route_name]
        self.via_obj = self.route_obj
        self.target_obj = self.labels[self.target_name]
        self.route_mode = task_def.route_mode
        self.place_mode = task_def.place_mode
        self.arm_tag = ArmTag(task_def.arm)

        self._expected_place_center_z = self._get_expected_place_center_z()
        self._target_yaw = self._get_target_yaw()
        self._route_initial_z = float(self.source_obj.get_pose().p[2])
        self._route_samples = []

        print(f"action type: {self.action_type}")
        print(f"source: {self.source_name}")
        print(f"route ref: {self.route_name}")
        print(f"target: {self.target_name}")
        print(f"route mode: {self.route_mode}")
        print(f"place mode: {self.place_mode}")
        print(f"arm: {self.arm_tag}")

    def _get_expected_place_center_z(self):
        source_hz = self._object_defs[self.source_name].half_size[2]
        label_hz = self._label_defs[self.target_name].half_size[2]
        return self._table_height + label_hz * 2 + source_hz + 0.001

    def _is_rotated_target(self):
        return self.place_mode in ("long_front", "short_right")

    def _get_place_quat(self):
        if self._is_rotated_target():
            return self.GRASP_QUAT_LONG_FRONT
        return self.GRASP_QUAT_SHORT_FRONT

    def _get_target_yaw(self):
        if self._is_rotated_target():
            return np.pi / 2
        return 0.0

    @staticmethod
    def _nlerp_quat(q0, q1, t: float):
        """Normalized linear interpolation for quaternions (good enough for short rotations)."""
        qa = np.asarray(q0, dtype=np.float64)
        qb = np.asarray(q1, dtype=np.float64)
        # Keep interpolation on the shorter hemisphere.
        if np.dot(qa, qb) < 0:
            qb = -qb
        q = (1.0 - t) * qa + t * qb
        n = np.linalg.norm(q)
        if n < 1e-8:
            return qa.astype(np.float32).tolist()
        return (q / n).astype(np.float32).tolist()

    def _try_grasp_source(self, pre_grasp_dis=0.08, grasp_dis=0.0):
        source_pose = self.source_obj.get_pose()
        cx, cy, cz = source_pose.p[0], source_pose.p[1], source_pose.p[2]
        grasp_quat = self.GRASP_QUAT_SHORT_FRONT

        pre_grasp_pose = [cx, cy, cz + self.GRIPPER_OFFSET + pre_grasp_dis] + grasp_quat
        grasp_pose = [cx, cy, cz + self.GRIPPER_OFFSET + grasp_dis] + grasp_quat

        grasp_action = (
            self.arm_tag,
            [
                Action(self.arm_tag, "move", target_pose=pre_grasp_pose),
                Action(self.arm_tag, "move", target_pose=grasp_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
                Action(self.arm_tag, "close", target_gripper_pos=0.0),
            ],
        )
        return self.arm_tag, grasp_action

    def _build_transfer_waypoints(self, extra_transit_z=0.0):
        place_quat = self._get_place_quat()
        rotating_on_descent = self._is_rotated_target()
        transit_quat = self.GRASP_QUAT_SHORT_FRONT if rotating_on_descent else place_quat

        route_pose = self.route_obj.get_pose()
        rx, ry = route_pose.p[0], route_pose.p[1]

        target_pose = self.target_obj.get_pose()
        tx, ty = target_pose.p[0], target_pose.p[1]

        travel_ee_z = self._table_height + self.GRIPPER_OFFSET + 0.09 + extra_transit_z
        place_ee_z = self._expected_place_center_z + self.GRIPPER_OFFSET
        release_ee_z = place_ee_z + (0.012 if rotating_on_descent else 0.0)

        if self.arm_tag == "left":
            current_ee = self.robot.get_left_ee_pose()
        else:
            current_ee = self.robot.get_right_ee_pose()

        if self.route_mode == "around":
            # "around X" means do NOT pass over X; pass over another object instead.
            over_name = self.AROUND_OVER_MAP.get(self.route_name, self.route_name)
            over_pose = self.object_dict[over_name].get_pose()
            route_x = float(over_pose.p[0])
            route_y = float(over_pose.p[1])
        else:
            route_x = float(rx)
            route_y = float(ry)

        waypoints = []
        constraints = []

        # Lift first, then follow a denser arc from source->route->target to avoid abrupt turns.
        current_xy = np.asarray([float(current_ee[0]), float(current_ee[1])], dtype=np.float64)
        route_xy = np.asarray([route_x, route_y], dtype=np.float64)
        target_xy = np.asarray([float(tx), float(ty)], dtype=np.float64)

        lift_z = max(float(current_ee[2]) + 0.05, travel_ee_z - 0.035)
        lift_pose = [float(current_xy[0]), float(current_xy[1]), float(lift_z)] + transit_quat

        entry_xy_1 = current_xy * 0.78 + route_xy * 0.22
        entry_xy_2 = current_xy * 0.52 + route_xy * 0.48
        exit_xy_1 = route_xy * 0.52 + target_xy * 0.48
        exit_xy_2 = route_xy * 0.22 + target_xy * 0.78

        arc_z_mid = float(travel_ee_z + 0.006)
        arc_z_high = float(travel_ee_z + 0.012)
        route_peak_z = float(travel_ee_z + 0.022)

        arc_entry_pose_1 = [float(entry_xy_1[0]), float(entry_xy_1[1]), arc_z_mid] + transit_quat
        arc_entry_pose_2 = [float(entry_xy_2[0]), float(entry_xy_2[1]), arc_z_high] + transit_quat
        route_peak_pose = [route_x, route_y, route_peak_z] + transit_quat
        arc_exit_pose_1 = [float(exit_xy_1[0]), float(exit_xy_1[1]), arc_z_high] + transit_quat
        arc_exit_pose_2 = [float(exit_xy_2[0]), float(exit_xy_2[1]), arc_z_mid] + transit_quat
        pre_place_pose = [float(tx), float(ty), float(travel_ee_z)] + transit_quat
        place_pose = [float(tx), float(ty), float(release_ee_z)] + place_quat

        waypoints.extend(
            [
                lift_pose,
                arc_entry_pose_1,
                arc_entry_pose_2,
                route_peak_pose,
                arc_exit_pose_1,
                arc_exit_pose_2,
                pre_place_pose,
            ]
        )
        constraints.extend([None, None, None, None, None, None, None])

        if rotating_on_descent:
            # Rotate while descending.
            z_span = max(travel_ee_z - release_ee_z, 1e-3)
            descend_pose_1 = [float(tx), float(ty), float(travel_ee_z - z_span * 0.35)] + self._nlerp_quat(
                transit_quat,
                place_quat,
                0.45,
            )
            descend_pose_2 = [float(tx), float(ty), float(travel_ee_z - z_span * 0.70)] + self._nlerp_quat(
                transit_quat,
                place_quat,
                0.80,
            )
            waypoints.extend([descend_pose_1, descend_pose_2, place_pose])
            constraints.extend([None, None, None])
        else:
            waypoints.append(place_pose)
            constraints.append([1, 1, 1, 0, 0, 0])

        return waypoints, constraints

    def _retime_joint_path(self, arm_tag, joint_path):
        """Smooth waypoint trajectory so intermediate waypoints do not cause stop-and-go."""
        planner = self.robot.left_mplib_planner if arm_tag == "left" else self.robot.right_mplib_planner
        if planner is None or joint_path is None or len(joint_path) < 3:
            return None, None
        try:
            _, pos, vel, _, _ = planner.TOPP(joint_path, 1 / 250, verbose=False)
            if pos is None or vel is None:
                return None, None
            if len(pos) == 0 or len(vel) == 0:
                return None, None
            return np.asarray(pos), np.asarray(vel)
        except Exception:
            return None, None

    def _speedup_joint_path(self, pos, speedup=1.0):
        """Resample trajectory with fewer points while keeping smooth velocity profile."""
        if pos is None:
            return None, None
        if len(pos) < 2:
            return np.asarray(pos, dtype=np.float32), np.zeros_like(pos, dtype=np.float32)
        if len(pos) < 3 or speedup <= 1.01:
            vel = np.gradient(pos, axis=0) * 250.0
            return np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32)

        new_n = max(3, int(len(pos) / speedup))
        src_idx = np.arange(len(pos), dtype=np.float32)
        dst_idx = np.linspace(0, len(pos) - 1, new_n, dtype=np.float32)
        interp_pos = np.zeros((new_n, pos.shape[1]), dtype=np.float32)
        for j in range(pos.shape[1]):
            interp_pos[:, j] = np.interp(dst_idx, src_idx, pos[:, j])
        interp_vel = np.gradient(interp_pos, axis=0) * 250.0
        return interp_pos, np.asarray(interp_vel, dtype=np.float32)

    def _execute_waypoints_continuous(self, arm_tag, waypoints, constraints):
        if arm_tag not in ("left", "right") or not waypoints or len(waypoints) != len(constraints):
            raise ValueError("Invalid transfer arm or waypoints")
        path = self.left_joint_path if arm_tag == "left" else self.right_joint_path
        if self.need_plan:
            result = self._plan_transfer(arm_tag, waypoints, constraints)
            if result is None:
                return False
        else:
            index = self.left_cnt if arm_tag == "left" else self.right_cnt
            if index >= len(path) or path[index].get("robofollow_segment") != "scene3_transfer_v1":
                raise ValueError("Missing saved Scene3 transfer; regenerate the expert plans")
            result = deepcopy(path[index])

        position, velocity = np.asarray(result["position"]), np.asarray(result["velocity"])
        if (result["status"] != "Success" or position.ndim != 2 or position.shape[1] != 6
                or not len(position) or position.shape != velocity.shape
                or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
            raise ValueError("Invalid saved/planned Scene3 transfer")
        if self.need_plan:
            path.append(deepcopy(result))
        elif arm_tag == "left":
            self.left_cnt += 1
        else:
            self.right_cnt += 1
        control_seq = {"left_arm": None, "right_arm": None,
                       "left_gripper": None, "right_gripper": None}
        control_seq[f"{arm_tag}_arm"] = result
        self.take_dense_action(control_seq)
        return True

    def _take_picture(self):
        # Record route evidence on the same sampling schedule in both passes.
        if not self.eval_mode and hasattr(self, "_route_samples"):
            alternative = self.AROUND_OVER_MAP[self.route_name]
            correct = alternative if self.route_mode == "around" else self.route_name
            wrong = self.route_name if self.route_mode == "around" else alternative
            self._route_samples.append({
                "source": self.source_obj.get_pose().p.tolist(),
                "correct": self.object_dict[correct].get_pose().p.tolist(),
                "wrong": self.object_dict[wrong].get_pose().p.tolist(),
            })
        super()._take_picture()

    def _plan_transfer(self, arm_tag, waypoints, constraints):
        if arm_tag == "left":
            entity = self.robot.left_entity
            active_joints = self.robot.left_active_joints
            arm_joints = self.robot.left_arm_joints
            plan_fn = self.robot.left_plan_path
        elif arm_tag == "right":
            entity = self.robot.right_entity
            active_joints = self.robot.right_active_joints
            arm_joints = self.robot.right_arm_joints
            plan_fn = self.robot.right_plan_path
        else:
            self.plan_success = False
            return None

        qpos_full = np.array(entity.get_qpos(), dtype=np.float32)
        active_joint_names = [j.get_name() for j in active_joints]
        arm_joint_names = [j.get_name() for j in arm_joints]
        arm_indices = [active_joint_names.index(name) for name in arm_joint_names]
        start_arm_q = np.array([qpos_full[i] for i in arm_indices], dtype=np.float32)
        joint_knots = [start_arm_q]
        fallback_pos_segments = []
        fallback_vel_segments = []

        for idx, (pose, constraint_pose) in enumerate(zip(waypoints, constraints)):
            result = plan_fn(
                pose,
                constraint_pose=constraint_pose,
                last_qpos=qpos_full,
            )
            if result["status"] != "Success":
                self.plan_success = False
                print(f"  [ERROR] continuous plan failed at waypoint {idx}")
                return None

            pos = result["position"]
            vel = result["velocity"]
            if idx > 0:
                # Remove duplicated first sample to keep the trajectory continuous.
                pos = pos[1:] if len(pos) > 1 else pos
                vel = vel[1:] if len(vel) > 1 else vel
            fallback_pos_segments.append(pos)
            fallback_vel_segments.append(vel)

            # Keep using full-qpos seed for next segment (dual-arm articulation safe).
            last_pos = np.array(result["position"][-1], dtype=np.float32)
            if last_pos.shape[0] == qpos_full.shape[0]:
                qpos_full = last_pos
                arm_last = np.array([qpos_full[i] for i in arm_indices], dtype=np.float32)
            elif last_pos.shape[0] == len(arm_indices):
                qpos_full = qpos_full.copy()
                qpos_full[arm_indices] = last_pos
                arm_last = last_pos
            else:
                self.plan_success = False
                print(
                    "  [ERROR] unexpected planner output dim:",
                    last_pos.shape[0],
                    "(full qpos dim:",
                    qpos_full.shape[0],
                    ")",
                )
                return None
            joint_knots.append(arm_last)

        merged_pos = np.concatenate(fallback_pos_segments, axis=0)
        merged_vel = np.concatenate(fallback_vel_segments, axis=0)

        # Retiming on waypoint knots keeps path smooth across local segment boundaries.
        joint_knots = np.vstack(joint_knots)
        smoothed_pos, smoothed_vel = self._retime_joint_path(arm_tag, joint_knots)
        if smoothed_pos is not None and smoothed_vel is not None:
            merged_pos = smoothed_pos
            merged_vel = smoothed_vel
        merged_pos = np.asarray(merged_pos, dtype=np.float32)
        sped_pos, sped_vel = self._speedup_joint_path(merged_pos, speedup=self.TRANSFER_SPEEDUP)
        if sped_pos is not None and sped_vel is not None:
            merged_pos = sped_pos
            merged_vel = sped_vel
        if merged_vel is None:
            if len(merged_pos) < 2:
                merged_vel = np.zeros_like(merged_pos, dtype=np.float32)
            else:
                merged_vel = np.gradient(merged_pos, axis=0) * 250.0

        return {"status": "Success", "position": np.asarray(merged_pos, dtype=np.float32),
                "velocity": np.asarray(merged_vel, dtype=np.float32),
                "robofollow_segment": "scene3_transfer_v1"}

    def play_once(self):
        if self.action_type in ("place_on_label_route", "place_on_label_via"):
            return self._play_place_on_label_route()
        print(f"[ERROR] Unknown action type: {self.action_type}")
        return self.info

    def _play_place_on_label_route(self):
        arm_tag, grasp_action = self._try_grasp_source()
        self.info["info"] = {
            "{source}": self.source_name,
            "{route}": self.route_name,
            "{route_mode}": self.route_mode,
            "{target}": self.target_name,
            "{arm}": str(arm_tag),
            "{action}": "place_on_label_route",
            "{place_mode}": self.place_mode,
            "{stage}": "init",
        }

        print("  [stage] grasp")
        self.info["info"]["{stage}"] = "grasp"
        self.move(grasp_action)
        if not self.plan_success:
            print("  [ERROR] Grasp planning failed!")
            self.info["info"]["{error}"] = "grasp_failed"
            return self.info

        print("  [stage] transfer(primary)")
        self.info["info"]["{stage}"] = "transfer_primary"
        transfer_waypoints, transfer_constraints = self._build_transfer_waypoints(extra_transit_z=0.0)
        self._execute_waypoints_continuous(arm_tag, transfer_waypoints, transfer_constraints)
        if not self.plan_success:
            print("  [WARN] primary transfer failed, retry with slightly higher transit.")
            self.plan_success = True
            transfer_waypoints, transfer_constraints = self._build_transfer_waypoints(extra_transit_z=0.04)
            print("  [stage] transfer(retry)")
            self.info["info"]["{stage}"] = "transfer_retry"
            self._execute_waypoints_continuous(arm_tag, transfer_waypoints, transfer_constraints)
            if not self.plan_success:
                self.info["info"]["{error}"] = "transfer_retry_failed"
                return self.info

        self.move((arm_tag, [Action(arm_tag, "open", target_gripper_pos=1.0)]))
        if not self.plan_success:
            self.info["info"]["{error}"] = "open_failed"
            return self.info

        self.info["info"]["{stage}"] = "retreat"
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))
        if not self.plan_success:
            self.info["info"]["{error}"] = "retreat_failed"
            return self.info

        self.info["info"]["{stage}"] = "return_home"
        self.together_move_to_pose(
            left_target_pose=self.robot.left_original_pose,
            right_target_pose=self.robot.right_original_pose,
        )
        if not self.plan_success:
            self.info["info"]["{error}"] = "return_home_failed"
            return self.info

        self.info["info"]["{stage}"] = "done"
        return self.info

    @staticmethod
    def _wrap_to_pi(angle: float):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _quat_to_yaw(q):
        # q is [w, x, y, z]
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _axis_yaw_error(self, yaw: float, target: float):
        candidates = [target, target + np.pi, target - np.pi]
        return min(abs(self._wrap_to_pi(yaw - c)) for c in candidates)

    def check_success(self):
        if self.action_type not in ("place_on_label_route", "place_on_label_via"):
            return False

        source_pos = self.source_obj.get_pose().p
        target_pos = self.target_obj.get_pose().p

        xy_dist = np.linalg.norm(source_pos[:2] - target_pos[:2])
        z_err = abs(float(source_pos[2]) - float(self._expected_place_center_z))

        source_yaw = self._quat_to_yaw(self.source_obj.get_pose().q)
        yaw_err = self._axis_yaw_error(source_yaw, self._target_yaw)

        if self.arm_tag == "left":
            gripper_open = float(self.robot.get_left_gripper_val())
        else:
            gripper_open = float(self.robot.get_right_gripper_val())

        route_ok = True
        if not self.eval_mode:
            route_check = check_training_route(self._route_samples, self._route_initial_z, self.route_mode)
            self.info["info"]["route_check"] = route_check
            self.info["info"]["route_samples"] = self._route_samples
            route_ok = route_check["passed"]
        return bool(
            route_ok
            and xy_dist < 0.045
            and z_err < 0.03
            and yaw_err < 0.50
            and gripper_open > 0.6
        )
