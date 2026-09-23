from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
import os
from dataclasses import dataclass, field

from envs.utils.action import Action
from robofollow.tasks.scene4 import *
from robofollow.geometry import create_box, create_sphere, create_standing_cylinder, create_actor


class scene4(Base_Task):
    GRASP_QUAT_TOP_DOWN = [-0.5, 0.5, -0.5, -0.5]

    GRIPPER_OFFSET = 0.12
    NORMAL_PRE_GRASP_DIS = 0.08
    LIFT_DIS = 0.10
    # Raise grasp point for all cubes/cylinders to reduce grabbing support objects.
    CUBE_GRASP_BIAS_BASE = 0.008
    CYLINDER_GRASP_BIAS_BASE = 0.010
    STACKED_CUBE_GRASP_BIAS_MAX = 0.020
    STACKED_CUBE_GRASP_HEIGHT_RATIO = 0.70
    # For cylinders stacked on another object, grasp even higher to avoid
    # collision contact with the lower support object during finger closing.
    STACKED_CYLINDER_GRASP_BIAS_MAX = 0.026
    STACKED_CYLINDER_GRASP_HEIGHT_RATIO = 0.90

    APPROACH_DIS = 0.14
    TCP_TO_FINGERTIP_BIAS_Z = 0.04
    CUBE_GRASP_HEIGHT_RATIO = 0.75
    MIN_EXPOSED_ABOVE_RIM = 0.012
    BOWL_GRASP_EXTRA_LOWER_Z = 0.005

    BOWL_DROP_MARGIN = 0.015
    TABLE_CLEARANCE = 0.001
    STACK_GAP = 0.002

    PLACE_APPROACH_DIS = 0.06
    PLACE_RETREAT_DIS = 0.08

    SETTLE_STEPS = 180
    POST_REPOSITION_SETTLE_STEPS = 40

    @staticmethod
    def _normalize_instruction_type(instruction_type: str):
        if instruction_type is None:
            return instruction_type
        clean_type = instruction_type.strip()
        if clean_type.startswith("scene4_"):
            clean_type = clean_type[len("scene4_") :]
        return clean_type

    def setup_demo(self, **kwargs):
        raw_instruction_type = kwargs.get(
            "instruction_type",
            os.environ.get("ROBOTWIN_INSTRUCTION_TYPE", "s41_l_pick_cube_in_bowl"),
        )
        self.instruction_type = self._normalize_instruction_type(raw_instruction_type)
        self._scene_override = kwargs.get("scene_name", None)
        print(f"Instruction type: {self.instruction_type}")
        super()._init_task_env_(**kwargs)

    def _estimate_bowl_rim_z(self, bowl_actor, bowl_center_z, table_height):
        cfg = getattr(bowl_actor, "config", None) or {}
        try:
            center = np.asarray(cfg.get("center", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            extents = np.asarray(cfg.get("extents", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(-1)
            scale = np.asarray(cfg.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)
            if scale.size == 1:
                scale = np.repeat(scale, 3)
            local_top_y = float((center[1] + 0.5 * extents[1]) * scale[1])
            rim_z = float(bowl_center_z + local_top_y)
            return max(rim_z, float(table_height + 0.02))
        except Exception:
            return float(table_height + 0.03)

    def _settle_scene(self, steps):
        for _ in range(int(steps)):
            self.scene.step()

        if self.render_freq and hasattr(self, "viewer"):
            self._update_render()
            self.viewer.render()

    def _sample_slot_xy(self, slot_id: int, allow_perturb: bool = True):
        x, y = SLOT_XY[slot_id]
        if not allow_perturb:
            return float(x), float(y)

        px = getattr(self._scene_cfg, "layout_perturb_x", 0.0)
        py = getattr(self._scene_cfg, "layout_perturb_y", 0.0)
        x += np.random.uniform(-px, px)
        y += np.random.uniform(-py, py)
        return float(x), float(y)

    @staticmethod
    def _half_height_from_def(obj_def: ObjectDef):
        if obj_def.shape == "cube":
            return float(obj_def.size[0])
        if obj_def.shape == "sphere":
            return float(obj_def.size[0])
        if obj_def.shape == "cylinder":
            return float(obj_def.size[1])
        raise ValueError(f"Unsupported shape: {obj_def.shape}")

    def _object_half_height(self, obj_name: str):
        return self._half_height_from_def(self._object_defs[obj_name])

    def _create_object_actor(self, obj_def: ObjectDef, x: float, y: float, z: float):
        if obj_def.shape == "cube":
            hs = float(obj_def.size[0])
            return create_box(
                scene=self,
                pose=sapien.Pose([x, y, z]),
                half_size=(hs, hs, hs),
                color=obj_def.color,
                name=obj_def.name,
                friction=obj_def.friction,
                mass=obj_def.mass,
                boxtype=obj_def.boxtype,
            )

        if obj_def.shape == "sphere":
            radius = float(obj_def.size[0])
            return create_sphere(
                scene=self,
                pose=sapien.Pose([x, y, z]),
                radius=radius,
                color=obj_def.color,
                name=obj_def.name,
                friction=obj_def.friction,
                mass=obj_def.mass,
            )

        if obj_def.shape == "cylinder":
            radius, half_height = float(obj_def.size[0]), float(obj_def.size[1])
            return create_standing_cylinder(
                scene=self,
                pose=sapien.Pose([x, y, z]),
                radius=radius,
                half_height=half_height,
                color=obj_def.color,
                name=obj_def.name,
                friction=obj_def.friction,
                mass=obj_def.mass,
            )

        raise ValueError(f"Unsupported object shape: {obj_def.shape}")

    def _register_object(self, obj_def: ObjectDef, obj_actor):
        self.objects[obj_def.name] = obj_actor
        self.object_dict[obj_def.name] = obj_actor
        self._object_defs[obj_def.name] = obj_def
        self.add_prohibit_area(obj_actor, padding=0.08)

    def _ensure_bowl_cube_exposure(self, obj_name: str, container_name: str):
        obj_def = self._object_defs[obj_name]
        if obj_def.shape != "cube":
            return

        obj_actor = self.objects[obj_name]
        settled_pose = obj_actor.get_pose()
        hz = float(obj_def.size[0])
        settled_top_z = float(settled_pose.p[2] + hz)

        rim_z = float(self._bowl_rim_z.get(container_name, self._table_height + 0.02))
        min_top_needed = float(rim_z + self.MIN_EXPOSED_ABOVE_RIM)
        if settled_top_z >= min_top_needed:
            return

        dz = float(min_top_needed - settled_top_z)
        lifted_p = settled_pose.p.copy()
        lifted_p[2] = float(lifted_p[2] + dz)
        obj_actor.actor.set_pose(sapien.Pose(lifted_p, settled_pose.q))
        print(
            f"cube auto-lift for exposure: {obj_name}, dz={dz:.4f}, "
            f"top_z {settled_top_z:.4f} -> {min_top_needed:.4f}"
        )
        self._settle_scene(self.POST_REPOSITION_SETTLE_STEPS)

    def load_actors(self):
        task_def = TASKS[self.instruction_type]
        scene_name = getattr(self, "_scene_override", None) or task_def.scene
        scene_cfg = SCENES[scene_name]

        self._scene_cfg = scene_cfg
        self._task_def = task_def
        self._table_height = 0.74 + self.table_z_bias

        self.object_dict = {}
        self.objects: dict[str, object] = {}
        self.actors: dict[str, object] = {}
        self._object_defs: dict[str, ObjectDef] = {}
        self._actor_defs: dict[str, ActorDef] = {}
        self._bowl_rim_z: dict[str, float] = {}

        # 1) create actors (bowl)
        for actor_def in scene_cfg.actors:
            ax, ay = self._sample_slot_xy(actor_def.slot, allow_perturb=True)
            az = self._table_height + float(actor_def.center_z_offset)

            actor = create_actor(
                self,
                pose=sapien.Pose([ax, ay, az], actor_def.quat),
                modelname=actor_def.modelname,
                model_id=actor_def.model_id,
                convex=True,
                is_static=actor_def.is_static,
                scale_multiplier=actor_def.scale,
            )
            actor.set_name(actor_def.name)

            self.actors[actor_def.name] = actor
            self.object_dict[actor_def.name] = actor
            self._actor_defs[actor_def.name] = actor_def
            self.add_prohibit_area(actor, padding=0.09)

            rim_z = float(self._estimate_bowl_rim_z(actor, az, self._table_height))
            self._bowl_rim_z[actor_def.name] = rim_z
            print(
                f"create actor: {actor_def.name} at ({ax:.3f}, {ay:.3f}, {az:.3f}), "
                f"rim_z={rim_z:.3f}"
            )

        # 2) create base objects on table
        stacked_defs = []
        in_container_defs = []

        for obj_def in scene_cfg.objects:
            if obj_def.on_top_of is not None:
                stacked_defs.append(obj_def)
                continue
            if obj_def.in_container is not None:
                in_container_defs.append(obj_def)
                continue

            x, y = self._sample_slot_xy(obj_def.slot, allow_perturb=True)
            z = float(self._table_height + self._half_height_from_def(obj_def) + self.TABLE_CLEARANCE)
            obj = self._create_object_actor(obj_def, x, y, z)
            self._register_object(obj_def, obj)
            print(f"create object: {obj_def.name} at ({x:.3f}, {y:.3f}, {z:.3f})")

        # 3) create objects in bowl
        for obj_def in in_container_defs:
            container_name = obj_def.in_container
            container_obj = self.object_dict[container_name]
            container_pose = container_obj.get_pose().p
            hz = self._half_height_from_def(obj_def)
            rim_z = float(self._bowl_rim_z.get(container_name, self._table_height + 0.03))
            z = float(rim_z + hz + self.BOWL_DROP_MARGIN)

            obj = self._create_object_actor(obj_def, float(container_pose[0]), float(container_pose[1]), z)
            self._register_object(obj_def, obj)
            print(
                f"create in-container object: {obj_def.name} in {container_name} "
                f"at ({container_pose[0]:.3f}, {container_pose[1]:.3f}, {z:.3f})"
            )

        self._settle_scene(self.SETTLE_STEPS)

        for obj_def in in_container_defs:
            self._ensure_bowl_cube_exposure(obj_def.name, obj_def.in_container)

        # 4) create stacked objects after base objects settle
        for obj_def in stacked_defs:
            base_name = obj_def.on_top_of
            base_pose = self.objects[base_name].get_pose().p
            base_h = self._object_half_height(base_name)
            obj_h = self._half_height_from_def(obj_def)

            x = float(base_pose[0])
            y = float(base_pose[1])
            z = float(base_pose[2] + base_h + obj_h + self.STACK_GAP)

            obj = self._create_object_actor(obj_def, x, y, z)
            self._register_object(obj_def, obj)
            print(f"create stacked object: {obj_def.name} on {base_name} at ({x:.3f}, {y:.3f}, {z:.3f})")

        self._settle_scene(self.SETTLE_STEPS)

        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(3):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        self.action_type = task_def.action
        self.source_name = task_def.source
        self.target_name = task_def.target
        self.place_slot = task_def.place_slot
        self.arm_tag = ArmTag(task_def.arm)
        self.second_source_name = task_def.second_source
        self.second_arm_tag = ArmTag(task_def.second_arm) if task_def.second_arm else None

        self.source_obj = self.object_dict[self.source_name]
        self.target_obj = self.object_dict[self.target_name] if self.target_name else None
        self.container_name = self._object_defs[self.source_name].in_container

        self._pickup_baseline_z: dict[str, float] = {}

        print(f"action type: {self.action_type}")
        print(f"source: {self.source_name}")
        print(f"target: {self.target_name}")
        print(f"arm: {self.arm_tag}")
        if self.second_source_name:
            print(f"second source: {self.second_source_name}")
            print(f"second arm: {self.second_arm_tag}")

    def _is_precision_pick(self, obj_name: str):
        obj_def = self._object_defs[obj_name]
        return obj_def.shape == "cube" and obj_def.in_container is not None

    def _record_pickup_baseline(self, obj_name: str):
        self._pickup_baseline_z[obj_name] = float(self.object_dict[obj_name].get_pose().p[2])

    def _arm_tcp_pose(self, arm_tag: ArmTag):
        return self.robot.get_left_tcp_pose() if arm_tag == "left" else self.robot.get_right_tcp_pose()

    def _arm_ee_pose(self, arm_tag: ArmTag):
        return self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose()

    def _arm_gripper_val(self, arm_tag: ArmTag):
        return float(self.robot.get_left_gripper_val()) if arm_tag == "left" else float(self.robot.get_right_gripper_val())

    def _grasp_normal(self, obj_name: str, arm_tag: ArmTag):
        obj_pose = self.object_dict[obj_name].get_pose().p
        cx, cy, cz = float(obj_pose[0]), float(obj_pose[1]), float(obj_pose[2])

        grasp_bias_z = 0.0
        obj_def = self._object_defs[obj_name]
        if obj_def.shape == "cube":
            grasp_bias_z = self.CUBE_GRASP_BIAS_BASE
        if obj_def.shape == "cylinder":
            grasp_bias_z = self.CYLINDER_GRASP_BIAS_BASE

        # For a cube stacked on another object, grasp higher to reduce grabbing the lower object.
        if obj_def.shape == "cube" and obj_def.on_top_of is not None:
            grasp_bias_z = max(
                grasp_bias_z,
                min(
                self.STACKED_CUBE_GRASP_BIAS_MAX,
                self._object_half_height(obj_name) * self.STACKED_CUBE_GRASP_HEIGHT_RATIO,
                ),
            )
        # For a stacked cylinder, use a larger upward bias than cubes.
        if obj_def.shape == "cylinder" and obj_def.on_top_of is not None:
            grasp_bias_z = max(
                grasp_bias_z,
                min(
                self.STACKED_CYLINDER_GRASP_BIAS_MAX,
                self._object_half_height(obj_name) * self.STACKED_CYLINDER_GRASP_HEIGHT_RATIO,
                ),
            )

        pre_pose = [cx, cy, cz + self.GRIPPER_OFFSET + self.NORMAL_PRE_GRASP_DIS + grasp_bias_z] + self.GRASP_QUAT_TOP_DOWN
        grasp_pose = [cx, cy, cz + self.GRIPPER_OFFSET + grasp_bias_z] + self.GRASP_QUAT_TOP_DOWN

        self.move(
            (
                arm_tag,
                [
                    Action(arm_tag, "move", target_pose=pre_pose),
                    Action(arm_tag, "move", target_pose=grasp_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
                    Action(arm_tag, "close", target_gripper_pos=0.0),
                ],
            )
        )
        return bool(self.plan_success)

    def _grasp_precision_in_bowl(self, obj_name: str, arm_tag: ArmTag):
        obj_pose = self.object_dict[obj_name].get_pose().p
        cx, cy, cz = float(obj_pose[0]), float(obj_pose[1]), float(obj_pose[2])

        hz = self._object_half_height(obj_name)
        cube_bottom_z = float(cz - hz)
        cube_top_z = float(cz + hz)
        cube_side = float(hz * 2.0)

        fingertip_z = float(cube_bottom_z + self.CUBE_GRASP_HEIGHT_RATIO * cube_side)
        fingertip_z = min(fingertip_z, float(cube_top_z - 0.001))
        fingertip_z = max(fingertip_z, float(self._table_height + 0.005))

        container_name = self._object_defs[obj_name].in_container
        if container_name:
            rim_z = float(self._bowl_rim_z.get(container_name, self._table_height + 0.02))
            fingertip_z = max(fingertip_z, float(rim_z + self.MIN_EXPOSED_ABOVE_RIM * 0.5))

        # Slightly lower in-bowl grasp target for more stable contacts.
        fingertip_z = float(fingertip_z - self.BOWL_GRASP_EXTRA_LOWER_Z)
        fingertip_z = min(fingertip_z, float(cube_top_z - 0.001))
        fingertip_z = max(fingertip_z, float(cube_bottom_z + 0.003))
        fingertip_z = max(fingertip_z, float(self._table_height + 0.005))

        target_tcp_z = float(fingertip_z + self.TCP_TO_FINGERTIP_BIAS_Z)
        approach_pose = [cx, cy, target_tcp_z + self.APPROACH_DIS] + self.GRASP_QUAT_TOP_DOWN

        self.move((arm_tag, [Action(arm_tag, "move", target_pose=approach_pose)]))
        if not self.plan_success:
            return False

        tcp_now = self._arm_tcp_pose(arm_tag)
        ee_now = self._arm_ee_pose(arm_tag)

        dz = float(np.clip(target_tcp_z - float(tcp_now[2]), -0.20, 0.05))
        descend_pose = ee_now.copy()
        descend_pose[0] = cx
        descend_pose[1] = cy
        descend_pose[2] = float(descend_pose[2] + dz)
        descend_pose[3:] = self.GRASP_QUAT_TOP_DOWN

        self.move(
            (
                arm_tag,
                [Action(arm_tag, "move", target_pose=descend_pose, constraint_pose=[1, 1, 1, 0, 0, 0])],
            )
        )
        if not self.plan_success:
            return False

        self.move((arm_tag, [Action(arm_tag, "close", target_gripper_pos=0.0)]))
        return bool(self.plan_success)

    def _grasp_and_lift(self, obj_name: str, arm_tag: ArmTag):
        self._record_pickup_baseline(obj_name)

        if self._is_precision_pick(obj_name):
            success = self._grasp_precision_in_bowl(obj_name, arm_tag)
        else:
            success = self._grasp_normal(obj_name, arm_tag)

        if not success:
            return False

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.LIFT_DIS))
        return bool(self.plan_success)

    def _place_held_object(self, arm_tag: ArmTag, px: float, py: float, pz: float):
        pre_place = [px, py, pz + self.GRIPPER_OFFSET + self.PLACE_APPROACH_DIS] + self.GRASP_QUAT_TOP_DOWN
        place_pose = [px, py, pz + self.GRIPPER_OFFSET] + self.GRASP_QUAT_TOP_DOWN

        self.move(
            (
                arm_tag,
                [
                    Action(arm_tag, "move", target_pose=pre_place),
                    Action(arm_tag, "move", target_pose=place_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
                    Action(arm_tag, "open", target_gripper_pos=1.0),
                ],
            )
        )
        if not self.plan_success:
            return False

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.PLACE_RETREAT_DIS))
        return bool(self.plan_success)

    def _return_home(self):
        self.together_move_to_pose(
            left_target_pose=self.robot.left_original_pose,
            right_target_pose=self.robot.right_original_pose,
        )
        return bool(self.plan_success)

    def play_once(self):
        if self.action_type == "pickup":
            return self._play_pickup()
        if self.action_type == "unstack_to_slot":
            return self._play_unstack_to_slot()
        if self.action_type == "stack_on_target":
            return self._play_stack_on_target()
        if self.action_type == "sequence_pickups":
            return self._play_sequence_pickups()

        print(f"[ERROR] Unknown action type: {self.action_type}")
        return self.info

    def _play_pickup(self):
        self.info["info"] = {
            "{scene}": self._scene_cfg.name,
            "{source}": self.source_name,
            "{arm}": str(self.arm_tag),
            "{action}": "pickup",
        }

        if not self._grasp_and_lift(self.source_name, self.arm_tag):
            self.info["info"]["{error}"] = "pickup_failed"
            return self.info

        if not self._return_home():
            self.info["info"]["{error}"] = "return_home_failed"
            return self.info

        self.info["info"]["{stage}"] = "done"
        return self.info

    def _play_unstack_to_slot(self):
        self.info["info"] = {
            "{scene}": self._scene_cfg.name,
            "{source}": self.source_name,
            "{slot}": self.place_slot,
            "{arm}": str(self.arm_tag),
            "{action}": "unstack_to_slot",
        }

        if not self._grasp_and_lift(self.source_name, self.arm_tag):
            self.info["info"]["{error}"] = "grasp_failed"
            return self.info

        px, py = SLOT_XY[self.place_slot]
        source_h = self._object_half_height(self.source_name)
        pz = float(self._table_height + source_h + self.TABLE_CLEARANCE)

        if not self._place_held_object(self.arm_tag, float(px), float(py), pz):
            self.info["info"]["{error}"] = "place_failed"
            return self.info

        if not self._return_home():
            self.info["info"]["{error}"] = "return_home_failed"
            return self.info

        self.info["info"]["{stage}"] = "done"
        return self.info

    def _play_stack_on_target(self):
        self.info["info"] = {
            "{scene}": self._scene_cfg.name,
            "{source}": self.source_name,
            "{target}": self.target_name,
            "{arm}": str(self.arm_tag),
            "{action}": "stack_on_target",
        }

        if not self._grasp_and_lift(self.source_name, self.arm_tag):
            self.info["info"]["{error}"] = "grasp_failed"
            return self.info

        target_pos = self.object_dict[self.target_name].get_pose().p
        target_h = self._object_half_height(self.target_name)
        source_h = self._object_half_height(self.source_name)
        place_z = float(target_pos[2] + target_h + source_h + self.STACK_GAP)

        if not self._place_held_object(self.arm_tag, float(target_pos[0]), float(target_pos[1]), place_z):
            self.info["info"]["{error}"] = "place_failed"
            return self.info

        if not self._return_home():
            self.info["info"]["{error}"] = "return_home_failed"
            return self.info

        self.info["info"]["{stage}"] = "done"
        return self.info

    def _play_sequence_pickups(self):
        self.info["info"] = {
            "{scene}": self._scene_cfg.name,
            "{source_1}": self.source_name,
            "{arm_1}": str(self.arm_tag),
            "{source_2}": self.second_source_name,
            "{arm_2}": str(self.second_arm_tag),
            "{action}": "sequence_pickups",
        }

        if not self._grasp_and_lift(self.source_name, self.arm_tag):
            self.info["info"]["{error}"] = "first_pick_failed"
            return self.info

        if not self._grasp_and_lift(self.second_source_name, self.second_arm_tag):
            self.info["info"]["{error}"] = "second_pick_failed"
            return self.info

        if not self._return_home():
            self.info["info"]["{error}"] = "return_home_failed"
            return self.info

        self.info["info"]["{stage}"] = "done"
        return self.info

    def _check_pickup_success(self, obj_name: str, arm_tag: ArmTag):
        baseline = float(self._pickup_baseline_z.get(obj_name, self.object_dict[obj_name].get_pose().p[2]))
        obj_z = float(self.object_dict[obj_name].get_pose().p[2])
        lifted = obj_z > (baseline + 0.04)
        gripper_closed = self._arm_gripper_val(arm_tag) < 0.55
        return bool(lifted and gripper_closed)

    def _check_place_to_slot_success(self):
        source_pos = self.object_dict[self.source_name].get_pose().p
        px, py = SLOT_XY[self.place_slot]
        source_h = self._object_half_height(self.source_name)
        expected_z = float(self._table_height + source_h + self.TABLE_CLEARANCE)

        xy_dist = np.linalg.norm(np.array([source_pos[0] - px, source_pos[1] - py], dtype=np.float32))
        z_err = abs(float(source_pos[2]) - expected_z)
        gripper_open = self._arm_gripper_val(self.arm_tag) > 0.6
        return bool(xy_dist < 0.05 and z_err < 0.05 and gripper_open)

    def _check_stack_success(self):
        source_pos = self.object_dict[self.source_name].get_pose().p
        target_pos = self.object_dict[self.target_name].get_pose().p
        source_h = self._object_half_height(self.source_name)
        target_h = self._object_half_height(self.target_name)
        expected_z = float(target_pos[2] + target_h + source_h + self.STACK_GAP)

        xy_dist = np.linalg.norm(np.array([source_pos[0] - target_pos[0], source_pos[1] - target_pos[1]], dtype=np.float32))
        z_err = abs(float(source_pos[2]) - expected_z)
        gripper_open = self._arm_gripper_val(self.arm_tag) > 0.6
        return bool(xy_dist < 0.045 and z_err < 0.04 and gripper_open)

    def check_success(self):
        if not self.plan_success:
            return False

        if self.action_type == "pickup":
            return self._check_pickup_success(self.source_name, self.arm_tag)

        if self.action_type == "sequence_pickups":
            ok1 = self._check_pickup_success(self.source_name, self.arm_tag)
            ok2 = self._check_pickup_success(self.second_source_name, self.second_arm_tag)
            return bool(ok1 and ok2)

        if self.action_type == "unstack_to_slot":
            return self._check_place_to_slot_success()

        if self.action_type == "stack_on_target":
            return self._check_stack_success()

        return False
