from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, Union
import transforms3d as t3d

from envs.utils.action import Action
from robofollow.tasks.scene1 import *
from robofollow.geometry import create_box, create_sphere, create_standing_cylinder, create_actor


class scene1(Base_Task):
    def setup_demo(self, **kwargs):
        self.instruction_type = kwargs.get(
            "instruction_type",
            os.environ.get("ROBOTWIN_INSTRUCTION_TYPE", "pickup_yellow_block_1_left")
        )
        self._scene_override = kwargs.get("scene_name", None)
        print(f"Instruction type: {self.instruction_type}")
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        task_def = TASKS[self.instruction_type]
        scene_name = getattr(self, "_scene_override", None) or task_def.scene
        scene_cfg = SCENES[scene_name]

        self._scene_cfg = scene_cfg
        self._task_def = task_def

        table_height = 0.74 + self.table_z_bias

        self.objects = []
        self.actors = []
        self.object_dict = {}  # Object and actor names mapped to entities.
        self._object_defs: dict[str, ObjectDef] = {}

        for obj_def in scene_cfg.objects:
            rule = scene_cfg.perturb_rules.get(obj_def.name, scene_cfg.default_perturb) 
            x, y = perturb_position(*obj_def.position, rule)

            if obj_def.shape == "sphere":
                radius = obj_def.size[0]
                z = table_height + radius
                sphere_pose = sapien.Pose([x, y, z])
                obj = create_sphere(
                    scene=self, pose=sphere_pose,
                    radius=radius, color=obj_def.color,
                    name=obj_def.name, friction=2.0, mass=0.05
                )
                print(f"create sphere: {obj_def.name} at ({x:.3f}, {y:.3f}, {z:.3f})")

            elif obj_def.shape == "cylinder":
                radius, half_height = obj_def.size
                z = table_height + half_height
                cyl_pose = sapien.Pose([x, y, z])
                obj = create_standing_cylinder(
                    scene=self, pose=cyl_pose,
                    radius=radius, half_height=half_height,
                    color=obj_def.color, name=obj_def.name,
                    friction=0.1, mass=0.2
                )
                print(f"create cylinder: {obj_def.name} at ({x:.3f}, {y:.3f}, {z:.3f})")

            else:  # cube
                half_size = obj_def.size[0]
                z = table_height + half_size + 0.001
                block_pose = sapien.Pose([x, y, z])
                obj = create_box(
                    scene=self, pose=block_pose,
                    half_size=(half_size, half_size, half_size),
                    color=obj_def.color, name=obj_def.name,
                    friction=0.1, mass=0.2,
                )
                print(f"create box: {obj_def.name} at ({x:.3f}, {y:.3f}, {z:.3f})")

            self.objects.append(obj)
            self.object_dict[obj_def.name] = obj
            self._object_defs[obj_def.name] = obj_def
            self.add_prohibit_area(obj, padding=0.10)

        if scene_cfg.actors:
            print(f"\ncreate actors:")
            for actor_def in scene_cfg.actors:
                ax, ay, az = actor_def.position
                if az < 0.1: 
                    az = 0.76
                
                actor_pose = sapien.Pose([ax, ay, az + self.table_z_bias], actor_def.quat)
                actor = create_actor(
                    self, pose=actor_pose,
                    modelname=actor_def.modelname,
                    model_id=actor_def.model_id,
                    convex=True, is_static=actor_def.is_static,
                    scale_multiplier=actor_def.scale
                )
                
                self.actors.append(actor)
                self.object_dict[actor_def.name] = actor

                print(f"  actor: {actor_def.name} at ({ax:.2f}, {ay:.2f})")

        self.action_type = task_def.action
        self.source_name = task_def.source
        self.target_name = task_def.target
        self.source_obj = self.object_dict.get(task_def.source)
        self.target_obj = self.object_dict.get(task_def.target)
        self._source_initial_z = float(self.source_obj.get_pose().p[2])

        print(f"action type: {self.action_type}")
        print(f"source: {self.source_name}")
        print(f"target: {self.target_name}")

    def play_once(self):
        if self.action_type == "pickup":
            return self._play_pickup()
        elif self.action_type == "stack":
            return self._play_stack()
        elif self.action_type == "place_beside":
            return self._play_place_beside()
        elif self.action_type == "place_behind":
            return self._play_place_behind()
        elif self.action_type == "place_in_front":
            return self._play_place_in_front()
        elif self.action_type == "place_left_of":
            return self._play_place_left_of()
        else:
            print(f"[ERROR] Unknown action type: {self.action_type}")
            return self.info

    def _get_arm_for_object(self, obj, use_right=False):
        if use_right:
            return ArmTag("right")
        return ArmTag("left")

    def _use_right_arm(self):
        """Return True if this task uses the right arm."""
        return self._task_def.arm == "right"

    def _try_grasp(self, obj, pre_grasp_dis=0.08, grasp_dis=0.0, use_right=False):

        arm_tag = self._get_arm_for_object(obj, use_right=use_right)
        obj_pose = obj.get_pose()
        cx, cy, cz = obj_pose.p[0], obj_pose.p[1], obj_pose.p[2]
        grasp_quat = [-0.5, 0.5, -0.5, -0.5]  # Top-down orientation.
        gripper_offset = 0.12  # Approximate fingertip-to-end-effector distance in meters.

        pre_grasp_pose = [cx, cy, cz + gripper_offset + pre_grasp_dis] + grasp_quat
        grasp_pose = [cx, cy, cz + gripper_offset + grasp_dis] + grasp_quat

        grasp_action = (arm_tag, [
            Action(arm_tag, "move", target_pose=pre_grasp_pose),
            Action(arm_tag, "move", target_pose=grasp_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
            Action(arm_tag, "close", target_gripper_pos=0.0),
        ])

        return arm_tag, grasp_action

    def _grasp_lift_source(self):
        """Grasp the source object and lift it. Returns arm_tag or None on failure."""
        use_right = self._use_right_arm()
        arm_tag, grasp_action = self._try_grasp(self.source_obj, use_right=use_right)

        if arm_tag is None:
            self.plan_success = False
            return None

        self.move(grasp_action)

        if not self.plan_success:
            print(f"  [ERROR] Grasp planning failed!")
            return None

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        if not self.plan_success:
            return None

        return arm_tag

    def _place_at_position(self, arm_tag, px, py, pz):
        """Move the grasped object to (px, py, pz) and release."""
        grasp_quat = [-0.5, 0.5, -0.5, -0.5]
        gripper_offset = 0.12

        pre_place_pose = [px, py, pz + gripper_offset + 0.06] + grasp_quat
        place_pose = [px, py, pz + gripper_offset] + grasp_quat

        place_action = (arm_tag, [
            Action(arm_tag, "move", target_pose=pre_place_pose),
            Action(arm_tag, "move", target_pose=place_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
            Action(arm_tag, "open", target_gripper_pos=1.0),
        ])

        self.move(place_action)

        if not self.plan_success:
            return

        # Lift clear of the object before returning home.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))

        self.together_move_to_pose(
            left_target_pose=self.robot.left_original_pose,
            right_target_pose=self.robot.right_original_pose,
        )

    def _play_pickup(self):
        use_right = self._use_right_arm()
        arm_tag, grasp_action = self._try_grasp(self.source_obj, use_right=use_right)

        if arm_tag is None:
            self.plan_success = False
            return self.info

        self.move(grasp_action)

        if not self.plan_success:
            print(f"  [ERROR] Grasp planning failed!")
            return self.info

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.together_move_to_pose(
            left_target_pose=self.robot.left_original_pose,
            right_target_pose=self.robot.right_original_pose,
        )

        self.info["info"] = {
            "{source}": self.source_name,
            "{arm}": str(arm_tag),
            "{action}": "pickup",
        }
        return self.info

    def _play_stack(self):
        arm_tag = self._grasp_lift_source()
        if arm_tag is None:
            return self.info

        target_pose = self.target_obj.get_pose()
        tx, ty, tz = target_pose.p[0], target_pose.p[1], target_pose.p[2]

        # Cylinder size stores radius and half-height separately.
        target_def = self._object_defs[self.target_name]
        if target_def.shape == "cylinder":
            target_height = target_def.size[1] * 2
        elif target_def.shape == "sphere":
            target_height = target_def.size[0] * 2
        else:
            target_height = target_def.size[0] * 2  

        source_def = self._object_defs[self.source_name]
        if source_def.shape == "cylinder":
            source_half_height = source_def.size[1]
        else:
            source_half_height = source_def.size[0]  

        table_height = 0.74 + self.table_z_bias
        place_z = table_height + target_height + source_half_height + 0.01 

        self._place_at_position(arm_tag, tx, ty, place_z)

        self.info["info"] = {
            "{source}": self.source_name,
            "{target}": self.target_name,
            "{arm}": str(arm_tag),
            "{action}": "stack",
        }
        return self.info

    def _play_place_beside(self):
        arm_tag = self._grasp_lift_source()
        if arm_tag is None:
            return self.info

        target_pose = self.target_obj.get_pose()
        tx, ty, tz = target_pose.p[0], target_pose.p[1], target_pose.p[2]

        # Use both object widths to leave clearance along positive x.
        target_def = self._object_defs[self.target_name]
        if target_def.shape == "cylinder":
            target_radius = target_def.size[0]
        elif target_def.shape == "sphere":
            target_radius = target_def.size[0]
        else:
            target_radius = target_def.size[0]

        source_def = self._object_defs[self.source_name]
        source_half = source_def.size[0]

        offset_x = target_radius + source_half + 0.06  # 6 cm clearance.

        table_height = 0.74 + self.table_z_bias
        place_z = table_height + source_half + 0.001

        self._place_at_position(arm_tag, tx + offset_x, ty, place_z)

        self.info["info"] = {
            "{source}": self.source_name,
            "{target}": self.target_name,
            "{arm}": str(arm_tag),
            "{action}": "place_beside",
        }
        return self.info

    def _play_place_behind(self):
        arm_tag = self._grasp_lift_source()
        if arm_tag is None:
            return self.info

        target_pose = self.target_obj.get_pose()
        tx, ty, tz = target_pose.p[0], target_pose.p[1], target_pose.p[2]

        source_def = self._object_defs[self.source_name]
        source_half = source_def.size[0]

        offset_y = 0.06  # Behind is positive y, away from the robot.

        table_height = 0.74 + self.table_z_bias
        place_z = table_height + source_half + 0.001

        self._place_at_position(arm_tag, tx, ty + offset_y, place_z)

        self.info["info"] = {
            "{source}": self.source_name,
            "{target}": self.target_name,
            "{arm}": str(arm_tag),
            "{action}": "place_behind",
        }
        return self.info

    def _play_place_in_front(self):
        arm_tag = self._grasp_lift_source()
        if arm_tag is None:
            return self.info

        target_pose = self.target_obj.get_pose()
        tx, ty, tz = target_pose.p[0], target_pose.p[1], target_pose.p[2]

        source_def = self._object_defs[self.source_name]
        source_half = source_def.size[0]

        offset_y = -0.05  # In front is negative y, toward the robot.

        table_height = 0.74 + self.table_z_bias
        place_z = table_height + source_half + 0.001

        self._place_at_position(arm_tag, tx, ty + offset_y, place_z)

        self.info["info"] = {
            "{source}": self.source_name,
            "{target}": self.target_name,
            "{arm}": str(arm_tag),
            "{action}": "place_in_front",
        }
        return self.info

    def _play_place_left_of(self):
        arm_tag = self._grasp_lift_source()
        if arm_tag is None:
            return self.info

        target_pose = self.target_obj.get_pose()
        tx, ty, tz = target_pose.p[0], target_pose.p[1], target_pose.p[2]

        # Use both object widths to leave clearance along negative x.
        target_def = self._object_defs[self.target_name]
        if target_def.shape == "cylinder":
            target_radius = target_def.size[0]
        elif target_def.shape == "sphere":
            target_radius = target_def.size[0]
        else:
            target_radius = target_def.size[0]

        source_def = self._object_defs[self.source_name]
        source_half = source_def.size[0]

        offset_x = -(target_radius + source_half + 0.04)  # 4 cm clearance.

        table_height = 0.74 + self.table_z_bias
        place_z = table_height + source_half + 0.001

        self._place_at_position(arm_tag, tx + offset_x, ty, place_z)

        self.info["info"] = {
            "{source}": self.source_name,
            "{target}": self.target_name,
            "{arm}": str(arm_tag),
            "{action}": "place_left_of",
        }
        return self.info

    def check_success(self):
        if self.action_type == "pickup":
            return self._check_pickup()
        elif self.action_type == "stack":
            return self._check_stack()
        elif self.action_type in ("place_beside", "place_behind", "place_in_front", "place_left_of"):
            return self._check_place_relative()
        else:
            return False

    def _check_pickup(self):
        source_pos = self.source_obj.get_pose().p
        return source_pos[2] > 0.85

    def _check_stack(self):
        source_pos = self.source_obj.get_pose().p
        target_pos = self.target_obj.get_pose().p

        xy_distance = np.sqrt(
            (source_pos[0] - target_pos[0])**2 +
            (source_pos[1] - target_pos[1])**2
        )
        z_diff = source_pos[2] - target_pos[2]

        return (xy_distance < 0.03 and z_diff > 0.02 and
                self.is_left_gripper_open() and self.is_right_gripper_open())

    def _check_place_relative(self):
        """Require a released object on the table in the requested relative direction."""
        if not (self.plan_success and self.target_obj is not None and
                self.is_left_gripper_open() and self.is_right_gripper_open()):
            return False
        source = np.asarray(self.source_obj.get_pose().p)
        target = np.asarray(self.target_obj.get_pose().p)
        if not (np.isfinite(source).all() and np.isfinite(target).all()):
            return False
        if abs(float(source[2]) - self._source_initial_z) > 0.02:
            return False
        directions = {
            "place_left_of": (-1., 0.),
            "place_beside": (1., 0.),
            "place_behind": (0., 1.),
            "place_in_front": (0., -1.),
        }
        direction = directions.get(self.action_type)
        if direction is None:
            return False
        delta = source[:2] - target[:2]
        distance = float(np.linalg.norm(delta))
        if not 1e-6 < distance <= 0.12:
            return False
        return float(np.dot(delta / distance, direction)) >= float(np.cos(np.deg2rad(45.)))
