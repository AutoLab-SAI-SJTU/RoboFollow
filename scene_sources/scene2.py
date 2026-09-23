from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, Union
import transforms3d as t3d

from envs.utils.action import Action
from robofollow.tasks.scene2 import *
from robofollow.geometry import create_box, create_sphere, create_standing_cylinder, create_actor


class scene2(Base_Task):
    def setup_demo(self, **kwargs):
        self.instruction_type = kwargs.get(
            "instruction_type",
            os.environ.get("ROBOTWIN_INSTRUCTION_TYPE", "pickup_red_cylinder")
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

        self.objects = []             # objects: sphere, box, cylinder
        self.actors = []              # actors          
        self.object_dict = {}         # name -> entity (both objects and actors)
        self._object_defs: dict[str, ObjectDef] = {}  # name -> definition

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
                    friction=0.12, mass=0.2
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

        print(f"action type: {self.action_type}")
        print(f"source: {self.source_name}")
        print(f"target: {self.target_name}")

    def play_once(self):
        if self.action_type == "pickup":
            return self._play_pickup()
        elif self.action_type == "stack":
            return self._play_stack()
        elif self.action_type == "push":
            return self._play_push()
        elif self.action_type == "place_bowl":
            return self._play_place_bowl()
        else:
            print(f"[ERROR] Unknown action type: {self.action_type}")
            return self.info

    def _get_arm_for_object(self, obj, use_right=False):
        if use_right:
            return ArmTag("right")
        return ArmTag("left")

    def _try_grasp(self, obj, pre_grasp_dis=0.08, grasp_dis=0.0, use_right=False):

        arm_tag = self._get_arm_for_object(obj, use_right=use_right)
        obj_pose = obj.get_pose()
        cx, cy, cz = obj_pose.p[0], obj_pose.p[1], obj_pose.p[2]
        grasp_quat = [-0.5, 0.5, -0.5, -0.5]   # Top-down 
        gripper_offset = 0.12 #The distance from the claw to the end effector is approximately 0.12m

        pre_grasp_pose = [cx, cy, cz + gripper_offset + pre_grasp_dis] + grasp_quat
        grasp_pose = [cx, cy, cz + gripper_offset + grasp_dis] + grasp_quat

        grasp_action = (arm_tag, [
            Action(arm_tag, "move", target_pose=pre_grasp_pose),
            Action(arm_tag, "move", target_pose=grasp_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
            Action(arm_tag, "close", target_gripper_pos=0.0),
        ])

        return arm_tag, grasp_action

    def _play_pickup(self):
        arm_tag, grasp_action = self._try_grasp(self.source_obj)

        if arm_tag is None:
            self.plan_success = False
            return self.info

        self.move(grasp_action)

        if not self.plan_success:
            print(f"  [ERROR] Grasp planning failed!")
            return self.info

        # lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        # back to original pose
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


        arm_tag, grasp_action = self._try_grasp(self.source_obj)

        if arm_tag is None:
            self.plan_success = False
            return self.info

        self.move(grasp_action)

        if not self.plan_success:
            return self.info

        # lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        if not self.plan_success:
            return self.info
  
        target_pose = self.target_obj.get_pose()
        tx, ty, tz = target_pose.p[0], target_pose.p[1], target_pose.p[2]

        # get target object height
        target_def = self._object_defs[self.target_name]
        if target_def.shape == "cylinder":
            target_height = target_def.size[1] * 2
        else:
            target_height = target_def.size[0] * 2  

        source_def = self._object_defs[self.source_name]
        if source_def.shape == "cylinder":
            source_half_height = source_def.size[1]
        else:
            source_half_height = source_def.size[0]  

        table_height = 0.74 + self.table_z_bias
        place_z = table_height + target_height + source_half_height + 0.01 

        grasp_quat = [-0.5, 0.5, -0.5, -0.5]
        gripper_offset = 0.12

        pre_place_pose = [tx, ty, place_z + gripper_offset + 0.06] + grasp_quat
        place_pose = [tx, ty, place_z + gripper_offset] + grasp_quat

        place_action = (arm_tag, [
            Action(arm_tag, "move", target_pose=pre_place_pose),
            Action(arm_tag, "move", target_pose=place_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
            Action(arm_tag, "open", target_gripper_pos=1.0),
        ])

        self.move(place_action)

        if not self.plan_success:
            return self.info

        # lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))

        self.together_move_to_pose(
            left_target_pose=self.robot.left_original_pose,
            right_target_pose=self.robot.right_original_pose,
        )

        self.info["info"] = {
            "{source}": self.source_name,
            "{target}": self.target_name,
            "{arm}": str(arm_tag),
            "{action}": "stack",
        }
        return self.info

    def _get_object_radius(self, obj_name):

        obj_def = self._object_defs.get(obj_name)
        if obj_def is None:
            return 0.025  

        if obj_def.shape == "sphere":
            return obj_def.size[0]
        elif obj_def.shape == "cylinder":
            return obj_def.size[0]
        else:  # cube
            return obj_def.size[0] * np.sqrt(2)

    def _play_push(self):

        source_pose = self.source_obj.get_pose()
        target_pose = self.target_obj.get_pose()

        sx, sy, sz = source_pose.p[0], source_pose.p[1], source_pose.p[2]
        tx, ty, tz = target_pose.p[0], target_pose.p[1], target_pose.p[2]

        distance = np.sqrt((tx - sx)**2 + (ty - sy)**2)

        if distance < 0.01:
            print(f"  [WARNING] Source and target too close!")
            self.plan_success = False
            return self.info

        push_distance = 0.05

        arm_tag = self._get_arm_for_object(self.source_obj, use_right=False) 
        grasp_quat = [-0.5, 0.5, -0.5, -0.5]
        gripper_offset = 0.12

        source_radius = self._get_object_radius(self.source_name)

        gripper_half_width = 0.01
        contact_offset = source_radius + gripper_half_width + 0.005

        # push direction
        dir_x = (tx - sx) / distance
        dir_y = (ty - sy) / distance

        approach_offset = contact_offset + 0.02
        approach_x = sx - dir_x * approach_offset
        approach_y = sy - dir_y * approach_offset

        contact_x = sx - dir_x * contact_offset
        contact_y = sy - dir_y * contact_offset
        push_end_x = contact_x + dir_x * push_distance
        push_end_y = contact_y + dir_y * push_distance

        print(f"  push {self.source_name} -> {self.target_name}")
        print(f"  source: ({sx:.3f}, {sy:.3f}), target: ({tx:.3f}, {ty:.3f})")
        print(f"  push distance: {push_distance:.3f}m, radius: {source_radius:.3f}m")
        print(f"  approach position: ({approach_x:.3f}, {approach_y:.3f})")

        table_height = 0.74 + self.table_z_bias
        push_height = table_height + 0.02  

        # move to approach position
        pre_approach_pose = [approach_x, approach_y, push_height + gripper_offset + 0.10] + grasp_quat

        approach_action = (arm_tag, [
            Action(arm_tag, "move", target_pose=pre_approach_pose),
        ])

        self.move(approach_action)
        if not self.plan_success:
            print(f"    ✗ move to approach position failed")
            return self.info

        # close gripper
        close_action = (arm_tag, [
            Action(arm_tag, "close", target_gripper_pos=0.0),
        ])
        self.move(close_action)

        # down
        approach_pose = [approach_x, approach_y, push_height + gripper_offset] + grasp_quat

        descend_action = (arm_tag, [
            Action(arm_tag, "move", target_pose=approach_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
        ])
        self.move(descend_action)
        if not self.plan_success:
            print(f"    ✗ Failed to reach the required pushing height")
            return self.info

        # push
        push_pose = [push_end_x, push_end_y, push_height + gripper_offset] + grasp_quat

        push_action = (arm_tag, [
            Action(arm_tag, "move", target_pose=push_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
        ])
        self.move(push_action)
        if not self.plan_success:
            print(f"    ✗ Failed to push")
            return self.info

        # Move a little backward in the opposite direction to the pushing direction
        retreat_x = push_end_x - dir_x * 0.03
        retreat_y = push_end_y - dir_y * 0.03
        retreat_pose = [retreat_x, retreat_y, push_height + gripper_offset] + grasp_quat

        retreat_action = (arm_tag, [
            Action(arm_tag, "move", target_pose=retreat_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
        ])
        self.move(retreat_action)

        # lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        # return to original pose
        self.together_move_to_pose(
            left_target_pose=self.robot.left_original_pose,
            right_target_pose=self.robot.right_original_pose,
        )

        self.info["info"] = {
            "{source}": self.source_name,
            "{target}": self.target_name,
            "{arm}": str(arm_tag),
            "{action}": "push",
        }
        return self.info




    def _play_place_bowl(self):


        arm_tag, grasp_action = self._try_grasp(self.source_obj, use_right=True)

        if arm_tag is None:
            self.plan_success = False
            return self.info

        self.move(grasp_action)

        if not self.plan_success:
            return self.info

        # lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        if not self.plan_success:
            return self.info

        bowl_pose = self.target_obj.get_pose()
        bx, by = bowl_pose.p[0], bowl_pose.p[1]
        table_height = 0.74 + self.table_z_bias

        # calculate place height (bowl inner bottom + object height)
        source_def = self._object_defs[self.source_name]

        # bowl inner bottom is about 0.02m above table height
        bowl_inner_z = table_height + 0.02

        if source_def.shape == "sphere":
            place_z = bowl_inner_z + source_def.size[0]  # + radius
        elif source_def.shape == "cylinder":
            place_z = bowl_inner_z + source_def.size[1]  # + half height
        else:
            place_z = bowl_inner_z + source_def.size[0]  # + half side length

        # directly calculate place pose
        grasp_quat = [-0.5, 0.5, -0.5, -0.5]
        gripper_offset = 0.12

        pre_place_pose = [bx, by, place_z + gripper_offset + 0.08] + grasp_quat
        place_pose = [bx, by, place_z + gripper_offset + 0.02] + grasp_quat

        place_action = (arm_tag, [
            Action(arm_tag, "move", target_pose=pre_place_pose),
            Action(arm_tag, "move", target_pose=place_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
            Action(arm_tag, "open", target_gripper_pos=1.0),
        ])

        self.move(place_action)

        if not self.plan_success:
            return self.info

        # lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))

        # return to original pose
        self.together_move_to_pose(
            left_target_pose=self.robot.left_original_pose,
            right_target_pose=self.robot.right_original_pose,
        )

        self.info["info"] = {
            "{source}": self.source_name,
            "{target}": self.target_name,
            "{arm}": str(arm_tag),
            "{action}": "place_bowl",
        }
        return self.info


    def check_success(self):
        if self.action_type == "pickup":
            return self._check_pickup()
        elif self.action_type == "stack":
            return self._check_stack()
        elif self.action_type == "push":
            return self._check_push()
        elif self.action_type == "place_bowl":
            return self._check_place_bowl()
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

    def _check_push(self):
        return self.plan_success

    def _check_place_bowl(self):
        source_pos = self.source_obj.get_pose().p
        bowl_pos = self.target_obj.get_pose().p

        xy_distance = np.sqrt(
            (source_pos[0] - bowl_pos[0])**2 +
            (source_pos[1] - bowl_pos[1])**2
        )

        return (xy_distance < 0.03 and
                self.is_left_gripper_open() and self.is_right_gripper_open())
