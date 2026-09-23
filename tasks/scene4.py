from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

@dataclass
class ObjectDef:
    name: str
    shape: str  # "cube" | "sphere" | "cylinder"
    size: tuple
    color: tuple
    slot: int | None = None
    on_top_of: str | None = None
    in_container: str | None = None
    friction: float = 1.0
    mass: float = 0.10
    boxtype: str = "default"

@dataclass
class ActorDef:
    name: str
    modelname: str
    model_id: int
    slot: int
    scale: float = 0.7
    quat: tuple = (0.5, 0.5, 0.5, 0.5)
    is_static: bool = True
    center_z_offset: float = 0.02

@dataclass
class SceneConfig:
    name: str
    objects: list[ObjectDef] = field(default_factory=list)
    actors: list[ActorDef] = field(default_factory=list)
    layout_perturb_x: float = 0.01
    layout_perturb_y: float = 0.01

@dataclass
class TaskDef:
    name: str
    scene: str
    action: str  # "pickup" | "unstack_to_slot" | "stack_on_target" | "sequence_pickups"
    source: str
    arm: str
    target: str | None = None
    place_slot: int | None = None
    second_source: str | None = None
    second_arm: str | None = None
    instruction: str = ""
    level: int = 0

SCENES: dict[str, SceneConfig] = {}

TASKS: dict[str, TaskDef] = {}

def register_scene(cfg: SceneConfig):
    SCENES[cfg.name] = cfg
    return cfg

def register_task(task: TaskDef):
    TASKS[task.name] = task
    return task

def _reg(name, scene, action, source, arm, target=None, place_slot=None, second_source=None, second_arm=None):
    return register_task(
        TaskDef(
            name=name,
            scene=scene,
            action=action,
            source=source,
            arm=arm,
            target=target,
            place_slot=place_slot,
            second_source=second_source,
            second_arm=second_arm,
        )
    )

SLOT_XY = {
    1: (-0.14, -0.06),
    2: (-0.02, -0.06),
    3: (0.10, -0.06),
    4: (-0.14, -0.18),
    5: (-0.02, -0.18),
    6: (0.10, -0.18),
}

CUBE_HALF = 0.024

CUBE_FRICTION = 0.1

CUBE_MASS = 0.2

SPHERE_RADIUS = 0.024

SPHERE_FRICTION = 1.0

SPHERE_MASS = 0.10

CYLINDER_RADIUS = 0.024

CYLINDER_HALF_HEIGHT = 0.025

CYLINDER_FRICTION = 0.1

CYLINDER_MASS = 0.2

register_scene(
    SceneConfig(
        name="scene4_1",
        actors=[
            ActorDef(name="green_bowl", modelname="002_bowl", model_id=3, slot=1, scale=0.7),
        ],
        objects=[
            ObjectDef(
                name="red_cube_in_bowl",
                shape="cube",
                size=(CUBE_HALF,),
                color=(0.95, 0.20, 0.20),
                slot=1,
                in_container="green_bowl",
                friction=CUBE_FRICTION,
                mass=CUBE_MASS,
            ),
            ObjectDef(
                name="yellow_cylinder",
                shape="cylinder",
                size=(CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT),
                color=(1.0, 0.85, 0.10),
                slot=2,
                friction=CYLINDER_FRICTION,
                mass=CYLINDER_MASS,
            ),
            ObjectDef(
                name="blue_cube_right",
                shape="cube",
                size=(CUBE_HALF,),
                color=(0.10, 0.30, 0.95),
                slot=3,
                friction=CUBE_FRICTION,
                mass=CUBE_MASS,
            ),
            ObjectDef(
                name="green_cylinder_top",
                shape="cylinder",
                size=(CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT),
                color=(0.10, 0.80, 0.20),
                slot=3,
                on_top_of="blue_cube_right",
                friction=CYLINDER_FRICTION,
                mass=CYLINDER_MASS,
            ),
            ObjectDef(
                name="blue_sphere",
                shape="sphere",
                size=(SPHERE_RADIUS,),
                color=(0.10, 0.30, 0.95),
                slot=4,
                friction=SPHERE_FRICTION,
                mass=SPHERE_MASS,
            ),
            ObjectDef(
                name="red_sphere",
                shape="sphere",
                size=(SPHERE_RADIUS,),
                color=(0.95, 0.20, 0.20),
                slot=5,
                friction=SPHERE_FRICTION,
                mass=SPHERE_MASS,
            ),
        ],
    )
)

register_scene(
    SceneConfig(
        name="scene4_2",
        actors=[
            ActorDef(name="green_bowl", modelname="002_bowl", model_id=3, slot=1, scale=0.7),
        ],
        objects=[
            ObjectDef(
                name="yellow_cylinder",
                shape="cylinder",
                size=(CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT),
                color=(1.0, 0.85, 0.10),
                slot=2,
                friction=CYLINDER_FRICTION,
                mass=CYLINDER_MASS,
            ),
            ObjectDef(
                name="blue_cube_right",
                shape="cube",
                size=(CUBE_HALF,),
                color=(0.10, 0.30, 0.95),
                slot=3,
                friction=CUBE_FRICTION,
                mass=CUBE_MASS,
            ),
            ObjectDef(
                name="green_cylinder_top",
                shape="cylinder",
                size=(CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT),
                color=(0.10, 0.80, 0.20),
                slot=3,
                on_top_of="blue_cube_right",
                friction=CYLINDER_FRICTION,
                mass=CYLINDER_MASS,
            ),
            ObjectDef(
                name="red_sphere",
                shape="sphere",
                size=(SPHERE_RADIUS,),
                color=(0.95, 0.20, 0.20),
                slot=4,
                friction=SPHERE_FRICTION,
                mass=SPHERE_MASS,
            ),
            ObjectDef(
                name="blue_sphere",
                shape="sphere",
                size=(SPHERE_RADIUS,),
                color=(0.10, 0.30, 0.95),
                slot=5,
                friction=SPHERE_FRICTION,
                mass=SPHERE_MASS,
            ),
        ],
    )
)

register_scene(
    SceneConfig(
        name="scene4_3",
        actors=[
            ActorDef(name="green_bowl", modelname="002_bowl", model_id=3, slot=1, scale=0.7),
        ],
        objects=[
            ObjectDef(
                name="yellow_cylinder",
                shape="cylinder",
                size=(CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT),
                color=(1.0, 0.85, 0.10),
                slot=2,
                friction=CYLINDER_FRICTION,
                mass=CYLINDER_MASS,
            ),
            ObjectDef(
                name="blue_cube_on_yellow",
                shape="cube",
                size=(CUBE_HALF,),
                color=(0.10, 0.30, 0.95),
                slot=2,
                on_top_of="yellow_cylinder",
                friction=CUBE_FRICTION,
                mass=CUBE_MASS,
            ),
            ObjectDef(
                name="green_cylinder_right",
                shape="cylinder",
                size=(CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT),
                color=(0.10, 0.80, 0.20),
                slot=3,
                friction=CYLINDER_FRICTION,
                mass=CYLINDER_MASS,
            ),
            ObjectDef(
                name="blue_sphere",
                shape="sphere",
                size=(SPHERE_RADIUS,),
                color=(0.10, 0.30, 0.95),
                slot=4,
                friction=SPHERE_FRICTION,
                mass=SPHERE_MASS,
            ),
            ObjectDef(
                name="red_sphere",
                shape="sphere",
                size=(SPHERE_RADIUS,),
                color=(0.95, 0.20, 0.20),
                slot=6,
                friction=SPHERE_FRICTION,
                mass=SPHERE_MASS,
            ),
        ],
    )
)

_reg("s41_l_pick_cube_in_bowl", "scene4_1", "pickup", "red_cube_in_bowl", "left")

_reg("s41_l_pick_yellow_cylinder", "scene4_1", "pickup", "yellow_cylinder", "left")

_reg("s41_r_pick_green_cylinder_top", "scene4_1", "pickup", "green_cylinder_top", "right")

_reg("s41_l_pick_blue_sphere", "scene4_1", "pickup", "blue_sphere", "left")

_reg("s41_l_pick_red_sphere", "scene4_1", "pickup", "red_sphere", "left")

_reg(
    "s41_r_unstack_green_cylinder_to_slot6",
    "scene4_1",
    "unstack_to_slot",
    "green_cylinder_top",
    "right",
    place_slot=6,
)

_reg("s41_r_stack_green_cylinder_on_yellow", "scene4_1", "stack_on_target", "green_cylinder_top", "right", target="yellow_cylinder")

_reg("s41_l_pick_blue_not_cube", "scene4_1", "pickup", "blue_sphere", "left")

_reg("s41_l_pick_red_not_in_bowl", "scene4_1", "pickup", "red_sphere", "left")

_reg("s41_l_pick_ball_not_leftmost", "scene4_1", "pickup", "red_sphere", "left")

_reg("s41_if_bowl_nonempty_then_l_pick_blue_else_r_stack", "scene4_1", "pickup", "blue_sphere", "left")

_reg(
    "s41_if_blue_right_of_red_then_l_pick_else_r_stack",
    "scene4_1",
    "stack_on_target",
    "green_cylinder_top",
    "right",
    target="yellow_cylinder",
)

_reg(
    "s41_if_object_on_right_cube_then_r_stack_else_l_pick_blue",
    "scene4_1",
    "stack_on_target",
    "green_cylinder_top",
    "right",
    target="yellow_cylinder",
)

_reg("s41_if_yellow_right_of_green_then_r_stack_else_l_pick_blue", "scene4_1", "pickup", "blue_sphere", "left")

_reg(
    "s41_seq_l_bowl_then_r_green_cylinder",
    "scene4_1",
    "sequence_pickups",
    "red_cube_in_bowl",
    "left",
    second_source="green_cylinder_top",
    second_arm="right",
)

_reg(
    "s41_seq_r_green_cylinder_then_l_bowl",
    "scene4_1",
    "sequence_pickups",
    "green_cylinder_top",
    "right",
    second_source="red_cube_in_bowl",
    second_arm="left",
)

_reg("s42_l_pick_yellow_cylinder", "scene4_2", "pickup", "yellow_cylinder", "left")

_reg(
    "s42_r_unstack_green_cylinder_to_slot6",
    "scene4_2",
    "unstack_to_slot",
    "green_cylinder_top",
    "right",
    place_slot=6,
)

_reg(
    "s42_r_stack_green_cylinder_on_yellow",
    "scene4_2",
    "stack_on_target",
    "green_cylinder_top",
    "right",
    target="yellow_cylinder",
)

_reg(
    "s42_if_bowl_nonempty_then_l_pick_blue_else_r_stack",
    "scene4_2",
    "stack_on_target",
    "green_cylinder_top",
    "right",
    target="yellow_cylinder",
)

_reg("s42_if_blue_right_of_red_then_l_pick_blue_else_r_stack", "scene4_2", "pickup", "blue_sphere", "left")

_reg(
    "s42_seq_l_red_ball_then_r_green_cylinder",
    "scene4_2",
    "sequence_pickups",
    "red_sphere",
    "left",
    second_source="green_cylinder_top",
    second_arm="right",
)

_reg(
    "s42_seq_r_green_cylinder_then_l_red_ball",
    "scene4_2",
    "sequence_pickups",
    "green_cylinder_top",
    "right",
    second_source="red_sphere",
    second_arm="left",
)

_reg(
    "s43_r_unstack_blue_cube_from_yellow_to_slot5",
    "scene4_3",
    "unstack_to_slot",
    "blue_cube_on_yellow",
    "right",
    place_slot=5,
)

_reg(
    "s43_r_stack_blue_cube_on_right_cylinder",
    "scene4_3",
    "stack_on_target",
    "blue_cube_on_yellow",
    "right",
    target="green_cylinder_right",
)

_reg("s43_if_bowl_nonempty_then_l_pick_blue_else_r_pick_red", "scene4_3", "pickup", "red_sphere", "right")

_reg("s43_r_pick_object_not_touching_table", "scene4_3", "pickup", "blue_cube_on_yellow", "right")
