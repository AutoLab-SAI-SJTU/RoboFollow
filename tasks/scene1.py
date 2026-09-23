from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

@dataclass
class ObjectDef:
    name: str        # "red_sphere"
    shape: str       # "sphere" | "cube" | "cylinder"
    size: tuple      # sphere: (radius,) | cube: (half_size,) | cylinder: (radius, half_height)
    color: tuple     # (r, g, b) 0-1
    position: tuple  # (x, y) 

@dataclass
class ActorDef:
    name: str               # "green_bowl"
    modelname: str          # "002_bowl"
    position: tuple         # (x, y, z)
    model_id: int = 0 
    quat: tuple = (0.707107, 0.707107, 0, 0)  # default upright
    is_static: bool = False
    scale: float = 1.0

@dataclass
class PerturbRule:
    x_range: float = 0.01
    y_range: float = 0.01
    y_direction: str = "both"   # "up" | "down" | "both"

@dataclass 
class SceneConfig:
    name: str
    objects: list[ObjectDef] = field(default_factory=list)
    actors: list[ActorDef] = field(default_factory=list)
    perturb_rules: dict[str, PerturbRule] = field(default_factory=dict)
    default_perturb: PerturbRule = field(default_factory=PerturbRule)

@dataclass
class TaskDef:
    name: str              
    scene: str              
    source: str              
    target: Optional[str]   
    arm: str      # "left" | "right"
    action: str   # "pickup" | "stack" | "place_beside" | "place_behind" | "place_in_front" | "place_left_of"
    instruction: str = ""   
    level: int = 0          

SCENES: dict[str, SceneConfig] = {}

TASKS: dict[str, TaskDef] = {}

def register_scene(cfg: SceneConfig):
    SCENES[cfg.name] = cfg
    return cfg

def register_task(t: TaskDef):
    TASKS[t.name] = t
    return t

def _reg(name, scene, source, target, arm, action):
    return register_task(TaskDef(name, scene, source, target, arm, action))

register_scene(SceneConfig(
    name="default",
    objects=[
        # Row 1 (back row)
        ObjectDef("red_sphere",      "sphere",   (0.02,),         (1.0, 0.0, 0.0), (-0.14, -0.04)),  # left/back
        ObjectDef("yellow_block_1",  "cube",     (0.02,),         (1.0, 1.0, 0.0), (-0.02, -0.04)),  # mid/back
        ObjectDef("red_cylinder",    "cylinder", (0.025, 0.025),   (1.0, 0.0, 0.0), ( 0.10, -0.04)),  # right/back
        # Row 2 (front row)
        ObjectDef("yellow_block_2",  "cube",     (0.02,),         (1.0, 1.0, 0.0), (-0.14, -0.18)),  # left/front
        ObjectDef("green_sphere",    "sphere",   (0.02,),         (0.0, 1.0, 0.0), (-0.02, -0.18)),  # mid/front
        ObjectDef("yellow_block_3",  "cube",     (0.02,),         (1.0, 1.0, 0.0), ( 0.10, -0.18)),  # right/front
    ],
    actors=[],
    perturb_rules={
        # back row perturb down, front row perturb up, to avoid overlap
        "red_sphere":      PerturbRule(x_range=0.015, y_range=0.015, y_direction="down"),
        "yellow_block_1":  PerturbRule(x_range=0.015, y_range=0.015, y_direction="down"),
        "red_cylinder":    PerturbRule(x_range=0.015, y_range=0.015, y_direction="down"),
        "yellow_block_2":  PerturbRule(x_range=0.015, y_range=0.015, y_direction="up"),
        "green_sphere":    PerturbRule(x_range=0.015, y_range=0.015, y_direction="up"),
        "yellow_block_3":  PerturbRule(x_range=0.015, y_range=0.015, y_direction="up"),
    },
))

_reg("pickup_yellow_block_1_left",    "default", "yellow_block_1",  None, "left",  "pickup")

_reg("pickup_yellow_block_1_right",   "default", "yellow_block_1",  None, "right", "pickup")

_reg("pickup_yellow_block_2",         "default", "yellow_block_2",  None, "left",  "pickup")

_reg("pickup_yellow_block_3",         "default", "yellow_block_3",  None, "right", "pickup")

_reg("place_yellow_block_1_behind_red_cylinder",   "default", "yellow_block_1", "red_cylinder",  "right", "place_behind")

_reg("place_yellow_block_1_beside_red_cylinder",   "default", "yellow_block_1", "red_cylinder",  "right", "place_beside")

_reg("place_yellow_block_1_behind_red_sphere",     "default", "yellow_block_1", "red_sphere",    "left",  "place_behind")

_reg("place_yellow_block_1_left_of_red_sphere",    "default", "yellow_block_1", "red_sphere",    "left",  "place_left_of")

_reg("place_yellow_block_2_behind_green_sphere",   "default", "yellow_block_2", "green_sphere",  "left",  "place_behind")

_reg("place_yellow_block_2_in_front_of_green_sphere", "default", "yellow_block_2", "green_sphere", "left", "place_in_front")

_reg("place_yellow_block_2_behind_red_sphere",     "default", "yellow_block_2", "red_sphere",    "left",  "place_behind")

_reg("place_yellow_block_2_left_of_red_sphere",    "default", "yellow_block_2", "red_sphere",    "left",  "place_left_of")

_reg("place_yellow_block_3_behind_red_cylinder",   "default", "yellow_block_3", "red_cylinder",  "right", "place_behind")

_reg("place_yellow_block_3_beside_red_cylinder",   "default", "yellow_block_3", "red_cylinder",  "right", "place_beside")

_reg("place_yellow_block_3_behind_green_sphere",   "default", "yellow_block_3", "green_sphere",  "right", "place_behind")

_reg("place_yellow_block_3_in_front_of_green_sphere", "default", "yellow_block_3", "green_sphere", "right", "place_in_front")

def perturb_position(cx, cy, rule: PerturbRule):
    x = cx + np.random.uniform(-rule.x_range, rule.x_range)
    if rule.y_direction == "up":
        y = cy + np.random.uniform(0, rule.y_range)
    elif rule.y_direction == "down":
        y = cy - np.random.uniform(0, rule.y_range)
    else:
        y = cy + np.random.uniform(-rule.y_range, rule.y_range)
    return x, y
