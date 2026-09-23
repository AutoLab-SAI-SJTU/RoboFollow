from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

@dataclass
class ObjectDef:
    name: str        # "large_red_sphere"
    shape: str       # "sphere" | "cube" | "cylinder"
    size: tuple      # sphere: (radius,) | cube: (half_size,) | cylinder: (radius, half_height)
    color: tuple     # (r, g, b) 0-1
    position: tuple  # (x, y) 

@dataclass
class ActorDef:
    name: str               # "bottle_left"
    modelname: str          # "001_bottle"
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
    action: str   # "pickup" | "stack" | "push" | "place_bowl"
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
        ObjectDef("large_red_cylinder","cylinder",(0.030, 0.035),(1.0, 0.0, 0.0), (-0.14, -0.04)), #左上
        ObjectDef("large_blue_cube","cube",(0.03,),(0.1, 0.1, 1.0), (-0.14, -0.13)), #左下
        ObjectDef("small_red_cube","cube",(0.02,),(1.0, 0.0, 0.0), (-0.03, -0.04)),  #右上
        ObjectDef("small_blue_cylinder","cylinder",(0.02, 0.025),(0.1, 0.1, 1.0), (-0.03, -0.13)),#右下
    ],
    actors=[
        ActorDef("green_bowl", "002_bowl", (0.07, -0.10, 0.76), model_id=3, scale=0.6, is_static=False, quat=(0.5, 0.5, 0.5, 0.5)),
        ActorDef("yellow_bowl", "002_bowl", (0.16, -0.10, 0.76), model_id=1, scale=0.8, is_static=False, quat=(0.5, 0.5, 0.5, 0.5)),
    ],
    perturb_rules={
        "large_blue_cube":   PerturbRule(y_direction="up"),
        "small_blue_cylinder": PerturbRule(y_direction="up"),
        "large_red_cylinder":  PerturbRule(y_direction="down"),
        "small_red_cube":    PerturbRule(y_direction="down"),
    },
))

_reg("pickup_red_cylinder",  "default", "large_red_cylinder",  None, "left", "pickup")

_reg("pickup_left_cube",     "default", "large_blue_cube",     None, "left", "pickup")

_reg("pickup_right_cube",    "default", "small_red_cube",      None, "left", "pickup")

_reg("pickup_blue_cylinder", "default", "small_blue_cylinder", None, "left", "pickup")

_reg("stack_left_cube_on_red_cylinder",   "default", "large_blue_cube",     "large_red_cylinder",  "left", "stack")

_reg("stack_left_cube_on_blue_cylinder",  "default", "large_blue_cube",     "small_blue_cylinder", "left", "stack")

_reg("stack_blue_cylinder_on_left_cube",  "default", "small_blue_cylinder", "large_blue_cube",     "left", "stack")

_reg("stack_blue_cylinder_on_right_cube", "default", "small_blue_cylinder", "small_red_cube",      "left", "stack")

_reg("push_left_cube_to_red_cylinder",    "default", "large_blue_cube",     "large_red_cylinder",  "left", "push")

_reg("push_left_cube_to_blue_cylinder",   "default", "large_blue_cube",     "small_blue_cylinder", "left", "push")

_reg("push_red_cylinder_to_left_cube",    "default", "large_red_cylinder",  "large_blue_cube",     "left", "push")

_reg("push_red_cylinder_to_right_cube",   "default", "large_red_cylinder",  "small_red_cube",      "left", "push")

_reg("place_right_cube_in_green_bowl",    "default", "small_red_cube",      "green_bowl",  "right", "place_bowl")

_reg("place_right_cube_in_yellow_bowl",   "default", "small_red_cube",      "yellow_bowl", "right", "place_bowl")

_reg("place_blue_cylinder_in_green_bowl", "default", "small_blue_cylinder", "green_bowl",  "right", "place_bowl")

_reg("place_blue_cylinder_in_yellow_bowl","default", "small_blue_cylinder", "yellow_bowl", "right", "place_bowl")

def perturb_position(cx, cy, rule: PerturbRule):
    x = cx + np.random.uniform(-rule.x_range, rule.x_range)
    if rule.y_direction == "up":
        y = cy + np.random.uniform(0, rule.y_range)
    elif rule.y_direction == "down":
        y = cy - np.random.uniform(0, rule.y_range)
    else:
        y = cy + np.random.uniform(-rule.y_range, rule.y_range)
    return x, y
