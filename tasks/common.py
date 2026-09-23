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
