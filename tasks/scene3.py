from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

@dataclass
class ObjectDef:
    name: str
    half_size: tuple  # (hx, hy, hz)
    color: tuple      # (r, g, b) in [0,1]
    position: tuple   # (x, y)
    yaw: float = 0.0
    friction: float = 0.8
    mass: float = 0.18
    boxtype: str = "default"

@dataclass
class LabelDef:
    name: str
    half_size: tuple  # (hx, hy, hz)
    color: tuple      # (r, g, b) in [0,1]
    position: tuple   # (x, y)

@dataclass
class SceneConfig:
    name: str
    objects: list[ObjectDef] = field(default_factory=list)
    labels: list[LabelDef] = field(default_factory=list)
    layout_perturb_x: float = 0.01
    layout_perturb_y: float = 0.01

@dataclass
class TaskDef:
    name: str
    scene: str
    source: str
    via: str
    target: str
    arm: str
    action: str
    route_mode: str  # "via" | "around"
    place_mode: str  # "short_front" | "long_front" | "short_right" | "long_right"
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

def _reg(name, scene, source, via, target, arm, route_mode, place_mode):
    return register_task(
        TaskDef(
            name=name,
            scene=scene,
            source=source,
            via=via,
            target=target,
            arm=arm,
            action="place_on_label_route",
            route_mode=route_mode,
            place_mode=place_mode,
        )
    )

def yaw_to_quat(yaw: float):
    # Quaternion in [w, x, y, z].
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]

BAR_HALF_SIZE = (0.018, 0.028, 0.013)

LABEL_HALF_SIZE = (0.03475, 0.03475, 0.001)

register_scene(
    SceneConfig(
        name="default",
        objects=[
            ObjectDef(
                name="red_bar_2",
                half_size=BAR_HALF_SIZE,
                color=(0.95, 0.25, 0.25),
                position=(-0.09, -0.17),
                yaw=0.0,
            ),
            ObjectDef(
                name="green_bar_3",
                half_size=BAR_HALF_SIZE,
                color=(0.10, 0.80, 0.20),
                position=(-0.09, -0.09),
                yaw=0.0,
            ),
            ObjectDef(
                name="yellow_bar_4",
                half_size=BAR_HALF_SIZE,
                color=(1.0, 0.85, 0.10),
                position=(0.01, -0.17),
                yaw=0.0,
            ),
            ObjectDef(
                name="blue_bar_5",
                half_size=BAR_HALF_SIZE,
                color=(0.10, 0.30, 0.95),
                position=(0.01, -0.09),
                yaw=0.0,
            ),
        ],
        labels=[
            LabelDef(
                name="green_label_1",
                half_size=LABEL_HALF_SIZE,
                color=(0.0, 0.8, 0.0),
                position=(-0.19, -0.13),
            ),
            LabelDef(
                name="blue_label_6",
                half_size=LABEL_HALF_SIZE,
                color=(0.10, 0.30, 0.95),
                position=(0.11, -0.13),
            ),
        ],
    )
)

_reg("r_pick2_via4_to6_short_front", "default", "red_bar_2", "yellow_bar_4", "blue_label_6", "right", "via", "short_front")

_reg("r_pick2_via4_to6_long_front", "default", "red_bar_2", "yellow_bar_4", "blue_label_6", "right", "via", "long_front")

_reg("r_pick2_via5_to6_short_front", "default", "red_bar_2", "blue_bar_5", "blue_label_6", "right", "via", "short_front")

_reg("r_pick2_via5_to6_long_front", "default", "red_bar_2", "blue_bar_5", "blue_label_6", "right", "via", "long_front")

_reg("r_pick3_via4_to6_short_front", "default", "green_bar_3", "yellow_bar_4", "blue_label_6", "right", "via", "short_front")

_reg("r_pick3_via4_to6_long_front", "default", "green_bar_3", "yellow_bar_4", "blue_label_6", "right", "via", "long_front")

_reg("r_pick3_via5_to6_short_front", "default", "green_bar_3", "blue_bar_5", "blue_label_6", "right", "via", "short_front")

_reg("r_pick3_via5_to6_long_front", "default", "green_bar_3", "blue_bar_5", "blue_label_6", "right", "via", "long_front")

_reg("l_pick5_around2_to1_short_right", "default", "blue_bar_5", "red_bar_2", "green_label_1", "left", "around", "short_right")

_reg("l_pick5_around2_to1_long_right", "default", "blue_bar_5", "red_bar_2", "green_label_1", "left", "around", "long_right")

_reg("l_pick5_around3_to1_short_right", "default", "blue_bar_5", "green_bar_3", "green_label_1", "left", "around", "short_right")

_reg("l_pick5_around3_to1_long_right", "default", "blue_bar_5", "green_bar_3", "green_label_1", "left", "around", "long_right")

_reg("l_pick4_around2_to1_short_right", "default", "yellow_bar_4", "red_bar_2", "green_label_1", "left", "around", "short_right")

_reg("l_pick4_around2_to1_long_right", "default", "yellow_bar_4", "red_bar_2", "green_label_1", "left", "around", "long_right")

_reg("l_pick4_around3_to1_short_right", "default", "yellow_bar_4", "green_bar_3", "green_label_1", "left", "around", "short_right")

_reg("l_pick4_around3_to1_long_right", "default", "yellow_bar_4", "green_bar_3", "green_label_1", "left", "around", "long_right")
