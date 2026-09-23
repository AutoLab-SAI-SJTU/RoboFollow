from __future__ import annotations
import numpy as np
import sapien
from envs._base_task import Base_Task
from robofollow.geometry import create_box, create_sphere, create_standing_cylinder
from robofollow.tasks.common import *
from .common import perturb_position

class Scene2EvalEnv(Base_Task):
    """Scene 2 evaluation with expert object mass and friction settings."""
    def __init__(self):
        super().__init__()
        self.object_dict: dict[str, sapien.Entity] = {}
        self.label_dict:  dict[str, sapien.Entity] = {}
        self.bowl_dict:   dict[str, sapien.Entity] = {}

    # required overrides
    def load_actors(self): pass
    def play_once(self):   pass
    def check_success(self): return False

    def get_object_position(self, name):
        ent = self.object_dict.get(name) or self.label_dict.get(name) or self.bowl_dict.get(name)
        return np.array(ent.get_pose().p) if ent else None

    def populate_scene(self, scene_cfg):
        from robofollow.geometry import create_actor, create_box, create_sphere, create_standing_cylinder

        th = 0.74 + self.table_z_bias
        default_rule = scene_cfg.default_perturb

        # Reset per-episode registries.
        self.object_dict = {}
        self.label_dict = {}
        self.bowl_dict = {}

        for obj in scene_cfg.objects:
            rule = scene_cfg.perturb_rules.get(obj.name, default_rule)
            x, y = perturb_position(*obj.position, rule)

            if obj.shape == "sphere":
                radius = obj.size[0]
                z = th + radius
                ent = create_sphere(
                    self,
                    sapien.Pose([x, y, z]),
                    radius,
                    obj.color,
                    name=obj.name,
                    friction=2.0,
                    mass=0.05,
                )
            elif obj.shape == "cylinder":
                radius, half_height = obj.size
                z = th + half_height
                ent = create_standing_cylinder(
                    self,
                    sapien.Pose([x, y, z]),
                    radius,
                    half_height,
                    obj.color,
                    name=obj.name,
                    friction=0.12,
                    mass=0.2,
                )
            else:  # cube
                half_size = obj.size[0]
                z = th + half_size + 0.001
                ent = create_box(
                    self,
                    sapien.Pose([x, y, z]),
                    (half_size, half_size, half_size),
                    obj.color,
                    name=obj.name,
                    friction=0.1,
                    mass=0.2,
                )

            self.object_dict[obj.name] = ent

        for lb in getattr(scene_cfg, "labels", []):
            z = th + 0.001
            ent = create_box(
                self,
                sapien.Pose([lb.position[0], lb.position[1], z]),
                lb.half_size,
                lb.color,
                is_static=True,
                name=lb.name,
            )
            self.label_dict[lb.name] = ent

        for actor_def in scene_cfg.actors:
            ax, ay, az = actor_def.position
            actor_pose = sapien.Pose([ax, ay, az + self.table_z_bias], actor_def.quat)
            ent = create_actor(
                self,
                pose=actor_pose,
                modelname=actor_def.modelname,
                model_id=actor_def.model_id,
                convex=True,
                is_static=actor_def.is_static,
                scale_multiplier=actor_def.scale,
            )
            if "bowl" in actor_def.modelname:
                self.bowl_dict[actor_def.name] = ent
            else:
                self.object_dict[actor_def.name] = ent

        for _ in range(500):
            self.scene.step()
