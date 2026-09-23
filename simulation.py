"""Construct RoboTwin environments for benchmark evaluation."""
from __future__ import annotations

import copy
import importlib
import random

import numpy as np

from .config import environment_args
from .evaluation.environments import Scene2EvalEnv


def create_environment(scene, module, task, seed, runtime="fixed"):
    if runtime != "fixed":
        raise ValueError("Only the fixed evaluation runtime is supported")
    args = environment_args(scene, seed=seed, evaluation=True)
    args["task_name"] = "batch_eval"
    random.seed(seed)
    np.random.seed(seed)
    cfg = copy.deepcopy(module.SCENE_VARIANTS[task.scene])
    if scene == "scene2":
        env = Scene2EvalEnv()
        env._init_task_env_(**args)
        env.populate_scene(cfg)
    else:
        definitions = importlib.import_module(f"robofollow.tasks.{scene}")
        env_module = importlib.import_module(f"envs.{scene}")
        cfg.name = task.scene
        previous_cfg = definitions.SCENES.get(task.scene)
        definitions.SCENES[task.scene] = cfg
        previous_dummy = None
        if scene == "scene4":
            instruction_type = "__robofollow_eval_dummy"
            previous_dummy = definitions.TASKS.get(instruction_type)
            definitions.TASKS[instruction_type] = definitions.TaskDef(
                name=instruction_type, scene=task.scene, action="pickup",
                source="blue_sphere", arm="left", instruction="eval dummy",
            )
        else:
            instruction_type = task.base_task_type
            # Scene 1 L3 composes new actions: a known train task initializes
            # the physics; the independent evaluator scores the actual task.
            if instruction_type not in definitions.TASKS:
                instruction_type = next(iter(definitions.TASKS))
        env_class = getattr(env_module, scene)
        class FixedHorizonEnv(env_class):
            def check_success(self):
                return False
        env = FixedHorizonEnv()
        try:
            env.setup_demo(instruction_type=instruction_type, scene_name=task.scene, **args)
        finally:
            if previous_cfg is None:
                definitions.SCENES.pop(task.scene, None)
            else:
                definitions.SCENES[task.scene] = previous_cfg
            if scene == "scene4":
                if previous_dummy is None:
                    definitions.TASKS.pop(instruction_type, None)
                else:
                    definitions.TASKS[instruction_type] = previous_dummy
        if scene == "scene1":
            env.label_dict = {}
            env.bowl_dict = {}
            env.get_object_position = lambda name: np.array(env.object_dict[name].get_pose().p)
    env.step_lim = 2**31 - 1
    return env
