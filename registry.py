"""Public registry interface; importing it does not load the simulator."""
import dataclasses
import importlib
import json
from pathlib import Path

SCENES = ("scene1", "scene2", "scene3", "scene4")


def evaluation(scene):
    if scene not in SCENES:
        raise ValueError(f"Unknown scene: {scene}")
    module = importlib.import_module(f"robofollow.evaluation.{scene}")
    module.init_registry(*([0] if scene in {"scene2", "scene3"} else []))
    return module


def training(scene):
    if scene not in SCENES:
        raise ValueError(f"Unknown scene: {scene}")
    module = importlib.import_module(f"robofollow.tasks.{scene}")
    templates = json.loads((Path(__file__).parent / "tasks" / f"{scene}_instructions.json").read_text())
    if set(templates) != set(module.TASKS):
        raise ValueError(f"Training templates/tasks mismatch: {scene}")
    return module, templates


def snapshot(scene):
    module = evaluation(scene)
    return {
        "tasks": [dataclasses.asdict(task) for task in module.TASKS.values()],
        "layouts": {name: dataclasses.asdict(cfg) for name, cfg in module.SCENE_VARIANTS.items()},
    }
