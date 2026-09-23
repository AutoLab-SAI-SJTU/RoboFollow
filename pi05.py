"""OpenPI configuration, short/full training, and the RoboFollow pi05 adapter.

Run in the separate OpenPI environment. Simulator imports are not required.
"""
import argparse
import importlib.util
import json
import logging
from pathlib import Path
import sys

import numpy as np

from .runinfo import ROOT, write_json


def openpi_root(path=None):
    root = Path(path or ROOT / "XPolicyLab/policy/Pi_05/openpi").resolve()
    if not (root / "src/openpi/training/config.py").is_file():
        raise FileNotFoundError(f"OpenPI checkout missing: {root}")
    for entry in (root / "src", root / "packages/openpi-client/src"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    return root


def make_config(settings):
    openpi_root(settings.get("openpi_root"))
    from openpi.models import pi0_config
    from openpi.training import config, weight_loaders
    from openpi import transforms

    run_dir = Path(settings["run_dir"]).resolve()
    return config.TrainConfig(
        name="pi05_robofollow", exp_name=run_dir.name,
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=50,
            paligemma_variant=settings.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=settings.get("action_expert_variant", "gemma_300m"),
        ),
        data=config.LeRobotAlohaDataConfig(
            repo_id=settings["repo_id"], adapt_to_pi=False,
            use_delta_joint_actions=True,
            assets=config.AssetsConfig(asset_id=settings["repo_id"]),
            base_config=config.DataConfig(prompt_from_task=True),
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                "images": {"cam_high": "observation.images.cam_high",
                           "cam_left_wrist": "observation.images.cam_left_wrist",
                           "cam_right_wrist": "observation.images.cam_right_wrist"},
                "state": "observation.state", "actions": "action", "prompt": "prompt",
            })]),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(settings["base_weights"]) / "params")),
        assets_base_dir=str(run_dir / "assets"), checkpoint_dir_override=str(run_dir / "checkpoints"),
        batch_size=settings["batch_size"], num_workers=settings["num_workers"],
        num_train_steps=settings["steps"], fsdp_devices=settings["fsdp_devices"],
        log_interval=1, save_interval=settings["save_interval"], seed=settings["seed"],
        wandb_enabled=False, resume=settings.get("resume", False),
    )


def load_script(root, name):
    spec = importlib.util.spec_from_file_location(f"robofollow_openpi_{name}", root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compute_stats(config, root):
    from openpi.shared import normalize
    from openpi.training import data_loader
    script = load_script(root, "compute_norm_stats")
    data = config.data.create(config.assets_dirs, config.model)
    # Batch size 1 includes every frame, including small datasets and remainders.
    dataset = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    dataset = data_loader.TransformedDataset(dataset, [*data.repack_transforms.inputs,
                                                      *data.data_transforms.inputs, script.RemoveStrings()])
    count = len(dataset)
    if count < 1:
        raise ValueError("The training dataset is empty")
    # Statistics are CPU work; do not shard a one-frame batch across visible GPUs.
    loader = data_loader.TorchDataLoader(dataset, local_batch_size=1, num_batches=count,
                                        num_workers=config.num_workers, framework="pytorch")
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    print(f"Computing normalization over {count} frames", flush=True)
    for index, batch in enumerate(loader, start=1):
        for key, stat in stats.items():
            stat.update(np.asarray(batch[key]))
        if index % 100 == 0 or index == count:
            print(f"Normalization: {index}/{count} frames", flush=True)
    path = config.assets_dirs / data.asset_id
    normalize.save(path, {key: stat.get_statistics() for key, stat in stats.items()})
    print(f"Normalization saved: {path} ({count} frames)", flush=True)


class Pi05Policy:
    def __init__(self, checkpoint, config_path=None, num_steps=10):
        checkpoint = Path(checkpoint).resolve()
        config_path = Path(config_path) if config_path else checkpoint.parent.parent / "train_config.json"
        settings = json.loads(config_path.read_text())
        config = make_config(settings)
        from openpi.policies.policy_config import create_trained_policy
        self.policy = create_trained_policy(config, checkpoint, sample_kwargs={"num_steps": int(num_steps)})
        self.instruction = ""

    def reset(self):
        self.instruction = ""

    def set_instruction(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("pi05 requires a nonempty instruction")
        self.instruction = text

    def predict(self, observation):
        if not self.instruction:
            raise ValueError("Call set_instruction before predict")
        images = {}
        for target, source in (("cam_high", "head_camera"), ("cam_left_wrist", "left_camera"),
                               ("cam_right_wrist", "right_camera")):
            rgb = np.asarray(observation["observation"][source]["rgb"])
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"Expected HWC uint8 RGB for {source}, got {rgb.shape}/{rgb.dtype}")
            images[target] = rgb.transpose(2, 0, 1)
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite 14-dimensional state")
        return np.asarray(self.policy.infer({"images": images, "state": state,
                                            "prompt": self.instruction})["actions"], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stats", "train"))
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.fsdp_devices, args.save_interval) < 1 or args.num_workers < 0:
        parser.error("Counts must be positive (num-workers may be zero)")
    root = openpi_root(args.openpi_root)
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key != "command"}
    settings["openpi_root"] = str(root)
    if not (args.base_weights / "params").is_dir():
        parser.error("base-weights must contain params/")
    config = make_config(settings)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.run_dir / "train_config.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        changed = {key for key in settings if old.get(key) != settings[key]} - {"resume", "steps"}
        if changed:
            parser.error(f"Training configuration changed: {sorted(changed)}; choose a new run directory")
    write_json(manifest, settings)
    logging.basicConfig(level=logging.INFO, force=True)
    if args.command == "stats":
        compute_stats(config, root)
    else:
        load_script(root, "train").main(config)


if __name__ == "__main__":
    main()
