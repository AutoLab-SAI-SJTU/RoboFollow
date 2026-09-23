"""Collect expert plans and replay them into the RoboFollow raw HDF5 format."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from pathlib import Path

from .registry import SCENES, training
from .runinfo import ROOT, metadata, write_json
from .episodes import validate_episode, validate_joint_paths, write_episode


def collect_task(scene, task_name, output, episodes, seed_start, max_attempts, resume):
    from .assets import require_installed
    from .config import environment_args
    require_installed(ROOT / "assets")
    # Do not silently substitute a different expert motion planner.
    from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401
    from envs.utils import UnStableError
    env_class = getattr(importlib.import_module(f"envs.{scene}"), scene)
    _, templates = training(scene)
    directory = output / scene / f"{scene}_{task_name}"
    protocol = {"scene": scene, "task": task_name, "seed_start": seed_start,
                "instructions": templates[task_name], "data_format": "robofollow-raw-v1",
                "collection_version": "replay-raw-v1", "expert_success": "geometry-route-and-joint-limits", **metadata()}
    marker = directory / "collection.json"
    if directory.exists() and any(directory.iterdir()):
        if not resume or not marker.is_file():
            raise FileExistsError(f"Output exists; use --resume for matching runs: {directory}")
        if json.loads(marker.read_text()) != protocol:
            raise ValueError(f"Collection protocol/source changed: {marker}")
    directory.mkdir(parents=True, exist_ok=True)
    args = environment_args(scene, output=directory)
    args["instruction_type"] = task_name
    seed_file = directory / "seeds.json"
    seeds = json.loads(seed_file.read_text()) if seed_file.exists() else []
    if episodes < len(seeds):
        raise ValueError("Requested episode count is smaller than the saved seed count")
    write_json(marker, protocol)
    (directory / "complete.json").unlink(missing_ok=True)
    info_file = directory / "scene_info.json"
    scene_info = json.loads(info_file.read_text()) if info_file.exists() else {}
    env = env_class()
    next_seed = max(seeds) + 1 if seeds else seed_start
    attempts = 0
    while len(seeds) < episodes:
        if attempts >= max_attempts:
            raise RuntimeError(f"Seed budget exhausted: {len(seeds)}/{episodes} successful seeds")
        seed = next_seed + attempts
        attempts += 1
        try:
            random.seed(seed)
            env.setup_demo(**{**args, "seed": seed, "now_ep_num": len(seeds), "need_plan": True,
                              "left_joint_path": [], "right_joint_path": []})
            env.play_once()
            success = env.plan_success and env.check_success()
            if success:
                validate_joint_paths({"left_joint_path": env.left_joint_path, "right_joint_path": env.right_joint_path})
                success, _ = env.planned_joints_legal()
            if success:
                env.save_traj_data(len(seeds))
                seeds.append(seed)
                write_json(seed_file, seeds)
                (directory / "seed.txt").write_text(" ".join(map(str, seeds)) + "\n")
            print(f"{scene}/{task_name}: seed {seed}, success={success}, accepted={len(seeds)}/{episodes}", flush=True)
        except UnStableError as exc:
            print(f"Unstable seed {seed}: {exc}", flush=True)
        finally:
            if hasattr(env, "scene"):
                env.close_env()
    for index, seed in enumerate(seeds):
        target = directory / "data" / f"episode{index}.hdf5"
        identity = {"scene": scene, "task": task_name, "seed": seed, "episode": index}
        if target.exists():
            validate_episode(target, expected_instructions=templates[task_name], expected_identity=identity)
            if not (directory / "video" / f"episode{index}.mp4").is_file() or f"episode_{index}" not in scene_info:
                raise ValueError(f"Missing video or scene info for {target}")
            continue
        try:
            random.seed(seed)
            replay_args = {**args, "seed": seed, "now_ep_num": index, "need_plan": False,
                           "save_data": True, "left_joint_path": [], "right_joint_path": []}
            env.setup_demo(**replay_args)
            traj = env.load_tran_data(index)
            validate_joint_paths(traj)
            env.set_path_lst({**replay_args, "left_joint_path": traj["left_joint_path"],
                              "right_joint_path": traj["right_joint_path"]})
            info = env.play_once()
            if env.need_plan:
                raise RuntimeError("Expert replay unexpectedly enabled planning")
            if env.left_cnt != len(env.left_joint_path) or env.right_cnt != len(env.right_joint_path):
                raise RuntimeError("Expert replay did not consume all saved trajectory segments")
            if not (env.plan_success and env.check_success()):
                raise RuntimeError(f"Expert replay failed: {scene}/{task_name}, seed={seed}")
            instruction_dir = directory / "instructions"
            instruction_dir.mkdir(exist_ok=True)
            # Keep every training paraphrase; never inject evaluation instructions.
            descriptions = {"seen": templates[task_name], "unseen": []}
            write_json(instruction_dir / f"episode{index}.json", descriptions)
            scene_info[f"episode_{index}"] = info
            write_json(info_file, scene_info)
        finally:
            if hasattr(env, "scene"):
                env.close_env(clear_cache=(index + 1) % args["clear_cache_freq"] == 0)
        write_episode(env.folder_path["cache"], target, directory / "video" / f"{target.stem}.mp4",
                      templates[task_name], identity, int(args["save_freq"] or 15))
        env.remove_data_cache()
    (directory / "seed.txt").write_text(" ".join(map(str, seeds)) + "\n")
    write_json(directory / "complete.json", {"episodes": len(seeds), "seeds": seeds})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES + ("all",), required=True)
    parser.add_argument("--tasks", help="Comma-separated training task IDs; default all")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-seed-attempts", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=ROOT / "data/robofollow")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_seed_attempts < 1 or args.seed_start < 0:
        parser.error("episodes/max-seed-attempts must be positive; seed-start must be nonnegative")
    scenes = SCENES if args.scene == "all" else (args.scene,)
    if args.tasks and args.scene == "all":
        parser.error("Use one scene with --tasks")
    output = args.output.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.chdir(ROOT)
    if not (args.list or args.dry_run):
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    for scene in scenes:
        module, templates = training(scene)
        tasks = args.tasks.split(",") if args.tasks else list(module.TASKS)
        unknown = set(tasks) - set(module.TASKS)
        if unknown:
            parser.error(f"Unknown training task IDs: {sorted(unknown)}")
        for name in tasks:
            if args.list or args.dry_run:
                print(f"{scene}/{name}: {args.episodes} episodes, {len(templates[name])} instructions -> {output / scene / (scene + '_' + name)}")
            else:
                collect_task(scene, name, output, args.episodes, args.seed_start, args.max_seed_attempts, args.resume)


if __name__ == "__main__":
    main()
