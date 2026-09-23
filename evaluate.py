"""Run four-scene evaluation through a generic policy API."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

from .registry import SCENES, evaluation
from .runinfo import ROOT, metadata, write_json


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[f"{row['scene']}/{row['level_tag']}"].append(row)
    summaries = {}
    for name, rows in groups.items():
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["task"]].append(row["completion_ok"])
        task_macro = sum(sum(values) / len(values) for values in by_task.values()) / len(by_task)
        episode_micro = sum(r["completion_ok"] for r in rows) / len(rows)
        summaries[name] = {
            "episodes": len(rows),
            "task_count": len(by_task),
            "mean_intent_score": sum(r["intent_score"] for r in rows) / len(rows),
            "mean_exec_score": sum(r["exec_score"] for r in rows) / len(rows),
            "intent_full_rate": sum(r["intent_ok"] for r in rows) / len(rows),
            "exec_full_rate": sum(r["exec_ok"] for r in rows) / len(rows),
            "completion_rate": task_macro,
            "task_macro_completion_rate": task_macro,
            "episode_micro_completion_rate": episode_micro,
        }
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES + ("all",), required=True)
    parser.add_argument("--levels", default="L0,L1,L2,L3")
    parser.add_argument("--tasks", help="Comma-separated evaluation IDs")
    parser.add_argument("--limit", type=int, help="Limit task count per scene for smoke tests")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=10, help="Policy calls per episode")
    parser.add_argument("--actions-per-step", type=int, default=50)
    parser.add_argument("--runtime", choices=("fixed",), default="fixed",
                        help="evaluation runtime (fixed only)")
    parser.add_argument("--direct-step", action="store_true", help="Explicit alternative to TOPP; scores are not interchangeable")
    parser.add_argument("--sim-steps", type=int, default=15)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=25, help="Playback FPS only")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--remote", action="store_true")
    group.add_argument("--policy-factory", help="module:function returning reset/set_instruction/predict adapter")
    parser.add_argument("--policy-kwargs", default="{}", help="JSON keyword arguments for factory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--policy-timeout", type=float, default=300.,
                        help="TCP reply timeout in seconds, including the first model compilation")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(value < 1 for value in (args.rounds, args.max_steps, args.actions_per_step, args.sim_steps, args.video_fps)) or (args.limit is not None and args.limit < 1):
        parser.error("Episode, action, step, limit and FPS counts must be positive")
    if args.policy_timeout <= 0 or args.base_seed < 0:
        parser.error("Policy timeout must be positive and base seed must be nonnegative")
    levels = args.levels.split(",")
    if not levels or set(levels) - {"L0", "L1", "L2", "L3"}:
        parser.error("Unknown evaluation levels")
    selected = []
    for scene in (SCENES if args.scene == "all" else (args.scene,)):
        module = evaluation(scene)
        names = args.tasks.split(",") if args.tasks else list(module.TASKS)
        unknown = set(names) - set(module.TASKS)
        if unknown:
            parser.error(f"Unknown evaluation IDs for {scene}: {sorted(unknown)}")
        names = [name for name in names if module.TASKS[name].level_tag in levels]
        selected.extend((scene, module, module.TASKS[name]) for name in names[:args.limit])
    if not selected:
        parser.error("No evaluation tasks selected")
    if args.list or args.dry_run:
        for scene, _, task in selected:
            print(f"{scene} {task.name}: {task.instruction}")
        print(f"{len(selected)} tasks x {args.rounds} rounds = {len(selected) * args.rounds} trials")
        return
    if not (args.remote or args.policy_factory):
        parser.error("Specify --remote or --policy-factory")
    if args.output is None:
        parser.error("Specify a new --output directory")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory must be empty: {output}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.chdir(ROOT)
    from .assets import require_installed
    require_installed(ROOT / "assets")
    from .policies import ValidatedPolicy, load_factory
    from .remote_policy import RemotePolicy
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    from .simulation import create_environment
    if args.direct_step:
        from .evaluation.common import _patch_robot_for_direct_step
        _patch_robot_for_direct_step()
    policy = ValidatedPolicy(RemotePolicy(host=args.host, port=args.port, timeout=args.policy_timeout) if args.remote else
                             load_factory(args.policy_factory, json.loads(args.policy_kwargs)))
    output.mkdir(parents=True, exist_ok=True)
    run = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
           "metric_version": "strict", "gate_mode": "strict", "runtime": args.runtime,
           "completion_metric_version": "legacy-final-layout-v1",
           "completion_aggregation": "task-macro",
           "stepper": "direct" if args.direct_step else "robotwin-topp",
           "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "selected_tasks": [{"scene": scene, "task": task.name} for scene, _, task in selected],
           "expected_trials": len(selected) * args.rounds, **metadata()}
    write_json(output / "run.json", run)
    results = []
    try:
        for scene, module, task in selected:
            for rd in range(1, args.rounds + 1):
                seed = args.base_seed + rd * 100 + module._stable_hash32(task.name) % 1000
                env = None
                try:
                    env = create_environment(scene, module, task, seed, runtime=args.runtime)
                    result = module.run_once(env, policy, task, args.max_steps, args.actions_per_step,
                                             direct_step=args.direct_step,
                                             sim_steps=args.sim_steps)
                    frames = result.pop("frames")
                    if args.video:
                        import imageio.v2 as imageio
                        videos = output / "videos" / scene
                        videos.mkdir(parents=True, exist_ok=True)
                        video = videos / f"{task.name}_r{rd}_s{seed}.mp4"
                        imageio.mimsave(video, frames, fps=args.video_fps)
                        result["video"] = str(video.relative_to(output))
                    result.update(scene=scene, layout=task.scene, seed=seed, round=rd,
                                  level_tag=task.level_tag, base_task_type=task.base_task_type)
                    results.append(result)
                    write_json(output / "results.json", results)
                    print(f"{scene}/{task.name} r{rd}: intent={result['intent_score']:.3f}, exec={result['exec_score']:.3f}", flush=True)
                finally:
                    if env is not None:
                        env.close_env()
        run["status"] = "complete"
    except BaseException:
        run["status"] = "failed"
        run["error"] = traceback.format_exc()
        raise
    finally:
        run["completed_trials"] = len(results)
        write_json(output / "run.json", run)
        write_json(output / "summary.json", {"status": run["status"], "groups": summarize(results)})
        policy.close()


if __name__ == "__main__":
    main()
