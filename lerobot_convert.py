"""Convert validated RoboFollow episodes to a local LeRobot image dataset."""
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from data.decode_image_bit import decode_image_bit
from .convert import CAMERAS
from .episodes import validate_episode, joint_pairs, camera_buffers, discover_episodes
from .registry import training
from .runinfo import write_json


def convert(source, output, repo_id, fps=17, all_instructions=True, image_size=(640, 480)):
    # LeRobot 0.4 uses the new namespace; older OpenPI environments use common.
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        if exc.name != "lerobot.datasets":
            raise
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new dataset directory: {output}")
    if fps <= 0:
        raise ValueError("FPS must be positive")
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("Image dimensions must be positive")
    paths = discover_episodes(source)
    checked = []
    shapes = {}
    # Validate the entire selection before creating a dataset. No silent skips.
    for path, scene, task, _ in paths:
        _, templates = training(scene)
        info = validate_episode(path, expected_instructions=templates[task])
        checked.append((path, info))
    shapes = {camera: (image_size[1], image_size[0], 3) for camera in CAMERAS}
    names = [*[f"left_joint_{i}" for i in range(6)], "left_gripper",
             *[f"right_joint_{i}" for i in range(6)], "right_gripper"]
    features = {key: {"dtype": "float32", "shape": (14,), "names": names}
                for key in ("observation.state", "action")}
    for camera, mapped in CAMERAS.items():
        features[f"observation.images.{mapped}"] = {
            "dtype": "image", "shape": shapes[camera],
            "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(repo_id=repo_id, root=output, fps=fps,
                                   robot_type="aloha_agilex", features=features,
                                   use_videos=False, image_writer_threads=4)
    rows = []
    for path, info in checked:
        with h5py.File(path, "r") as episode:
            state, action = joint_pairs(episode)
            images = {mapped: np.stack([cv2.resize(decode_image_bit(b), image_size) for b in camera_buffers(episode, camera)])
                      for camera, mapped in CAMERAS.items()}
        instructions = info["instructions"] if all_instructions else info["instructions"][:1]
        # One instruction per episode keeps language stable across an action chunk.
        for instruction in instructions:
            for index in range(len(state)):
                dataset.add_frame({"observation.state": state[index], "action": action[index],
                                   "task": instruction, **{
                    f"observation.images.{camera}": frames[index] for camera, frames in images.items()
                }})
            dataset.save_episode()
            rows.append({"source": str(path.relative_to(source)), "instruction": instruction,
                         "frames": len(state), "identity": info["identity"]})
        print(f"Converted {path.name}: {len(state)} frames x {len(instructions)} instructions", flush=True)
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    write_json(output / "robofollow_conversion.json", {
        "repo_id": repo_id, "source_episodes": len(paths), "episodes": rows,
        "frames": sum(row["frames"] for row in rows), "fps": fps,
        "timing": "nominal integer frame-index rate; no resampling; not measured control frequency",
        "action_alignment": "raw frames paired once; stored pairs retained", "color_order": "RGB",
        "image_size": list(image_size),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory, normally $HF_LEROBOT_HOME/<repo-id>")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=17,
                        help="Nominal frame-index rate (250/15 rounded); no resampling")
    parser.add_argument("--first-instruction-only", action="store_true")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    args = parser.parse_args()
    convert(args.input, args.output, args.repo_id, args.fps, not args.first_instruction_only,
            (args.image_width, args.image_height))


if __name__ == "__main__":
    main()
