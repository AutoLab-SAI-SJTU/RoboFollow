"""Convert raw or paired RoboFollow episodes to Pi-style HDF5."""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from data.decode_image_bit import decode_image_bit
from .runinfo import write_json
from .episodes import validate_episode, joint_pairs, camera_buffers, discover_episodes
from .registry import training

JOINT_FIELDS = ("left_arm_joint_states", "left_ee_joint_states",
                "right_arm_joint_states", "right_ee_joint_states")
CAMERAS = {"cam_head": "cam_high", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}


def vector(group):
    return np.concatenate([np.asarray(group[name]).reshape(len(group[name]), -1)
                           for name in JOINT_FIELDS], axis=1).astype(np.float32)


def convert_episode(source, target, *, expected_instructions=None, image_size=(640, 480)):
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("Image dimensions must be positive")
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix(".hdf5.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    checked = validate_episode(source, expected_instructions=expected_instructions)
    try:
        with h5py.File(source, "r") as inp, h5py.File(temporary, "w") as out:
            state, action = joint_pairs(inp)
            if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14 or len(state) == 0:
                raise ValueError(f"Invalid state/action shape in {source}: {state.shape}, {action.shape}")
            instructions = checked["instructions"]
            out.attrs["image_codec"] = "legacy-opencv-rgb"
            out.attrs["source_format"] = checked["format"]
            for key, value in checked["identity"].items():
                out.attrs[f"robofollow_{key}"] = value
            out.create_dataset("action", data=action)
            obs = out.create_group("observations")
            obs.create_dataset("qpos", data=state)
            obs.create_dataset("left_arm_dim", data=np.repeat(6, len(state)))
            obs.create_dataset("right_arm_dim", data=np.repeat(6, len(state)))
            images = obs.create_group("images")
            for source_camera, target_camera in CAMERAS.items():
                buffers = camera_buffers(inp, source_camera)
                if len(buffers) != len(state):
                    raise ValueError(f"Camera/state length mismatch: {source_camera}")
                encoded = []
                for buffer in buffers:
                    rgb = cv2.resize(decode_image_bit(buffer), image_size)
                    # The Pi reader calls cv2.imdecode without BGR->RGB.
                    # Encode RGB directly here to preserve that reader contract.
                    ok, jpeg = cv2.imencode(".jpg", rgb)
                    if not ok:
                        raise ValueError("JPEG encoding failed")
                    encoded.append(jpeg.tobytes())
                width = max(map(len, encoded))
                images.create_dataset(target_camera, data=encoded, dtype=f"S{width}")
            out.create_dataset("instructions", data=json.dumps(instructions), dtype=h5py.string_dtype())
        temporary.replace(target)
        return instructions
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw robofollow output root")
    parser.add_argument("--output", type=Path, required=True, help="New processed root")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    args = parser.parse_args()
    if args.image_width <= 0 or args.image_height <= 0:
        parser.error("Image dimensions must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty")
    sources = discover_episodes(args.input)
    for source, scene, task, index in sources:
        filename = f"episode_{index}.hdf5"
        target_dir = args.output / scene / task
        target_dir.mkdir(parents=True, exist_ok=True)
        _, templates = training(scene)
        instructions = convert_episode(source, target_dir / filename, expected_instructions=templates[task],
                                       image_size=(args.image_width, args.image_height))
        instruction_file = target_dir / "instructions.json"
        payload = {"instructions": instructions}
        if instruction_file.exists() and json.loads(instruction_file.read_text()) != payload:
            raise ValueError(f"Inconsistent training instructions: {source}")
        write_json(instruction_file, payload)
    write_json(args.output / "conversion.json", {"episodes": len(sources), "format": "pi-hdf5",
               "image_codec": "legacy-opencv-rgb", "action_alignment": "raw frames paired once; stored pairs retained",
               "image_size": [args.image_width, args.image_height]})
    print(f"Converted {len(sources)} episodes")


if __name__ == "__main__":
    main()
