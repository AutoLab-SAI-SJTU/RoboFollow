"""Validate and publish complete training episodes."""
import json
import pickle
import re
from pathlib import Path
import tempfile

import cv2
import h5py
import numpy as np

from data.decode_image_bit import decode_image_bit

JOINT_WIDTHS = {
    "left_arm_joint_states": 6, "left_ee_joint_states": 1,
    "right_arm_joint_states": 6, "right_ee_joint_states": 1,
}
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
IDENTITY_FIELDS = ("scene", "task", "seed", "episode")


def validate_joint_paths(paths):
    for arm in ("left", "right"):
        segments = paths[f"{arm}_joint_path"]
        if not isinstance(segments, list):
            raise ValueError(f"Invalid {arm} trajectory list")
        for index, segment in enumerate(segments):
            position, velocity = np.asarray(segment["position"]), np.asarray(segment["velocity"])
            if (segment["status"] != "Success" or position.ndim != 2 or position.shape[1] != 6
                    or not len(position) or position.shape != velocity.shape
                    or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
                raise ValueError(f"Invalid {arm} trajectory segment {index}")


RAW_JOINTS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
RAW_CAMERAS = {"cam_head": "head_camera", "cam_left_wrist": "left_camera",
               "cam_right_wrist": "right_camera"}


def episode_format(data):
    raw = "joint_action" in data and "observation" in data
    paired = "state" in data and "action" in data and "vision" in data
    if raw == paired:
        raise ValueError("Expected one raw or paired episode schema")
    return "robofollow-raw-v1" if raw else "robotwin-2026-09"


def episode_instructions(data, path):
    if "instructions" in data:
        return json.loads(data["instructions"].asstr()[()])
    sidecar = Path(path).parent.parent / "instructions" / (Path(path).stem + ".json")
    return json.loads(sidecar.read_text())["seen"]


def joint_pairs(data):
    if episode_format(data) == "robofollow-raw-v1":
        group = data["joint_action"]
        states = np.concatenate([np.asarray(group[k]).reshape(len(group[k]), -1)
                                 for k in RAW_JOINTS], axis=1).astype(np.float32)
        return states[:-1], states[1:]
    def vector(group):
        return np.concatenate([np.asarray(group[k]).reshape(len(group[k]), -1)
                               for k in JOINT_WIDTHS], axis=1).astype(np.float32)
    return vector(data["state"]), vector(data["action"])


def camera_buffers(data, camera):
    if episode_format(data) == "robofollow-raw-v1":
        return data[f"observation/{RAW_CAMERAS[camera]}/rgb"][:-1]
    return data[f"vision/{camera}/colors"][:]


def discover_episodes(source):
    """Accept a collection root, scene directory, or individual task directory."""
    from .registry import SCENES, training
    source = Path(source)
    paths = set()
    for pattern in ("data/episode*.hdf5", "*/data/episode*.hdf5", "scene*/*/data/episode*.hdf5"):
        paths.update(source.glob(pattern))
    found, identities = [], set()
    for path in sorted(paths):
        match = re.fullmatch(r"episode_?(\d+)", path.stem)
        if not match:
            raise ValueError(f"Invalid episode filename: {path}")
        directory = path.parent.parent
        scene = next((x for x in SCENES if directory.name.startswith(x + "_")), None)
        scene = scene or (directory.parent.name if directory.parent.name in SCENES else None)
        if scene is None:
            raise ValueError(f"Cannot identify scene for {path}")
        task = directory.name.removeprefix(scene + "_")
        if task not in training(scene)[0].TASKS:
            raise ValueError(f"Unknown training task: {scene}/{task}")
        key = (scene, task, int(match[1]))
        if key in identities:
            raise ValueError(f"Duplicate episode identity: {key}")
        identities.add(key)
        found.append((path, scene, task, int(match[1])))
    if not found:
        raise ValueError(f"No RoboFollow episodes under {source}")
    return sorted(found, key=lambda row: (row[1], row[2], row[3]))


def _validate_raw(data, path, expected_instructions, decode_images):
    frames = None
    for name, width in zip(RAW_JOINTS, JOINT_WIDTHS.values()):
        values = np.asarray(data[f"joint_action/{name}"])
        if width == 1 and values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != width or len(values) < 2:
            raise ValueError(f"Invalid shape for joint_action/{name}: {values.shape}")
        if values.dtype.kind not in "fi" or not np.isfinite(values).all():
            raise ValueError(f"Invalid joint values in {name}")
        if frames is not None and len(values) != frames:
            raise ValueError(f"Frame count mismatch in {name}")
        frames = len(values)
    instructions = episode_instructions(data, path)
    if (not isinstance(instructions, list) or not instructions
            or not all(isinstance(x, str) and x.strip() for x in instructions)):
        raise ValueError("Instructions must be a nonempty list of nonempty strings")
    if expected_instructions is not None and instructions != list(expected_instructions):
        raise ValueError("Instructions do not match the training task")
    for camera in RAW_CAMERAS.values():
        buffers = data[f"observation/{camera}/rgb"]
        if buffers.ndim != 1 or len(buffers) != frames:
            raise ValueError(f"Frame count/shape mismatch in {camera}")
        shape = None
        if decode_images:
            for buffer in buffers:
                rgb = decode_image_bit(buffer)
                if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                    raise ValueError(f"Invalid RGB image in {camera}")
                if shape is not None and rgb.shape != shape:
                    raise ValueError(f"Inconsistent RGB shape in {camera}")
                shape = rgb.shape
    return frames - 1, instructions


def validate_episode(path, *, expected_instructions=None, expected_identity=None, decode_images=True):
    """Validate raw and paired episodes, reporting the number of training pairs."""
    path = Path(path)
    try:
        with h5py.File(path, "r") as data:
            format_name = episode_format(data)
            if format_name == "robofollow-raw-v1":
                length, instructions = _validate_raw(data, path, expected_instructions, decode_images)
            else:
                length = None
                for group in ("state", "action"):
                    for name, width in JOINT_WIDTHS.items():
                        values = np.asarray(data[f"{group}/{name}"])
                        if values.ndim != 2 or values.shape[1] != width or not len(values):
                            raise ValueError(f"Invalid shape for {group}/{name}: {values.shape}")
                        if values.dtype.kind not in "fi" or not np.isfinite(values).all():
                            raise ValueError(f"Invalid joint values in {group}/{name}")
                        if length is not None and len(values) != length:
                            raise ValueError(f"Frame count mismatch in {group}/{name}")
                        length = len(values)
                instructions = episode_instructions(data, path)
                if not isinstance(instructions, list) or not instructions or not all(
                    isinstance(text, str) and text.strip() for text in instructions
                ):
                    raise ValueError("Instructions must be a nonempty list of nonempty strings")
                if expected_instructions is not None and instructions != list(expected_instructions):
                    raise ValueError("Instructions do not match the training task")
                for camera in CAMERAS:
                    buffers = data[f"vision/{camera}/colors"]
                    if buffers.ndim != 1 or len(buffers) != length:
                        raise ValueError(f"Frame count/shape mismatch in {camera}")
                    shape = tuple(np.asarray(data[f"vision/{camera}/shape"]).tolist())
                    if len(shape) != 3 or shape[2] != 3 or any(x <= 0 for x in shape):
                        raise ValueError(f"Invalid RGB shape in {camera}: {shape}")
                    if decode_images:
                        for index, buffer in enumerate(buffers):
                            rgb = decode_image_bit(buffer)
                            if rgb is None or rgb.shape != shape or rgb.dtype != np.uint8:
                                raise ValueError(f"Invalid RGB image in {camera}, frame {index}")
            identity = {key: data.attrs[f"robofollow_{key}"] for key in IDENTITY_FIELDS
                        if f"robofollow_{key}" in data.attrs}
            # h5py returns NumPy integer scalars for seed/episode attributes.
            # Downstream conversion manifests must remain JSON serializable.
            identity = {key: value.item() if isinstance(value, np.generic) else value
                        for key, value in identity.items()}
            if expected_identity is not None:
                for key, value in expected_identity.items():
                    if identity.get(key) != value:
                        raise ValueError(f"Episode identity mismatch: {key}")
            return {"frames": length, "instructions": instructions, "identity": identity, "format": format_name}
    except (KeyError, OSError, TypeError, ValueError, cv2.error) as exc:
        raise ValueError(f"Invalid episode {path}: {exc}") from exc


def write_episode(cache, target, video, instructions, identity, frequency):
    """Publish the HDF5 only after writing and validating a temporary episode."""

    target, video = Path(target), Path(video)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".episode-", dir=target.parent) as temporary:
        temporary = Path(temporary)
        (temporary / "data").mkdir()
        (temporary / "instructions").mkdir()
        staged_hdf5, staged_video = temporary / "data" / target.name, temporary / video.name
        (temporary / "instructions" / (target.stem + ".json")).write_text(
            json.dumps({"seen": instructions, "unseen": []}))
        write_raw_cache(cache, staged_hdf5, staged_video)
        with h5py.File(staged_hdf5, "r+") as data:
            for key, value in identity.items():
                data.attrs[f"robofollow_{key}"] = value
        validate_episode(staged_hdf5, expected_instructions=instructions, expected_identity=identity)
        if target.exists():
            raise FileExistsError(target)
        staged_video.replace(video)
        staged_hdf5.replace(target)


def write_raw_cache(cache, target, video):
    """Write all observation fields, encoding RGB arrays as JPEG buffers."""
    from envs.utils.images_to_video import images_to_video
    paths = sorted((p for p in Path(cache).glob("*.pkl") if p.stem.isdigit()),
                   key=lambda p: int(p.stem))
    if len(paths) < 2 or [int(p.stem) for p in paths] != list(range(len(paths))):
        raise ValueError(f"Missing or insufficient cache frames: {cache}")
    frames = []
    for path in paths:
        with path.open("rb") as stream:
            frames.append(pickle.load(stream))
    def write(group, values):
        if not all(set(v) == set(values[0]) for v in values):
            raise ValueError("Observation fields changed within an episode")
        for key in values[0]:
            items = [v[key] for v in values]
            if isinstance(items[0], dict):
                write(group.create_group(key), items)
            elif key == "rgb":
                encoded = []
                for rgb in items:
                    ok, buffer = cv2.imencode(".jpg", rgb)
                    if not ok:
                        raise ValueError("JPEG encoding failed")
                    encoded.append(buffer.tobytes())
                group.create_dataset(key, data=encoded, dtype=f"S{max(map(len, encoded))}")
            else:
                group.create_dataset(key, data=np.asarray(items))
    with h5py.File(target, "w") as data:
        write(data, frames)
    images_to_video(np.asarray([f["observation"]["head_camera"]["rgb"] for f in frames]),
                    out_path=str(video))
