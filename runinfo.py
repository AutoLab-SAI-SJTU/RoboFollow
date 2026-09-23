"""Small, portable run manifests."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def code_digest():
    digest = hashlib.sha256()
    paths = list((ROOT / "robofollow").rglob("*.py")) + list((ROOT / "envs").rglob("*.py"))
    paths += list((ROOT / "robofollow/tasks").glob("*.json"))
    paths += [ROOT / "robofollow/protocol.yml", ROOT / "data/decode_image_bit.py"]
    for path in sorted(p for p in paths if "curobo" not in p.parts):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def metadata():
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, capture_output=True)
    versions = {}
    for name in ("numpy", "sapien", "torch", "mplib", "toppra", "h5py", "transforms3d", "imageio",
                 "imageio-ffmpeg", "nvidia-curobo", "warp-lang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    config_files = ("env_cfg/task_config/_camera_config.yml", "assets/embodiments/aloha-agilex/config.yml")
    config_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in config_files if (ROOT / name).is_file()}
    # Editable cuRobo installs often report version 0.0.0, so record the actual
    # planner Python source, not just its distribution version.
    planner_hash = None
    spec = importlib.util.find_spec("curobo")
    if spec is not None and spec.origin:
        planner_root = Path(spec.origin).parent
        digest = hashlib.sha256()
        for path in sorted(planner_root.rglob("*.py")):
            digest.update(str(path.relative_to(planner_root)).encode())
            digest.update(path.read_bytes())
        planner_hash = digest.hexdigest()
    return {"upstream_commit": result.stdout.strip() or None, "code_sha256": code_digest(),
            "versions": versions, "curobo_source_sha256": planner_hash, "config_sha256": config_hashes,
            "asset_overrides": json.loads((ROOT / "robofollow/asset_overrides/manifest.json").read_text())}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
