"""Install, verify, and restore the two benchmark visual overrides.

All files are validated before mutation. Originals are backed up separately;
unknown versions are rejected. No simulator dependencies are imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

PACKAGE = Path(__file__).resolve().parent
OVERRIDES = PACKAGE / "asset_overrides"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest():
    return json.loads((OVERRIDES / "manifest.json").read_text())["files"]


def atomic_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".robofollow-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as out, Path(source).open("rb") as inp:
            while block := inp.read(1024 * 1024):
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(name, 0o644)
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def operate(assets_root, action="status", dry_run=False):
    root = Path(assets_root).resolve()
    plans = []
    for row in manifest():
        target = root / row["target"]
        source = OVERRIDES / row["replacement"]
        backup = root / ".robofollow-backups" / row["target"]
        if target.is_symlink() or backup.is_symlink():
            raise ValueError(f"Refusing symlink target/backup: {target}")
        if not target.resolve().is_relative_to(root) or not backup.resolve().is_relative_to(root):
            raise ValueError(f"Asset path escapes asset root: {target}")
        if sha256(source) != row["replacement_sha256"]:
            raise ValueError(f"Bundled replacement checksum mismatch: {source}")
        current = sha256(target) if target.is_file() else None
        state = ("installed" if current == row["replacement_sha256"] else
                 "original" if current == row["original_sha256"] else
                 "missing" if current is None else "unknown")
        if backup.exists() and sha256(backup) != row["original_sha256"]:
            raise ValueError(f"Original backup checksum mismatch: {backup}")
        if action != "status" and state in {"missing", "unknown"}:
            raise ValueError(f"{target}: {state}; install the matching original assets first")
        if action == "restore" and state == "installed" and not backup.is_file():
            raise ValueError(f"Cannot restore without original backup: {backup}")
        plans.append((row, target, source, backup, state))
    # Preflight above covers both bowls; a mismatch cannot cause a partial install.
    for row, target, source, backup, state in plans:
        print(f"{row['color']}: {state} ({row['target']})")
        if dry_run:
            continue
        if action == "install" and state == "original":
            if not backup.exists():
                atomic_copy(target, backup)
            atomic_copy(source, target)
        elif action == "restore" and state == "installed":
            atomic_copy(backup, target)
    return {row["color"]: state for row, _, _, _, state in plans}


def require_installed(assets_root):
    states = operate(assets_root)
    if set(states.values()) != {"installed"}:
        raise RuntimeError("Run python -m robofollow.assets install before collection/evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "install", "restore"))
    parser.add_argument("--assets-root", type=Path, default=PACKAGE.parent / "assets")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    operate(args.assets_root, args.action, args.dry_run)


if __name__ == "__main__":
    main()
