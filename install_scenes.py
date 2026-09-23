"""Copy the four bundled RoboFollow scenes into a RoboTwin envs directory.

Place robofollow/ inside RoboTwin, then run:
    python robofollow/install_scenes.py

Only Python's standard library is needed; no simulator is imported.
"""
import argparse
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
SCENE_FILES = tuple(f"scene{i}.py" for i in range(1, 5))


def install(target=None, *, dry_run=False):
    target = Path(target).expanduser().resolve() if target is not None else PACKAGE.parent
    envs = target / "envs"
    if envs.is_symlink() or not (envs / "_base_task.py").is_file():
        raise ValueError(f"Not a RoboTwin root with an envs/_base_task.py file: {target}")
    if not (target / "robofollow/collect.py").is_file():
        raise ValueError(f"Place the complete robofollow directory in {target} first")

    # Check every source and destination before adding any files.
    plans = []
    unchanged = []
    for name in SCENE_FILES:
        source = PACKAGE / "scene_sources" / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing bundled scene (a regular file is required): {source}")
        data = source.read_bytes()
        destination = envs / name
        if destination.is_symlink():
            raise ValueError(f"Refusing to replace a symlink: {destination}")
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != data:
                raise ValueError(
                    f"Existing scene differs from the bundled copy: {destination}. "
                    "Back up and remove that file yourself before installing this version."
                )
            unchanged.append(destination)
        else:
            plans.append((destination, data))

    for destination in unchanged:
        print(f"Unchanged: {destination}")
    for destination, data in plans:
        if dry_run:
            print(f"Would add: {destination}")
            continue
        # Exclusive creation also protects files created after the checks above.
        stream = destination.open("xb")
        try:
            with stream:
                stream.write(data)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        print(f"Added: {destination}")
    action = "Would add" if dry_run else "Added"
    print(f"{action} {len(plans)} scenes; {len(unchanged)} already match.")
    return len(plans)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=PACKAGE.parent,
        help="RoboTwin root; defaults to the parent of this robofollow directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="Check and show the plan without writing files")
    args = parser.parse_args()
    try:
        install(args.target, dry_run=args.dry_run)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
