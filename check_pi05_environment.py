"""Check OpenPI dependencies, allowing the optional Rerun NumPy conflict."""
from importlib import metadata
from pathlib import Path
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    root = Path(sys.argv[1])
    project = tomllib.loads((root / "pyproject.toml").read_text())
    overrides = project.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    numpy_override = any(
        canonicalize_name(Requirement(value).name) == "numpy"
        and str(Requirement(value).specifier) == "<2.0.0,>=1.22.4"
        for value in overrides
    )
    problems = []
    for distribution in metadata.distributions():
        for value in distribution.requires or ():
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                problems.append(f"{distribution.name} requires missing {requirement.name}")
                continue
            if not requirement.specifier or requirement.specifier.contains(installed, prereleases=True):
                continue
            message = f"{distribution.name}=={distribution.version} requires {requirement}; installed {installed}"
            # Narrowly preserve the explicit override in the pinned upstream
            # project. Never silently waive unrelated dependency conflicts.
            if (numpy_override and canonicalize_name(distribution.name) == "rerun-sdk"
                    and distribution.version == "0.26.2"
                    and canonicalize_name(requirement.name) == "numpy"
                    and str(requirement.specifier) == ">=2" and installed == "1.26.4"):
                print(f"KNOWN UPSTREAM OVERRIDE (Rerun viewer dependency unsatisfied): {message}", file=sys.stderr)
            else:
                problems.append(message)
    if problems:
        raise SystemExit("\n".join(problems))
    print("Installed requirements checked against the upstream override policy.")


if __name__ == "__main__":
    main()
