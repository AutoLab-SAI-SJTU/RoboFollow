#!/usr/bin/env bash
# Install into a separate environment; run from any working directory.
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${PACKAGE_DIR}/../XPolicyLab/policy/Pi_05/openpi}"
if ! command -v uv >/dev/null 2>&1; then
    echo 'Install uv first, then rerun this script.' >&2
    exit 1
fi
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${OPENPI_ROOT}/.venv}"
cd "$OPENPI_ROOT"
uv sync --frozen --python 3.11 --group lerobot --no-dev
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -r "$PACKAGE_DIR/requirements-pi05.txt"
"$UV_PROJECT_ENVIRONMENT/bin/python" "$PACKAGE_DIR/check_pi05_environment.py" "$OPENPI_ROOT"
printf 'Model interpreter: %s/bin/python\n' "$UV_PROJECT_ENVIRONMENT"
