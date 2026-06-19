#!/usr/bin/env bash
# Thin shell wrapper around run.py.
#
# Prerequisites:
#   - open3d and numpy installed in ../../.venv
#   - the SpatialRust Python extension installed in ../../.venv (maturin develop)
#
# Usage: bench/open3d_comparison/run.sh [N_POINTS]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python"

exec "$PY" "$HERE/run.py" "$@"
