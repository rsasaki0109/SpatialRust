#!/usr/bin/env bash
# Thin shell wrapper around run.py.
#
# Prerequisites:
#   - open3d installed in ../../.venv
#   - NumPy + the SpatialRust Python extension only for --synthetic-points
#
# Usage: bench/open3d_comparison/run.sh [--input cloud.pcd|--synthetic-points N]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="${PYTHON:-python3}"
fi

exec "$PY" "$HERE/run.py" "$@"
