#!/usr/bin/env bash
# CPU vs GPU RANSAC plane benchmark (Unix).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 bench/ransac_plane/run.py "$@"
