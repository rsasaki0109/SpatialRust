#!/usr/bin/env bash
# CPU vs GPU Euclidean clustering benchmark (Unix).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 bench/euclidean_cluster/run.py "$@"
