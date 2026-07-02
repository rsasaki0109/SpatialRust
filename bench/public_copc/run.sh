#!/usr/bin/env bash
# Epic 60: public COPC validation harness (Unix).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 bench/public_copc/run.py "$@"
