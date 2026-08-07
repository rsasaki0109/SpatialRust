#!/usr/bin/env bash
# Runs the SpatialRust-vs-PDAL benchmark on an identical public PCL cloud and
# prints a side-by-side timing table.
#
# Prerequisites:
#   - pdal (>= 2.4) on PATH, Python, and a Rust toolchain
#
# Usage:
#   bench/pdal_comparison/run.sh
#   bench/pdal_comparison/run.sh --input cloud.pcd
#   bench/pdal_comparison/run.sh --synthetic 200000
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PCD="$ROOT/target/bench-data/table_scene_lms400.pcd"
PY="${PYTHON:-python3}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --input)
      PCD="$2"
      shift 2
      ;;
    --synthetic)
      N="$2"
      PCD="/tmp/bench_cloud_${N}.pcd"
      echo "== generating $N-point synthetic cloud =="
      "$PY" "$ROOT/bench/pcl_comparison/gen_cloud.py" --points "$N" --out "$PCD"
      shift 2
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        N="$1"
        PCD="/tmp/bench_cloud_${N}.pcd"
        echo "== generating $N-point synthetic cloud =="
        "$PY" "$ROOT/bench/pcl_comparison/gen_cloud.py" --points "$N" --out "$PCD"
        shift
      else
        PCD="$1"
        shift
      fi
      ;;
  esac
done

if [ ! -f "$PCD" ]; then
  echo "== fetching public PCL table_scene_lms400 sample =="
  "$PY" "$ROOT/bench/pcl_comparison/fetch_public_cloud.py" --out "$PCD"
else
  echo "== using input cloud $PCD =="
fi

if ! command -v pdal >/dev/null 2>&1; then
  echo "error: pdal is required (install pdal >= 2.4)" >&2
  exit 1
fi

echo "== building SpatialRust bench_ops =="
cargo build --release --manifest-path "$ROOT/Cargo.toml" -p spatialrust \
  --example bench_ops --features mvp,filter-outlier,transform-ops >/dev/null 2>&1

echo "== running SpatialRust =="
"$ROOT/target/release/examples/bench_ops" "$PCD" > /tmp/sr_pdal_out.csv
echo "== running PDAL =="
"$PY" "$HERE/pdal_bench.py" "$PCD" > /tmp/pdal_out.csv

echo
printf '%-30s %14s %14s %10s\n' "operation" "SpatialRust(s)" "PDAL(s)" "speedup"
printf '%-30s %14s %14s %10s\n' "------------------------------" "--------------" "--------------" "----------"
while IFS=, read -r op sr_t sr_n; do
  pdal_line="$(grep "^$op," /tmp/pdal_out.csv || true)"
  pdal_t="$(echo "$pdal_line" | cut -d, -f2)"
  if [ -n "$pdal_t" ]; then
    speedup="$(awk -v a="$pdal_t" -v b="$sr_t" 'BEGIN{ if(b>0) printf "%.2fx", a/b; else print "n/a" }')"
  else
    speedup="n/a"
  fi
  printf '%-30s %14s %14s %10s\n' "$op" "$sr_t" "${pdal_t:-n/a}" "$speedup"
done < /tmp/sr_pdal_out.csv
