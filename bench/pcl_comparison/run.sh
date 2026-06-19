#!/usr/bin/env bash
# Runs the SpatialRust-vs-PCL benchmark on an identical public PCL cloud and
# prints a side-by-side timing table.
#
# Prerequisites:
#   - libpcl-dev (headers + libs), g++, eigen3, Python
#   - NumPy + the SpatialRust Python extension only for --synthetic
#
# Usage:
#   bench/pcl_comparison/run.sh
#   bench/pcl_comparison/run.sh --input cloud.pcd
#   bench/pcl_comparison/run.sh --synthetic 200000
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PCD="$ROOT/target/bench-data/table_scene_lms400.pcd"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="${PYTHON:-python3}"
fi

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
      "$PY" "$HERE/gen_cloud.py" --points "$N" --out "$PCD"
      shift 2
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        N="$1"
        PCD="/tmp/bench_cloud_${N}.pcd"
        echo "== generating $N-point synthetic cloud =="
        "$PY" "$HERE/gen_cloud.py" --points "$N" --out "$PCD"
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
  "$PY" "$HERE/fetch_public_cloud.py" --out "$PCD"
else
  echo "== using input cloud $PCD =="
fi

echo "== building benches =="
cargo build --release --manifest-path "$ROOT/Cargo.toml" -p spatialrust \
  --example bench_ops --features mvp,filter-outlier >/dev/null 2>&1

PCL_INC="-I/usr/include/pcl-1.14 -I/usr/include/eigen3"
PCL_LIBS="-lpcl_common -lpcl_io -lpcl_filters -lpcl_features -lpcl_search -lpcl_kdtree -lpcl_octree"
g++ -O2 -std=c++17 $PCL_INC "$HERE/pcl_bench.cpp" -o /tmp/pcl_bench $PCL_LIBS

echo "== running SpatialRust =="
"$ROOT/target/release/examples/bench_ops" "$PCD" > /tmp/sr_out.csv
echo "== running PCL =="
/tmp/pcl_bench "$PCD" > /tmp/pcl_out.csv

echo
printf '%-30s %14s %14s %10s\n' "operation" "SpatialRust(s)" "PCL(s)" "speedup"
printf '%-30s %14s %14s %10s\n' "------------------------------" "--------------" "--------------" "----------"
while IFS=, read -r op sr_t sr_n; do
  pcl_line="$(grep "^$op," /tmp/pcl_out.csv || true)"
  pcl_t="$(echo "$pcl_line" | cut -d, -f2)"
  if [ -n "$pcl_t" ]; then
    speedup="$(awk -v a="$pcl_t" -v b="$sr_t" 'BEGIN{ if(b>0) printf "%.2fx", a/b; else print "n/a" }')"
  else
    speedup="n/a"
  fi
  printf '%-30s %14s %14s %10s\n' "$op" "$sr_t" "${pcl_t:-n/a}" "$speedup"
done < /tmp/sr_out.csv
