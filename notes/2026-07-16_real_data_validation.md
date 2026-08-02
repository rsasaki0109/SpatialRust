# Public real-data end-to-end validation

Date: 2026-07-16

SpatialRust 1.1.0 was validated on two public, non-sensitive point-cloud
datasets. No dataset bytes are committed to the repository.

## Host and execution

- Host: Windows x86_64, local default thread pools.
- Rust: 1.97.0, `--release`.
- Python: CPython 3.12, NumPy 2.4.6, locally built SpatialRust 1.1.0 release
  extension under `C:\Users\rsasa\Workspace\SpatialRust\.venv`.
- Repository commit before the validation change: `951d7db`.

## PCL table scene

Source:
`https://raw.githubusercontent.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd`

| Field | Observed value |
| --- | ---: |
| File bytes | 5,649,007 |
| SHA-256 | `E285D415641E0D9DE695B611DB874CC8FE995E8089B77A50D6056D24D8CBCC58` |
| Source points | 460,400 |
| COPC ROI points | 76,919 |
| COPC ROI + spacing×4 points | 26 |
| MVP plane inliers | 54 |
| MVP clusters | 36 |

The release Rust path read PCD, wrote COPC, re-read the full cloud, applied
bounds and resolution queries, then ran the MVP pipeline.

The Python path used leaf size 0.02:

| Stage | Observed value |
| --- | ---: |
| Statistical outlier result | 456,396 points; 4,004 removed |
| Voxel result | 9,840 points |
| DBSCAN | 4 clusters; 6 noise points |
| Registration cloud | 456 points |
| FPFH convergence / max transform error | true / 0.1124 |
| ICP convergence / refined max transform error | true / `0.0000105873` |
| One measured Python process run | 2.274 seconds |
| Harness-internal pipeline interval on repeat | 0.389 seconds |

The timing is a single end-to-end observation, not a portable performance
claim. The committed Python harness asserts the structural counts, convergence,
coarse error below 0.2, and refined error below `1e-4`.

## Remote Autzen COPC

Source: `https://s3.amazonaws.com/hobu-lidar/autzen-classified.copc.laz`

The original ignored smoke selected a corner ROI with a five-meter Z slice.
That query returned zero points while the test still passed. The strengthened
test uses the central 25% of the root XY bounds, retains the complete Z range,
and requires `0 < roi_count < header_point_count`.

| Field | Observed value |
| --- | ---: |
| Header points | 10,653,336 |
| Central XY ROI points | 889,058 |

The strengthened query completed successfully through the HTTP byte-source and
COPC hierarchy path.

## Reproduction

```powershell
python bench/public_copc/run.py
python bench/public_copc/run.py --http
.venv/Scripts/python.exe bench/public_copc/validate_python.py
```

The HTTP check is ignored in ordinary CI because it depends on an external
network service. Local PCD/COPC integration remains deterministic after the
public sample is fetched.
