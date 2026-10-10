# Managed environment and comparison-data recovery

## Verified recovery

The installed toolchain was outside PATH, under
`/workspace/.spatialrust-env`. Activating its existing `env.sh` restores
Cargo/rustc 1.99.0. The earlier conclusion that this environment had no Rust
installation was based only on PATH and is superseded by these checks.

In `/workspace/SpatialRust-recovery`, core compilation and the 36 core/math
all-feature library tests pass. A frozen default Python wheel builds offline
with `maturin build --release --locked --offline`. Its installed native/stub/marker
bytes match the wheel; 565 Python tests pass with exactly three expected ONNX
skips. The new wheel does not require ONNX shared libraries. The older comparison
installation imports after setting `LD_LIBRARY_PATH` to
`/workspace/SpatialRust-python-delivery/target/onnxruntime`.

The restored local activation entry point is:

```bash
source /workspace/.spatialrust-env/recovery-env.sh
cd /workspace/SpatialRust-recovery
```

This selects the new default wheel and Open3D 0.19.0. Its virtual environment
reuses existing comparison dependencies through a `.pth` file; this is a local
recovery environment, not a separately resolved clean dependency installation.
The activation file and data/build artifacts are local and are not committed.

## Network findings

Executor HTTPS calls must request network permission. With that permission,
GitHub git fetch, public file downloads and `https://3dmatch.cs.princeton.edu/`
succeed. Earlier proxy-connect failures from calls without permission do not
establish a broken proxy. Current explicit CONNECT denials remain for:

- `redwood-data.org`: independent Redwood/ICL-NUIM comparison source;
- `3dvision.princeton.edu`: official 3DMatch fragment archives;
- `autoware-auto.s3.us-east-2.amazonaws.com`: canonical public rosbag archive;
- `api.github.com` from the executor (GitHub connector API access works).

No proxy route, mirror or domain workaround was used for denied downloads.
Current configured custom hosts contain `3dmatch.cs.princeton.edu`; a historical
environment draft in earlier notes is not evidence of an active allowlist change.
Publishing a managed network-policy change requires the environment configuration
workflow; no supported configuration/publish tool is exposed in this session.

## Public data and actual comparison

Numeric demo files were freshly downloaded from the publisher's GitHub repository,
at commit `e7a135af4c318ff3b8d7f6c963df094d7e4ea540`, using the paths
`data/demo/src.npy`, `ref.npy`, and `gt.npy`. No repository code or pickle was run.

| File | SHA-256 |
| --- | --- |
| src.npy | `266f01df95375b33b4d259c40d9009f2f86dd7219ec23a9fb4fca92e3b3a1b02` |
| ref.npy | `030c5e827916c99684ae4a50435581bbf6caeb80f918de705a738beb007d827d` |
| gt.npy | `04ab380fe5152dfe70a54dc5a852fdfa0bba73eed1943be660b2462f6a80f237` |

`scripts/prepare_numpy_reference.py` validates 15,953 source and 18,977 target
points, disables pickle, verifies PCD roundtrips and binds reference/input hashes.
The supplied near-rigid rotation uses the existing explicit correction option.
This reference's independent sensor lineage remains unverified. Re-downloading
the same demo is not an independent second dataset.

With seeds 7, 8 and 9, the freshly rebuilt native wheel and Open3D complete six
global-registration rows. Native translation errors are 0.169811–0.206075 m;
Open3D errors are 0.175508–0.218663 m. Executing a row successfully is not an
accuracy success. The shared partial-cloud drift remains visible; no accuracy
ranking, default-policy change or speed claim follows from this recovery.

Local review artifacts:

- `/workspace/SpatialRust-recovery/target/recovery-data/` — originals and hash-bound prepared pair;
- `/workspace/SpatialRust-recovery/target/recovery-comparison/comparison.json` and `report.html`;
- `/workspace/SpatialRust-recovery/target/recovery-python-junit.xml`;
- `/workspace/SpatialRust-recovery/target/recovery-wheel-receipt.json`;
- `/workspace/SpatialRust-recovery/target/recovery-inventory.json`.

## Operational calibration remains unavailable

The user does not know the source storage location. The documented external disk
`/media/sasaki/aiueo` is not mounted, and no matching rosbag or measured clock/frame
artifact was found in the available workspace. The historical canonical input is
713,670,656 bytes, SHA-256
`b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`.
The official Autoware archive URL is
`https://autoware-auto.s3.us-east-2.amazonaws.com/rosbag2/rosbag2-astuff-1-lidar-only.tar.gz`;
the current proxy denies that host. Even retrieving this archive would not close
calibration: the prior source survey found no clock or TF artifacts in it.

To close this gate, obtain measured evidence matching the actual bag's path,
size and hash, using the existing `rosbag2_calibration_evidence` contract:

- clock evidence: source/target time domains, method, basis, sample count,
  offsets, drift and uncertainty;
- frame evidence: declared root, explicit root-to-`lidar_front`/`lidar_rear`
  edges, translations in metres and rotations as xyzw quaternions, with method.

An unrelated capture's transforms, generic vehicle configuration or an identity
transform cannot establish this binding. No calibration values were fabricated
or registered. Operational full-pipeline validation remains blocked on this input.
The existing calibration-evidence example's four contract tests pass, including
missing-artifact handling and exact source path/size/hash matching. This verifies
the evidence validator, not measured calibration or operation on the absent bag.

## Hosted CI and HTTP fixture repair

On PR #112 head `0cc8b08`, all four installed-wheel runtime jobs (Windows/macOS,
Linux Python 3.8/3.14), sdist rebuild/runtime and the strict rendered Web/WASM
browser assertion pass. A complete 103-job CI observation finds nine successes,
one failure, one active and 92 queued jobs; this is not a complete green matrix.

The failed macOS streaming job `114088624425` reports binary bytes interpreted
as an HTTP status line. The shared local test server incorrectly sends a response
body for HEAD and omits `Connection: close` despite closing each connection.
A raw TCP regression deterministically fails before the repair and passes after
it. The server now advertises the GET length on HEAD but sends no payload and
declares connection closure. The all-format IO suite passes 63 library tests,
two HTTP integration tests and one PCD integration test. Production HTTP reader
code is unchanged. A hosted macOS rerun on the repaired head is still required.
The exact streaming IO feature set also passes 61 library and 11 integration
tests. Rust 1.99 strict Clippy exposed an existing indexed loop in a node-reader
test; replacing it with `enumerate` preserves the assertions and clears the lint.

The provisional OSS-relative maturity estimate remains **82%**. Environment
recovery and replay restore reproducibility; independent-dataset accuracy,
calibrated operational validation and the complete final-head CI matrix remain
requirements toward 90%.
