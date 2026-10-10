# SR2 handoff: exact-head CI evidence and canonical bag recovery

This task continues PR #112 at
`ce1fcd8a5e4da126423bb186ebcbafae6c9411ad` in the existing checkout
`/workspace/SpatialRust`. The former `/workspace/SpatialRust-recovery` machine
and its installed files are not present in this task. The environment and
artifacts below were prepared and exercised here, rather than inferred from
the earlier task's paths.

## CI observation tied to the actual commit

At 2026-10-10 01:50:30 UTC (10:50:30 Asia/Tokyo), run `38011747105` contains
103 unique jobs across both API pages: 24 successes and 79 queued jobs with
no assigned runners. macOS, Windows and Ubuntu Streaming 1.2 conformance jobs
all succeed. Run `38011747222` contains all eight jobs: seven successes and
one deliberately skipped publication job. Both runs belong to the full
`ce1fcd8a5e4da126423bb186ebcbafae6c9411ad` commit and attempt 1. PR #112 remains
open; a complete green main CI matrix is not established, so it was not merged.
Queue metadata alone does not identify a runner-allocation cause.

`inspect_github_ci.py` now accepts `--expected-head-sha`, validates every job's
run/head/attempt, and rereads the run identity after collecting pages. It rejects
a changed total across pages rather than accepting the latest total as a complete
snapshot. It still reports queued, skipped and failed outcomes separately. A
skipped publication job does not count as a successful test. Status observations
remain sampled across requests, not an atomic GitHub snapshot.

```bash
cd /workspace/SpatialRust
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
python3 scripts/inspect_github_ci.py --repo rsasaki0109/SpatialRust \
  --run-ids 38011747105 38011747222 \
  --expected-head-sha ce1fcd8a5e4da126423bb186ebcbafae6c9411ad \
  --output target/handoff-ci-observation.json
```

The output must be new for each observation. Nine stdlib tests pass, including
the previous two runtime CI regressions and seven checks for missing pages,
changing totals, old commits, mixed attempts, and incomplete outcomes. The
Web/WASM CI step discovers these new regressions too.

## Current-instance Python validation

Rust 1.99.0 and a clean CPython 3.12 virtual environment build the default abi3
wheel with `maturin build --release --locked`. Installed native/stub/marker bytes
match the wheel. All 565 applicable Python tests pass, with exactly three
expected ONNX skips. The environment has Open3D 0.19.0, NumPy 2.5.3 and PyArrow
26.0.0; it does not use the earlier task's `.pth` recovery shortcut.

The publisher's pinned GeoTransformer demo is downloaded again and all three
source hashes match the earlier SR2 receipt. With the same seeds 7/8/9, all six
native/Open3D registrations execute. Native translation errors remain
0.169811–0.206075 m and Open3D errors remain 0.175508–0.218663 m. This is replay
of the same demo, not an independent dataset or an accuracy/speed ranking.

Local artifacts, retained outside version control:

- `/workspace/SpatialRust/target/handoff-python-junit.xml`
- `/workspace/SpatialRust/target/handoff-wheel-receipt-final.json`
- `/workspace/SpatialRust/target/handoff-ci-observation.json`
- `/workspace/SpatialRust/target/handoff-comparison/comparison.json`
- `/workspace/SpatialRust/target/handoff-data/geotransformer/`

Activate with `source /workspace/.spatialrust-env/recovery-env.sh`.
The reusable install/start draft includes the Rust and Python preparation.
Saving the draft does not demonstrate publication or restoration in another task.

## Canonical rosbag recovered, calibration still missing

The official download now succeeds:
`https://autoware-auto.s3.us-east-2.amazonaws.com/rosbag2/rosbag2-astuff-1-lidar-only.tar.gz`.
The archive is 335,328,022 bytes with SHA-256
`317f5d3b249bb81fdb00e259552ebc6c1c21b8016a4c2491f5cc5889aa9746ac`.
Its SQLite file is exactly 713,670,656 bytes with the historical source hash
`b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`.
The archive contains only its bag directory, metadata YAML and SQLite file.
Read-only SQLite inspection confirms 768 front and 767 rear PointCloud2 messages;
there are no `/clock`, `/tf`, `/tf_static` or `/odom` topics. Timestamp extrema
are retained as integer nanoseconds in the local recovery receipt.

The actual recovered bag passes source identity checks. Existing
`rosbag2_calibration_readiness` returns its documented blocked exit 1;
`rosbag2_calibration_evidence` writes a hash-checked JSON/HTML/manifest and returns
its documented blocked exit 2. `identity_matches` is true and
`registration_ready` is false. Four existing calibration-evidence tests pass.
No clock offset or front/rear extrinsic was fabricated or applied. The declared
frame names identify missing graph endpoints, not measured transforms.

Artifacts are under `/workspace/SpatialRust/target/handoff-data/rosbag/`.
Sensor data and native binaries are not committed. Recovering the original bytes
closes the missing-file prerequisite, but not the measured calibration gate.

## Remaining gates and maturity

The environment draft preserves existing settings and the package-manager preset,
and adds `api.github.com`, `redwood-data.org`, `www.redwood-data.org`, and
`autoware-auto.s3.us-east-2.amazonaws.com`. Actual GitHub API and Autoware downloads
succeed; runtime configuration metadata remains an incomplete description of the
observed access. Redwood HTTPS on both official hosts returns a proxy 503 with an
upstream `CERTIFICATE_VERIFY_FAILED`. TLS verification remains enabled; no denied
route, alternate mirror or insecure download is used. Independent Redwood
accuracy validation remains blocked on a working verified publisher connection.

The provisional OSS-relative assessment stays **82%**. New-instance reproducibility,
correct evidence binding and bag recovery do not prove independent accuracy or
calibrated operation. Reaching 90% still requires the complete latest-head main
CI matrix, independently acquired dataset accuracy, and source-bound measured
clock/front/rear calibration and operational validation.
