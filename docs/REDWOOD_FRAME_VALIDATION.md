# Fixed Redwood frame validation

Open3D's official distribution makes the augmented ICL-NUIM livingroom1 noisy
depth and reference trajectory available through verified HTTPS. The original
`redwood-data.org` service still fails upstream certificate verification in this
environment. Using Open3D's declared distribution does not repair that service;
no proxy bypass or disabled TLS verification was used.

The dataset descriptors, reference convention, and camera defaults were checked
against Open3D commit
[`1e7b17438687a0b0c1e5a7187321ac7044afe275`](https://github.com/isl-org/Open3D/tree/1e7b17438687a0b0c1e5a7187321ac7044afe275):
`RedwoodIndoorLivingRoom1.cpp`, `FileLOG.cpp`, `PinholeCameraIntrinsic.cpp`,
`RGBDImage.h`, and the dataset binding documentation. This is noisy **synthetic**
data, separate from the prior GeoTransformer/3DMatch data. It is not measured
multi-sensor calibration or an independent real-sensor trajectory.

## Protocol and preparation

The [plan](../benchmarks/redwood-livingroom1-plan.json) was committed at `d7e4c78`
before fitting or viewing accuracy. Four fixed pairs `(0,30)`, `(900,930)`,
`(1800,1830)`, `(2700,2730)` cover different parts of one 2,870-frame scene.
Each method runs seeds 7, 8, 9 with the existing global registration controls.
No pair is selected using geometry, overlap or reference motion, and every
failure remains in the denominator. Passing requires translation error at most
0.05 m **and** rotation error at most 5°. This is a frame-relative pose criterion,
not the publisher's fragment-registration recall metric.

Only the noisy-depth ZIP and trajectory are needed. No reference surface, clean
depth, color or ONI input reaches the algorithms. The 260,777,252-byte ZIP matches
the official MD5 `2fec03a29258a29b9ffedde88fddd676`; its SHA-256 is
`e5f0497abdd77bf8ca16d3216ef3642f1041e743a31519a071562597fcb65acd`.
The 748,425-byte trajectory matches MD5 `601ac4b51aba2455a90aed8aa1158c6a` and
SHA-256 `81148ea03d897a30034a634c08b55264dabff8185f2fc98880fc96b4aa227a51`.
Checksums are taken from the pinned official descriptor, not replaced after
download failure. The ZIP member inventory is exact and duplicate-free; only
the eight declared images are read, without filesystem extraction.

Backprojection uses PrimeSense 640×480 intrinsics (525, 525, 319.5, 239.5),
millimetres to metres, and `0 < depth < 3 m`. Coordinates are optical x-right,
y-down, z-forward. Eight clouds contain 1,898,532 points in total; each PCD's
f32 coordinates roundtrip exactly. Every input PNG and generated PCD is hashed.
No sensor extrinsic or world transform is applied to input points.

`run_redwood_reference.py` calls the existing `compare_public_global.run`
without references or initial poses. It saves all raw registration rows and
hashes the frozen file **before reading the trajectory**. LOG matrices are
`world_from_camera`; evaluation uses
`inverse(world_from_target) @ world_from_source`. Native libraries, helpers,
plan, input preparation and frozen output are bound in the receipts. Output
directories must be new so a later invocation cannot replace prior evidence.

## Results and limits

The [committed summary receipt](receipts/2026-10-10_redwood_frame_validation.json)
contains every pair/seed error and the full source/input/output hash bindings.

| Method | Generated | Meets fixed accuracy limits | Translation error | Rotation error |
| --- | --- | --- | --- | --- |
| SpatialRust | 12/12 | 12/12 | 0.012995–0.047019 m | 0.110528–1.172285° |
| Open3D 0.19.0 | 12/12 | 12/12 | 0.014619–0.042791 m | 0.110206–0.928084° |

The independent audit verifies all 2,870 LOG matrices against Open3D's LOG
reader, all eight backprojections against Open3D, and all 24 saved pose scores.
Maximum geometry difference is `1.217071e-7 m`. Trace/acos audit scoring projects
only the audit rotation onto SO(3) to remove f32 orthogonality noise; it never
changes saved poses or the primary atan2 score. Maximum projection norm is
`1.059413e-7`, and angular scores differ by at most `2.729053e-8°`.

This closes the inability to run a separate dataset through a legitimate
official distribution. It does not prove broad robustness: four nearby frame
pairs in one synthetic scene are the independent cases; repeated seeds are
not extra scenes. Both methods pass the declared criterion, without an accuracy
or speed superiority claim. Calibration, complete hosted CI, and broader
operational evidence remain necessary for the 90% target. This data-only
checkpoint kept maturity at 82%; the subsequent complete hosted CI checkpoint
is recorded in [the current assessment](MATURITY.md).

Local validation: 18 new correctness tests; the full default-wheel Python suite
has **583 passed, three expected ONNX skips**; nine CI-helper tests pass. Existing
fragment-reference functions and their 20 tests remain unchanged.

## Reproduce and audit without refitting

Use the selected comparison environment (`numpy`, Pillow, SpatialRust, Open3D):

```bash
source /workspace/.spatialrust-env/recovery-env.sh
cd /workspace/SpatialRust
mkdir -p target/redwood-inputs
curl --fail --location --proto '=https' --tlsv1.2 \
  https://github.com/isl-org/open3d_downloads/releases/download/augmented-icl-nuim/livingroom1-depth-simulated.zip \
  --output target/redwood-inputs/livingroom1-depth-simulated.zip
curl --fail --location --proto '=https' --tlsv1.2 \
  https://github.com/isl-org/open3d_downloads/releases/download/augmented-icl-nuim/livingroom1-traj.txt \
  --output target/redwood-inputs/livingroom1-traj.txt
python scripts/run_redwood_reference.py \
  --plan benchmarks/redwood-livingroom1-plan.json --data-dir target/redwood-inputs \
  --output-dir target/redwood-run
python scripts/audit_redwood_frames.py \
  --plan benchmarks/redwood-livingroom1-plan.json --data-dir target/redwood-inputs \
  --frozen-dir target/redwood-run --output-dir target/redwood-audit
```

The audit performs no registration. It copies identical trajectory bytes to a
`.log` filename because Open3D selects its LOG parser by extension (`.txt` selects
TUM). In this task the actual run and audit are retained at
`/workspace/SpatialRust/target/blocker-resolution/redwood-validation-v2` and
`/workspace/SpatialRust/target/blocker-resolution/redwood-audit`.
Raw sensor/image data and native binaries are not committed.
