# Complete TUM sequence protocol

Preparation, CPU motion generation and evaluation now have separate commands for
the single fixed official sequence `rgbd_dataset_freiburg1_xyz`. The implementation
and plan are committed as `b2506d4f1ae92f3bd89ac88f5b4b5dbbd0dff6af` before any
real TUM data is acquired or evaluated. **No real TUM trajectory accuracy has
been measured.** The maturity estimate remains 86%, provisional.

## Verified work and remaining acquisition

Both Python 3.8.20 / NumPy 1.24.4 / PyArrow 17.0.0 / Pillow 10.4.0 and Python
3.12.14 / NumPy 2.5.3 / PyArrow 26.0.0 / Pillow 12.3.0 pass 679 applicable tests,
with three expected ONNX skips. The 32 new TUM regressions check complete
40-frame preparation, integer times beyond f64 epoch precision, missing images,
archive boundaries/links/budgets, hash changes, native projection, native ICP
direction, noncommuting camera composition, tracking loss, typed IO and post-fit
evaluation. Pillow remains a benchmark/test dependency, lazily imported by the
decoder; the SDK's runtime dependencies are unchanged. Hosted test jobs install
it so those decoding checks are executed rather than skipped.

All three actual CLIs also complete a retained **static synthetic 40-frame**
fixture: 40 prepared, generated and evaluated poses, 12,288,000 native projected
points, and exact typed IO at the first/final frames. Its zero ATE follows from
identical artificial images and identity truth; it is not a real motion accuracy
result. Its 0.39-second span has zero eligible one-second RPE pairs; the separate
analytic trajectory regressions exercise RPE. Artifacts are under
`/workspace/SpatialRust/target/goal-continuation/tum-cli-fixture-40`.

The RealSense attribute audit's use of `StructType.names` failed on the supported
PyArrow 17 endpoint. Iterating Arrow Fields restores the same exact schema/value
checks. The full endpoint suite passes, and all 18 retained real L515 input and
restored point-cloud pairs pass the corrected audit on Python 3.8 / PyArrow 17.
Historical run-v3 hashes are preserved; the new audit is
`/workspace/SpatialRust/target/real-sensor-reference/arrow17-saved-cloud-audit.json`.
PR #116's corrected head still requires its complete hosted CI gate before merge.

Publication applied the two previously requested TUM domains to runtime revision
7. Verified HTTPS fetches of the official format/download pages now succeed.
The official archive URL returns HTTP 302 to
`https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz`;
the proxy rejects that destination's CONNECT with HTTP 403. The exact redirect
hostname was added to the saved custom allowlist, preserving the prior six
entries and all presets. The save result requires publication; it is not proof
of running-instance access. Proxy and TLS verification remain enabled. The
download page advertises 0.47 GB for this sequence; its retrieved page does not
provide an archive checksum. After official HTTPS acquisition, record the full
SHA-256 before preparation; do not describe a locally recorded hash as a
publisher checksum.

## Fixed input and coordinate contract

`benchmarks/tum-fr1-xyz-plan.json` binds the retrieved official format-page bytes
by SHA-256. That page defines 640×480 16-bit PNG depth, counts / 5000 in metres,
depth already registered into RGB coordinates, and reference poses at the RGB
optical center in the mocap world. It recommends ROS default intrinsics
525/525/319.5/239.5 without additional undistortion. This documented approximation
is used unchanged; it is not new measured calibration. No extra IR-to-RGB
transform, Freiburg depth correction factor or sensor mounting transform is
applied. Counts convert to metres once in f64 before the f32 native projection,
so exact 0.1/5.0 m boundary pixels are retained. An independent f64 pinhole
formula checks all valid native coordinates and the invalid mask within 1e-6 m.
This numerical agreement does not establish physical calibration accuracy.

Preparation hashes the entire archive before and after a streaming tar pass.
It allows at most 30,000 members, 16 MiB per member and 8 GiB expanded bytes;
the complete RGB and depth indexes each allow 10,000 frames. It refuses path
aliases/traversal, links, special files, duplicate members, missing/unindexed
images and unsupported PNG headers. It retains every depth timestamp as int64
nanoseconds and every image's original bytes/hash. Nearest RGB association has a
20 ms limit with deterministic earlier-time ties; RGB may be reused or absent
and is not used by depth fitting. Full image decoding occurs during generation.
The reference member is excluded from the prepared directory and is not
interpreted during preparation or generation.

Generation keeps current/previous clouds and one reusable dense buffer, with
bounded per-frame metadata rather than accumulating all point clouds. Every
indexed depth frame remains in the output. Depth bounds are 0.1..5.0 m. Native
CPU point-to-point ICP has identity pair initialization, two fixed voxel/gate
stages (0.1/0.2 m and 0.05/0.1 m, 20 iterations each), no trimming and fixed
stopping criteria. Acceptance requires at least three supported points and 0.5
source support at the final 0.1 m gate; convergence/support are not truth labels.
Registration maps current into previous camera coordinates; composition is
`world_from_current = world_from_previous @ previous_from_current`. Only
validated f32 rotation roundoff is projected back to SO(3); no scale fitting
or reference alignment is used. A 100 ms inter-frame gap loses tracking. Failed
initial images remain failed, and the first valid image establishes the sole
identity anchor. After a tracking/IO failure, subsequent frames retain errors
without starting a disconnected trajectory at a new origin.

Every 60th/final successful frame preserves xyz, original `raw_depth:u16` and
pixel indices through binary PCD → native read → typed Arrow audit → native
write → reread. Per-frame JSONL checkpoints retain errors. Prepared source,
plan, calibration, used images, helper/native bytes and output hashes bind the
frozen estimates; code and used inputs are checked again before freezing.
Output directories cannot be overwritten. This runner has no loop closure,
reference relocalization, clock synchronization measurement or photometric RGB
odometry. It does not certify the original Autoware bag's missing extrinsics.

## Run after official acquisition

Activate `/workspace/.spatialrust-env/recovery-env.sh` and work in
`/workspace/SpatialRust`. Use distinct new output directories; replace the hash
placeholders with the actual complete SHA-256 values, never guessed values.

```bash
python scripts/prepare_tum_sequence.py \
  --archive target/tum-reference/rgbd_dataset_freiburg1_xyz.tgz \
  --archive-sha256 ARCHIVE_SHA256 \
  --plan benchmarks/tum-fr1-xyz-plan.json \
  --output-dir target/tum-reference/prepared-v1

python scripts/run_tum_sequence.py \
  --prepared-dir target/tum-reference/prepared-v1 \
  --manifest-sha256 MANIFEST_SHA256 \
  --output-dir target/tum-reference/generated-v1

python scripts/evaluate_tum_sequence.py \
  --archive target/tum-reference/rgbd_dataset_freiburg1_xyz.tgz \
  --estimates target/tum-reference/generated-v1/estimates.json \
  --estimates-sha256 FROZEN_ESTIMATES_SHA256 \
  --plan benchmarks/tum-fr1-xyz-plan.json \
  --output-dir target/tum-reference/evaluation-v1
```

The evaluator checks the supplied frozen estimate hash and the generation plan
before interpreting `groundtruth.txt` from the original hash-bound archive.
It reports every planned failure/unmatched pose, raw and first-pose-aligned ATE,
bounded interpolation without extrapolation, and planned/evaluated one-second
RPE counts. First-pose SE(3) alignment changes only the world gauge; it does not
fit scale or optimize trajectory alignment. Do not revise the plan after seeing
real scores and present a tuned replay as an independent blind result.

The compact evidence record is
[`receipts/2026-10-10_tum_preparation.json`](receipts/2026-10-10_tum_preparation.json).
