# Source-bound RealSense operational validation

All 890 recorded depth frames across two Open3D-distributed Intel RealSense L515
bags complete native projection and independent coordinate checks. This validates
depth ingestion and geometry/IO execution on actual sensor data. It does not
measure motion accuracy, clock synchronization, color alignment or the original
Autoware bag's mounting calibration.

## Fixed source and protocol

The protocol is `benchmarks/realsense-l515-plan.json`, committed as `4ce5a76`
before processing. The tested runner/helper is committed as `136ddd1` before the
real execution. Publisher descriptors come from Open3D commit
`1e7b17438687a0b0c1e5a7187321ac7044afe275`; their source hashes are in the receipt.
HTTPS certificate verification and both official MD5 and frozen SHA-256 checks
remain enabled. No input archive, bag, generated point cloud or native binary is
committed. The MIT license on descriptor source does not substitute for any
original data distribution terms.

| Recording | Publisher MD5 | Frozen SHA-256 |
| --- | --- | --- |
| SampleL515Bag.zip | `9770eeb194c78103037dbdbec78b9c8c` | `3b8c34c7f4e22a7817b3e9cc513f6d894943645f23dafbc2f5a013d4dd06a32a` |
| JackJackL515Bag.bag | `9f670dc92569b986b739c4179a659176` | `7e75130b2375bccbc633adcab65c24273a83fc2b1f94d06183e2239071527f5e` |

URLs are the official `isl-org/open3d_downloads` GitHub release, tag
`20220301-data`. The ZIP contains exactly `L515_test_s.bag`; the extracted bag
is also hash-bound (`38365d600d0d4f1ae9e76d435571a4d6fa20d1ccf0db8e3eac18440d27709f63`).
The larger bag is 1,132,964,755 bytes and contains 884 depth, 884 infrared and
441 color frames. Only its depth stream is projected. Depth intrinsics and the
recorded depth unit, approximately 0.00025 m per count, are taken from that
bag's initial metadata. The depth calibration declares no distortion. Other
distortion models, late/changing calibration and unsupported image layouts are
rejected; no rectification is assumed.

All recorded depth frames use fixed 0.1..3.0 m bounds. A single native f32 dense
output buffer is reused; a separate f64 pinhole formula checks the invalid mask
and every valid optical xyz coordinate, with preregistered 1e-6 m absolute
tolerance. Optical axes are x right, y down, z forward. Recorded extrinsics are
retained with `applied:false`; no world/body transform is introduced. Color is
unregistered and has different intrinsics/resolution, so RGB-D alignment is not
claimed.

Every 60th frame plus the final frame is saved as a binary PCD with xyz,
`raw_depth:u16`, `pixel_u:u32` and `pixel_v:u32`. Native read, typed Arrow export,
native write and native reread must preserve all types and values exactly.
Every 120th frame runs the existing CPU MVP pipeline with 0.05 m voxels,
0.02 m plane gate, 0.1 m cluster tolerance and minimum cluster size 10. That
pipeline receives xyz only; the separate IO audit is the attribute-preservation
claim. Pipeline completion is not a scene-label accuracy result.

Bag timestamps and header timestamps are retained as separate integer
nanoseconds. All raw per-frame depth metadata strings are kept, including the
declared `Hardware Clock` domain. No offset, epoch equivalence, drift correction
or exposure-time synchronization is inferred from those labels. Source bytes,
serialized frames, camera metadata, transforms, helper/native code and saved
outputs are SHA-256 bound. Per-frame failures remain in the denominator and
checkpoint JSONL; completed output directories cannot be overwritten.

The reader has explicit 10,000-frame and 2,073,600-pixel limits. Geometry is
processed one frame at a time, while inventory and result metadata grow with
the frame count inside that bound. This is a bounded benchmark runner, not an
unlimited ROS recording ingestion service or a native ROS1 API.

## Observed results

| Recording | Successful / planned depth frames | Projected points | Exact IO frames | MVP runs | Max coordinate difference |
| --- | --- | --- | --- | --- | --- |
| L515_test_s | 6 / 6 | 442,633 | 2 | 1 | 1.968e-7 m |
| JackJackL515Bag | 884 / 884 | 263,672,519 | 16 | 8 | 2.834e-7 m |

The long recording's sensor timestamps span 29,306,871,000 ns; its depth bag
timestamps span 29,408,950,688 ns. These are distinct clocks. Both recordings
have zero skipped sequence counters. No frame is selected by a reference pose
or a favorable score. The recorded process peak is 172,268 KiB (about 168 MiB).
The long bag's processing loop takes 24.765 s on this machine, excluding initial
inventory, source verification and startup. No real-time or cross-library speed
claim follows from that local interval.

Raw results are retained at
`/workspace/SpatialRust/target/real-sensor-reference/run-v1` with all frame
outcomes, calibration inventory, 18 input/restored PCD pairs and source bindings.
The compact reviewable record is
[`receipts/2026-10-10_realsense_operation.json`](receipts/2026-10-10_realsense_operation.json).

Reproduce in a new output directory:

```sh
cd /workspace/SpatialRust
source /workspace/.spatialrust-env/recovery-env.sh
python -m pip install 'rosbags==0.11.6'
python scripts/run_realsense_reference.py \
  --plan benchmarks/realsense-l515-plan.json \
  --data-dir target/real-sensor-reference/official \
  --output-dir target/real-sensor-reference/run-new
```

Optional rosbags is used only by this benchmark and imported lazily by the
inventory/runner. The native library's dependencies and APIs are unchanged.
Core correctness tests use bounded fixtures and run without rosbags installed.

## Independent trajectory evaluation preparation

`evaluate_timestamped_trajectory.py` scores already frozen estimates against a
separate TUM-format trajectory. It requires both complete file hashes, matching
explicit sensor frames, metre units, source/calibration/plan hashes and a
declaration of reference-free generation. Those declarations and hashes bind
the inputs; they do not independently certify calibration or earlier generation.

Decimal epochs are converted exactly into int64 nanoseconds; subnanosecond,
nonfinite and out-of-range times are rejected. References require strictly
increasing times and proper poses. Association uses bounded interpolation,
shortest-path quaternion SLERP and no extrapolation. The defaults are a 20 ms
maximum gap to both bracket ends and 1 s RPE interval, with 20 ms pair tolerance.
Failed estimates and unmatched reference coverage remain visible. ATE is
reported in the original coordinates and after a first jointly valid pose
SE3 gauge alignment. No scale or best-fit trajectory alignment is used. RPE is
scored from relative motions, with planned and evaluated pair counts separate.

The frozen estimate schema is `spatialrust.timestamped-trajectory.v1` with
`length_unit:"m"`, `sensor_frame`, `reference_used_for_generation:false`,
`source_sha256`, `calibration_sha256`, `plan_sha256`, and ordered `poses`.
Each pose contains integer `timestamp_ns` and either `status:"success"` with
proper `world_from_sensor`, or `status:"error"` with a nonempty `error`.

```sh
python scripts/evaluate_timestamped_trajectory.py \
  --estimates frozen-estimates.json --estimates-sha256 VERIFIED_ESTIMATES_SHA256 \
  --reference groundtruth.txt --reference-sha256 VERIFIED_REFERENCE_SHA256 \
  --reference-frame DECLARED_CAMERA_FRAME --output new-accuracy.json
```

These commands require actual independently sourced inputs. Current validation
is analytic/unit/CLI regression only: no TUM accuracy result is reported. The
official URL attempted is
`https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz`.
The current proxy returns CONNECT HTTP 403; it does not establish whether the
file itself exists. Host additions `cvg.cit.tum.de` and `vision.in.tum.de` are
saved in the environment draft, preserving the existing destinations. Draft
persistence does not apply/publish the settings or verify the connection.

## Validation and maturity

The complete installed default-wheel Python suite passes **639 tests**, with
**three expected ONNX skips**, zero failures and zero errors (642 planned).
The 56 new tests cover exact epochs, endian/padded rows, unsupported calibration,
late/missing source evidence, typed native IO, bounded archives, noncommuting
pose gauge, retained tracking failures, missed reference coverage, scale drift,
and frozen CLI hashes. The 12 standard-library CI helper tests also pass.

The provisional estimate is **86%**, from 85%, on the limited real-sensor
ingestion/geometry/IO evidence. It is not measured PCL/Open3D/OpenCV parity.
Independent real trajectory accuracy, measured synchronization, longer/broader
operation and the canonical Autoware bag's source-bound clock/mounting evidence
remain unresolved. The 90% target is not claimed.
