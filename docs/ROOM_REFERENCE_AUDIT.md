# Complete frozen room reference audit

Slice 146W completes one prospectively fixed public CPU reference run over all
1,360 original room frames. The reference has first-anchor ATE **0.378945m**
and 1s translation RPE **0.058388m**; original reference acceptance is
**false**. The native model retains its original failure.

The fixed public reference also misses at least one original accuracy limit. This observation limits a native-only explanation for the room failure; it does not prove identical internal arithmetic or rule out native defects.

The [complete fixed protocol](../benchmarks/room-reference-audit-plan.json),
[exact child plan](../benchmarks/tum-fr1-room-reference-comparator-plan.json)
and [hash-bound evidence](receipts/2026-10-20_room_reference_audit.json)
retain the original calibration, complete timeline and every raw/causal/world/IO
and accuracy gate. The protocol was committed at `fc411dfdcddc0b5a0e4ec47b3c15d8e4a58cef76`;
Python implementation and all numerical observations bind
`81b082b4faaacdf99a1640ae6c3482538ad5c671`. No production Rust, native bridge/stub,
core/math/GPU tree, old estimator module, dependency or default changed.

## Complete real observations

| Frozen model | Generated / planned | Reference matched | First-anchor ATE (m) | 1s RPE (m) | Original accuracy |
| --- | ---: | ---: | ---: | ---: | --- |
| Native ReferencePrecision, already frozen | 1360 / 1360 | 1359 | 0.378652661 | 0.058375326 | Fail |
| Public Open3D CPU legacy hybrid | 1360 / 1360 | 1359 | 0.378944850 | 0.058387685 | Fail |

Original limits remain 100% generated, at least 95% reference matches,
ATE <=0.10m and 1s RPE <=0.05m. All 1359 successful adjacent
pairs independently pass original raw bilinear count/support/depth admission,
causal seed and proper/world-direction checks. Complete cloud/typed IO stays
in the unchanged generator. The independent SciPy trajectory audit reproduces
the reference scores. A complete trajectory was hash-frozen **before** the
first score; there was one fitting attempt, with no retry, reset, trimming,
rescaling, ground truth in fitting, parameter sweep or post-score policy change.

The actual public CPU API is the separately frozen Open3D 0.19.0 module,
using the unchanged `PreviousMotionOpen3dRgbd` adapter: hybrid Jacobian,
depth [0.1,5.0]m, depth difference 0.03m and [20,10,5] iterations. Identity seeds
the first pair and each returned current-from-previous increment seeds the next.
The public 6x6 information matrix is diagnostic; it is not measured uncertainty.
The original native IO module is reused for the real run. SDK validation uses
the already frozen latest wheel so all existing Huber API tests remain covered.

## Complete native/reference differences

All **1360 / 1360** planned timestamps are compared
in the unchanged common first-frame optical world, with no fitted alignment.
Translation difference is **0.000618069m RMS**,
**0.002157494m maximum**; proper rotation difference is
**0.019933053 degrees RMS**,
**0.137703268 degrees maximum**. Every per-row difference and
any missing row remain explicit. Independent NumPy translations and SciPy
rotation magnitudes check all rows and aggregate statistics. Disagreement is an
observation, not a claim about an unobserved internal Jacobian or correspondence.

## Seven source-model cases

The pinned [upstream odometry source](https://github.com/isl-org/Open3D/blob/1e7b17438687a0b0c1e5a7187321ac7044afe275/cpp/open3d/pipelines/odometry/Odometry.cpp)
indexes its correspondence map by source pixel. Its depth replacement branch
does not establish unique target pixels. Both source-directed equation models
can retain multiple sources at one target, so a missing target z-buffer is not
established by this inspection.

Seven prospectively fixed analytical cases agree on identity, contraction,
selected-target/source NaN holes, signed truncation and exact/next-down depth
boundaries. The contraction retains 16 rows at four targets. These are bounded
**source-modeled equations only**; actual runtime pair arrays remain unexposed
and unobserved. They do not instrument the frozen binary or explain a full
trajectory difference by themselves.

## Verification and storage

- 37 new correctness-critical math/schema/IO cases; 167 focused passes.
- 1,374 complete SDK passes and the same three pre-existing skips.
- Installed frozen wheel and stub verified; no new native build or installation.
- All 477 prior scientific JSON, 42 frozen native artifacts, seven pinned
  upstream source files and the completed room audit remain hash-identical.
- Reclaim 696,265,324 bytes of exact terminal passing pytest scratch after
  XML, absolute containment and recursive no-reparse checks.
- New build metadata: 336,599 bytes; hard-link input view:
  837,562,811 logical bytes; results:
  259,870,199 bytes; combined 1,097,769,609 bytes.
- Zero sensor downloads or payload copies, zero Cargo-cache growth; bounded
  results <=512MiB, combined <=2.5GiB, four workers and >=20GiB free throughout.

The owned evidence lives at `E:/SpatialRustData/maturity-80/room-reference-audit-v1`,
the hard-link view at `E:/SpatialRustData/maturity-80/room-reference-input-v1`
and execution records at `E:/SpatialRustBuild/room-reference-audit-v1`.
All command, log, source/module/input/output identities are retained in the
public receipt. The actual controlled tests and local study are separate from
the repository's cross-platform CI.

## Remaining operational target

Reference results cannot qualify the native solver or replace any condition of
the operational 80% goal. Native room robustness and genuinely unobserved motion,
both raw Linux hardware classes, matching canonical measured clock uncertainty
and physical frame accuracy, and complete canonical mapping/semantic/Viewer/
interchange/release acceptance remain open. This recording is observed
development data; it is not a new held-out success.
