# Full-scene Open3D RGB-D comparison

Operational slice 146K. Commit this contract and both
`tum-fr1-{xyz,rpy}-open3d-dense-plan.json` plans before any new comparator
estimation. Native causal controls remain frozen at 798/798 xyz and 722/722
rpy poses. XYZ passes; rpy ATE 0.114793m exceeds the existing 0.10m limit even
though its 1s RPE 0.048145m passes. This study seeks a full external comparison
before attributing the gap to a particular preprocessing difference.

## Data and execution contract

Use installed Open3D 0.19.0 CPU module SHA-256
`ab1b2f4a1126a11886a2a9f3eeab5ceaeac0bab1b53ee319eb363df7ef629d88`.
The existing 151,434,752-byte module stays in place; download or copy no package.
Keep the frozen SpatialRust numerical extension for shared input projection,
grayscale, geometric support and typed IO. Add no Rust, core, GPU or production
dependency, native rebuild, implicit device transfer or solver change.

The comparator adapter owns one float64 proper 4x4 previous source-to-target
increment (128 bytes) and a successful-pair count. First seed is identity; later
seeds are the exact preceding returned increment. Pass an owned seed and owned
float32 grayscale/metric-depth CPU images to the legacy CPU hybrid API. Retain
no image or unbounded pose history. Record the seed, preceding pair index,
proper returned/inverse transforms, package/model identity and diagnostic 6x6
information matrix. Advance state only after successful finite/proper output
and unchanged raw final projective validation. A later generator rejection
permanently loses the chain and makes no further estimator call.

Use 20/10/5 iterations, 0.03m depth difference, 0.1..5m depth and the same
525/525/319.5/239.5 registered-depth intrinsics. Original grayscale and depth
units are unchanged. The pinned implementation internally filters full-resolution
gray/depth with Gaussian3 and normalizes gray means over initial correspondences
to 0.5 before its own pyramid. Sampling, derivatives and stopping differ from
the native bilinear estimator. This compares implementations; it does not claim
equivalence or establish that one preprocessing step caused the observed drift.
See the [pinned source](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/pipelines/odometry/Odometry.cpp)
and [versioned API](https://www.open3d.org/docs/0.19.0/python_api/open3d.pipelines.odometry.compute_rgbd_odometry.html).

## Fixed full-scene gates

Reuse the already checked PNG/index payloads under
`E:/SpatialRustData/maturity-80/tum-fr1-{xyz,rpy}/prepared-causal-dense-v1/`
through checked hard links to new prepared directories. Never fall back to image
copies. Preserve original archives, manifests, all native/GPU binaries, wheels,
failed/successful estimates, metrics and audits. Read no reference pose during
preparation or estimation; freeze both complete timelines before either score.

Process all 798 xyz and 722 rpy rows with no retry, reset, dropped failure,
future-frame selection, scale fitting or reference initializer. Use unchanged
common raw full-resolution final checks: at least 30 correspondences and 0.5
fraction, with independent 0.5 full-cloud support at 0.1m. Keep camera/world
composition, timestamp association and typed IO exactly as the native controls.
Producer validation uses bounded NumPy row tiles; a separate auditor recomputes
full-frame projective counts/residuals and seed/world lineage. These raw metrics
are common admission measurements, not Open3D's internal filtered objective.

Keep 100% pose generation, >=95% reference coverage, first-pose SE3-aligned ATE
RMSE <=0.10m and one-second translation RPE RMSE <=0.05m. Retain three xyz/six
rpy reference gaps and all unmatched RPE pairs. Audit statistics, transformations
and scores at the existing tolerances. An estimator success flag or surviving
prefix does not satisfy these unchanged full-scene gates. Never use the returned
information matrix as sensor calibration or clock uncertainty evidence.

## Resource and scope boundary

The initial implementation commit `b42d755` reached neither odometry nor
reference scoring: the installed legacy binding accepts a default RGBDImage
constructor with separate owned color/depth properties. Both attempted runs
retain their first identity row and TypeError logs under the original v1
directories; they are preparation/API failures, not completed trajectory
measurements. A separate implementation repair uses that exact API and tests
actual float-buffer ownership without fitting. API TypeError/RuntimeError now
becomes a reported lost-chain failure. Preserve the v1 preparation/results and
reuse their unchanged prepared inputs in a separately committed v2 study/output
directory. Keep all plans, seeds, admission thresholds and scoring gates fixed;
the combined v1/v2 new-output budget remains 1GiB.

Write new results only under the E drive, using an
`open3d-dense-v1` suffix and study root
`E:/SpatialRustData/maturity-80/open3d-dense-study-v1/`. Keep at least 20GiB free,
cap combined new study output at 1GiB and each owned command at one hour. Use
at most four CPU workers through scoped thread controls. Freeze package/native,
code, plan, input and output identities. Open3D internal allocator/pyramid memory
is not a measured or equivalent native working-memory bound. No speed claim.

Only exact owned passing pytest scratch can be removed after successful XML,
absolute-parent and full no-reparse checks. Preserve all scientific outcomes and
logs, including failures. Both datasets are observed development data. Canonical
measured clock/physical calibration, both raw Linux hardware classes and full
mapping/semantic/release acceptance remain separate open 80% requirements.

## Observed complete runs (2026-10-20 JST)

The model/plans were fixed at `b4cfad9`; repaired implementation `7b08961`
generated both complete timelines on the existing Windows CPU packages. Both
estimate files were frozen at `2026-10-19T19:50:05.632263+00:00`, before either
new reference score. The [source/storage/measurement receipt](receipts/2026-10-20_open3d_dense_rgbd.json)
binds all original archives, manifests, plans, binaries, commands and audits.

| Implementation / scene | Generated | Reference matched | First-pose ATE RMSE (m) | 1s translation RPE RMSE (m) | Fixed gates |
| --- | ---: | ---: | ---: | ---: | --- |
| Native causal / xyz (preserved control) | 798/798 | 795/798 | 0.089993 | 0.026591 | Pass |
| Open3D CPU / xyz | 798/798 | 795/798 | 0.051174 | 0.023110 | Pass |
| Native causal / rpy (preserved control) | 722/722 | 716/722 | 0.114793 | 0.048145 | ATE fails |
| Open3D CPU / rpy | 722/722 | 716/722 | 0.099684 | 0.048494 | Pass |

These are complete planned trajectories. Three xyz and six rpy publisher gaps
remain explicit; evaluated/planned 1s pairs are 762/768 and 680/692. The external
rpy ATE margin is only 0.000316m below the 0.10m limit, and its RPE is slightly
higher than the native control. No repetition, selection or threshold change was
made after scoring. This gives a complete external control for the same dataflow;
it does not establish held-out robustness or identify one causal solver difference.
Native rpy acceptance remains open. The next numerical investigation can isolate
full-resolution preprocessing while retaining raw final checks and these controls.

Independent full-frame NumPy auditing checks all 797+721 causal/projective pairs:
counts/fractions and prior seeds agree exactly; maximum gray/depth RMSE
differences are 2.78e-17 / 3.47e-18 and world-chain difference is 2.70e-14.
Independent SciPy trajectory auditing agrees within 1.17e-15m / 2.85e-13 degrees.
All original three/six reference gaps and unmatched RPE pairs stay in denominators.

The shared input retains the publisher's recommended registered-PNG approximation:
see [TUM file formats](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).
No extra depth rescale, undistortion or sensor extrinsic was applied. This input
model remains separate from measured canonical clock/physical calibration.

All eight generation/audit/score commands exit successfully. The final Python
suite has 999 passes and three existing optional ONNX skips, including 25 new
comparator cases and installed legacy-image buffer ownership without fitting.
Changed Python files parse as Python 3.8. Native/core/GPU trees and 26 frozen
native/module/library files remain unchanged; 256 prior JSON files and five
wheels retain their original hashes. No native rebuild or package download.

Combined v1/v2 scientific output and new preparation metadata occupy 336,933,306
logical bytes, below 1GiB. Existing PNG/index hard links share 918,149,680 logical
bytes; v2 adds no links or image payload. Only the three verified passing pytest
scratch directories were removed, reclaiming 754,445,485 logical bytes. The
receipt observes 678,105,968,640 bytes free on E after completion. Preserve both
initial API failures and all successful results. This comparison leaves the
remaining canonical, hardware-class and full mapping/semantic/release gates open.

### Checkout identity portability follow-up

Initial Linux CI found that the fixed child plans recorded the pre-existing
Windows parent's CRLF SHA despite the repository's LF checkout rule. Parent
validation now accepts only those identical bytes in uniform LF or CRLF form,
retaining the original declared identity. Mixed endings, stray CR, extra
whitespace, wrong hashes and changed model/acceptance fields reject. The child
plans, all scientific outputs and original generation/auditor bindings remain
fixed at their recorded source versions; this repair changes no fitting or
scoring math and requires no new run.

Ten checkout regressions bring local validation to 1,009 Python passes and three
existing optional ONNX skips, including 35 comparator cases. Their separate
passing scratch cleanup reclaims another 316,349,249 logical bytes, bringing
total passing-scratch reclamation to 1,070,794,734 bytes. Original measurements
and the initial three cleanup records stay intact; the receipt appends separate
source identities, XML, cleanup and scientific-output preservation checks.

### Optional external test in installed-wheel validation

Linux then passed every test but the wheel receipt expected only the existing
three optional ONNX skips. The external image-ownership smoke also skips when
Open3D is absent from that fresh environment. Wheel validation now identifies
only this exact test module/name and its unavailable `open3d.cpu.pybind` import
reason as one additional optional skip. Baseline skip counts stay fixed; wrong
names/modules/reasons, duplicates, absent cases, failures and all other extra
skips reject. Installed package/native bytes, ICP/support and exact PCD checks
still run.

Nine wheel-admission regressions bring full local validation to 1,018 passes and
three existing optional ONNX skips. A separate process deliberately hiding only
Open3D produces 34 passes/one identified external skip and successfully validates
the actual installed frozen wheel and runtime. This simulates module absence;
it is separate from the real Linux CI. Both passing scratch roots were removed
after XML/path/reparse checks, bringing total logical reclamation to
1,427,823,793 bytes. The receipt appends their separate evidence; no scientific
plan, estimate, score or numerical binary changed and no fitting was repeated.

The real ONNX job also has a zero-baseline JUnit admission contract. It now uses
the same exact external classifier with `expected_skips=0`: the single unavailable
external smoke is identified, while any skipped ONNX test still rejects. The
prior real CI ran 1,020 tests successfully with only that external skip. The
updated consumer parses as YAML, accepts the actual isolated absent-module XML,
and rejects the local full XML containing three absent ONNX checks. Python and
numerical source remain those of the 1,018-pass suite; no repeated fit or build.
