# Maturity assessment

Current assessment (2026-10-10): **82%, provisional**. This is a subjective
engineering progress estimate, not measured feature parity with PCL, OpenCV or
Open3D, not the fraction of their functionality implemented, and not a claim
of equivalent production readiness. The long-term target remains 90%.

Open3D's official verified-HTTPS distribution now supplies a separate noisy
synthetic Redwood/augmented ICL-NUIM dataset, despite the original Redwood
service's unresolved certificate error. A precommitted four-pair protocol
runs 24 registrations with no reference initialization; both SpatialRust and
Open3D pass 12/12 fixed accuracy checks. All inputs, frozen estimates and scoring
are hash-bound; independent backprojection and pose-score audits agree. The
default Python suite passes 583 tests with three expected ONNX skips. This is
one synthetic scene with nearby frames, not real-sensor calibration or broad
library parity. Complete hosted CI and source-bound operational calibration
remain open, so maturity stays **82%**. See
[the fixed protocol and evidence](REDWOOD_FRAME_VALIDATION.md).

The earlier SR2 handoff restored a clean Python comparison environment and the exact
canonical rosbag bytes in a new task. All 565 applicable Python tests pass,
with three expected ONNX skips; the same public demo comparison replays without
adding an independent dataset. Exact-head CI observation covers all 103 main
jobs (24 successful, 79 queued) and all eight wheel jobs (seven successful,
publication intentionally skipped) on `ce1fcd8`. The macOS streaming regression
is repaired, but the incomplete matrix prevents merge. The recovered bag still
lacks measured clock and front/rear frame evidence; its registration gate stays
blocked. Original Redwood HTTPS failed upstream certificate verification. These outcomes
leave the assessment at **82%**; see
[the handoff evidence](../notes/2026-10-10_sr2_handoff_validation.md).

Environment/data recovery leaves the estimate at **82%**. Rust 1.99 is restored
from the existing local installation; network-authorized GitHub downloads work.
A fresh frozen default wheel passes 565 tests with three expected ONNX skips,
exact installed-byte/runtime checks, and six actual native/Open3D registrations
on a freshly downloaded hash-bound publisher demo. This restores prior comparison
data, not a second independent benchmark. Operational source-bound clock and
front/rear LiDAR calibration remain unavailable. The final prior CI head verifies
Windows/macOS/Linux wheel runtimes and strict browser output, but the full matrix
has a macOS HTTP fixture failure. Its HEAD-body bug is now reproduced and repaired
locally; hosted verification of the repaired head remains necessary. See
`notes/2026-10-10_environment_data_recovery.md`.

The 82% estimate adds a precommitted fitting-scene holdout: nine pairs in three
new scenes, 54 baseline registrations plus 54 fixed-prior replays. The previously
chosen 0.8 trim improves 17/27 to 22/27 with five gains on two pairs and zero
correctness losses; all untrimmed controls exactly reproduce original poses.
The frozen 0.2 m guard provides no correctness gain, and pooled proximity
selection loses one available correct candidate. These negative findings limit
policy claims rather than being hidden by aggregate rankings. Input/helper/native
bindings, independent saved-pose score recomputation, checkpoint resume and
common-axis geometric visualization make the evidence reviewable. This remains
one benchmark with selected positive pairs and repeated seeds, not general
library parity or a validated universal default. All 561 ONNX-enabled Python
tests pass; the default wheel passes 558 tests with three expected ONNX skips
on both CPython 3.8 and 3.14. See
`/workspace/SpatialRust-python-delivery/docs/3DMATCH_SCENE_HOLDOUT.md`.

A previous Linux runtime checkpoint runs the entire current suite on the default
abi3 wheel at both supported endpoints: CPython 3.8 and 3.14 each pass 537 tests
with three expected ONNX skips; the ONNX-enabled 3.12 suite passes all 540.
This strengthens local regression coverage but leaves maturity at 80%. At
2026-10-09 22:48 UTC, the seven wheel jobs and first 30 returned CI jobs lacked assigned
runners; this first-page observation did not cover the complete CI job set. Main CI now cancels superseded runs by workflow/ref, preserving tags;
this does not solve external runner allocation. Reaching 90% still requires
successful external platform gates, independent accuracy validation and broader
operational evidence; none is inferred from a configured job or a queue status.

The fixed other-pair trim validation adds 66 actual refinements on the other
11 official pairs, excluding the exploratory hotel pair. With the same 33
generated priors, retaining 0.8 improves publisher correctness from 23/33 to
24/33: one gain, zero losses, eight failures in both settings. Some scores worsen
without crossing the correctness threshold. This is conditional evidence on
previously examined pairs, not an independent blind benchmark or a reason to
change the default. Hash-bound per-case checkpoints resume the complete real
study without fitting again; standalone PNG/SVG and exact tables expose both
improvements and regressions. Local verification is 515 ONNX-enabled Python
tests and 11 focused batch/plot tests on the Python 3.8 default wheel. The
assessment remains 80%; broader initialization and platform gates remain open.

Reference-free selection over the 72 saved poses succeeds on 9/12 pairs with
three native seeds, 10/12 with three Open3D seeds and 11/12 with all six poses.
Forward and balanced proximity rules reach the same correctness counts. All
selections precede publisher evaluation; post-fit labels are excluded from the
ranking function. This identifies complementary global candidates and the one
pair with no correct final candidate, rather than proving an independent
production selector. The exploratory evidence leaves maturity at 80%.
Local verification is 526 ONNX-enabled Python tests and 11 additional selection
checks on the default CPython 3.8 wheel. Broader platform and dataset gates remain.

Reference-free stage protection exposes a limit rather than closing an accuracy
gate: on 36 native outputs, forward and bidirectional proximity still lose the
three good hotel initializations. A fixed 0.2 m source-centroid motion guard
saves those three but suppresses two legitimate recoveries (24/36 versus 23/36).
Smaller limits score 22/36. All alternatives are exploratory on previously
examined data, with no default change. Local verification is 540 ONNX-enabled
Python tests and 14 stage/motion tests on the default Python 3.8 wheel. Maturity
remains 80%; a validated drift guard and independent datasets are still needed.

The 80% estimate adds hash-bound post-fit stage attribution: among 36 native
outputs, refinement retains 18 correct initial poses, recovers 5, loses 3 and
leaves 10 incorrect. All losses occur on one hotel pair, separating refinement
drift from initializer failures without claiming a physical cause. Reference
binding is also enforced before fixed-initialization replay. All 494 Python tests
pass with the ONNX-enabled extension. The same default abi3 wheel passes 491
tests with 3 expected ONNX skips on CPython 3.8/NumPy 1.24 and CPython 3.14/NumPy
2.5; archive/native/type bytes and runtime checks match in both isolated environments.
These are Linux x86_64 checks. Remote jobs remain queued without assigned runners;
cross-platform/GPU and broader accuracy/performance evidence are still open.

The subsequent exploratory hotel replay holds all three generated initializations
fixed. All-pair ICP reproduces the three original failing poses exactly; retaining
0.8, 0.6 or 0.4 recovers 3/3 under the publisher criterion in each setting.
Translation improves while median rotation error increases. The hash-bound
post-fit evaluator rejects changed priors, gates, iteration caps, stopping rules,
trim settings and missing/duplicated rows. All 502 ONNX-enabled Python tests pass.
This selected failure case supports a retained-residual intervention, not a
default trim change or general accuracy claim, so the assessment stays at 80%.
The frozen source archive additionally rebuilds with `--locked` under Python
3.14 in a separate Cargo target and passes 499 tests plus three expected ONNX
skips as a newly installed wheel, with exact archive/native/type/runtime checks.
Remote runtime jobs still have no assigned runners; this local evidence does
not close the Windows/macOS/ARM or GPU gates.

Locale hardening keeps the estimate at 80%: two reference/report CLIs had confirmed
ASCII-locale failures on Japanese provenance and mathematical symbols. All study
tools and alignment examples now use explicit UTF-8 text IO. The two failing CLI
regressions pass, and all 504 ONNX-enabled Python tests pass. Cross-platform runtime
CI still needs assigned runners; fixing a local encoding defect does not certify
those platform gates or justify claiming 90%.

The 78% estimate adds actual official 3DMatch fragments across three scenes,
12 pairs fixed before fitting, 72 native/Open3D registrations, and publisher
information-score evaluation. Native succeeds on 23/36 planned rows and Open3D
on 27/36; these selected positive pairs are not full benchmark recall/precision.
Shared failures and complementary outputs are recorded without an oracle selector
claim. Complete per-case receipts can be validated and aggregated after an
interruption without rerunning fitting. The separately distributed Python build
now tracks its frozen dependency graph and verifies identical sdist lock bytes;
Windows/macOS installed-wheel runtime gates are configured. Remote runtime
success, additional datasets, throughput/memory and operational evidence still
require verification before 90%. See `docs/3DMATCH_LIVE_COMPARISON.md`.

The 75% estimate adds strict Redwood/3DMatch reference binding with verified
source/target convention, pair-ID matching, original file hashes, and saved-log
evaluation using the publisher information metric. 2,563 GT/information records
across 12 official metadata scenes validate. Historical log analysis demonstrates
complementary correct-pair sets without claiming a deployable oracle selector.
470 Python tests pass; 20 new checks pass with the isolated default wheel too.
At that checkpoint the actual fragment download host, 3dvision.princeton.edu,
was blocked; the later 78% checkpoint successfully acquired and evaluated three
official fragment scenes. The 90% runtime/data/operational evidence gates remain open.

The 74% estimate repairs an observed Rust 1.75 compatibility failure by pinning
the compatible thiserror release and adding a fresh-resolution CI gate. Rust
1.75 checks the default meta crate and geometry surface, and passes 36 core/math
all-feature tests. Current-stable tests pass 174 core/math/full-vision and 29 CPU
registration tests; the rebuilt extension passes 450 Python tests, 20 optional
comparison tests, strict Clippy and stubtest. Its new default wheel passes 447
tests with 3 expected ONNX skips and byte/runtime verification. Full features
still need newer Rust, and the verified scope is explicit. The 90% evidence gate
is not met: primary 3DMatch downloads are network-blocked, remote/platform/GPU
runtime and operational stability evidence remain incomplete.

The 72% estimate adds a public supplied-reference pair, safe NumPy/PCD preparation,
explicitly recorded near-rotation correction, ten controlled native/Open3D
global registrations and twenty fixed-initialization trim replays. The study
exposes shared drift on partial clouds and explains why proximity-based support
can favor a worse pose; trimming improves median reference error but can worsen
individual initializations. 450 Python tests and 20 optional comparison tests
pass. The reference's original sensor lineage remains unverified, and a single
public demonstration is not multiple-dataset physical ground-truth validation.

The subsequent reference-pose evaluation slice leaves this assessment at 70%:
all four file alignment workflows bind reports to source/target byte hashes,
and an independent post-fit evaluator records declared reference provenance,
rotation/translation errors and an HTML comparison. 440 ONNX-enabled Python
tests pass. The reference fixture is synthetic; no public sensor ground truth
has yet been independently verified. Evaluation infrastructure alone does not
close that evidence gap or justify a higher score.

The 70% estimate closes a failure discovered by the external comparison:
planar/near-planar PnP now uses scale-normalized geometry detection and homography
initialization, then refines against the actual 3D points. On unchanged inputs
and controls, plain recovery improves from 15/60 to 28/60 and robust recovery
from 20/60 to 39/60, with zero recovery regressions and exactly unchanged OpenCV
rows. Archived baseline hashes and calculation fingerprints validate the pairing.
Verification passes 421 ONNX-enabled Python tests, 142 full-feature Rust vision
tests, 18 optional comparison/report tests, strict extension Clippy and stubtest.
A fresh default wheel passes 418 tests with exactly 3 expected ONNX skips;
installed native/type bytes and ICP/support/PCD checks match the wheel. Noisy
planar failures, collinear ambiguity, real ground-truth datasets and platform
coverage remain gaps. Reaching this checkpoint does not establish library parity.

The 68% estimate adds owned, validated, GIL-releasing Python robust PnP and a
240-run OpenCV 4.12.0 failure comparison. All 397 Python tests, 11 Rust geometry
tests, 2 optional comparison tests, strict extension Clippy and stubtest pass.
With 30% wrong volumetric correspondences, native plain fitting recovers 0/5
and native RANSAC 5/5. Planar/near-planar noisy cases still expose a DLT
initialization gap versus OpenCV, and collinear poses are ambiguous. This is
verified robust functionality and external failure evidence, not parity.

The 66% estimate adds geometry-conditioned failure diagnostics and visualization:
point-to-point and point-to-plane information spectra, weak directions, and
origin/scale normalization. Analytical line/plane/coincident ranks and independent
finite-difference Jacobians verify the math; 380 Python tests pass. Optional
global reports preserve poses/support while adding full-cloud spectra. Public
source/target and target-normal analysis run successfully. Local fixed-pair
information does not establish global identifiability, actual retained-pair
observability, noise covariance or confidence; those limits are explicit.

The 64% estimate adds PCL 1.15.0 to the controlled external ICP study: 702
registrations, 234 paired conditions, all three libraries recover 138 poses
with no classification disagreement. Native comparator and runner hashes,
actual PCL update counts and common full-source evaluation are recorded; 5
optional comparison tests pass. Inclusive zero-delta stopping and numerical
precision differ, and this remains synthetic local ICP rather than real-sensor
or global-registration parity.

The 63% estimate adds isolated wheel and source-distribution runtime validation
and fixes a confirmed release defect: source-rebuilt wheels omitted the type
stub and PEP 561 marker. Both normal and repaired source-rebuilt wheels pass
356 tests with exactly 3 expected ONNX skips; native/type archive bytes match
installed imports, and conversion/ICP/support/PCD checks pass. CI now repeats
the two release paths before publishing. Local evidence is Linux x86_64,
CPython 3.12, NumPy 2.2.6; remote CI and other platform runtimes remain unverified.

The 61% estimate adds a complete no-caller-pose FPFH/RANSAC → multiscale ICP
file workflow: bounded coarse descriptor matching, deterministic multi-seed
generation, failed-hypothesis isolation, optional minimum proximity support,
full-source candidate selection and visualization of initialization provenance.
All 359 Python tests pass. It recovers a large known synthetic pose without
truth entering initialization, and runs on the public cloud_bin_0/1 pair while
preserving 198,835 points, field schema and exact saved XYZ. Real-pair ground
truth remains unavailable; successful output and support do not certify accuracy.

The 59% estimate adds validated native FPFH/RANSAC configuration, finite/unit
input checks, rejection of overflowing geometry, a corrected zero RNG state,
Python reproducible seeds, bounded normal-estimation scratch, and GIL release.
Validation includes 29 CPU-registration tests, 334 Python tests, strict Clippy
and stubtest. Global hypothesis quality and the separate keypoint Python path
remain limitations; boundary correctness is not evidence of better accuracy.

The 58% assessment adds a direct controlled Open3D 0.19.0 comparison: 468 runs,
234 paired conditions, shared original-source evaluation and source/native
fingerprints, with 3 optional comparison tests passing. All paired recovery
decisions matched (138 recovered per library), exposing shared gate/outlier/
initialization tradeoffs. This establishes an external comparison method and
failure evidence, not superiority or broad real-sensor parity.

The explicit multiscale file workflow adds independently configurable stage
resolution, gates, trimming and stopping, with target-frame composition,
original-source attribute preservation and per-stage visualization. All 314
Python tests pass. A three-stage run on the public cloud_bin_0/1 pair preserves
198,835 source points and exact XYZ after PCD roundtrip. This supports the 57%
assessment; one public pair without verified ground truth does not establish
accuracy superiority, external-library parity or broad performance gains.

Earlier 2026-10-10 update at 55%: optional ICP trimming and a 702-run
paired synthetic study are now available, with both improvements and regressions
documented. Local verification is 290 Python tests and 25 CPU-registration tests,
plus strict Clippy and stubtest. This native addition alone did not establish
greater overall deployment readiness.

The subsequent file-workflow integration and controlled public-pair validation
support the current 56% estimate: both CLIs propagate trimming, record/display
settings, validate before IO, and preserve full-source output/support. Public
candidate runs match independent runs and exact XYZ roundtrips. Comparison checks
input, prior, parameter, native-extension and pipeline-source fingerprints. All
298 Python tests pass. On this one public pair, retaining 80% slightly worsens
proximity metrics, so it remains opt-in; no true-pose improvement is claimed.

The recent assessment moved from 50% to 55% after these completed and merged
capabilities, rather than changing the score merely to meet a requested target:

| Capability | Verified evidence |
| --- | --- |
| Inspectable ICP behavior | Owned optional per-update history and explicit stop reasons, sharing ordinary updates; NumPy distance checks and trace/plain equality |
| Usable convergence visualization | File and candidate CLI integration, standalone SVG and exact measurement tables, old-report compatibility and malformed-report rejection |
| Controllable stopping and safer boundaries | Separate translation/angular/absolute-MSE thresholds, explicit recorded settings, Rust/Python invalid inputs and transform overflow rejected |
| Scale-independent rigid estimation | Proper quaternion fit with f64 centroids and normalized eigensystem; scale/plane/large-origin regressions; 120 independent NumPy SVD comparisons |
| Reproducible failure assessment | 567 controlled runs over scale, noise, replacements, partial clouds and symmetry; saved update histories, source/native hashes, CI smoke tests and full-study artifacts |

Local verification for this milestone: 280 Python tests with the rebuilt
ONNX-enabled release extension, 23 registration tests with all CPU registration
features, strict registration and extension Clippy, and Python stubtest. Public
cloud_bin_0/1 optimized candidate runs match independent runs and full aligned
XYZ roundtrips exactly. There is no independently verified real-pair ground truth.
Remote CI cannot currently be inspected because GitHub API access is denied;
configured CI steps and successful local checks do not establish remote success.

Important remaining gaps toward 90%:

- Direct, controlled PCL/Open3D/OpenCV comparisons on multiple public datasets
  with verified reference poses, failure regimes, memory and throughput evidence.
- Stronger treatment of outliers, correspondence ambiguity, degenerate geometry
  and global initialization. Stopping thresholds do not solve these problems.
- Broader confirmed platform, release, MSRV and GPU/backend conformance. Existing
  workflows and features are not evidence that every deployment works.
- Continued API compatibility, packaging, documentation and operational validation
  as features evolve. f32 XYZ precision still limits large-origin sensor geometry.

See the dated notes for limits of individual experiments. Reported geometric
support is distance-gated proximity, not measured physical overlap. A met
convergence criterion does not establish the correct pose.

Remote evidence update (2026-10-10): main `767f128` has successful macOS vision,
visual, streaming and installed-wheel runtime gates, plus Linux Python 3.12
bindings. Full pagination observes 103 CI jobs; earlier blanket queue claims
from the first 30 jobs are corrected. Windows wheel construction succeeds but
runtime validation exposes Linux-only native fingerprint discovery in two study
CLIs. The reader is repaired and locally regression-tested, with remote Windows
verification still required. The assessment stays at 82% until that repair is
confirmed and broader dataset/operational gates are verified. The next synthetic
Redwood/ICL-NUIM dataset is externally blocked by its denied official domain;
the additive network draft is saved but not applied or published.

Follow-up on main `a2fc890` (2026-10-10): installed default wheels pass actual
runtime validation on Linux Python 3.8 and 3.14 and macOS Python 3.12. Windows
runtime still fails after the fingerprint repair; public annotations do not yet
identify the remaining failing test, and the job-log host is proxy-denied even
after an approved retry. The wheel workflow has eight observed jobs (six success,
one Windows failure, one skipped publication), including the conditional publish
job. CI has 103 jobs and includes a failed Web/WASM browser gate; it is not green.
JUnit failures now become check annotations so the next run can expose test names
and tracebacks without accessing the denied log host. Focused diagnostic and
fingerprint tests pass locally (7 tests). These diagnostics add observability;
they do not establish that Windows is repaired. Maturity remains 82%.

Runtime repair follow-up (2026-10-10): logs from main `b1a9acb` identify the
remaining Windows failure as an implicit cp1252 read of UTF-8 alignment HTML
(564 passes, one failure, three expected ONNX skips). The report annotation
CLI also fails when printing a Unicode arrow through redirected cp1252 stdout.
The Web/WASM gate stops before browser execution because the compiled schema
is wasm-bindgen 0.2.129 while the installed CLI is 0.2.126. Tests now read
alignment artifacts explicitly as UTF-8, annotations emit UTF-8, and Web CI
selects its CLI from the generated workspace lockfile, rejecting missing or
ambiguous versions. Two stdlib regression checks pass locally, and a real
native alignment CLI reproduces the cp1252 mismatch and passes the UTF-8 read.
The local native wheel is a previous frozen Linux build; this check does not
verify the latest Windows wheel. Remote Windows and Web/WASM results remain
required before counting either gate as repaired. Maturity remains **82%**.

Future progress reports include this provisional engineering percentage relative
to PCL, Open3D, and OpenCV evidence at the end, together with the main remaining
gaps. It is not a measured percentage of those projects' functionality.

Verified PR #112 runtime checkpoint, commit `21333bc` (2026-10-10): the
Windows CPython 3.12 installed-wheel gate passes 565 tests with three expected
ONNX skips; exact installed native bytes and the wheel runtime receipt match.
Both Linux endpoint gates, macOS, x86_64/aarch64 wheel builds, and the frozen
sdist rebuild/runtime gate succeed. Publication is intentionally skipped on a
PR. Web/WASM installs the resolved 0.2.129 CLI, passes the two regressions,
generates bindings, and completes real Chrome smoke with the rendered status
element showing PASS. Evidence: Python wheels run `38009690111`, Windows job
`114086565729`, and Web job `114086565967` in CI run `38009690109`.

The browser gate previously searched the entire DOM for a PASS string also
present in its script source. Its follow-up assertion requires the exact
rendered status element; failed and running fixtures containing the script's
PASS literal are rejected, while a successful status is accepted. That stricter
assertion still requires CI on the new PR head. The full 103-job CI run was not
yet complete at this checkpoint, so no aggregate green claim is made. Broader
dataset, performance, and operational evidence remains open; the provisional
assessment stays at **82%**, rather than treating two repaired gates as 90%.

