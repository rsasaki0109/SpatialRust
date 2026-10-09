# Optional correspondence trimming and paired failure study

Rust IcpRegistration::with_trim_fraction and keyword-only trim_fraction on both
Python ICP functions retain floor(fraction * gated pair count) lowest-distance
correspondences for transform estimation. Equal distances follow source order.
The default 1.0 preserves the original update path, does not sort, and does not
allocate a ranking buffer. No public Rust configuration/result/history fields
were added or changed. Invalid fractions or too few retained correspondences
produce errors rather than silently increasing the retained count.

History correspondences still means estimator pairs (now after trimming).
evaluated_correspondences and fitness still evaluate all post-update gated points;
absolute fitness stopping also uses this full gated population. Trimming therefore
does not silently improve reported fitness by shrinking its denominator. A
regression fixture retains exact clean matches, preserves the correct identity
pose, and deliberately has higher full-population fitness than biased ordinary
least squares. Other tests cover deterministic ties, floor retention, defaults,
invalid values, ordinary/trace equality, read-only inputs and concurrent reads.

Validation: 25 CPU registration tests, 290 Python tests with rebuilt release
ONNX-enabled extension, strict registration/extension Clippy, and Python stubtest.
Clippy exemptions are local to the two Python entry points to preserve existing
positional arguments plus explicit keyword options. Remote CI cannot currently
be inspected because GitHub API access is denied by policy.

The study uses 702 runs: five seeds each for clean/noisy volumes, 20% near-region
replacement outliers, 20% random replacement outliers, 60% partial source, plus
one 180-point symmetric ring; prior errors 0/20/40 degrees; gates .3/.6/1.2 m;
retention 1/.8/.5. Native work is single-stage full-resolution ICP, cap100,
translation/rotation stopping 1e-4 m/rad and absolute fitness disabled. Evaluation
uses the entire original source at .05 m. Source has a fixed generating 60-degree
rotation and translation (5,-2,.5); priors have an additional small translation.
Truth labels assess outputs and initial retention only, never native ranking or
pose selection. Recovery means generating-pose error <1 degree and <.01 m.

There are 234 matched environment/seed/prior/gate groups. Retaining .8 gains 49
recoveries and loses 29 versus all pairs; retaining .5 gains 42 and loses 42.
All runs completed without native errors. These are paired controlled-fixture
counts, not an external-library ranking or universal accuracy claim.

At gate .6 m with near outliers and zero angular prior error, all-pair ICP recovers
0/5 versus 5/5 with .8 retention. Initially it removes the 8–19 gated outliers
but also retains only 262–271 of 320 true members. At 40-degree prior error,
both settings recover 0/5 despite .8 retaining very few outliers. On clean data
with 40-degree error and gate .6, all pairs recover 5/5 while .8 recovers 0/5.
Noisy and partial data have similar lost-basin cases. Residual ranking can remove
the leverage needed to correct a poor initial pose. Geometric true membership
also does not certify correspondence to its generating target point.

Ring results remain ambiguous from geometry; generating-pose disagreement is
not observable physical error. All-pair and .8 retention recover its generating
pose for only the zero-angle prior. Trimming is an explicit option, not a new
default or an automatic fallback chosen using ground truth.

The report visualizes paired gained/lost recoveries and first-update retained
true/false members. JSON records full-source support, errors, final pair counts,
fixture/native/source hashes and settings. Source hashes were verified after the
final measurement. CI runs the study and saves JSON/HTML for 14 days; a smaller
paired study in the binding suite protects both improvement and counterexample.

Artifacts:
/workspace/SpatialRust-python-delivery/target/trimmed-icp-study-final/study.json
/workspace/SpatialRust-python-delivery/target/trimmed-icp-study-final/report.html

Maturity remains 55%, provisional. Public real-pair testing of trimming and file
workflow integration are subsequent work; no new real-sensor accuracy or speed
claim is made here.
