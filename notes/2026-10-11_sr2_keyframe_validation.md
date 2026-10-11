# SR2 bounded native keyframe validation — 2026-10-11

The new retained-keyframe adapters do not establish better real motion.
All 12 fixed study/comparison cases fail at least one complete-coverage or
accuracy condition. Keep the maturity at **87%, provisional**; target 90% is
unsupported. The default estimator stays unchanged.

The implementation and full outcomes are in
`/workspace/SpatialRust/docs/TUM_KEYFRAME_VALIDATION.md`; compact source/hash,
failure-range, missing-pair and independent-audit evidence is in
`/workspace/SpatialRust/docs/receipts/2026-10-11_tum_keyframe_validation.json`.
Inclusive RPE pair runs are lossless: both source and target indices increase
by one from their separate first to last endpoints. Decode validation agrees
with every unchanged full local evaluation pair list.

## Fixed chronology

- Static 30-frame keyframes: `aff896602f490eb996436417fbef029d3eec535e`,
  2026-10-11T00:39:58Z, before acquiring the official desk sequence.
- Input-only overlap-margin promotion: `fdc52e6fbd4e5066b80fa552c5ffe0d91236c69d`,
  00:47:18Z, after desk acquisition but before desk preparation/fitting or
  reference scoring. The static failures were preserved.
- Coarse native SVD ICP then native point-to-plane:
  `2248b315674c36dfacf38cca7c70b6b2698071b8`, 01:09:53Z, after all seven
  preceding results were frozen/evaluated. XYZ/rpy/desk are exploratory here.
  Fresh desk2 was fixed before requesting it at 01:10:22Z, completing at
  01:12:39Z. All five new estimates froze before desk2 reference interpretation.

The old desk acquisition receipt names its completion timestamp
`requested_at_utc`. It remains unchanged; the new metadata receipt identifies
the correction explicitly and does not invent a request timestamp. Desk2
separately records its actual request and completion times. Both archives
arrive through the inherited proxy with TLS verification and HTTP 200;
SHA-256 records downloaded bytes, not a publisher-supplied archive checksum.

## What was actually verified

Both local Python endpoints pass 731 tests with three expected ONNX skips
(734 total), including actual native multi-promotion motion recovery,
unobservable geometry, permanent lost-chain handling, immutable protocol
conditions and exact helper/native bindings. Real xyz/rpy/desk/desk2 outcomes
are frozen and independently recomputed with SciPy at the original numerical
tolerances. All 12 score audits pass; this verifies calculation, not motion.

Fresh desk2 retains all 639 depth / 640 RGB image indices over 21.288169 s.
Each estimator processes 639 poses and 146,714,333 points, with 12 exact typed
IO audits and maximum independent optical-projection discrepancy 3.729e-7 m.
The reference lacks 15 matches; 624 poses and 591/609 one-second RPE pairs
remain evaluable. Hybrid ATE/RPE is 1.045931/0.179715 m, versus the unchanged
plane baseline's 0.523587/0.102466 m. Both fail their 0.10/0.05 m limits.
Hybrid desk stops at 265/595, with support 0.486208 below the fixed 0.5 limit.

PNG/SVG/HTML comparisons are local under
`/workspace/SpatialRust/target/tum-keyframe-reference/*-comparison-report-v4/`.
The first exports are preserved; a visually observed clipped rpy legend is
fixed with explicit canvas margins and stable 3D title sizing. Plot hashes bind the exact
estimates/scores; export framing does not change accuracy or coverage.

The source-bound measured clock/front/rear mounting calibration for the exact
canonical Autoware recording remains unavailable. The user's answer was
unknown; do not repeat that question or infer measured calibration from ICP.
Approximate TUM intrinsics and L515 factory calibration cannot certify that
recording. Accurate broader motion remains a separate open gate.

No sensor archives, depth/RGB images, point-cloud dumps or native binaries
are committed. TUM data attribution: Sturm et al., IROS 2012, CC BY 4.0.
The separate GPU stage-timing changes from main PR #122 are retained.

## Published environment restoration and Windows diagnosis

Runtime revision 16 restores all 12 frozen cases, all four final plot receipts
and historical helper/native hashes. Its 78 relevant tests pass and a fresh
desk2 hybrid independent audit re-executes for 639 planned / 624 scored poses.
These restore existing evidence and do not increase maturity.

The first PR #126 head `69ecb43` has 88 successful main jobs / 15 queued at the
diagnosis snapshot. Wheels have six successes, one Windows failure and the
intentionally skipped publication job. All three Windows test failures are
slash-only adapter/policy presence assertions, confirmed through the normal
GitHub check-run annotations API. The log-download redirect is HTTP 403;
annotations provide the exact failure evidence without bypassing network
policy or adding domains. Repair `4cbf8d8` checks resolved absolute keys and
source hashes directly, preserving every physical/numerical condition and all
frozen outputs. Verify the repaired head's own complete matrices before merge.

Subsequent main PR #124 GPU bounded retention and PR #125 parallel contrast
changes are preserved. CHANGELOG conflicts retain both independent additions.

The premerge main refresh also finds PR #127 bounded parallel point queries
and PR #128 resident GPU image histograms. Their source, documentation and CI
profiles are preserved; only the overlapping CHANGELOG insertion needs manual
resolution, retaining both entries. These add four feature-matrix jobs, so the
integrated head requires **107 main jobs**, seven required wheel jobs and only
the intentionally skipped PyPI publication. Older 103-job runs cannot certify
this integrated head. Frozen study inputs, helpers, estimates and scores remain
unchanged; the historical native SDK stays preserved for exact replay.

## COPC hosted failure and subsequent main integration

The integrated `83e0869` head's complete explicit-attempt snapshot at
`/workspace/SpatialRust/target/goal-continuation/pr126-ci-main127128-watch-20.json`
records 86 main successes, one COPC HTTP failure and 20 queued jobs; all seven
required wheel jobs succeed, with only PyPI publication intentionally skipped.
The failed job is `114374492305`, during `Build and test`, exit 101. Toolchain
setup and cache restore succeeded. GitHub annotations expose no underlying
diagnostic. Log retrieval redirects to `productionresultssa11.blob.core.windows.net`,
which the session proxy rejects with HTTP 403; TLS and policy are preserved.

The same COPC profile on the integrated local source passes 34 tests, including
both local HTTP fixtures, with no failures or ignores. The restored local lock
first needs updating for newer main manifests: the initial `--locked` command
does not execute tests. Its exact bytes are backed up, and offline resolution
then executes the profile successfully. This local lock prerequisite does not
explain the hosted failure, whose command does not pass `--locked`. Neither a
fixed defect nor a transient failure is inferred from the local pass. Detailed
receipts are under `/workspace/SpatialRust/target/goal-continuation/`:
`pr126-copc-failure-annotations.json`, `pr126-copc-log-access.json`,
`pr126-copc-integrated-local-offline.log`, `pr126-copc-local-diagnosis.json`.

The fresh main revision also contains PR #129 resident GPU histogram CDF.
All its source, API notes and CI commands are retained; both CHANGELOG entries
survive the insertion conflict. Required counts remain 107 main and seven wheel
jobs, but the resulting head needs its own complete matrices. Its 78 relevant
TUM tests pass; generation/evaluation helpers, Python tests and study metadata
remain unchanged, as does the SHA-verified historical native SDK. Hosted COPC
diagnosis remains incomplete while detailed logs are inaccessible. Do not merge
using the failed/superseded head or call the real-motion/calibration gates closed.

## Published log access and complete-header fixture repair

The newly published environment includes the exact log destination. A normal
verified-TLS request now downloads the 50,062-byte failed-job log without bypassing
the proxy. The log identifies `local_http_head_has_no_body`, at the client's
`read_to_end`, with OS error 104 / `ConnectionReset`. The fixture had processed
only one TCP read: it could respond and close while later request-header bytes
were still unread. Its earlier no-body HEAD repair did not address fragmentation.

Two real TCP regressions split the final HEAD delimiter and the Range endpoint.
Both fail before the repair because the fixture responds before complete headers.
The server now reads through CRLF-CRLF within the existing 4 KiB buffer, retries
interrupted reads, and bounds incomplete requests with a five-second read timeout.
After repair, the COPC profile passes 36 tests with no failures or ignores,
including strict HEAD no-body and exact four-byte partial-content assertions.
The exact originally failing HEAD assertion also passes 100 separate executions.
The production COPC reader and sensor/trajectory algorithms are unchanged.

The restored runtime verifies all 12 estimate/score/audit/plan records, historical
Git helper blobs and the native SDK, all four final plot sets, and all four source
archives (1,551,929,060 bytes). Its 78 relevant TUM tests pass. Local receipts are
under `/workspace/SpatialRust/target/goal-continuation/`:
`pr126-copc-failed-job-published-runtime.log`,
`pr126-copc-published-runtime-log-access.json`,
`pr126-fragmented-headers-before-fix.log`,
`pr126-fragmented-headers-after-fix-profile.log`,
`pr126-head-repeat-after-fix.json`, `pr126-main135-tum-junit.xml`, and
`published-runtime-copc-resume-frozen-verification.json`.

The previous `055c70b` head completes its own 107 main and seven required wheel
jobs successfully; only PyPI publication is intentionally skipped. That result
does not certify this fixture repair or subsequent main changes. Main PRs #130
through #135 are also preserved, including their GPU shaders, runtime ownership,
benchmarks and CI commands; required counts remain 107 main and seven wheel jobs.
Only CHANGELOG insertion conflicts need resolution, retaining both additions.
Require the resulting head's own gates and a fresh identical candidate merge tree.
No real estimate, score or physical limit changes. All 12 cases still fail at
least one fixed condition; source-bound canonical calibration is unavailable.
The maturity remains **87%, provisional**, and 90% remains unsupported.
