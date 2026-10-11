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
`/workspace/SpatialRust/target/tum-keyframe-reference/*-comparison-report-v2/`.
The first exports are preserved; a visually observed clipped rpy legend is
fixed by including outer artists in saved bounds. Plot hashes bind the exact
estimates/scores; export framing does not change accuracy or coverage.

The source-bound measured clock/front/rear mounting calibration for the exact
canonical Autoware recording remains unavailable. The user's answer was
unknown; do not repeat that question or infer measured calibration from ICP.
Approximate TUM intrinsics and L515 factory calibration cannot certify that
recording. Accurate broader motion remains a separate open gate.

No sensor archives, depth/RGB images, point-cloud dumps or native binaries
are committed. TUM data attribution: Sturm et al., IROS 2012, CC BY 4.0.
The separate GPU stage-timing changes from main PR #122 are retained.
