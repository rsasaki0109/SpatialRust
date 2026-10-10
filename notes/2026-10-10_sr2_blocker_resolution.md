# SR2 blocker resolution

## Completed independent data validation

The original Redwood endpoints still fail proxy upstream certificate
verification. Pinned Open3D v0.19.0 source declares an official GitHub release
distribution for augmented ICL-NUIM. That verified HTTPS route successfully
supplied livingroom1 simulated-depth ZIP and reference trajectory, both matching
the publisher descriptors' MD5 values; SHA-256 identities are retained.
No allowlist broadening, proxy bypass, TLS exception or hash replacement was
needed. This resolves access to the needed separate benchmark inputs while
leaving the original server trust failure recorded.

Protocol commit `d7e4c78` fixes four frame pairs, three seeds, registration
controls and 0.05 m / 5° accuracy limits before accuracy inspection. Runner
commit `b2b592a` prepares hash-bound camera-frame inputs and freezes all saved
poses before reading the reference trajectory. Both methods generate 12/12
poses and pass 12/12 fixed accuracy checks. Native translation errors range
0.012995–0.047019 m; Open3D errors range 0.014619–0.042791 m.
All 2,870 trajectory poses, eight backprojections and 24 saved scores are audited
against Open3D and independent arithmetic. See
[protocol, limits and reproduction](../docs/REDWOOD_FRAME_VALIDATION.md) and
[the committed receipt](../docs/receipts/2026-10-10_redwood_frame_validation.json).

The full default-wheel Python suite passes 583 tests with three expected ONNX
skips, and nine CI-observation tests pass. Eighteen new tests verify depth units,
axis signs, truncation, strict LOG identity, relative-transform direction,
non-rigid rejection, input hashes and complete failure denominators. Existing
fragment-reference code and its 20 tests are preserved unchanged.

## Remaining external gates

Exact-head all-page CI observation remains incomplete. The retained observations
are `/workspace/SpatialRust/target/blocker-resolution/pr112-ci-final-review.json`
and `pr113-ci-during-redwood.json`: at 02:50 UTC PR #112 has 75/103 successful main jobs and
28 queued; at 02:46 UTC PR #113 has 56/103 successful main jobs and 47 queued. Both wheel runs
have seven successes and one intentionally skipped publication job. No observed
job failed. This does not establish why allocation is delayed. GitHub Actions
permission inspection was denied to the integration; no credentials or limits
were changed. The unresolved latest-head matrix prevents a safe merge.

The exact canonical Autoware bag is already recovered (713,670,656 bytes,
SHA-256 `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`).
Its official archive contains only the DB3 and metadata, with no calibration
attachment. SQLite has no clock/TF/odom messages. The existing source-bound
evidence remains `registration_ready:false`; measured clock and root-to-front/
rear extrinsic artifacts are missing. Public bucket listing returned S3
`AccessDenied` (HTTP 403); this is evidence that enumeration is unavailable,
not proof that no other object exists. A generic vehicle configuration or a
different capture's transforms cannot satisfy this exact-input calibration
gate. The recovered evidence and four-file manifest remain at
`/workspace/SpatialRust/target/handoff-data/rosbag/calibration-evidence`.

## Maturity

The provisional OSS-relative assessment remains **82%**, target 90%. A separate
synthetic benchmark is now demonstrated, but one scene and nearby frame pairs
do not establish independent real-sensor accuracy, broad robustness or calibrated
full-bag operation. No score increase is inferred from reaching 12/12 on this
limited protocol, a queued CI matrix or an unregistered calibration artifact.
