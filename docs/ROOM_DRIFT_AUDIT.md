# Complete frozen room motion audit

Slice 146U compares the complete frozen rounded-sobel and reference-precision
room trajectories using their original reference, first-pose gauges, raw/causal
audits and fixed 1s pair denominator. It identifies where error is concentrated
without estimating another trajectory or changing acceptance.

The [complete prospective protocol](../benchmarks/room-drift-audit-plan.json)
was committed as `34ba6abbf37305571daade8897ed85f7c56e7ea4` before implementation.
Initial implementation is `b7c68c3b41499fe5693037016003feca692ecf9d`; final analysis
and strict entry-point validation are `2c47a138cd2747dd365ebc508c4ddb6c08c22956`.
The [public receipt](receipts/2026-10-20_room_drift_audit.json) binds every input,
output, original native module, source, test and resource observation.

## Coverage and unchanged acceptance

Each candidate retains 1,360 generated/planned rows, 1,359 reference matches,
1,357 reference-evaluable adjacent pairs and all 1,330 planned 1s pairs, of which
1,328 have reference endpoints. The one missing reference sample and all affected
pairs remain explicit. No reset, trimming, new gauge, rescale or refit occurs.

Independent Decimal-time/SciPy SLERP and rotation-magnitude calculations check
all original translation/rotation rows and 1s pairs before descriptive output.
Causal seeds, inverse directions, world composition and every recorded raw
count/statistic are checked against the original immutable audits. Source images
and archives are not opened. Original score/coverage limits remain 100% generated,
>=95% reference matched, ATE RMSE <=0.10m and 1s translation RPE RMSE <=0.05m.

| Frozen model | ATE RMSE (m) | Orientation RMS (degrees) | Adjacent translation RMS (m) | 1s translation RPE RMS (m) | Original acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| rounded-sobel | 0.379777450 | 12.826148 | 0.007771012 | 0.058405114 | Fail |
| reference-precision | 0.378652661 | 12.803079 | 0.007771551 | 0.058375326 | Fail |

Orientation and adjacent-step summaries are descriptive; they introduce no new
acceptance threshold. Both existing models still fail room. Their positions
differ by only 0.002007649782161814m RMS (maximum 0.004472554293637627m) and
orientations by 0.06021369415405687 degrees RMS under the original gauges.
The precision/retraction change therefore leaves a very similar observed room
error trajectory; this does not identify an error cause.

## Eight prospectively fixed chronological bins

Assign frame index `i` to `floor(8*i/1360)`, adjacent pairs by target row and
1s pairs by source row. All eight bins are shown. Binned squared errors and
counts reconstruct every original denominator and full RMSE. Each-bin values
are descriptive, without separate pass/fail decisions.

| Frame indices | Reference matches | ATE RMS (m) | Orientation RMS (degrees) | 1s translation RMS (m) | Share of full 1s squared error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–169 | 170/170 | 0.087732 | 3.3642 | 0.056162 | 11.85% |
| 170–339 | 170/170 | 0.350402 | 7.4316 | 0.099784 | 37.18% |
| 340–509 | 169/170 | 0.417956 | 7.0269 | 0.043376 | 7.03% |
| 510–679 | 170/170 | 0.424921 | 16.7465 | 0.057705 | 12.51% |
| 680–849 | 170/170 | 0.314232 | 18.5997 | 0.055896 | 11.74% |
| 850–1019 | 170/170 | 0.416176 | 12.7486 | 0.049006 | 9.02% |
| 1020–1189 | 170/170 | 0.414801 | 13.1644 | 0.043263 | 7.03% |
| 1190–1359 | 170/170 | 0.466346 | 15.2489 | 0.034313 | 3.64% |

The second fixed bin (170–339) contributes 37.18% of 1s translation squared
error. This concentration is retained inside the complete original score; no
interval is removed or selected for fitting. Later low relative error can
coexist with substantial accumulated position/orientation error.

For the new model, adjacent translation error has descriptive Pearson values
of 0.658955 with raw photometric RMSE and -0.516518 with correspondence fraction.
The 1s translation values with complete-window mean photometric/depth/support
RMSE are 0.090956/0.562222/0.682475. All five covariates, both error components
and both time scales remain in the receipt. Overlapping/time-correlated windows
are observations, not independent trials; no significance or causal claim is
made. Hidden runtime H/J remain unobserved. A separately prospective robust-loss
comparison is a next investigation, with the original full gates preserved.

## Plot, tests and storage

The four-panel full-timeline SVG and PNG show original-gauge position/orientation,
adjacent position error and original 1s position error for both candidates.
Every planned sample is supplied; missing references create NaN gaps and path
simplification is disabled. The generated 226,821-byte PNG was visually checked.

| Artifact | Absolute path |
| --- | --- |
| Complete rows, pairs, bins and associations | `E:/SpatialRustData/maturity-80/room-drift-audit-v1/audit/audit.json` |
| Vector figure | `E:/SpatialRustData/maturity-80/room-drift-audit-v1/audit/drift.svg` |
| PNG preview | `E:/SpatialRustData/maturity-80/room-drift-audit-v1/audit/preview.png` |

Forty initial math/IO tests and 41 final tests pass, including the exact fixed
plan entry guard, closed-form position/rotation drift, 180-degree/large-epoch
cases, missing/failing coverage, malformed input, reset/direction/seed changes
and full plot counts. Existing SDK tests pass 1,310 cases with three existing
skips. Optional validation tests live outside the SDK wheel suite. A mandatory
`Frozen room drift audit` CI job uses the tested NumPy/SciPy/Matplotlib/pytest
versions; SDK runtime dependencies stay unchanged.

All 437 prior scientific JSON and 39 native artifacts remain hash-identical.
All Rust crates, native Python bridge/stub and production dependency manifests
are byte-identical to the parent. No new native build/install, real estimation,
sensor image/archive read, payload copy or download occurs.

New owned roots are `E:/SpatialRustBuild/room-drift-audit-v1` and
`E:/SpatialRustData/maturity-80/room-drift-audit-v1`. Complete inventories retain
88,437/15,222,221 logical bytes, respectively (15,310,658 combined), within the
prospective 1GiB scratch/16MiB result/2GiB combined caps. Exactly 560,792,891
bytes of passing pytest scratch were reclaimed after terminal/XML/absolute/
no-reparse checks. The physical free-space floor stays above 822GiB against
the required 20GiB minimum.

Room robustness, genuinely unobserved motion qualification, both raw Linux
hardware classes, matching canonical measured clock/physical accuracy and full
canonical mapping/semantic/Viewer/interchange/release acceptance remain open.
Neither this audit nor green CI proves operational 80%.
