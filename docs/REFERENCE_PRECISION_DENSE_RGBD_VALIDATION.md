# Explicit reference-precision dense RGB-D model

Slice 146T adds the opt-in Rust `DenseRgbdFittingModel::ReferencePrecision` and
Python `fitting_model="reference-precision"`. Float32 intermediate image/XYZ
arithmetic and proper left `Rz * Ry * Rx` increments reproduce all six frozen
CPU reference one-step transforms to at most 1.11e-16 per component. The native
rpy development trajectory now passes the original accuracy gate. Room remains
outside both accuracy limits; operational 80% acceptance remains open.

The [complete protocol](../benchmarks/reference-precision-dense-plan.json) was
committed as `0ab2b6e753cbf6dacf290048a9015aaef28defb5` before implementation,
builds or new fitting. Numerical source is
`356717b98f68e9784067b2f3e5cb507f0510bbfb`. The
[public receipt](receipts/2026-10-20_reference_precision_dense_rgbd.json) binds
sources, actual installed wheel/extension, frozen native tests, all trajectories,
original source archives, independent audits and resource records.

## Numerical and ownership contract

Each Gaussian/Sobel pass has float32 tap products, ordered f64 accumulation and
a float32 store. Locally recomputed horizontal samples avoid another owned
plane. Pair-mean gains use f64 means and float32 stores; floor pyramids use
left-associated float32 2x2 means. NaN holes and replicated borders remain
explicit. Fitting XYZ is quantized before the f64 proper pose action; intensity
subtraction is float32. Sqrt-weighted rows feed the existing checked f64 solver.

Matching retains native unquantized f64 geometry and signed rounded lookup.
The finite/pivoted Gaussian solve remains checked. These are distinct from the
external projection association and solver behavior; six synthetic matches are
not a general bitwise-equivalence or real-error causal claim.

Owned level buffers still store f64 values. Checked logical working bytes are
`32*sum(floor-pyramid pixels) + 64*base_pixels +
max(ceil(height/16)*sizeof(Normal), 48*sizeof(f64))`, covering retained raw
levels, fitting pyramids, current gradients/smoothing and nonconcurrent ordered
reduction/solver scratch. Caller/Python input copies, allocator metadata and
stacks are separate. Python owns inputs before releasing the GIL. CPU workers
are bounded by four; WASM is serial.

The raw/bilinear default, all three existing filtered choices and rounded-sobel
delegate unchanged. New-model preprocessing must be `raw`. Every scheduled
update executes; final admission uses original full-resolution raw images.
Core, math, GPU, heavy dependencies and explicit CPU/device transfer contracts
are unchanged.

## Fixed complete development outcomes

All four recordings were already observed before this model. Plant and room
are development data here. All four complete trajectories were frozen before
any score. The first seed is identity and subsequent seeds use the preceding
relative increment. No retry, reset, trimming, rescale or post-score model
change occurred. All 4,015 raw/causal pairs and four score audits pass.

Acceptance remains 100% generated rows, at least 95% reference matches,
first-pose aligned ATE RMSE <=0.10m and 1s RPE RMSE <=0.05m.

| Recording | Generated | Reference matches | ATE RMSE (m) | 1s RPE RMSE (m) | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| xyz | 798/798 | 795 | 0.050581566 | 0.023074869 | Pass |
| rpy | 722/722 | 716 | 0.099679364 | 0.048502927 | Pass |
| plant | 1139/1139 | 1130 | 0.078227299 | 0.037715335 | Pass |
| room | 1360/1360 | 1359 | 0.378652661 | 0.058375326 | Fail |

The exact receipt determines acceptance. Rpy ATE is 0.09967936372592444m,
only 0.00032063627407556m below the original limit. The prior rounded-sobel
rpy result (0.10032699404221075m) and every prior failure stay preserved.
Room ATE/RPE are 0.37865266052617225/0.05837532600255425m and fail both limits.
All 19 unmatched reference timestamps remain explicit; none were trimmed.

## Verification and storage

Native tests pass 35 cases (28 existing, 7 new); the separately frozen test
executable passes the same 35. Python passes 163 focused and 1,310 full tests
with the three existing skips. Clippy, WASM compilation, stub/runtime ownership
checks and installed-wheel byte validation pass. All 25 cases across the five
old choices match the frozen prior extension exactly. The six new synthetic
poses also match the separate source-model predictions within 9.81e-17.

All 399 prior scientific JSON files and 36 native artifacts remain hash-identical.
One receipt assembly attempt stopped on an untracked root lockfile before
writing a public receipt; correcting that provenance inventory reused every
complete scientific output. Its script and diagnostic note remain available.

| Owned root | Logical regular-file bytes |
| --- | ---: |
| `E:/SpatialRustBuild/native-dense-reference-precision-v1` | 23,238,057 |
| `E:/SpatialRustData/maturity-80/reference-precision-dense-input-v1` | 2,502,838,373 |
| `E:/SpatialRustData/maturity-80/reference-precision-dense-study-v1` | 810,123,525 |

Positive shared Cargo-cache growth is 15,793,322 bytes;
combined new logical bytes and positive cache growth are 3,351,993,277
bytes against the prospective 8GiB cap. Input PNG/index views are hard links:
new image payload copies and sensor downloads are zero. Complete inventories
charge each linked pathname logically. Physical free space remains over 822GiB,
above the 20GiB floor. Exactly 722,806,354 bytes of successful pytest scratch
were reclaimed after terminal/XML/absolute/no-reparse checks; all failures and
scientific outputs remain. The study's conservative result counter additionally
charges prepared-view metadata; the table counts each root separately.

Independent genuinely unobserved motion qualification, both raw Linux hardware
classes, matching canonical measured clock uncertainty and physical frame
accuracy, and complete canonical mapping/semantic/Viewer/interchange/release
acceptance remain required. CI and development passes do not establish 80%.
