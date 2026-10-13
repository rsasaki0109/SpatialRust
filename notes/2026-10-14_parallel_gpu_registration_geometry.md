# Shared GPU registration geometry preparation (2026-10-14)

## Scope and representation

`GpuIcpInput::upload` prepares finite XYZ bounds with bounded native CPU workers
for large point batches. This boundary is shared by ordinary/trimmed ICP, GICP,
point-to-plane and GPU-generated covariance constructors. Device kernels,
queries, input positions, payload layouts and transfer receipts are unchanged.

The final policy keeps inputs below 262,144 points serial, without querying CPU
parallelism. Above that boundary, worker count is bounded by available CPU
parallelism and floor(points/65,536), with at least one worker. A one-worker or
WASM call uses serial scans. Source and target are prepared sequentially.
Each worker returns only six f32 bounds (24 bytes), plus its scoped thread and
handle; no new point-sized staging allocation or device copy is introduced.
Existing count/device limits precede geometry preparation. Validation errors
still return before device input allocation.

Finite extrema combine by minimum/maximum; f64 midpoint, difference, square and
axis accumulation order remain. Nonzero anchors and f64 extents match the frozen
parent exactly. The **intended representation change** is that every derived
zero frame anchor is now +0, on both serial/native and WASM paths. Original
input point bits, including -0, remain unchanged. Existing scale overflow and
underflow rejection is retained.

Rust min/max permit either input when equal positive/negative zeros are compared.
An attempt to replay identical serial folds matched initial tests but failed after
a rebuild: copying the same expression did not establish a stable zero sign.
The final code defines that sign rather than relying on compiler choices. See
[Rust f32::min](https://doc.rust-lang.org/std/primitive.f32.html#method.min) and
[f32::max](https://doc.rust-lang.org/std/primitive.f32.html#method.max).
The public raw 176-byte ICP moment buffer exposes frame anchors, so its zero
anchor signs can differ from older builds. Numeric values, nonzero words,
status/counts, normalized sums and receipts are separately verified. This is
not a claim of full raw-byte parity for derived zero anchors.

## Validation

Production sources:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_reduction.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_geometry_preparation.rs`

Frozen validation/upload and benchmark sources:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_geometry_preparation_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_geometry_preparation_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/tests/gpu_gicp_reuse_query_benchmark.rs`

The serial geometry and upload references come from merged parent
`b1f58d5a177b0fa053b7c43813071447dbd41248` (PR #189). Device kernels, target tree
and query upload remain shared and unchanged. Tests deliberately normalize only
zero signs in the frozen anchor reference; no floating epsilon is introduced.

- 76,032,240 input points across frozen geometry calls: exact nonzero anchor
  bits and f64 extent bits, canonical +0 anchors. Sizes include empty/single,
  small and previous worker boundaries, 131,071/72/73, 262,143/44/45 and 524,289;
  eight random/planar/line/constant/signed-zero/large-offset/subnormal profiles;
  requested workers 0/1/2/6 and automatic staging.
- All 256 eight-point signed-zero patterns repeat through 262,145-point parallel
  chunks with 2/6 requested workers (134,218,240 input points). Four concurrent
  262,145-point calls add 1,048,580 points. Uniform negative zeros and rounded
  negative-subnormal zero anchors explicitly become +0, while input bits stay
  unchanged. Total valid geometry comparison inputs: 211,299,060 points.
- Independent analytic bounds/extent, zero/one/many worker caps through
  `usize::MAX` without allocation, NaN/infinities in every axis at chunk/end
  positions, extreme constant offsets, overflowing spans and upload underflow.
- 234 native uploaded moment conditions compare 10,296 raw words. Status/count/
  reserved fields and all compensated sums are bit-exact. Exactly 180 anchor
  words differ only as +0 versus -0; all other anchor words match exactly.
  Decoded values, scales and receipts match, and queued results survive input
  owner drop. No meaningful math tolerance was widened.

## Measurement method and decisions

The initial one-worker-per-16,384 policy was rejected: 32,768, 65,537 and
262,145-point completed calls regressed in both surveys. For example, the last
full old/new ratio was 0.887x / 0.834x. Increasing chunk size to 65,536 improved
262,145-point host preparation but still regressed at the 131,072-point start
(0.943x / 0.927x completed-call ratios). The final minimum is therefore 262,144,
while retaining the bounded 65,536-point worker policy. Initial and intermediate
adverse tables/logs remain under local `-initial` and `-64k-grain` prefixes.

Those two earlier surveys used signed-zero fold replay. Their zero-sign/math
behavior and timing are **not** final production evidence. The final canonical
zero version removes bit summaries and replay passes, and is tested/measured
again with its own native executable. The unsuccessful exact-zero-sign rebuild
is documented above; it was never committed or published as a production claim.

## Final validation and completed-call results

The final implementation passed 52 native ICP/GICP/plane kernel tests and eight
registration integration targets (29 executions; shared reference tests repeat).
The latter cover 1,269 trajectory conditions: 54 ICP, 486 trimmed ICP, 27 large
trimmed-precision trajectories, 108 point-to-plane and 594 GICP covariance-route
conditions. Together they evaluate 4,040,037 source points. Existing meaningful
numeric tolerances were not widened. Maximum normalized pose-coordinate errors:
ICP 1.4305115e-6, trimmed 1.9073486e-6, precise large trajectories 7.6293945e-6,
plane and GICP 9.536743e-7. The prior explicit f32 precision mode's adverse
8.154977e-4 large-trajectory error is retained and unchanged.

GPU all-feature/all-target Clippy, registration GPU-feature Clippy, CPU-only GPU
check, native minimal facade, combined ICP/GICP/plane WASM facade and rustfmt
checks passed. The final iterator-only lint repair was checked again with the
234 upload conditions; no additional production math changed. No new dependency,
feature or unsafe boundary was introduced, and core remains untouched.

Two final native release surveys use Vulkan discrete GPU execution, 12 available
CPU workers and the default 64 MiB scratch budget. Every complete survey includes
15 fixtures, three warmups and 20 paired samples for four rotated routes:
frozen/current geometry preparation and frozen/current completed upload plus
one ordinary ICP moment evaluation. Both final surveys were accepted on their
first attempts, with test durations 8.18 / 8.14 seconds. Full timing includes
input validation/packing/upload, moment submission, explicit 176-byte download/
wait/decode and input/output owner drop. Caller point fixtures and CPU target-tree
construction are outside the timer. Host timing ends at the owned geometry result,
with comparisons outside timing. Decoded values and receipts match every call;
host nonzero anchors and extent bits match exactly, and new zero signs are +0.

Profiles: 0 deterministic 3D hash grid, 1 planar +0 z, 2 line with mixed signed
zero y/z, 3 coincident +0, 4 coincident mixed signed zeros. The five 1,048,577-
source/one-target conditions improve full calls in both surveys: 1.269 / 1.268x,
1.249 / 1.241x, 1.243 / 1.243x, 1.230 / 1.198x and 1.208 / 1.199x. All ten
paired intervals are above one. For profile 0, full medians change from
17.774 / 18.703 ms to 14.316 / 14.805 ms. Host geometry improves by about
2.4–2.7x across these large fixtures. The 262,144/262,145-point host conditions
also improve, but their completed intervals include one. Small/medium completed
calls show no repeated adverse interval below one after the policy correction.
These are selected synthetic uploaded-moment calls with one target, not evidence
that a million-point full registration algorithm is 1.27x faster.

The identical public GICP stdin/stdout benchmark was built at the parent before
production edits and saved. Final/current builds use the same bare register-gicp-gpu
feature set. Two live processes receive one command at a time; neither startup,
runtime construction nor caller cloud generation enters samples. Each actual
timed alignment includes CPU tree/covariance preparation, validation, packing,
transfers, four evaluations/downloads, three solves/updates and owned cleanup.
Config: k=6, distance=0.25, radius=0.8 for radius/grid, three forced updates,
zero stopping thresholds, translation (0.03125,-0.015625,0.0078125), and 17 extra
targets for duplicate profiles. Complete pose/fitness/stopping/history/covariance
batches/receipt strings match across all 640 accepted timed public calls.

All final public conditions were accepted on their first attempts. They cover
257–65,537 logical source points, including GPU constructors with combined
source/target geometry; all physical geometry batches remain below the final
262,144-point cutoff. They are serial-placement controls, not measurements of
large parallel geometry within the full alignment. No public condition establishes
a repeated speedup. The 65,537-point GPU k-NN route is adverse in survey 1
(0.936x [0.859,0.995]) and favorable in survey 2 (1.088x [1.010,1.191]); retain
both. Survey 2's 16,384-point grid route is also adverse (0.988x [0.976,0.999]),
while survey 1 includes one. Do not infer a general full-registration improvement.

The process monitor rejects foreign Cargo/compiler/formatter/linker and named
benchmark/test executables, preceded by three quiet observations. No accepted
benchmark overlapped our builds or statistical analysis; foreign work was never
stopped. This does not prove the whole computer idle or remove all background
variability. Earlier overlapping attempts and the corrected old-control zero-sign
assertion failure remain in local logs; they are excluded from final tables.

Times below are medians; ratios are geometric means of paired old/current times.
The unadjusted 95% intervals use 10,000 bootstrap resamples of 20 paired log ratios,
sorted endpoints 249/9749. Final primitive seeds are
`20261014 + 30*(survey-1) + 2*condition_index + stage_index` (host=0/full=1);
public seeds are `20261014 + 8*(survey-1) + condition_index`.
Individual phase components add to each full sample; phase medians need not add
to the full median. The download/wait phase includes device computation/copies;
it does not separately prove shader or allocator timing. In survey 2's 1,048,577-
point 3D fixture, upload medians decrease 13.682 to 9.852 ms; download/wait stays
4.687 / 4.666 ms, consistent with a host-preparation improvement.

Local SHA-256 provenance:

- Parent public executable: `4F342DB3284DC5BBA71E0A2D37D7F80DA87D214486F9A272A2D13A9E0830D14E`
- Identical public benchmark source: `34343FCC04D26E903FFA5B764BDF5C27EAEE8470D9CCB7416AD2539389F6C0D8`
- Final public executable: `22D24CDB3CD9D4039ABBE30D51F4BBE42C28ADA8B4B8861D1AE111AB0BD9A785`
- Final native primitive/reference executable: `B96908D16421872DACC0A818B3E2E4B24E402BC304512C255AF6FAE605FB29B2`

Local raw arrays, attempt/overlap logs, controllers and analysis are untracked in
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`:

- `parallel-gpu-registration-geometry-baseline.json`
- `run-parallel-gpu-registration-geometry-survey.ps1`
- `parallel-gpu-registration-geometry-benchmark-1.log` / `-2.log`
- `parallel-gpu-registration-geometry-measurements.json`
- `run-parallel-registration-geometry-public-survey.ps1`
- `parallel-registration-geometry-public-benchmark-1.json` / `-2.json`
- `parallel-registration-geometry-public-measurements.json`
- `analyze-parallel-gpu-registration-geometry.py`
- `spatialrust_geometry_previous.exe`
- `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/spatialrust_gpu-fafe8a8488d734a8.exe`
- `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/gpu_gicp_reuse_query_benchmark-2e751f82c032c4ce.exe`

The local adapter model name is omitted. Frozen Rust fixtures and all final
condition/phase tables are committed; executables/logs remain local. Ignored
benchmarks require their explicit exact native release test/controller.

## All final primitive conditions

| Survey | Sources / targets / profile | Workers | Stage | Old ms | New ms | Old/new paired ratio [95%] |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 257 / 257 / 0 | 1 | host | 0.001 | 0.001 | 1.059x (1.024-1.097) |
| 1 | 257 / 257 / 0 | 1 | full | 1.000 | 1.005 | 0.995x (0.974-1.017) |
| 1 | 4,097 / 4,097 / 0 | 1 | host | 0.016 | 0.015 | 1.031x (1.003-1.067) |
| 1 | 4,097 / 4,097 / 0 | 1 | full | 1.075 | 1.088 | 1.020x (0.972-1.099) |
| 1 | 32,767 / 1 / 0 | 1 | host | 0.136 | 0.125 | 1.070x (1.022-1.120) |
| 1 | 32,767 / 1 / 0 | 1 | full | 1.448 | 1.514 | 1.006x (0.971-1.045) |
| 1 | 32,768 / 1 / 0 | 1 | host | 0.133 | 0.125 | 1.050x (1.020-1.084) |
| 1 | 32,768 / 1 / 0 | 1 | full | 1.364 | 1.382 | 0.996x (0.968-1.023) |
| 1 | 65,537 / 65,537 / 0 | 1 | host | 0.261 | 0.252 | 1.035x (1.015-1.053) |
| 1 | 65,537 / 65,537 / 0 | 1 | full | 4.329 | 4.342 | 1.000x (0.983-1.016) |
| 1 | 131,071 / 1 / 0 | 1 | host | 0.539 | 0.518 | 1.030x (0.968-1.092) |
| 1 | 131,071 / 1 / 0 | 1 | full | 2.853 | 2.815 | 1.010x (0.975-1.051) |
| 1 | 131,072 / 1 / 0 | 1 | host | 0.562 | 0.534 | 1.025x (0.962-1.085) |
| 1 | 131,072 / 1 / 0 | 1 | full | 2.852 | 2.947 | 0.996x (0.969-1.023) |
| 1 | 262,143 / 1 / 0 | 1 | host | 1.143 | 1.097 | 1.062x (1.000-1.116) |
| 1 | 262,143 / 1 / 0 | 1 | full | 4.772 | 4.698 | 0.935x (0.854-1.008) |
| 1 | 262,144 / 1 / 0 | 4 | host | 1.145 | 0.972 | 1.280x (1.203-1.379) |
| 1 | 262,144 / 1 / 0 | 4 | full | 4.713 | 4.800 | 1.005x (0.970-1.040) |
| 1 | 262,145 / 1 / 0 | 4 | host | 1.257 | 0.935 | 1.404x (1.271-1.597) |
| 1 | 262,145 / 1 / 0 | 4 | full | 4.847 | 4.661 | 1.036x (0.949-1.119) |
| 1 | 1,048,577 / 1 / 0 | 12 | host | 6.024 | 2.289 | 2.717x (2.605-2.845) |
| 1 | 1,048,577 / 1 / 0 | 12 | full | 17.774 | 14.316 | 1.269x (1.248-1.292) |
| 1 | 1,048,577 / 1 / 1 | 12 | host | 5.916 | 2.255 | 2.636x (2.514-2.744) |
| 1 | 1,048,577 / 1 / 1 | 12 | full | 18.627 | 14.882 | 1.249x (1.234-1.264) |
| 1 | 1,048,577 / 1 / 2 | 12 | host | 5.855 | 2.320 | 2.509x (2.437-2.582) |
| 1 | 1,048,577 / 1 / 2 | 12 | full | 18.791 | 15.124 | 1.243x (1.225-1.260) |
| 1 | 1,048,577 / 1 / 3 | 12 | host | 6.016 | 2.323 | 2.615x (2.520-2.722) |
| 1 | 1,048,577 / 1 / 3 | 12 | full | 19.243 | 15.600 | 1.230x (1.216-1.243) |
| 1 | 1,048,577 / 1 / 4 | 12 | host | 5.918 | 2.324 | 2.593x (2.492-2.697) |
| 1 | 1,048,577 / 1 / 4 | 12 | full | 19.772 | 16.419 | 1.208x (1.191-1.225) |
| 2 | 257 / 257 / 0 | 1 | host | 0.001 | 0.001 | 1.093x (1.057-1.135) |
| 2 | 257 / 257 / 0 | 1 | full | 1.022 | 1.006 | 1.013x (0.998-1.028) |
| 2 | 4,097 / 4,097 / 0 | 1 | host | 0.015 | 0.015 | 1.010x (0.999-1.021) |
| 2 | 4,097 / 4,097 / 0 | 1 | full | 1.102 | 1.110 | 1.011x (0.988-1.035) |
| 2 | 32,767 / 1 / 0 | 1 | host | 0.127 | 0.124 | 1.035x (1.013-1.066) |
| 2 | 32,767 / 1 / 0 | 1 | full | 1.272 | 1.272 | 0.998x (0.972-1.027) |
| 2 | 32,768 / 1 / 0 | 1 | host | 0.138 | 0.125 | 1.081x (1.058-1.104) |
| 2 | 32,768 / 1 / 0 | 1 | full | 1.373 | 1.368 | 0.986x (0.958-1.014) |
| 2 | 65,537 / 65,537 / 0 | 1 | host | 0.261 | 0.251 | 1.018x (0.981-1.046) |
| 2 | 65,537 / 65,537 / 0 | 1 | full | 4.272 | 4.328 | 1.031x (0.988-1.098) |
| 2 | 131,071 / 1 / 0 | 1 | host | 0.520 | 0.506 | 1.039x (1.021-1.059) |
| 2 | 131,071 / 1 / 0 | 1 | full | 2.813 | 2.833 | 0.993x (0.974-1.013) |
| 2 | 131,072 / 1 / 0 | 1 | host | 0.523 | 0.501 | 1.050x (1.027-1.075) |
| 2 | 131,072 / 1 / 0 | 1 | full | 2.781 | 2.823 | 1.002x (0.976-1.030) |
| 2 | 262,143 / 1 / 0 | 1 | host | 1.157 | 1.057 | 1.109x (1.046-1.195) |
| 2 | 262,143 / 1 / 0 | 1 | full | 4.785 | 4.774 | 0.961x (0.885-1.018) |
| 2 | 262,144 / 1 / 0 | 4 | host | 1.141 | 0.940 | 1.242x (1.162-1.354) |
| 2 | 262,144 / 1 / 0 | 4 | full | 4.705 | 4.717 | 1.000x (0.924-1.064) |
| 2 | 262,145 / 1 / 0 | 4 | host | 1.178 | 0.940 | 1.246x (1.213-1.285) |
| 2 | 262,145 / 1 / 0 | 4 | full | 4.829 | 4.737 | 0.985x (0.918-1.037) |
| 2 | 1,048,577 / 1 / 0 | 12 | host | 6.064 | 2.303 | 2.627x (2.497-2.758) |
| 2 | 1,048,577 / 1 / 0 | 12 | full | 18.703 | 14.805 | 1.268x (1.251-1.287) |
| 2 | 1,048,577 / 1 / 1 | 12 | host | 5.961 | 2.384 | 2.452x (2.311-2.592) |
| 2 | 1,048,577 / 1 / 1 | 12 | full | 18.552 | 15.197 | 1.241x (1.220-1.263) |
| 2 | 1,048,577 / 1 / 2 | 12 | host | 5.938 | 2.299 | 2.657x (2.554-2.766) |
| 2 | 1,048,577 / 1 / 2 | 12 | full | 19.030 | 15.512 | 1.243x (1.222-1.267) |
| 2 | 1,048,577 / 1 / 3 | 12 | host | 5.943 | 2.391 | 2.564x (2.417-2.722) |
| 2 | 1,048,577 / 1 / 3 | 12 | full | 19.491 | 16.215 | 1.198x (1.164-1.228) |
| 2 | 1,048,577 / 1 / 4 | 12 | host | 5.875 | 2.471 | 2.382x (2.241-2.523) |
| 2 | 1,048,577 / 1 / 4 | 12 | full | 20.086 | 16.786 | 1.199x (1.178-1.221) |

## All final public registration conditions

| Survey | Sources | Profile | Covariance route | Old ms | New ms | Old/new paired ratio [95%] |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 257 | lattice | CPU k-NN | 7.542 | 7.477 | 0.988x (0.949-1.025) |
| 1 | 4,097 | lattice | CPU k-NN | 18.710 | 18.769 | 1.006x (0.984-1.029) |
| 1 | 4,097 | duplicate/unequal | GPU k-NN | 16.233 | 16.029 | 1.019x (0.993-1.045) |
| 1 | 16,384 | lattice | GPU grid | 36.654 | 36.636 | 0.988x (0.962-1.011) |
| 1 | 65,537 | lattice | CPU k-NN | 135.507 | 134.936 | 1.000x (0.984-1.015) |
| 1 | 65,537 | lattice | GPU k-NN | 56.020 | 58.937 | 0.936x (0.859-0.995) |
| 1 | 65,537 | lattice | GPU radius | 74.176 | 75.279 | 1.009x (0.971-1.073) |
| 1 | 65,537 | duplicate/unequal | GPU grid | 84.515 | 84.663 | 0.995x (0.987-1.002) |
| 2 | 257 | lattice | CPU k-NN | 7.311 | 7.612 | 0.970x (0.921-1.025) |
| 2 | 4,097 | lattice | CPU k-NN | 18.725 | 18.779 | 0.993x (0.973-1.014) |
| 2 | 4,097 | duplicate/unequal | GPU k-NN | 16.268 | 15.939 | 1.008x (0.983-1.030) |
| 2 | 16,384 | lattice | GPU grid | 36.855 | 36.864 | 0.988x (0.976-0.999) |
| 2 | 65,537 | lattice | CPU k-NN | 135.924 | 136.055 | 1.013x (0.986-1.040) |
| 2 | 65,537 | lattice | GPU k-NN | 56.553 | 56.754 | 1.088x (1.010-1.191) |
| 2 | 65,537 | lattice | GPU radius | 75.511 | 75.564 | 0.996x (0.908-1.090) |
| 2 | 65,537 | duplicate/unequal | GPU grid | 84.773 | 84.738 | 1.001x (0.997-1.005) |

## All final primitive phases

| Survey | Sources / targets / profile | Variant | Upload ms | Submit ms | Download/wait ms | Drop ms | Full ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 257 / 257 / 0 | old | 0.024 | 0.113 | 0.853 | 0.003 | 1.000 |
| 1 | 257 / 257 / 0 | new | 0.024 | 0.108 | 0.867 | 0.003 | 1.005 |
| 1 | 4,097 / 4,097 / 0 | old | 0.102 | 0.111 | 0.856 | 0.003 | 1.075 |
| 1 | 4,097 / 4,097 / 0 | new | 0.101 | 0.108 | 0.862 | 0.003 | 1.088 |
| 1 | 32,767 / 1 / 0 | old | 0.285 | 0.132 | 1.023 | 0.004 | 1.448 |
| 1 | 32,767 / 1 / 0 | new | 0.289 | 0.135 | 1.044 | 0.004 | 1.514 |
| 1 | 32,768 / 1 / 0 | old | 0.264 | 0.123 | 0.960 | 0.004 | 1.364 |
| 1 | 32,768 / 1 / 0 | new | 0.273 | 0.122 | 0.965 | 0.003 | 1.382 |
| 1 | 65,537 / 65,537 / 0 | old | 2.453 | 0.146 | 1.738 | 0.004 | 4.329 |
| 1 | 65,537 / 65,537 / 0 | new | 2.453 | 0.160 | 1.693 | 0.004 | 4.342 |
| 1 | 131,071 / 1 / 0 | old | 1.444 | 0.153 | 1.228 | 0.003 | 2.853 |
| 1 | 131,071 / 1 / 0 | new | 1.442 | 0.129 | 1.229 | 0.003 | 2.815 |
| 1 | 131,072 / 1 / 0 | old | 1.498 | 0.172 | 1.242 | 0.004 | 2.852 |
| 1 | 131,072 / 1 / 0 | new | 1.573 | 0.167 | 1.235 | 0.004 | 2.947 |
| 1 | 262,143 / 1 / 0 | old | 2.848 | 0.195 | 1.694 | 0.004 | 4.772 |
| 1 | 262,143 / 1 / 0 | new | 2.856 | 0.142 | 1.688 | 0.003 | 4.698 |
| 1 | 262,144 / 1 / 0 | old | 2.881 | 0.171 | 1.647 | 0.004 | 4.713 |
| 1 | 262,144 / 1 / 0 | new | 2.911 | 0.192 | 1.652 | 0.004 | 4.800 |
| 1 | 262,145 / 1 / 0 | old | 2.989 | 0.211 | 1.660 | 0.004 | 4.847 |
| 1 | 262,145 / 1 / 0 | new | 2.774 | 0.164 | 1.671 | 0.004 | 4.661 |
| 1 | 1,048,577 / 1 / 0 | old | 13.661 | 0.246 | 3.946 | 0.006 | 17.774 |
| 1 | 1,048,577 / 1 / 0 | new | 9.970 | 0.254 | 4.053 | 0.007 | 14.316 |
| 1 | 1,048,577 / 1 / 1 | old | 13.694 | 0.252 | 4.763 | 0.008 | 18.627 |
| 1 | 1,048,577 / 1 / 1 | new | 9.923 | 0.250 | 4.707 | 0.007 | 14.882 |
| 1 | 1,048,577 / 1 / 2 | old | 13.454 | 0.268 | 4.985 | 0.008 | 18.791 |
| 1 | 1,048,577 / 1 / 2 | new | 9.839 | 0.258 | 4.996 | 0.008 | 15.124 |
| 1 | 1,048,577 / 1 / 3 | old | 13.445 | 0.270 | 5.473 | 0.007 | 19.243 |
| 1 | 1,048,577 / 1 / 3 | new | 9.840 | 0.259 | 5.472 | 0.007 | 15.600 |
| 1 | 1,048,577 / 1 / 4 | old | 13.414 | 0.261 | 6.058 | 0.008 | 19.772 |
| 1 | 1,048,577 / 1 / 4 | new | 10.052 | 0.257 | 5.950 | 0.008 | 16.419 |
| 2 | 257 / 257 / 0 | old | 0.023 | 0.106 | 0.886 | 0.003 | 1.022 |
| 2 | 257 / 257 / 0 | new | 0.023 | 0.107 | 0.875 | 0.003 | 1.006 |
| 2 | 4,097 / 4,097 / 0 | old | 0.100 | 0.112 | 0.893 | 0.003 | 1.102 |
| 2 | 4,097 / 4,097 / 0 | new | 0.100 | 0.110 | 0.892 | 0.003 | 1.110 |
| 2 | 32,767 / 1 / 0 | old | 0.252 | 0.113 | 0.904 | 0.003 | 1.272 |
| 2 | 32,767 / 1 / 0 | new | 0.256 | 0.109 | 0.900 | 0.003 | 1.272 |
| 2 | 32,768 / 1 / 0 | old | 0.278 | 0.131 | 0.960 | 0.003 | 1.373 |
| 2 | 32,768 / 1 / 0 | new | 0.280 | 0.125 | 0.960 | 0.004 | 1.368 |
| 2 | 65,537 / 65,537 / 0 | old | 2.450 | 0.150 | 1.713 | 0.004 | 4.272 |
| 2 | 65,537 / 65,537 / 0 | new | 2.428 | 0.162 | 1.704 | 0.003 | 4.328 |
| 2 | 131,071 / 1 / 0 | old | 1.438 | 0.127 | 1.219 | 0.003 | 2.813 |
| 2 | 131,071 / 1 / 0 | new | 1.457 | 0.131 | 1.241 | 0.003 | 2.833 |
| 2 | 131,072 / 1 / 0 | old | 1.432 | 0.137 | 1.219 | 0.003 | 2.781 |
| 2 | 131,072 / 1 / 0 | new | 1.452 | 0.131 | 1.230 | 0.003 | 2.823 |
| 2 | 262,143 / 1 / 0 | old | 2.928 | 0.189 | 1.650 | 0.004 | 4.785 |
| 2 | 262,143 / 1 / 0 | new | 2.963 | 0.169 | 1.654 | 0.004 | 4.774 |
| 2 | 262,144 / 1 / 0 | old | 2.876 | 0.174 | 1.642 | 0.004 | 4.705 |
| 2 | 262,144 / 1 / 0 | new | 2.889 | 0.198 | 1.605 | 0.004 | 4.717 |
| 2 | 262,145 / 1 / 0 | old | 2.967 | 0.208 | 1.640 | 0.004 | 4.829 |
| 2 | 262,145 / 1 / 0 | new | 2.920 | 0.169 | 1.648 | 0.004 | 4.737 |
| 2 | 1,048,577 / 1 / 0 | old | 13.682 | 0.282 | 4.687 | 0.007 | 18.703 |
| 2 | 1,048,577 / 1 / 0 | new | 9.852 | 0.267 | 4.666 | 0.007 | 14.805 |
| 2 | 1,048,577 / 1 / 1 | old | 13.354 | 0.260 | 4.864 | 0.008 | 18.552 |
| 2 | 1,048,577 / 1 / 1 | new | 10.042 | 0.259 | 4.896 | 0.008 | 15.197 |
| 2 | 1,048,577 / 1 / 2 | old | 13.455 | 0.272 | 5.133 | 0.006 | 19.030 |
| 2 | 1,048,577 / 1 / 2 | new | 10.058 | 0.279 | 5.232 | 0.008 | 15.512 |
| 2 | 1,048,577 / 1 / 3 | old | 13.420 | 0.271 | 5.743 | 0.007 | 19.491 |
| 2 | 1,048,577 / 1 / 3 | new | 10.117 | 0.272 | 5.756 | 0.007 | 16.215 |
| 2 | 1,048,577 / 1 / 4 | old | 13.467 | 0.228 | 6.377 | 0.008 | 20.086 |
| 2 | 1,048,577 / 1 / 4 | new | 10.096 | 0.195 | 6.405 | 0.007 | 16.786 |
