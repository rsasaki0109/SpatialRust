# Resident GPU image CDF — 2026-10-11

Workspace: `C:\Users\rsasa\Workspace\SpatialRust`.
Parent: `318fa829df5f257c87b28829573d2d33c0bb0cb8`, PR 128,
`codex/resident-gpu-image-histograms`. One focused GPU histogram-consumer slice.
The preceding CPU contrast PR 125 merged as
`4cc7a4da21539dc3b43ef8313127622346c4b470` before this slice started.

## Contract and implementation

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\histogram\cdf.rs`
adds `histogram_cdf_gpu`, `histogram_cdf_gpu_profiled` and `GpuImageCdf` under
`gpu-image`. Input is a resident `GpuImageHistogram`; no host-bin input or raw
buffer escape hatch is added. Entry i is the exact number of selected-channel
pixels at or below byte value i (inclusive). Values are monotonically
nondecreasing, and entry 255 equals `pixel_count()`. The source logical channel
is retained. Counts are unnormalized u32, with no float conversion or rounding.

One 256-lane compute pass uses the same work-efficient Blelloch tree structure
as the point-cloud scan, specialized to one full block. Each lane retains its
original bin, eight upsweep rounds build totals, eight downsweep rounds build
exclusive prefixes, and adding the original bin produces inclusive counts.
Every lane participates in all barriers; every tree and output slot is written.
Every partial sum/prefix is bounded by the source's checked u32 pixel total,
including u32::MAX. The shader uses 1 KiB workgroup-local memory, no uniform or
global temporary buffer, and a separate 1 KiB result. Device lane, local-storage,
workgroup and two-buffer limits are checked before pipeline/output allocation.

The original histogram stays immutable. Source reads are submitted before the
CDF call returns, so explicit source recycling is safe before the CDF completes:
subsequent pool reuse is ordered after those reads on the runtime queue. Each
result independently owns its output, retains the runtime identity across
moves, and only explicitly recycles to the original bounded buffer pool.
Foreign sources/readbacks are rejected and foreign recycling discards storage.
The per-result 1 KiB size and retained-pool budget do not bound caller-held live
result counts. Default dropping releases handles without implicit recycling.
The additional pipeline family is lazily cached per runtime.

CDF construction adds no payload upload/readback or host wait. Receipts inherit
the source history at construction and add 1,024 device-write bytes and stage
`histogram_cdf_u32_gpu`. Prior explicit source-bin readbacks remain in that
cumulative history; later source receipt changes do not mutate the CDF receipt.
Native `readback_u32` explicitly reads `[u32; 256]` and records another 1,024 bytes
and `readback_cdf_u32` on each successful call. Histogram and CDF now share a
private native staging helper; histogram readback behavior/receipts are unchanged.
Blocking readback remains absent on browser WASM. Browser runtime/async readback
is not tested; device computation is compile-checked.

Profiling reserves one unused stage on the same runtime before allocating work.
Explicit collection transfers 16 query bytes for one CDF pass, or 48 bytes for
histogram (two passes) plus CDF, separately from payload receipts. Capacity
rejection preserves prior samples; inner recording failure invalidates the
profiler. GPU timestamps exclude CPU setup, waits, transfers and inter-pass gaps.

The public APIs are provisional. No dependency is added, and default/core
production graphs stay unchanged. Normalized CDF and GPU equalization are future
work: their CPU-compatible rounding/degenerate-image contract must be defined
before implementing lookup application.

## Validation

`cargo test -p spatialrust-gpu --all-features -- --test-threads=2` passed all
77 unit, four new public CDF integration, three histogram integration and seven
existing image/point timing integration tests: 91 total on the local GPU/Vulkan.
The five new unit tests cover seven insufficient device-limit fields, exact
one-hot/leading/trailing/mixed/full-width bins through u32::MAX, immutable source
bins, repeated explicit readbacks, stale output reuse, retained outputs after
source recycling, zero retention budgets, runtime moves/foreign runtimes, and
four concurrent callers retaining 12 independently verified CDFs.

Public tests cover all 1/2/3/4 logical channels on a strided 17x19 ROI, an explicit
RGB-to-gray-to-histogram-to-CDF chain with no intermediate downloads, independent
cumulative receipt history, one/three-pass timestamp collection, unchanged
output/receipts under profiling, insufficient capacity without allocation,
foreign-profiler rejection and failed inner recording.

Default GPU tests, image-only compilation and the adapter-free CDF limit test
passed. Full/default all-target Clippy, facade `gpu-image`, minimal WGPU and
image+AoSoA WASM library checks, and Rustdoc passed with warnings denied.
Rustfmt, diff whitespace, CI YAML and benchmark compilation are checked.
Hosted GPU-less CI compiles shaders' calling code/tests and executes the two
adapter-free planners; actual shader execution is validated locally. The
existing benchmark-compile job now includes `image_cdf`; the required named
check count stays 114.

## Completed-work measurements

Windows, Rust 1.98.1 release/thin LTO, i7-9750H (6 cores/12 logical CPUs),
local GPU, Vulkan NVIDIA 596.36. Other user Python computations remained running;
none were stopped or reprioritized. Only the benchmark process sets
`RAYON_NUM_THREADS=6`; its image CPU reference is production
`spatialrust_vision::histogram_u8`, and the 256-bin CPU scan is a scalar inclusive
reference. GPU resident completion uses explicit `wait_idle` in timed regions.

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\examples\gpu_cdf.rs`
prebuilds and uploads images/histograms outside timing. Mixed values are
`(i*37 + i/997) % 256`; hot-bin images contain 43. It runs two warmups and ten
rounds rotating seven modes in one process. Every measured result is verified;
verification, destruction/recycling and optional timestamp collection are outside
the timed region. Resident modes explicitly read 1 KiB after timing for
verification, so the executable as a whole does perform that untimed transfer.
Separate sources keep the CPU readback receipt histories out of resident input.
All modes have zero timed H2D payload; source initialization uploads are excluded.

| Mode | Starting input | Timed output | Timed D2H payload |
| --- | --- | --- | --- |
| CpuScan | host 256-bin u32 histogram | host cumulative counts | none |
| Resident | resident histogram | completed resident CDF | none |
| CdfReadback | resident histogram | host CDF via GPU scan | 1,024 bytes |
| BinsReadbackCpu | resident histogram | host CDF via download + scalar scan | 1,024 bytes |
| ImageChainResident | resident image | completed resident histogram + CDF | none |
| ImageChainReadback | resident image | host CDF via histogram + GPU scan | 1,024 bytes |
| ImageReadbackCpu | resident image | host CPU histogram + CDF | full physical RGBA8 image |

All entries below are median [observed min, max] in ms. CPU scan values near the
native clock granularity and cache/scheduling changes require special care.
Ranges are observations of ten calls, not confidence intervals.

| Image / distribution | CpuScan | Resident | CdfReadback | BinsReadbackCpu | ImageChainResident | ImageChainReadback | ImageReadbackCpu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640x480 / mixed | 0.000400 [0.000300, 0.000500] | 0.162250 [0.120800, 0.266800] | 1.309700 [1.109100, 1.512900] | 1.225150 [1.111000, 1.326400] | 0.370350 [0.307900, 0.410400] | 1.400700 [1.195000, 1.562000] | 2.180200 [1.940500, 2.376400] |
| 640x480 / hot | 0.000400 [0.000300, 0.000600] | 0.164650 [0.124800, 0.231600] | 1.287100 [1.090100, 1.603600] | 1.080350 [0.989200, 1.670700] | 0.362850 [0.264800, 0.469200] | 1.262850 [1.209800, 1.512000] | 2.496050 [2.214800, 2.942200] |
| 3840x2160 / mixed | 0.001400 [0.000500, 0.002300] | 0.291800 [0.205900, 0.535100] | 5.959250 [5.466800, 6.854400] | 5.859100 [5.189500, 6.905700] | 0.636700 [0.551300, 0.678900] | 6.373350 [5.472900, 7.786500] | 31.154850 [29.306000, 32.296200] |
| 3840x2160 / hot | 0.000950 [0.000700, 0.002800] | 0.286400 [0.137600, 0.512400] | 5.897150 [5.348600, 6.795900] | 5.769800 [4.963700, 7.110900] | 0.611900 [0.496900, 0.711700] | 6.028700 [5.335200, 8.582700] | 31.139400 [26.676000, 36.295300] |

The standalone CPU scan is substantially faster than a completed GPU dispatch.
From a resident histogram to host CDF, the GPU scan plus readback also did not
beat bin readback plus CPU scan here (4K mixed 5.959 versus 5.859 ms). It provides
an explicit resident stage for subsequent GPU consumers, rather than an
argument to offload a host 256-bin array. Different starting/output placements
must not be treated as interchangeable latency comparisons.

For already-resident 4K mixed images returning host cumulative counts,
ImageChainReadback took 6.373 ms versus 31.155 ms for ImageReadbackCpu, about
4.9x locally. D2H payload was 1,024 versus 33,177,600 bytes (32,400x fewer).
The completed resident image chain took 0.637 ms. These are same-process local
observations under external load, not a portable speedup guarantee or evidence
for automatic GPU placement. Readback latency varies despite fixed 1 KiB payload;
shader time alone cannot explain completed-call latency.

All four profiles had zero new pooled scratch/output storage allocations after
warmup; existing image-chain scratch retained 262,144 bytes. This excludes
bind groups, encoders, readback staging, textures and driver allocations. The
CDF itself allocates no global scratch. Separate GPU pass timestamps were
5.184–5.344 us, with 16 explicit query-readback bytes per call.

### Criterion evidence

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\benches\image_cdf.rs`
measures the four histogram-input modes for mixed/hot distributions: eight
cases, 20 samples, one-second warmup and three-second measurement target
(extended by Criterion when necessary). It uses the same completed-work,
post-timing verification and explicit transfer boundary. Original images are
4K, but each scan has exactly 256 input bins. Cases run sequentially under
external load; 95% intervals do not account for between-case load variation.
Arithmetic-mean estimates and their 95% intervals below are in ms, not the slope
estimates Criterion may print during linear sampling. No adverse case is dropped.

| Distribution | CpuScan | Resident | CdfReadback | BinsReadbackCpu |
| --- | --- | --- | --- | --- |
| mixed | 0.000169 [0.000162, 0.000176] | 0.255589 [0.223908, 0.297221] | 1.405783 [1.367184, 1.455487] | 2.336414 [1.763387, 3.319238] |
| hot | 0.000275 [0.000233, 0.000328] | 0.314987 [0.166298, 0.549918] | 1.225546 [1.171799, 1.306538] | 1.108813 [1.083134, 1.136258] |

The mixed bin-readback/CPU case has a wide interval from desktop stalls. The
hot distribution again shows GPU-scan/readback taking longer than CPU scanning
after bin readback. All eight estimates are retained, with no rerun selection.

## Reproduction and receipts

```powershell
$env:RAYON_NUM_THREADS = '6'
cargo run -p spatialrust-gpu --features gpu-image --release --example gpu_cdf
cargo bench -p spatialrust-gpu --features gpu-image --bench image_cdf -- --noplot
```

Ignored local artifacts:

- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-cdf-tests.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-cdf-interleaved.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-cdf-criterion.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-cdf-estimates.json`

Delivery is stacked on frozen PR 128. A named-check merge guard waits for the
parent merge, retargets to main and requires all 114 successful checks, no
pending/failures and a clean merge state. It stops on changed source/dependency
heads or conflicts and preserves user body edits/local branches. Pending CI
is not reported as a completed merge.
