# Canny dense-noise frontier counterattack — 2026-07-16

## Scope

The 3×3 allocation-light Canny path previously pushed every initial strong
edge onto the hysteresis stack. Dense sensor noise produces many strong
non-maximum-suppressed pixels, so that strategy spent most of its time popping
strong pixels and rechecking eight neighbors even when no weak edge needed
promotion.

Classification now records weak candidates instead. After all parallel bands
finish, only weak candidates adjacent to an initial strong edge become flood
seeds. The existing depth-first frontier then promotes the remainder of each
connected weak component. Initial strong edges remain in the output whether or
not they have weak neighbors.

The production change is in
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\canny.rs`.
It remains safe Rust, uses caller-owned output and `CannyWorkspace`, and does
not change the public API or device ownership.

## Correctness

- All focused Rust Canny tests pass, including fast/inspectable parity for L1
  and L2 gradients, parallel ring parity, strided output padding, and borders.
- The focused OpenCV harness passed 300 randomized images bit-exact.
- VGA, 1080p, and 4K document-line and sensor-noise canonical outputs are
  bit-exact versus OpenCV 4.13.

## Measured Python boundary

CPython 3.12.10, OpenCV 4.13.0, Windows 11 10.0.26300, Intel 6-core/12-thread
host, OpenCV 12 threads, Rayon default, OpenCL disabled, thresholds 80/160,
3×3 aperture, L2 gradient, seeded interleaved samples, and minimum 20 ms timed
batches:

| Profile | Pattern | Mode | Previous SpatialRust | Current SpatialRust | SpatialRust improvement | Current OpenCV | Outcome |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1080p | sensor noise | allocate | 27.382 ms | 8.818 ms | 3.11× | 20.863 ms | SpatialRust 2.37× |
| 1080p | sensor noise | caller output | 27.106 ms | 8.172 ms | 3.32× | 21.138 ms | SpatialRust 2.59× |
| 4K | sensor noise | allocate | 120.087 ms | 33.554 ms | 3.58× | 87.621 ms | SpatialRust 2.61× |
| 4K | sensor noise | caller output | 116.506 ms | 31.473 ms | 3.70× | 86.448 ms | SpatialRust 2.75× |
| 1080p | document lines | caller output | — | 2.221 ms | — | 3.075 ms | SpatialRust 1.38× |
| 4K | document lines | caller output | — | 8.034 ms | — | 11.832 ms | SpatialRust 1.47× |

VGA sensor-noise caller output remains an OpenCV win: 4.694 ms versus
2.048 ms (OpenCV 2.29×). The claim is therefore limited to the measured
1080p and 4K dense-noise workloads.

The candidate frontier also reduces retained workspace on this input from
16,732,288 to 10,682,304 bytes at 1080p and from 66,342,904 to 42,169,888
bytes at 4K, because its capacity follows weak candidates instead of all
initial strong pixels.

The pre-change strict JSON receipt is
`C:\Users\rsasa\Workspace\SpatialRust\target\opencv-canny-noise-before.json`.
Post-change receipts are
`C:\Users\rsasa\Workspace\SpatialRust\target\opencv-canny-noise-after.json`
and
`C:\Users\rsasa\Workspace\SpatialRust\target\opencv-canny-noise-vga-after.json`.
Generated JSON files are not committed.

## Native probe

Criterion `canny/reuse` quick probes measured 2.095 ms at VGA, 4.717 ms at
1080p, and 16.514 ms at 4K. Relative to the immediately preceding Criterion
baseline, the 1080p and 4K noisy inputs improved by about 2.75× and 2.89×.

## Reproduction

```powershell
$env:PATH = "C:\Users\rsasa\.cargo\bin;$env:PATH"
cargo test -p spatialrust-vision --features imgproc-canny
cargo bench -p spatialrust-vision --features imgproc-canny --bench canny canny/reuse -- --quick
Set-Location crates/spatialrust-py
maturin develop --release
Set-Location ../..
.venv/Scripts/python.exe bench/opencv_canny_comparison/performance.py --profiles 1080p,4k --patterns sensor-noise,document-lines --output target/opencv-canny-noise-after.json
```
