# Gaussian band-local pipeline counterattack — 2026-07-16

## Scope

This slice targets the remaining standalone 5×5 Gaussian gap without changing
the public API or numerical contract. Large Q8 `u8` calls now partition the
output into worker-owned row bands. Each band materializes only its horizontal
rows plus the vertical halo, then immediately performs the vertical pass.
Runtime target dispatch is amortized once per band rather than once per row.

The implementation remains safe Rust, retains the `u16` intermediate, writes
caller-owned output, and keeps all scratch in `GaussianBlurU8Workspace`.
Small images retain the sequential path. The Q15 7×7 path is unchanged.

The production change is in
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\filter.rs`.
The focused harness is
`C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_gaussian_comparison\performance.py`.

## Correctness

- The existing 300 randomized RGB cases, including non-contiguous inputs and
  3×3/5×5/7×7 kernels, retained maximum absolute error 2/255 versus OpenCV.
- Canonical 1080p and 4K 5×5 output was bit-exact versus OpenCV.
- A 600×600 strided varying RGB test crosses parallel band seams and stays
  within 1/255 of the generic `f64` implementation.
- A large constant RGB image remains exact with a matching constant border,
  including top and bottom halo synthesis.

## Measured Python boundary

CPython 3.12.10, OpenCV 4.13.0, Windows 11 10.0.26300, Intel 6-core/12-thread
host, OpenCV 12 threads, Rayon default, OpenCL disabled, packed random RGB8,
5×5 sigma 1.2 Reflect101, seeded interleaved samples, and minimum 20 ms timed
batches:

| Profile | Mode | Previous SpatialRust | Current SpatialRust | SpatialRust improvement | OpenCV current | Current outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1080p | allocate | 6.188 ms | 3.443 ms | 1.80× | 1.983 ms | OpenCV 1.74× |
| 1080p | caller output | 5.469 ms | 3.054 ms | 1.79× | 1.473 ms | OpenCV 2.07× |
| 4K | allocate | 21.031 ms | 12.402 ms | 1.70× | 7.397 ms | OpenCV 1.68× |
| 4K | caller output | 20.586 ms | 10.635 ms | 1.94× | 5.169 ms | OpenCV 2.06× |

The previous values are the same-host receipt in
`C:\Users\rsasa\Workspace\SpatialRust\notes\2026-07-16_gaussian_acceleration.md`.
The current strict JSON receipt is generated at
`C:\Users\rsasa\Workspace\SpatialRust\target\opencv-gaussian-counterattack.json`
and is not committed.

## Native probe

Criterion `specialized_reuse` quick probes measured 1.297 ms at VGA, 2.548 ms
at 1080p, and 9.901 ms at 4K after the change. The immediately recorded
pre-change probes were 1.385 ms, 2.912 ms, and 10.985 ms. These short native
probes support the direction of the Python result but are not used for a
portable performance guarantee.

Band scratch remains explicit. With 12 workers, the Q8 horizontal capacity is
about 13.0 MB at 1080p and 50.9 MB at 4K, including four halo rows per band.

## Reproduction

```powershell
$env:PATH = "C:\Users\rsasa\.cargo\bin;$env:PATH"
cargo test -p spatialrust-vision --features imgproc-filter
cargo bench -p spatialrust-vision --features imgproc-filter --bench filter gaussian_blur_rgb8_5x5/specialized_reuse -- --quick
Set-Location crates/spatialrust-py
maturin develop --release
Set-Location ../..
.venv/Scripts/python.exe bench/opencv_gaussian_comparison/performance.py --profiles 1080p,4k --output target/opencv-gaussian-counterattack.json
```

This is a workload- and host-specific gap reduction. It does not claim that
SpatialRust wins standalone Gaussian blur against OpenCV.
