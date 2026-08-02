# CPU voxel centroid rebaseline — 2026-07-16

## Scope

This receipt remeasures the existing CPU centroid voxel downsampler. It does not
change the implementation, GPU threshold, or public API.

- Source revision: `334967e72c0e48e8c1877c076ea27e1706e03b91`
- Benchmark:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-filtering\benches\voxel_downsample.rs`
- Implementation:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-filtering\src\voxel.rs`
- Schema: `StandardSchemas::point_xyzi()`
- Mode: centroid, average attributes
- Leaf size: `4.0`
- Build: Criterion release profile

## Environment

- OS: Microsoft Windows 11 Pro Insider Preview `10.0.26300`
- Architecture: `x86_64-pc-windows-msvc`
- CPU: Intel Core i7-9750H at 2.60 GHz
- CPU topology: 6 cores / 12 logical processors
- Rust: `rustc 1.97.0 (2d8144b78 2026-07-07)`
- Cargo: `cargo 1.97.0 (c980f4866 2026-06-30)`
- Thread policy: benchmark default

## Command

```powershell
cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample -- cpu_centroid --noplot
```

## Criterion medians

| Input points | CPU centroid median |
| ---: | ---: |
| 10,000 | 252.35 us |
| 65,536 | 1.7191 ms |
| 100,000 | 2.6430 ms |
| 200,000 | 5.0882 ms |
| 500,000 | 11.555 ms |
| 750,000 | 18.280 ms |
| 1,000,000 | 23.887 ms |
| 2,000,000 | 47.290 ms |

The current 2M CPU median is about 8.2 times lower than the approximately
389 ms CPU value in the dated 2026-06-12 README comparison. This is a comparison
of repository receipts, not a portable cross-machine speedup claim.

## Optimization probe

A `point_xyzi`-specific fixed accumulator and compact `u32` key path was tested
against the generic implementation with exact output parity. It improved the
10k case by about 6%, was neutral around 100k–200k, and regressed the
500k–2M medians by roughly 3%–6%. The probe was therefore fully reverted and no
CPU implementation change landed in this slice.

## Claim boundary

These results apply only to this host, revision, schema, leaf size, build
profile, and benchmark generator. GPU timings were not collected in this run.
The 2026-07-16 CPU values must not be combined with the 2026-06-12 GPU values
to infer a new crossover point. The existing centroid Auto threshold remains
unchanged until a fresh matched CPU/GPU run is recorded.
