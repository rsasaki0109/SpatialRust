# Packed nearest and area resize acceleration — 2026-07-16

## Scope

This slice completes Epic 115 with reusable `NearestResizeU8Plan` and
`AreaResizeU8Plan` execution for packed one-channel and RGB8 images. Nearest
plans cache the half-pixel source indices. Area plans use integer block averages
for exact 2x and 4x shrinking and cache weighted source spans for other shrinking
ratios.

Both plans expose allocating and caller-owned `resize_into` entry points.
Explicitly strided input/output, enlargement for area, and other channel counts
fall back to the existing generic CPU implementation. No device transfer,
implicit GPU selection, or `unsafe` code is introduced.

## Correctness gate

A 300-case property test covers arbitrary input/output dimensions from 1 through
24 for gray and RGB8 nearest/area execution. Every planned result is bit-exact
with generic `resize`. Dedicated tests cover strided input and output, untouched
row padding, metadata propagation, and input/output dimension rejection.

`cargo test -p spatialrust-vision --all-features` passed 134 unit tests and 13
integration/property tests. The resize Criterion target also passed its `--test`
compile/execution check.

## Native measurement

Criterion measured packed RGB8 caller-owned half-scale output on Windows 10,
Intel Core i7-9750H, 12 logical processors, using the default Rayon thread
policy. These are short local runs with 500 ms warm-up, 1 s requested
measurement time, and 10 samples; the comparison is limited to planned reuse
versus the pre-existing generic reuse path.

| Filter/profile | Generic reuse | Planned reuse | Improvement |
| --- | ---: | ---: | ---: |
| nearest VGA | 1.705 ms | 0.0565 ms | 30.2x |
| nearest 1080p | 12.678 ms | 0.2236 ms | 56.7x |
| nearest 4K | 53.057 ms | 0.8079 ms | 65.7x |
| area VGA | 4.962 ms | 0.5084 ms | 9.76x |
| area 1080p | 43.661 ms | 2.776 ms | 15.7x |
| area 4K | 137.42 ms | 10.813 ms | 12.7x |

These measurements do not claim an OpenCV advantage. They establish improvement
against SpatialRust's generic implementation for the named packed, half-scale,
caller-owned workload only.
