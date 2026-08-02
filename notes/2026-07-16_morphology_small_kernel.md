# Epic 117E centered 5×5 morphology receipt (2026-07-16)

## Outcome

Centered 5×5 grayscale `u8` rectangular morphology with Replicate borders no
longer enters the large-window prefix/suffix engine or transposes two
full-image planes per stage. It uses fixed five-sample extrema, direct
row-major vertical passes, safe runtime SIMD dispatch, and bounded Rayon row
blocks. Other sizes, anchors, borders, masks, and component types retain their
existing exact paths.

## OpenCV comparison

Windows 11, Intel Family 6 Model 158, 6 cores / 12 logical CPUs, CPython
3.12.10, OpenCV 4.13.0, OpenCL disabled, and 12 OpenCV threads. Timings are
paired/interleaved public Python calls over seeded packed random grayscale
`u8` inputs.

| Profile | Allocate result | Caller-output result |
| --- | ---: | ---: |
| VGA | OpenCV 4.51× | OpenCV 1.90× |
| 1080p | OpenCV 1.98× | **SpatialRust 1.22×** |
| 4K | OpenCV 2.30× | OpenCV 1.50× |

The prior table showed OpenCV leads of 60.96×/13.34×/15.27× in allocate mode
and 60.32×/16.25×/17.78× in caller-output mode. This slice reduces those gaps
by 6.6×–31.8× and crosses OpenCV on the scoped 1080p reuse workload. It is not
a blanket morphology superiority claim.

## Validation

- 980 randomized OpenCV operation cases remain bit-exact
- Rust anchor/border/iteration/stride and workspace-capacity tests pass
- feature Clippy passes with warnings denied
- native Criterion covers allocate/reuse at VGA, 1080p, and 4K
- reproducible report:
  `C:\Users\rsasa\Workspace\SpatialRust\target\opencv-morphology-small-performance.json`
