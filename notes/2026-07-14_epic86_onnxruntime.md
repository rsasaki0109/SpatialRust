# Epic 86: inference contracts and ONNX Runtime CPU

Date: 2026-07-14

## Delivered contract

- `spatialrust-ai` has no runtime in its default build. It defines model source,
  named fixed/dynamic tensor specifications, session options, per-run copy
  policy, output destinations, and backend/session traits.
- `ai-onnxruntime` selects the CPU execution provider explicitly. CUDA,
  TensorRT, and DirectML are separate additive features and compile alone.
- Standard runs require both input and output copy permission. Bound runs accept
  compact typed storage and fail with `CopyRequired` instead of casting an
  under-aligned byte allocation.
- `TensorBuffer` can retain a runtime-owned host allocation behind
  `HostTensorStorage`. ONNX Runtime outputs therefore become another bound input
  without a host copy or a dependency from `spatialrust-tensor` to `ort`.
- Dynamic outputs can be runtime allocated. Compact u8/u16/f32 outputs can be
  preallocated by the caller; the f32 test proves the result pointer is exactly
  the pointer supplied before inference.
- The feature-gated Python `OnnxRuntimeSession` exposes named metadata and uses
  I/O Binding by default. `copy=True` is the explicit conversion path.

## Correctness evidence

- Rust ONNX tests cover named symbolic dimensions, standard copy refusal,
  f32 dynamic inference, u8/u16 typed binding, caller-preallocated pointer
  identity, runtime-owned output retention, output-to-input chaining, and
  rejection of raw f32 byte storage.
- The embedded dynamic Add model produces the same 4x3 float32 result through
  Rust binding, Rust copy mode, Python binding, and Python ONNX Runtime 1.24.4.
- `spatialrust-tensor --all-features`: 14 unit tests and 2 property tests pass.
- `spatialrust-ai --features onnxruntime`: 8 tests pass; all-target Clippy is
  clean with warnings denied.
- The Python extension builds both without and with `onnxruntime`; 62 pytest
  tests and `mypy.stubtest --ignore-missing-stub` pass for the feature build.

## CPU binding benchmark

Command:

```text
cargo bench -p spatialrust-ai --features onnxruntime --bench onnxruntime
```

Environment: Intel Core i7-9750H, Windows 11 Pro Insider Preview, rustc 1.92.0,
ONNX Runtime 1.24 through `ort` 2.0.0-rc.12. The model adds a dynamic `[pixels,
3]` float32 tensor to itself. Throughput counts one input and one output.

| Size | Explicit copy run median | I/O Binding median | Ratio |
| --- | ---: | ---: | ---: |
| 640x480 | 3.134 ms | 0.616 ms | 5.09x |
| 1920x1080 | 21.386 ms | 3.914 ms | 5.46x |
| 3840x2160 | 88.937 ms | 17.247 ms | 5.16x |

These are local measurements, not portable performance guarantees. The binding
benchmark still includes creation of SpatialRust and ONNX Runtime binding
objects per iteration; it isolates avoided tensor repacking and output copying,
not model kernel speedups.

## Toolchain note

`ort` and `ort-sys` 2.0.0-rc.12 declare Rust 1.88. This is a feature-specific
MSRV: `spatialrust-ai` without `onnxruntime`, the meta `ai` feature, and the
workspace default remain free of that dependency and retain the workspace MSRV.
