# Epic 90 AI vision pipeline completion record

Date: 2026-07-15 (Asia/Tokyo)

## Delivered contracts

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-ai\src\mock.rs`
  provides `MockInferenceBackend` and `MockProfile::SyntheticDepth`, selected
  through `ModelSource::Mock`, with explicit output-copy permissions.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\adapters.rs`
  (feature `ai-adapters`) letterboxes RGB into contiguous NCHW and decodes
  depth / score / detection tensors into dense vision types without depending
  on `spatialrust-ai`.
- Facade flag: `ai-vision-pipeline` = `ai` + `vision-ai-adapters` + `vision-spatial`.
- Integration test:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust\tests\vision_ai_pipeline.rs`
  runs RGB → mock depth → unproject → MVP without enabling `onnxruntime`.

## Verification

- `cargo test -p spatialrust-ai --lib`
- `cargo test -p spatialrust-vision --features ai-adapters --lib`
- `cargo test -p spatialrust --features ai-vision-pipeline,mvp --test vision_ai_pipeline`

## Notes

ONNX adapters remain opt-in. Real model weight packs are out of scope; mock
profiles are the correctness and demo path for Epic 90.
