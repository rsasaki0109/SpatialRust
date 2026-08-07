# Epic 150: real AI semantic meaning via ONNX entity embeddings

Date: 2026-08-07. Slices 150A–150C complete.

## Why

Epic 95 established semantic entities and embeddings with a deterministic mock
profile. Epic 150 connects real ONNX inference to spatial semantic entities so
point-cloud features become searchable embeddings without leaving the SpatialRust
data model.

## What was built

- `crates/spatialrust-semantic/src/model.rs` (`model` feature) — `OnnxEntityEmbedder`:
  - consumes an already-open `&mut dyn ModelSession` (backend/device chosen by
    the caller, never internally);
  - validates feature count against the input descriptor and finiteness;
  - runs with explicit `CopyPolicy` for input and output;
  - verifies the model output shape and bytes, then builds an `Embedding`.
  Nine unit tests cover identity round trips, feature-count mismatch,
  non-finite rejection, non-CPU rejection, and zero-dim output rejection.
- Facade feature `semantic-model` (`semantic` + `spatialrust-semantic/model` +
  `ai` + `tensor`) and `tests/onnx_semantic.rs` integration test:
  - committed fixture `crates/spatialrust/tests/fixtures/double_dynamic.onnx`
    (a 134-byte Add model mapping `input` [1,3] → `output` [1,3], doubling);
  - real ONNX Runtime CPU session embeds `[1,2,3]` → `[2,4,6]` exactly;
  - the embedding is inserted into `SemanticSearchIndex` and found by a
    query embedding, proving the full feature → model → search path.
- Python binding (150C): `PyOnnxEntityEmbedder` takes an existing
  `OnnxRuntimeSession`, input/output names and shapes, and `copy` policy;
  `embed()` returns a NumPy embedding plus its dimension. Gated by the
  `onnxruntime` wheel feature; a `double_dynamic.onnx`-style Python test
  doubles `[1,2,3]` → `[2,4,6]`.

## Contract notes

- The embedder never loads a model, selects a backend, or performs a hidden
  device transfer; copy permission is an explicit per-run choice.
- Heavy runtimes remain behind `ai-onnxruntime`; `semantic` default build and
  `spatialrust-semantic` default (no `model`) stay dependency-light.
- The committed fixture encodes no learned weights; it only proves the wiring.

## Verification

- `cargo test -p spatialrust-semantic --features model` — 9 tests.
- `cargo test -p spatialrust --features "ai-onnxruntime semantic-model" --test onnx_semantic` — 2 tests.
- clippy `-D warnings` clean for semantic and the facade test; `cargo fmt` clean.
- `cargo test --workspace` passes.

## Next slices

Epic 150 is complete. A follow-up note can document using a real public
open-vocabulary embedding model (e.g. CLIP image/text) with this surface; the
wiring and tests already cover the full feature → model → embedding → search
path in both Rust and Python.
