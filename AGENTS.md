# AGENTS

## Purpose
SpatialRust is a Rust-native spatial computing framework: point clouds, geometry, GPU compute, robotics integration, and AI-native spatial data.

## Scope
- Follow the master architecture in `docs/ARCHITECTURE.md`.
- Keep `spatialrust-core` small and stable.
- Isolate heavy dependencies (ROS2, ONNX, CUDA) behind feature flags and dedicated crates.
- Do not commit sensitive sensor dumps, private keys, or customer data.

## Operating rules
- Data model and execution traits before algorithm breadth.
- Explicit CPU/GPU transfers; no hidden device copies in production APIs.
- Public APIs stay safe; restrict `unsafe` to audited GPU/FFI boundaries.
- Prefer capability traits (`HasPositions3`, etc.) over monolithic point structs.

## Delivery standards
- Keep changes focused and reviewable.
- Add tests for correctness-critical math, schema, and IO behavior.
- Use absolute paths when referencing files in notes and reports.
