# 145E AI Semantic Overlay

Date: 2026-08-03

## Scope

This slice adds a portable, source-bound AI semantic overlay on top of the
existing bounded 143A E2E map. It intentionally keeps the model path light:
`spatialrust-ai::MockProfile::SemanticClasses` emits deterministic class IDs
and confidence from explicit CPU features. ONNX/CUDA dependencies are not
introduced into the ROS2 production dependency path.

The viewer contract stores quantized micrometre coordinates, confidence
millionths, class colors/statistics, model transfer accounting, source/frame
identity, and independent overlay/mapping admission flags. The semantic
adapter rejects a blocked state before producing renderer geometry.

## External evidence

Canonical source:

- `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`
- SHA-256: `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`

Positive run:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145e-ai-semantic-overlay/semantic-overlay.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145e-ai-semantic-overlay/semantic-overlay.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145e-ai-semantic-overlay/semantic-overlay.manifest.json`
- `overlay_ready=true`, `mapping_admitted=false`
- 1,064,304 input vertices, 4,094 sampled predictions, 3 classes
- model bytes: 65,504 host input, 32,752 host output, 0 device upload/readback

Negative probe:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145e-ai-semantic-overlay-validation-probe/semantic-overlay.json`
- wrong expected SHA; model runtime was not run, prediction count was zero,
  `overlay_ready=false`, and the CLI exited with status 2.

## Verification

Focused checks passed:

- `cargo test -p spatialrust-ai`
- `cargo test -p spatialrust-viewer --features semantic,serde`
- `cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_semantic_overlay`
- `cargo clippy -p spatialrust-ai --all-targets -- -D warnings`
- `cargo clippy -p spatialrust-viewer --features semantic,serde --all-targets -- -D warnings`
- `cargo clippy -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_semantic_overlay -- -D warnings`

The source/frame/calibration gates remain explicit: an attractive overlay is
inspection output only until source-bound clock/TF calibration and a
production model receipt are supplied.
