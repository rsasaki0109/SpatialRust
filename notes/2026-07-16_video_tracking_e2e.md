# Rust/Python video tracking E2E demo — 2026-07-16

## Scope

This final Vision 2 demo exercises one reproducible pipeline in both language
surfaces:

`generate PGM sequence → load frames → dense optical flow → threshold components → native IoU tracking → GIF`

The Rust example is
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust\examples\video_tracking_e2e.rs`.
The Python example and GIF renderer is
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-py\examples\video_tracking_e2e.py`.
The committed output is
`C:\Users\rsasa\Workspace\SpatialRust\docs\assets\video_tracking_e2e.gif`.

## Reproducible input

Both examples independently generate 12 Gray8 PGM frames at 96×72. The
background and both object textures are deterministic integer formulas. Object
class 1 translates by `(+2,+1)` pixels per frame; class 2 translates by
`(-2,-1)`.

The Rust output under `target/video-tracking-demo/frames` and Python output
under `target/video-tracking-demo/python-frames` contained 12 files each. The
ordered SHA-256 sequences were identical for all 12 files.

No external dataset, network download, codec runtime, or random seed is needed.

## Pipeline acceptance

- SpatialRust bounded image IO reloads every generated PGM as Gray8.
- `dense_flow_block_match` uses block radius 1 and search radius 3.
- Every one of the 11 frame pairs reports exact object-center vectors
  `(class 1, +2, +1)` and `(class 2, -2, -1)`.
- Thresholded connected components produce exactly two detections per frame.
- `MultiObjectTracker` preserves IDs 1 and 2 through the complete sequence.
- The Python binding exposes the same stateful native tracker and preserves
  integer IDs/classes plus float boxes/scores in typed track tuples.
- The committed GIF is 384×288, 12 frames, and 140 ms per frame.

## Verification

```powershell
$env:PATH = "C:\Users\rsasa\.cargo\bin;$env:PATH"
cargo run -p spatialrust --no-default-features --features image-io-standard,vision-video --example video_tracking_e2e
cargo check --manifest-path crates/spatialrust-py/Cargo.toml
maturin develop --release --manifest-path crates/spatialrust-py/Cargo.toml
.venv/Scripts/python.exe -m pytest crates/spatialrust-py/tests/test_bindings.py -k "dense_flow or multi_object_tracker" -q
.venv/Scripts/python.exe -m mypy.stubtest spatialrust --ignore-missing-stub
.venv/Scripts/python.exe crates/spatialrust-py/examples/video_tracking_e2e.py --frames-dir target/video-tracking-demo/python-frames --gif docs/assets/video_tracking_e2e.gif
```

Observed results: the Rust and Python E2E receipts both reported 12 frames, 11
flow pairs, and stable track IDs 1/2; the two focused Python tests passed; and
stubtest reported no issues.

CI runs the Python example with `--no-gif` on Python 3.8 and 3.12, and runs the
Rust example in the Linux/Windows/macOS Vision conformance matrix. GIF rendering
remains an explicit documentation step requiring Pillow.

The standalone `cargo clippy --manifest-path crates/spatialrust-py/Cargo.toml
-- -D warnings` command still encounters the repository's pre-existing
thread-local and morphology argument-count warnings. This slice adds no new
clippy warning.
