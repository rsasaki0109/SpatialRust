# Epic 84A shared linear filters

Date: 2026-07-14

## Artifacts

- Rust implementation: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\filter.rs`
- Shared border contract: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\border.rs`
- Criterion benchmark: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\benches\filter.rs`
- OpenCV comparison: `C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_vision_comparison\run.py`
- Python API and stubs: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-py`

## Contract

`filter2d` performs correlation, matching OpenCV. `convolve2d` reverses both the
coefficients and anchor explicitly. Integer outputs round and saturate;
`filter2d_f32` and `separable_filter_f32` retain signed and fractional results.
All neighborhood operations take `BorderMode` explicitly and accept packed or
strided `ImageView` input without copying it first.

## Reproduction

From `C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
& "$env:USERPROFILE\.cargo\bin\cargo.exe" test -p spatialrust-vision --features full
& "$env:USERPROFILE\.cargo\bin\cargo.exe" clippy -p spatialrust-vision --features full --all-targets --no-deps -- -D warnings
& "$env:USERPROFILE\.cargo\bin\cargo.exe" check -p spatialrust --no-default-features --features vision-imgproc-filter
& "$env:USERPROFILE\.cargo\bin\cargo.exe" bench -p spatialrust-vision --bench filter --features imgproc-filter
python bench\opencv_vision_comparison\run.py
```

The deterministic OpenCV 5.0 comparison measured maximum absolute uint8 error
of 1 for both filter2D and Gaussian blur. The Criterion suite covers a 5×5 RGB8
Gaussian blur at 640×480, 1920×1080, and 3840×2160.
