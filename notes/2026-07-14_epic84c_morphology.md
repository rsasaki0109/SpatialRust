# Epic 84C CPU morphology

Date: 2026-07-14

## Artifacts

- Rust implementation: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\morphology.rs`
- Python binding/stub: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-py`
- Criterion benchmark: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\benches\morphology.rs`
- Numerical harness: `C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_vision_comparison\run.py`

## Contract and verification

Every operation consumes a CPU `ImageView`, including a strided subview, and
returns a newly owned CPU image. Borders and iteration counts are explicit.
Structuring elements validate dimensions, anchor, mask length, and at least one
active sample before execution.

The OpenCV 5.0 harness compared erode, dilate, open, close, gradient, top-hat,
and black-hat with rectangular, cross, and elliptical 5×3 elements at two
iterations. All 21 cases had maximum uint8 error 0. Diamond and arbitrary masks
have Rust correctness coverage because diamond availability varies by older
OpenCV versions.

From `C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
& "$env:USERPROFILE\.cargo\bin\cargo.exe" test -p spatialrust-vision --features full
& "$env:USERPROFILE\.cargo\bin\cargo.exe" clippy -p spatialrust-vision --features full --all-targets --no-deps -- -D warnings
& "$env:USERPROFILE\.cargo\bin\cargo.exe" check -p spatialrust --no-default-features --features vision-imgproc-morphology
& "$env:USERPROFILE\.cargo\bin\cargo.exe" bench -p spatialrust-vision --bench morphology --features imgproc-morphology
python bench\opencv_vision_comparison\run.py
```
