# Epic 84D image analysis

Date: 2026-07-14

## Artifacts

- Rust: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\analysis.rs`
- Python: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-py`
- Benchmark: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\benches\analysis.rs`
- OpenCV comparison: `C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_vision_comparison\run.py`

## Results

Against OpenCV 5.0, fixed threshold, Otsu threshold and pixels, mean/Gaussian
adaptive thresholds, 256-bin histogram, histogram equalization, and float64
integral image matched exactly. CLAHE differed by at most one uint8 level due to
floating interpolation order.

Adaptive thresholding intentionally rounds its local u8 statistic before the
comparison and applies `ceil(C)` for Binary or `floor(C)` for BinaryInv, matching
OpenCV's integer boundary contract. Integral tables include a zero top row and
left column, and rectangle queries use half-open source coordinates.

From `C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
& "$env:USERPROFILE\.cargo\bin\cargo.exe" test -p spatialrust-vision --features full
& "$env:USERPROFILE\.cargo\bin\cargo.exe" clippy -p spatialrust-vision --features full --all-targets --no-deps -- -D warnings
& "$env:USERPROFILE\.cargo\bin\cargo.exe" check -p spatialrust --no-default-features --features vision-imgproc-analysis
& "$env:USERPROFILE\.cargo\bin\cargo.exe" bench -p spatialrust-vision --bench analysis --features imgproc-analysis
python bench\opencv_vision_comparison\run.py
```
