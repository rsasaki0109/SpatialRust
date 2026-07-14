# Epic 84B advanced CPU filters

Date: 2026-07-14

## Artifacts

- Implementation: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\advanced_filter.rs`
- Python surface: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-py\src\lib.rs`
- OpenCV harness: `C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_vision_comparison\run.py`
- Benchmarks: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\benches\filter.rs`

## Contracts

- Median filtering orders each channel independently and requires a positive
  odd aperture.
- Bilateral filtering uses a circular spatial neighborhood and the squared sum
  of absolute channel differences used by OpenCV's CPU path.
- Sobel, Scharr, and Laplacian return signed `f32` images, avoiding silent
  saturation of negative gradients.
- Pyramid operations use the canonical `[1, 4, 6, 4, 1]` kernel. `pyr_down`
  uses ceil-halving and `pyr_up` doubles each dimension.
- Every operation reads `ImageView` directly, including strided subviews, and
  creates an explicit CPU output image without hidden device transfers.

## Verification

OpenCV 5.0 deterministic comparison results:

| Operation | Maximum error |
| --- | ---: |
| median blur | 0 uint8 |
| bilateral filter | 0 uint8 |
| Sobel 5×5 | 0 f32 |
| Scharr 3×3 | 0 f32 |
| Laplacian 3×3 | 0 f32 |
| pyrDown | 0 uint8 |
| pyrUp | 0 uint8 |

Reproduce from `C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
& "$env:USERPROFILE\.cargo\bin\cargo.exe" test -p spatialrust-vision --features full
& "$env:USERPROFILE\.cargo\bin\cargo.exe" clippy -p spatialrust-vision --features full --all-targets --no-deps -- -D warnings
& "$env:USERPROFILE\.cargo\bin\cargo.exe" bench -p spatialrust-vision --bench filter --features imgproc-filter
python bench\opencv_vision_comparison\run.py
```
