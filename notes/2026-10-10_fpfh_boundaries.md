# FPFH/RANSAC boundaries

Native configuration now rejects nonfinite/zero/overflowing distance squares,
zero budgets, too-small/overflowing sample sizes and invalid edge tolerances.
Alignment requires enough points for the sample before descriptor work. FPFH
positions are finite, coordinate extent must have representable f32 squared
distances, and normals must be finite unit vectors (squared-norm tolerance 1e-3).
Previously, nonunit normals silently violated the Darboux angular assumptions;
they now return an explicit error. The seed equal to the RNG xor mask previously
froze the generator at zero; a nonzero fallback fixes that case.

Python `register_fpfh_ransac` retains positional compatibility and adds a
keyword-only seed with unchanged default. It validates configuration before
normal estimation, rejects nonfinite XYZ and fewer than three points, rejects
k<3, caps k scratch per cloud, and releases the GIL for the full native pipeline.
No new heavyweight runtime dependency. Native no-hypothesis behavior remains
identity/infinite fitness/not converged; callers must assess it before refinement.

Verified with 29 Rust CPU-registration tests, 334 Python tests, strict extension
and registration Clippy, and stubtest. The GIL regression prevents ordinary
bytecode switching and checks another Python thread executes during RANSAC.
Deterministic seeds, huge requested k, invalid distances and XYZ are exercised.
The ONNX-enabled release wheel was rebuilt locally; loading requires the
existing local ONNX library path because patchelf is unavailable here.

This does not establish global accuracy on real sensors or scale-invariant
FPFH neighborhoods. Estimated normal orientation remains dependent on its
viewpoint. The keypoint Python wrapper still has separate normal estimation
and GIL boundaries to improve. Maturity 59% remains provisional, target 90%.
