# GPU sparse radius grid (Epic 80 foundation)

## Outcome

`build_radius_grid_aoso_gpu` builds radius-sized uniform-grid cells directly
from a retained `GpuAoSoXyzBuffer`. Key assignment, deterministic sorting, and
segment compaction stay on the GPU and return `GpuAoSoRadiusGrid`.

The grid uses sparse sorted `(ix, iy, iz)` keys rather than allocating a dense
array from world bounds. A zero origin with floor division supports negative
coordinates without a CPU bounds pass. Original `point_index` order is retained
inside equal keys.

The test covers negative cells, points sharing a cell across original chunk
boundaries, a partial final chunk, and segment index ordering.

`estimate_normals_radius_grid_aoso_gpu` binary-searches the sparse sorted cell
keys for each of the 27 adjacent cells. Radius filtering, covariance, Jacobi
eigendecomposition, normals, and curvature all remain on the GPU. No CPU
neighbor list is created or uploaded.

## Files

- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/lib.rs`

## Validation

- Sparse AoSoA results match the established SoA GPU radius-grid path on a
  negative-coordinate planar patch split across multiple chunks.
- The sparse representation avoids world-extent cell caps; runtime remains
  sensitive to genuinely dense cells because every radius neighbor is visited.
