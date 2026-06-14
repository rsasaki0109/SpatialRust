"""Type stubs for the SpatialRust native extension.

These mirror the `#[pyfunction]` / `#[pyclass]` surface in `src/lib.rs` so that
editors and type checkers (mypy, pyright) understand the compiled module. Kept
in sync with the bindings by the pytest suite in `tests/`.
"""

from typing import Optional, Sequence, final

import numpy as np
from numpy.typing import NDArray

__version__: str

# Convenient aliases for the array shapes the bindings exchange.
_F32Array = NDArray[np.float32]  # positions, grids, range images, transforms
_I32Array = NDArray[np.int32]  # labels, edge_index
_Vec3 = tuple[float, float, float]

@final
class PointCloud:
    """A schema-aware point cloud backed by native Rust storage."""

    @staticmethod
    def from_xyz(points: _F32Array) -> PointCloud:
        """Builds a cloud from an (N, 3) float32 array of XYZ coordinates."""
        ...

    def xyz(self) -> _F32Array:
        """Returns the XYZ coordinates as an (N, 3) float32 array."""
        ...

    def labels(self) -> Optional[_I32Array]:
        """Per-point cluster labels as an (N,) int32 array, or None if unlabeled."""
        ...

    def field_names(self) -> list[str]:
        """Field names present in the cloud schema."""
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class PipelineResult:
    """Result of running the MVP pipeline."""

    @property
    def output(self) -> PointCloud: ...
    @property
    def downsampled(self) -> PointCloud: ...
    @property
    def cluster_count(self) -> int: ...
    @property
    def cluster_sizes(self) -> list[int]: ...
    @property
    def plane_inliers(self) -> int: ...
    @property
    def plane_normal(self) -> _Vec3: ...
    def labels(self) -> Optional[_I32Array]: ...
    def __repr__(self) -> str: ...

@final
class RegionResult:
    """Result of region growing segmentation."""

    @property
    def output(self) -> PointCloud: ...
    @property
    def cluster_count(self) -> int: ...
    @property
    def cluster_sizes(self) -> list[int]: ...
    def labels(self) -> Optional[_I32Array]: ...
    def __repr__(self) -> str: ...

@final
class DbscanResult:
    """Result of DBSCAN density-based clustering."""

    @property
    def output(self) -> PointCloud: ...
    @property
    def cluster_count(self) -> int: ...
    @property
    def cluster_sizes(self) -> list[int]: ...
    @property
    def noise_count(self) -> int: ...
    def labels(self) -> Optional[_I32Array]: ...
    def __repr__(self) -> str: ...

@final
class GroundResult:
    """Result of ground segmentation."""

    @property
    def ground(self) -> PointCloud: ...
    @property
    def non_ground(self) -> PointCloud: ...
    @property
    def ground_count(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class MultiPlaneResult:
    """Result of multi-plane segmentation."""

    @property
    def output(self) -> PointCloud: ...
    @property
    def plane_count(self) -> int: ...
    @property
    def plane_sizes(self) -> list[int]: ...
    @property
    def planes(self) -> list[tuple[float, float, float, float]]:
        """Each plane as (nx, ny, nz, d) in Hessian form n·p + d = 0."""
        ...

    def labels(self) -> Optional[_I32Array]: ...
    def __repr__(self) -> str: ...

@final
class SphereResult:
    """Result of fitting a RANSAC sphere."""

    @property
    def center(self) -> _Vec3: ...
    @property
    def radius(self) -> float: ...
    @property
    def inliers(self) -> PointCloud: ...
    @property
    def outliers(self) -> PointCloud: ...
    def __repr__(self) -> str: ...

@final
class CylinderResult:
    """Result of fitting a RANSAC cylinder."""

    @property
    def axis_point(self) -> _Vec3: ...
    @property
    def axis_direction(self) -> _Vec3: ...
    @property
    def radius(self) -> float: ...
    @property
    def inliers(self) -> PointCloud: ...
    @property
    def outliers(self) -> PointCloud: ...
    def __repr__(self) -> str: ...

@final
class RegistrationResult:
    """Result of a registration (alignment) run."""

    @property
    def fitness(self) -> float: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    def transform(self) -> _F32Array:
        """The 4x4 transform mapping source into the target frame."""
        ...

    def __repr__(self) -> str: ...

# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def read(path: str) -> PointCloud: ...
def write(path: str, cloud: PointCloud) -> None: ...

# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def voxel_downsample(
    cloud: PointCloud, leaf_size: float, policy: str = ...
) -> PointCloud: ...
def crop_box(
    cloud: PointCloud, min: _Vec3, max: _Vec3, invert: bool = ...
) -> PointCloud: ...
def pass_through(
    cloud: PointCloud, field: str, min: float, max: float, invert: bool = ...
) -> PointCloud: ...
def farthest_point_sampling(
    cloud: PointCloud, sample_size: int, seed_index: int = ...
) -> PointCloud: ...
def mls_smooth(
    cloud: PointCloud,
    search_radius: float = ...,
    polynomial_order: int = ...,
    min_neighbors: int = ...,
) -> PointCloud: ...
def statistical_outlier_removal(
    cloud: PointCloud, k_neighbors: int = ..., std_mul: float = ...
) -> PointCloud: ...
def radius_outlier_removal(
    cloud: PointCloud, radius: float = ..., min_neighbors: int = ...
) -> PointCloud: ...

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def iss_keypoints(
    cloud: PointCloud,
    salient_radius: float = ...,
    non_max_radius: float = ...,
    gamma_21: float = ...,
    gamma_32: float = ...,
    min_neighbors: int = ...,
) -> PointCloud: ...
def orient_normals(cloud: PointCloud, k_neighbors: int = ...) -> PointCloud: ...
def detect_boundary(
    cloud: PointCloud,
    search_radius: float = ...,
    angle_threshold: float = ...,
    min_neighbors: int = ...,
    k_neighbors: int = ...,
) -> PointCloud: ...

# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def dbscan(
    cloud: PointCloud, eps: float = ..., min_points: int = ...
) -> DbscanResult: ...
def segment_multi_plane(
    cloud: PointCloud,
    max_planes: int = ...,
    distance_threshold: float = ...,
    min_inliers: int = ...,
    max_iterations: int = ...,
) -> MultiPlaneResult: ...
def ground_segmentation(
    cloud: PointCloud,
    cell_size: float = ...,
    height_threshold: float = ...,
    erosion_cells: int = ...,
) -> GroundResult: ...
def ransac_sphere(
    cloud: PointCloud,
    distance_threshold: float = ...,
    max_iterations: int = ...,
    min_inliers: int = ...,
) -> SphereResult: ...
def ransac_cylinder(
    cloud: PointCloud,
    distance_threshold: float = ...,
    max_iterations: int = ...,
    min_inliers: int = ...,
    k_neighbors: int = ...,
) -> CylinderResult: ...
def region_growing(
    cloud: PointCloud,
    k_neighbors: int = ...,
    smoothness_deg: float = ...,
    min_region_size: int = ...,
) -> RegionResult: ...

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def chamfer_distance(a: PointCloud, b: PointCloud) -> float: ...
def hausdorff_distance(a: PointCloud, b: PointCloud) -> float: ...

# --------------------------------------------------------------------------- #
# Transform utilities
# --------------------------------------------------------------------------- #
def apply_transform(cloud: PointCloud, matrix: _F32Array) -> PointCloud: ...
def recenter(cloud: PointCloud) -> PointCloud: ...
def scale(cloud: PointCloud, factor: float) -> PointCloud: ...
def normalize_unit_sphere(cloud: PointCloud) -> PointCloud: ...
def merge(clouds: Sequence[PointCloud]) -> PointCloud: ...
def centroid(cloud: PointCloud) -> _Vec3: ...
def bounding_box(cloud: PointCloud) -> tuple[_Vec3, _Vec3]: ...
def oriented_bounding_box(
    cloud: PointCloud,
) -> tuple[_Vec3, _Vec3, list[_Vec3]]: ...

# --------------------------------------------------------------------------- #
# ML front-ends (voxel grids / range images / graphs)
# --------------------------------------------------------------------------- #
def voxelize(
    cloud: PointCloud, voxel_size: float = ..., mode: str = ...
) -> tuple[NDArray[np.float32], _Vec3, float]:
    """Dense (nz, ny, nx) grid plus (origin_xyz, voxel_size)."""
    ...

def range_image(
    cloud: PointCloud,
    width: int = ...,
    height: int = ...,
    fov_up_deg: float = ...,
    fov_down_deg: float = ...,
) -> _F32Array: ...
def knn_graph(cloud: PointCloud, k: int) -> _I32Array:
    """Directed k-NN graph as a (2, E) int32 edge_index (PyG convention)."""
    ...

def radius_graph(cloud: PointCloud, radius: float) -> _I32Array:
    """Directed radius graph as a (2, E) int32 edge_index (PyG convention)."""
    ...

# --------------------------------------------------------------------------- #
# Pipeline & registration
# --------------------------------------------------------------------------- #
def run_pipeline(
    cloud: PointCloud,
    leaf_size: float = ...,
    cluster_tolerance: Optional[float] = ...,
    min_cluster_size: Optional[int] = ...,
    plane_distance: Optional[float] = ...,
    policy: str = ...,
) -> PipelineResult: ...
def register_icp(
    source: PointCloud,
    target: PointCloud,
    max_correspondence_distance: float = ...,
    max_iterations: int = ...,
) -> RegistrationResult: ...
def register_point_to_plane(
    source: PointCloud,
    target: PointCloud,
    max_correspondence_distance: float = ...,
    max_iterations: int = ...,
    k_neighbors: int = ...,
) -> RegistrationResult: ...
def register_gicp(
    source: PointCloud,
    target: PointCloud,
    max_correspondence_distance: float = ...,
    max_iterations: int = ...,
    k_neighbors: int = ...,
) -> RegistrationResult: ...
def register_ndt(
    source: PointCloud,
    target: PointCloud,
    resolution: float = ...,
    max_iterations: int = ...,
) -> RegistrationResult: ...
def register_fpfh_ransac(
    source: PointCloud,
    target: PointCloud,
    feature_radius: float = ...,
    max_correspondence_distance: float = ...,
    ransac_iterations: int = ...,
    k_neighbors: int = ...,
) -> RegistrationResult: ...
def register_fpfh_keypoints(
    source: PointCloud,
    target: PointCloud,
    salient_radius: float = ...,
    feature_radius: float = ...,
    max_correspondence_distance: float = ...,
    ransac_iterations: int = ...,
    k_neighbors: int = ...,
) -> RegistrationResult: ...
