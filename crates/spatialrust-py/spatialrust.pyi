"""Type stubs for the SpatialRust native extension.

These mirror the `#[pyfunction]` / `#[pyclass]` surface in `src/lib.rs` so that
editors and type checkers (mypy, pyright) understand the compiled module. Kept
in sync with the bindings by the pytest suite in `tests/`.
"""

from typing import Optional, Sequence, final

import numpy as np
from numpy.typing import NDArray

__version__: str
__all__: list[str] = [
    "__version__", "ImageMetadata", "Tensor", "Keypoint2", "OnnxRuntimeSession",
    "DLPackTensorView", "PointCloud", "PipelineResult", "RegionResult",
    "DbscanResult", "GroundResult", "MultiPlaneResult", "SphereResult",
    "CylinderResult", "RegistrationResult", "read_image",
    "tensor_copy_from_numpy", "tensor_view_from_dlpack", "harris_keypoints",
    "shi_tomasi_keypoints", "fast_keypoints", "orb_features",
    "estimate_homography_ransac", "solve_pnp", "estimate_rgbd_odometry", "stereo_block_match",
    "match_binary_descriptors", "match_float_descriptors", "write_image", "read",
    "write", "voxel_downsample", "crop_box", "pass_through", "iss_keypoints",
    "orient_normals", "detect_boundary", "mls_smooth", "farthest_point_sampling",
    "statistical_outlier_removal", "radius_outlier_removal", "run_pipeline",
    "region_growing", "dbscan", "ground_segmentation", "segment_multi_plane",
    "ransac_sphere", "ransac_cylinder", "chamfer_distance", "hausdorff_distance",
    "apply_transform", "recenter", "scale", "normalize_unit_sphere", "merge",
    "centroid", "bounding_box", "oriented_bounding_box", "voxelize", "range_image",
    "rgbd_to_point_cloud", "depth_to_xyz", "calibrate_pinhole_camera",
    "calibrate_fisheye_angles", "dense_flow_image", "filter2d_image", "gaussian_blur_image",
    "median_blur_image", "bilateral_filter_image", "sobel_image", "scharr_image",
    "laplacian_image", "pyr_down_image", "pyr_up_image", "morphology_image",
    "threshold_image", "otsu_threshold_image", "adaptive_threshold_image",
    "histogram_image", "equalize_histogram_image", "clahe_image",
    "integral_image_u8", "canny_image", "resize_image", "letterbox_image",
    "normalize_image_chw", "rgb_to_gray_image", "rgb_to_hsv_image", "remap_image",
    "nms", "soft_nms", "connected_components_image", "find_mask_contours",
    "encode_mask_rle", "decode_mask_rle", "point_map_to_point_cloud", "knn_graph",
    "radius_graph", "register_icp", "register_point_to_plane", "register_gicp",
    "register_ndt", "register_fpfh_ransac", "register_fpfh_keypoints",
]

# Convenient aliases for the array shapes the bindings exchange.
_F32Array = NDArray[np.float32]  # positions, grids, range images, transforms
_F64Array = NDArray[np.float64]
_BoolArray = NDArray[np.bool_]
_I32Array = NDArray[np.int32]  # labels, edge_index
_U32Array = NDArray[np.uint32]
_Vec3 = tuple[float, float, float]
_U8Array = NDArray[np.uint8]
_U16Array = NDArray[np.uint16]

@final
class ImageMetadata:
    """Container, sample type, and Exif orientation from image decoding."""

    @property
    def format(self) -> str: ...
    @property
    def color_type(self) -> str: ...
    @property
    def orientation(self) -> int: ...
    @property
    def orientation_applied(self) -> bool: ...
    def __repr__(self) -> str: ...

def read_image(
    path: str, apply_orientation: bool = ...
) -> tuple[NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32], ImageMetadata]: ...
def write_image(
    path: str,
    image: NDArray[np.uint8] | NDArray[np.uint16],
    format: str,
    jpeg_quality: int = ...,
) -> None: ...

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

def depth_to_xyz(
    depth: _F32Array,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = ...,
    min_depth: float = ...,
    max_depth: float = ...,
    distortion: Optional[tuple[float, float, float, float, float]] = ...,
    out: Optional[_F32Array] = ...,
) -> _F32Array:
    """Convert depth to a dense ``(H, W, 3)`` XYZ image (invalid → NaN).

    If ``out`` is a contiguous ``(H, W, 3)`` float32 array it is filled in place.
    """
    ...

def calibrate_pinhole_camera(
    camera_points: _F64Array,
    pixels: _F64Array,
    width: int,
    height: int,
    huber_delta: float = ...,
    max_iterations: int = ...,
) -> tuple[float, float, float, float, float, float]: ...

def calibrate_fisheye_angles(
    theta: _F64Array, distorted_radius: _F64Array
) -> tuple[float, float, float, float, float]: ...
def dense_flow_image(
    previous: _U8Array,
    next: _U8Array,
    block_radius: int = ...,
    search_radius: int = ...,
) -> _F32Array: ...

def rgbd_to_point_cloud(
    depth: _F32Array,
    color: _U8Array,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = ...,
    min_depth: float = ...,
    max_depth: float = ...,
    distortion: Optional[tuple[float, float, float, float, float]] = ...,
) -> PointCloud:
    """Convert aligned depth and RGB images to an XYZRGB point cloud."""
    ...

# --------------------------------------------------------------------------- #
# Image preprocessing, detection, masks, and dense spatial data
# --------------------------------------------------------------------------- #
def filter2d_image(
    image: _U8Array, kernel: NDArray[np.float64], delta: float = ...
) -> _U8Array: ...
def gaussian_blur_image(
    image: _U8Array,
    kernel_width: int,
    kernel_height: int,
    sigma_x: float,
    sigma_y: Optional[float] = ...,
) -> _U8Array: ...
def median_blur_image(image: _U8Array, kernel_size: int) -> _U8Array: ...
def bilateral_filter_image(
    image: _U8Array,
    diameter: int,
    sigma_color: float,
    sigma_space: float,
) -> _U8Array: ...
def sobel_image(
    image: _U8Array,
    dx: int,
    dy: int,
    kernel_size: int = ...,
    scale: float = ...,
    delta: float = ...,
) -> _F32Array: ...
def scharr_image(
    image: _U8Array,
    dx: int,
    dy: int,
    scale: float = ...,
    delta: float = ...,
) -> _F32Array: ...
def laplacian_image(
    image: _U8Array,
    kernel_size: int = ...,
    scale: float = ...,
    delta: float = ...,
) -> _F32Array: ...
def pyr_down_image(image: _U8Array) -> _U8Array: ...
def pyr_up_image(image: _U8Array) -> _U8Array: ...
def morphology_image(
    image: _U8Array,
    operation: str,
    kernel_width: int,
    kernel_height: int,
    shape: str = ...,
    iterations: int = ...,
) -> _U8Array: ...
def threshold_image(
    image: _U8Array,
    threshold: float,
    max_value: int = ...,
    threshold_type: str = ...,
) -> _U8Array: ...
def otsu_threshold_image(
    image: _U8Array, max_value: int = ..., threshold_type: str = ...
) -> tuple[int, _U8Array]: ...
def adaptive_threshold_image(
    image: _U8Array,
    block_size: int,
    c: float,
    method: str = ...,
    max_value: int = ...,
    threshold_type: str = ...,
) -> _U8Array: ...
def histogram_image(image: _U8Array) -> NDArray[np.uint64]: ...
def equalize_histogram_image(image: _U8Array) -> _U8Array: ...
def clahe_image(
    image: _U8Array,
    clip_limit: float = ...,
    tiles_x: int = ...,
    tiles_y: int = ...,
) -> _U8Array: ...
def integral_image_u8(image: _U8Array) -> NDArray[np.float64]: ...
def canny_image(
    image: _U8Array,
    low_threshold: float,
    high_threshold: float,
    aperture_size: int = ...,
    l2_gradient: bool = ...,
) -> _U8Array: ...
def resize_image(
    image: _U8Array,
    width: int,
    height: int,
    interpolation: str = ...,
    out: Optional[_U8Array] = ...,
) -> _U8Array: ...
def letterbox_image(
    image: _U8Array,
    width: int,
    height: int,
    interpolation: str = ...,
    fill: Optional[tuple[int, int, int]] = ...,
) -> tuple[_U8Array, tuple[float, int, int, int, int]]: ...
def normalize_image_chw(
    image: _U8Array,
    scale: float = ...,
    mean: Optional[tuple[float, float, float]] = ...,
    std: Optional[tuple[float, float, float]] = ...,
    out: Optional[_F32Array] = ...,
) -> _F32Array: ...
def rgb_to_gray_image(image: _U8Array, out: Optional[_U8Array] = ...) -> _U8Array: ...
def rgb_to_hsv_image(image: _U8Array) -> _U8Array: ...
def remap_image(
    image: _U8Array,
    map_x: _F32Array,
    map_y: _F32Array,
    interpolation: str = ...,
    fill: Optional[tuple[int, int, int]] = ...,
) -> _U8Array: ...
def nms(
    boxes: _F32Array,
    scores: _F32Array,
    score_threshold: float = ...,
    iou_threshold: float = ...,
) -> NDArray[np.int64]: ...
def soft_nms(
    boxes: _F32Array,
    scores: _F32Array,
    score_threshold: float = ...,
    iou_threshold: float = ...,
    method: str = ...,
    sigma: float = ...,
) -> tuple[list[int], list[float]]: ...
def connected_components_image(
    mask: _U8Array,
    connectivity: int = ...,
) -> tuple[_U32Array, list[tuple[int, int, tuple[float, float, float, float]]]]: ...
def find_mask_contours(
    mask: _U8Array, epsilon: float = ...
) -> list[list[tuple[int, int]]]: ...
def encode_mask_rle(mask: _U8Array, coco: bool = ...) -> list[int]: ...
def decode_mask_rle(
    width: int, height: int, counts: Sequence[int], coco: bool = ...
) -> _U8Array: ...
def point_map_to_point_cloud(
    points: _F32Array,
    confidence: Optional[_F32Array] = ...,
    min_confidence: float = ...,
) -> PointCloud: ...

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
@final
class Tensor:
    @property
    def shape(self) -> list[int]: ...
    @property
    def dtype(self) -> str: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def __dlpack__(
        self,
        stream: object | None = ...,
        *,
        max_version: tuple[int, int] | None = ...,
        dl_device: tuple[int, int] | None = ...,
        copy: bool | None = ...,
    ) -> object: ...
    def copy(self) -> Tensor: ...

@final
class Keypoint2:
    @property
    def x(self) -> float: ...
    @property
    def y(self) -> float: ...
    @property
    def size(self) -> float: ...
    @property
    def angle_degrees(self) -> float | None: ...
    @property
    def response(self) -> float: ...
    @property
    def octave(self) -> int: ...
    @property
    def class_id(self) -> int | None: ...

def harris_keypoints(
    image: _U8Array,
    max_corners: int = ...,
    quality_level: float = ...,
    min_distance: float = ...,
    block_size: int = ...,
    gradient_size: int = ...,
    k: float = ...,
) -> list[Keypoint2]: ...
def shi_tomasi_keypoints(
    image: _U8Array,
    max_corners: int = ...,
    quality_level: float = ...,
    min_distance: float = ...,
    block_size: int = ...,
    gradient_size: int = ...,
) -> list[Keypoint2]: ...
def fast_keypoints(
    image: _U8Array,
    threshold: int = ...,
    nonmax_suppression: bool = ...,
) -> list[Keypoint2]: ...
def orb_features(
    image: _U8Array,
    max_features: int = ...,
    scale_factor: float = ...,
    levels: int = ...,
    edge_threshold: int = ...,
    fast_threshold: int = ...,
    patch_size: int = ...,
    score_type: str = ...,
) -> tuple[list[Keypoint2], _U8Array]: ...
def estimate_homography_ransac(
    source: _F64Array,
    target: _F64Array,
    threshold: float = ...,
    confidence: float = ...,
    max_iterations: int = ...,
    seed: int = ...,
) -> tuple[_F64Array, _BoolArray, _F64Array]: ...
def solve_pnp(
    object_points: _F64Array,
    image_points: _F64Array,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int = ...,
    height: int = ...,
) -> tuple[_F64Array, _F64Array]: ...
def estimate_rgbd_odometry(
    depth: _F32Array,
    source: _F64Array,
    target: _F64Array,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = ...,
    threshold: float = ...,
) -> tuple[_F64Array, _F64Array, _BoolArray, int]: ...
def stereo_block_match(
    left: _U8Array,
    right: _U8Array,
    window_size: int = ...,
    min_disparity: int = ...,
    num_disparities: int = ...,
    uniqueness_ratio: float = ...,
) -> _F32Array: ...
def match_binary_descriptors(
    query: _U8Array,
    train: _U8Array,
    cross_check: bool = ...,
    ratio: float | None = ...,
    max_distance: float | None = ...,
) -> list[tuple[int, int, float]]: ...
def match_float_descriptors(
    query: _F32Array,
    train: _F32Array,
    cross_check: bool = ...,
    ratio: float | None = ...,
    max_distance: float | None = ...,
) -> list[tuple[int, int, float]]: ...

@final
class OnnxRuntimeSession:
    def __new__(
        cls,
        path: str,
        *,
        intra_threads: int | None = ...,
        inter_threads: int | None = ...,
        deterministic: bool = ...,
    ) -> OnnxRuntimeSession: ...
    @property
    def inputs(self) -> list[tuple[str, str, list[str]]]: ...
    @property
    def outputs(self) -> list[tuple[str, str, list[str]]]: ...
    def run(self, inputs: dict[str, Tensor], *, copy: bool = ...) -> dict[str, Tensor]: ...

def tensor_copy_from_numpy(array: NDArray[np.generic]) -> Tensor: ...
@final
class DLPackTensorView:
    @property
    def shape(self) -> list[int]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def version(self) -> tuple[int, int]: ...
    def copy(self) -> Tensor: ...

def tensor_view_from_dlpack(producer: object) -> DLPackTensorView: ...
