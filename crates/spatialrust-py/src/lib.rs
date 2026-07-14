//! Python bindings for SpatialRust.
//!
//! Exposes the native-Rust point cloud pipeline (IO, voxel downsampling, RANSAC
//! plane segmentation, Euclidean clustering) to Python with zero-copy-friendly
//! NumPy interop. Build with `maturin develop`.

// PyO3's `#[pyfunction]` expansion emits `.into()` on already-`PyErr` results.
#![allow(clippy::useless_conversion)]

use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use spatialrust::core::{PointBuffer, PointBufferSet, SpatialMetadata};
use spatialrust::features::{
    orient_normals_consistent, BoundaryConfig, BoundaryDetector, FeatureEstimator,
    IssKeypointConfig, IssKeypointDetector, NormalEstimationConfig, NormalEstimator,
    NormalOrientationConfig,
};
use spatialrust::filtering::{
    Aabb, CropBox, FarthestPointSampling, FarthestPointSamplingConfig, MlsConfig, MlsSmoothing,
    PassThrough, PointCloudFilter, RadiusOutlierConfig, RadiusOutlierRemoval,
    StatisticalOutlierConfig, StatisticalOutlierRemoval, VoxelGridDownsample,
    VoxelGridDownsampleConfig,
};
use spatialrust::math::Mat4;
use spatialrust::metrics::{chamfer_distance as chamfer, hausdorff_distance as hausdorff};
use spatialrust::pipeline::{MvpPipeline, MvpPipelineConfig};
use spatialrust::registration::{
    FpfhRansacConfig, FpfhRansacRegistration, GicpConfig, GicpRegistration, IcpConfig,
    IcpRegistration, NdtConfig, NdtRegistration, PointCloudRegistration, PointToPlaneIcp,
    PointToPlaneIcpConfig, RegistrationResult,
};
use spatialrust::segmentation::{
    DbscanConfig, DbscanSegmenter, GroundConfig, GroundSegmenter, MultiPlaneConfig,
    MultiPlaneSegmenter, RansacCylinderSegmenter, RansacPrimitiveConfig, RansacSphereSegmenter,
    RegionGrowingConfig, RegionGrowingSegmenter,
};
use spatialrust::transform::{
    apply_transform as apply_tf, bounding_box as bbox, centroid as cloud_centroid, merge_clouds,
    normalize_unit_sphere as normalize_unit, oriented_bounding_box as obb, recenter as recenter_op,
    scale_cloud,
};
use spatialrust::vision::{
    approximate_polygon as approximate_contour, connected_components as label_components,
    decode_rle as decode_mask_runs, encode_rle as encode_mask_runs,
    find_contours as trace_contours, letterbox as letterbox_op, nms as nms_op,
    pack_chw as pack_chw_op, point_map_to_point_cloud as point_map_to_cloud, remap as remap_op,
    resize as resize_op, rgb_to_gray as rgb_to_gray_op, rgb_to_hsv as rgb_to_hsv_op,
    soft_nms as soft_nms_op, BinaryMask, BorderMode, BoundingBox2, ConfidenceMap, Connectivity,
    Interpolation, MaskRle, PointMap, RleOrder, SoftNmsMethod,
};
use spatialrust::voxelize::{
    range_image as range_image_proj, voxelize as voxelize_grid, RangeImageConfig, VoxelFill,
    VoxelGridConfig,
};
use spatialrust::{
    knn_graph as knn_graph_build, radius_graph as radius_graph_build, NeighborGraph,
};
use spatialrust::{
    read_point_cloud_file, write_point_cloud_file, ExecutionPolicy, HasPositions3, PointCloud,
    StandardSchemas,
};
use spatialrust::{
    rgbd_to_point_cloud as rgbd_to_cloud, BrownConrady, CameraIntrinsics, DepthConversionOptions,
    Image, PinholeCamera,
};

type Vec3Tuple = (f32, f32, f32);
type OrientedBoundingBoxTuple = (Vec3Tuple, Vec3Tuple, Vec<Vec3Tuple>);
type ComponentStats = Vec<(u32, usize, (f32, f32, f32, f32))>;

fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_policy(policy: &str) -> PyResult<ExecutionPolicy> {
    match policy.to_lowercase().as_str() {
        "auto" => Ok(ExecutionPolicy::Auto),
        "cpu" | "cpu-parallel" => Ok(ExecutionPolicy::CpuParallel),
        "cpu-single" => Ok(ExecutionPolicy::CpuSingle),
        other => Err(PyValueError::new_err(format!(
            "unknown execution policy `{other}` (expected: auto, cpu, cpu-single)"
        ))),
    }
}

fn parse_interpolation(interpolation: &str) -> PyResult<Interpolation> {
    match interpolation.to_lowercase().as_str() {
        "nearest" => Ok(Interpolation::Nearest),
        "bilinear" | "linear" => Ok(Interpolation::Bilinear),
        "bicubic" | "cubic" => Ok(Interpolation::Bicubic),
        "area" => Ok(Interpolation::Area),
        other => Err(PyValueError::new_err(format!(
            "unknown interpolation `{other}` (expected: nearest, bilinear, bicubic, area)"
        ))),
    }
}

fn rgb_image_from_numpy(array: PyReadonlyArray3<'_, u8>) -> PyResult<Image<u8, 3>> {
    let view = array.as_array();
    let shape = view.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("expected an (H, W, 3) uint8 RGB array"));
    }
    Image::try_new(shape[1], shape[0], view.iter().copied().collect()).map_err(to_py_err)
}

fn gray_u8_image_from_numpy(array: PyReadonlyArray2<'_, u8>) -> PyResult<Image<u8, 1>> {
    let view = array.as_array();
    let shape = view.shape();
    Image::try_new(shape[1], shape[0], view.iter().copied().collect()).map_err(to_py_err)
}

fn cloud_from_xyz(arr: PyReadonlyArray2<'_, f32>) -> PyResult<PointCloud> {
    let view = arr.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err("expected an (N, 3) float32 array of XYZ coordinates"));
    }
    let n = shape[0];
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    for i in 0..n {
        xs.push(view[[i, 0]]);
        ys.push(view[[i, 1]]);
        zs.push(view[[i, 2]]);
    }

    let mut buffers = PointBufferSet::new();
    buffers.insert("x", PointBuffer::from_f32(xs));
    buffers.insert("y", PointBuffer::from_f32(ys));
    buffers.insert("z", PointBuffer::from_f32(zs));
    PointCloud::try_from_parts(StandardSchemas::point_xyz(), buffers, SpatialMetadata::default())
        .map_err(to_py_err)
}

fn xyz_to_pyarray<'py>(py: Python<'py>, cloud: &PointCloud) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (xs, ys, zs) = cloud.positions3().map_err(to_py_err)?;
    let n = xs.len();
    let mut data = Vec::with_capacity(n * 3);
    for i in 0..n {
        data.push(xs[i]);
        data.push(ys[i]);
        data.push(zs[i]);
    }
    let arr = Array2::from_shape_vec((n, 3), data).map_err(to_py_err)?;
    Ok(arr.into_pyarray_bound(py))
}

fn labels_vec(cloud: &PointCloud) -> Option<Vec<i32>> {
    match cloud.field("label") {
        Ok(PointBuffer::I32(values)) => Some(values.clone()),
        _ => None,
    }
}

/// A schema-aware point cloud backed by native Rust storage.
#[pyclass(name = "PointCloud")]
#[derive(Clone)]
pub struct PyPointCloud {
    inner: PointCloud,
}

#[pymethods]
impl PyPointCloud {
    /// Builds a point cloud from an (N, 3) float32 NumPy array of XYZ.
    #[staticmethod]
    fn from_xyz(points: PyReadonlyArray2<'_, f32>) -> PyResult<Self> {
        Ok(Self { inner: cloud_from_xyz(points)? })
    }

    /// Returns the XYZ coordinates as an (N, 3) float32 NumPy array.
    fn xyz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        xyz_to_pyarray(py, &self.inner)
    }

    /// Returns per-point cluster labels as an (N,) int32 array, or None if unlabeled.
    fn labels<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i32>>> {
        labels_vec(&self.inner).map(|v| v.into_pyarray_bound(py))
    }

    /// Field names present in the cloud schema.
    fn field_names(&self) -> Vec<String> {
        self.inner.schema().fields().iter().map(|f| f.name.clone()).collect()
    }

    /// Number of points.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("PointCloud(points={}, fields={:?})", self.inner.len(), self.field_names())
    }
}

/// Result of running the MVP pipeline.
#[pyclass(name = "PipelineResult")]
pub struct PyPipelineResult {
    /// Labeled output cloud (cluster labels in the `label` field).
    #[pyo3(get)]
    output: PyPointCloud,
    /// Cloud after voxel downsampling.
    #[pyo3(get)]
    downsampled: PyPointCloud,
    /// Number of clusters found.
    #[pyo3(get)]
    cluster_count: usize,
    /// Size of each cluster, in label order.
    #[pyo3(get)]
    cluster_sizes: Vec<usize>,
    /// Number of points classified as the dominant plane.
    #[pyo3(get)]
    plane_inliers: usize,
    /// Unit normal of the dominant plane as (nx, ny, nz).
    #[pyo3(get)]
    plane_normal: (f32, f32, f32),
}

#[pymethods]
impl PyPipelineResult {
    /// Per-point cluster labels of the output cloud as an (N,) int32 array.
    fn labels<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i32>>> {
        self.output.labels(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineResult(points={}, clusters={}, plane_inliers={})",
            self.output.inner.len(),
            self.cluster_count,
            self.plane_inliers
        )
    }
}

/// Result of region growing segmentation.
#[pyclass(name = "RegionResult")]
pub struct PyRegionResult {
    /// Labeled output cloud (region labels in the `label` field).
    #[pyo3(get)]
    output: PyPointCloud,
    /// Number of smooth regions found.
    #[pyo3(get)]
    cluster_count: usize,
    /// Size of each region, in label order.
    #[pyo3(get)]
    cluster_sizes: Vec<usize>,
}

#[pymethods]
impl PyRegionResult {
    /// Per-point region labels of the output cloud as an (N,) int32 array.
    fn labels<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i32>>> {
        self.output.labels(py)
    }

    fn __repr__(&self) -> String {
        format!("RegionResult(points={}, regions={})", self.output.inner.len(), self.cluster_count)
    }
}

/// Result of DBSCAN density-based clustering.
#[pyclass(name = "DbscanResult")]
pub struct PyDbscanResult {
    /// Labeled output cloud (cluster labels in the `label` field, `-1` = noise).
    #[pyo3(get)]
    output: PyPointCloud,
    /// Number of clusters found.
    #[pyo3(get)]
    cluster_count: usize,
    /// Size of each cluster, in label order.
    #[pyo3(get)]
    cluster_sizes: Vec<usize>,
    /// Number of points classified as noise.
    #[pyo3(get)]
    noise_count: usize,
}

#[pymethods]
impl PyDbscanResult {
    /// Per-point cluster labels of the output cloud as an (N,) int32 array.
    fn labels<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i32>>> {
        self.output.labels(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "DbscanResult(points={}, clusters={}, noise={})",
            self.output.inner.len(),
            self.cluster_count,
            self.noise_count
        )
    }
}

/// DBSCAN density-based clustering. Groups points with at least `min_points`
/// neighbors within `eps`, labeling low-density points as noise (`-1`).
#[pyfunction]
#[pyo3(signature = (cloud, eps=0.5, min_points=10))]
fn dbscan(cloud: &PyPointCloud, eps: f32, min_points: usize) -> PyResult<PyDbscanResult> {
    let result = DbscanSegmenter::new(DbscanConfig::new(eps, min_points))
        .segment(&cloud.inner)
        .map_err(to_py_err)?;
    Ok(PyDbscanResult {
        output: PyPointCloud { inner: result.cloud },
        cluster_count: result.cluster_count,
        cluster_sizes: result.cluster_sizes,
        noise_count: result.noise_count,
    })
}

/// Result of multi-plane segmentation.
#[pyclass(name = "MultiPlaneResult")]
pub struct PyMultiPlaneResult {
    /// Labeled cloud: `label` field holds the plane index, `-1` = unassigned.
    #[pyo3(get)]
    output: PyPointCloud,
    /// Number of planes extracted.
    #[pyo3(get)]
    plane_count: usize,
    /// Point count of each plane, in extraction order.
    #[pyo3(get)]
    plane_sizes: Vec<usize>,
    /// Each plane as `(nx, ny, nz, d)` (Hessian form `n·p + d = 0`).
    #[pyo3(get)]
    planes: Vec<(f32, f32, f32, f32)>,
}

#[pymethods]
impl PyMultiPlaneResult {
    /// Per-point plane labels of the output cloud as an (N,) int32 array.
    fn labels<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i32>>> {
        self.output.labels(py)
    }

    fn __repr__(&self) -> String {
        format!("MultiPlaneResult(points={}, planes={})", self.output.inner.len(), self.plane_count)
    }
}

/// Sequentially extracts up to `max_planes` dominant planes (floor, walls,
/// ceiling, …) with RANSAC, labeling each point by plane index (`-1` =
/// unassigned).
#[pyfunction]
#[pyo3(signature = (cloud, max_planes=4, distance_threshold=0.02, min_inliers=100, max_iterations=1000))]
fn segment_multi_plane(
    cloud: &PyPointCloud,
    max_planes: usize,
    distance_threshold: f32,
    min_inliers: usize,
    max_iterations: usize,
) -> PyResult<PyMultiPlaneResult> {
    let config = MultiPlaneConfig {
        max_planes,
        distance_threshold,
        min_inliers,
        max_iterations,
        ..MultiPlaneConfig::default()
    };
    let result = MultiPlaneSegmenter::new(config).segment(&cloud.inner).map_err(to_py_err)?;
    let planes = result.planes.iter().map(|p| (p.normal.x, p.normal.y, p.normal.z, p.d)).collect();
    Ok(PyMultiPlaneResult {
        output: PyPointCloud { inner: result.labeled },
        plane_count: result.planes.len(),
        plane_sizes: result.plane_sizes,
        planes,
    })
}

/// Result of ground segmentation.
#[pyclass(name = "GroundResult")]
pub struct PyGroundResult {
    /// Points classified as ground.
    #[pyo3(get)]
    ground: PyPointCloud,
    /// Points classified as non-ground (objects, vegetation, structures).
    #[pyo3(get)]
    non_ground: PyPointCloud,
    /// Number of ground points.
    #[pyo3(get)]
    ground_count: usize,
}

#[pymethods]
impl PyGroundResult {
    fn __repr__(&self) -> String {
        format!(
            "GroundResult(ground={}, non_ground={})",
            self.ground.inner.len(),
            self.non_ground.inner.len()
        )
    }
}

/// Grid-based ground segmentation for outdoor scans. Splits the cloud into
/// ground and non-ground by comparing each point to a local minimum-height
/// estimate (eroded against neighbors). Assumes +Z is up.
#[pyfunction]
#[pyo3(signature = (cloud, cell_size=0.5, height_threshold=0.2, erosion_cells=1))]
fn ground_segmentation(
    cloud: &PyPointCloud,
    cell_size: f32,
    height_threshold: f32,
    erosion_cells: usize,
) -> PyResult<PyGroundResult> {
    let config =
        GroundConfig { cell_size, height_threshold, erosion_cells, ..GroundConfig::default() };
    let result = GroundSegmenter::new(config).segment(&cloud.inner).map_err(to_py_err)?;
    Ok(PyGroundResult {
        ground: PyPointCloud { inner: result.ground },
        non_ground: PyPointCloud { inner: result.non_ground },
        ground_count: result.ground_count,
    })
}

/// Result of fitting a RANSAC sphere.
#[pyclass(name = "SphereResult")]
pub struct PySphereResult {
    /// Sphere center as (x, y, z).
    #[pyo3(get)]
    center: (f32, f32, f32),
    /// Sphere radius.
    #[pyo3(get)]
    radius: f32,
    /// Points on the sphere surface.
    #[pyo3(get)]
    inliers: PyPointCloud,
    /// Points not on the sphere.
    #[pyo3(get)]
    outliers: PyPointCloud,
}

#[pymethods]
impl PySphereResult {
    fn __repr__(&self) -> String {
        format!(
            "SphereResult(center={:?}, radius={:.4}, inliers={})",
            self.center,
            self.radius,
            self.inliers.inner.len()
        )
    }
}

/// Result of fitting a RANSAC cylinder.
#[pyclass(name = "CylinderResult")]
pub struct PyCylinderResult {
    /// A point on the cylinder axis as (x, y, z).
    #[pyo3(get)]
    axis_point: (f32, f32, f32),
    /// Unit axis direction as (x, y, z).
    #[pyo3(get)]
    axis_direction: (f32, f32, f32),
    /// Cylinder radius.
    #[pyo3(get)]
    radius: f32,
    /// Points on the cylinder surface.
    #[pyo3(get)]
    inliers: PyPointCloud,
    /// Points not on the cylinder.
    #[pyo3(get)]
    outliers: PyPointCloud,
}

#[pymethods]
impl PyCylinderResult {
    fn __repr__(&self) -> String {
        format!(
            "CylinderResult(axis_point={:?}, radius={:.4}, inliers={})",
            self.axis_point,
            self.radius,
            self.inliers.inner.len()
        )
    }
}

/// Fits the dominant sphere with RANSAC and partitions inliers/outliers.
#[pyfunction]
#[pyo3(signature = (cloud, distance_threshold=0.02, max_iterations=1000, min_inliers=10))]
fn ransac_sphere(
    cloud: &PyPointCloud,
    distance_threshold: f32,
    max_iterations: usize,
    min_inliers: usize,
) -> PyResult<PySphereResult> {
    let config = RansacPrimitiveConfig {
        distance_threshold,
        max_iterations,
        min_inliers,
        ..RansacPrimitiveConfig::default()
    };
    let result = RansacSphereSegmenter::new(config).segment(&cloud.inner).map_err(to_py_err)?;
    let c = result.model.center;
    Ok(PySphereResult {
        center: (c.x, c.y, c.z),
        radius: result.model.radius,
        inliers: PyPointCloud { inner: result.inliers },
        outliers: PyPointCloud { inner: result.outliers },
    })
}

/// Fits the dominant cylinder with RANSAC. Normals are estimated on the cloud
/// from k-nearest neighbors (the axis is recovered from surface normals).
#[pyfunction]
#[pyo3(signature = (cloud, distance_threshold=0.02, max_iterations=1000, min_inliers=10, k_neighbors=20))]
fn ransac_cylinder(
    cloud: &PyPointCloud,
    distance_threshold: f32,
    max_iterations: usize,
    min_inliers: usize,
    k_neighbors: usize,
) -> PyResult<PyCylinderResult> {
    let with_normals = NormalEstimator::new(NormalEstimationConfig::k_neighbors(k_neighbors))
        .estimate(&cloud.inner)
        .map_err(to_py_err)?;
    let config = RansacPrimitiveConfig {
        distance_threshold,
        max_iterations,
        min_inliers,
        ..RansacPrimitiveConfig::default()
    };
    let result = RansacCylinderSegmenter::new(config).segment(&with_normals).map_err(to_py_err)?;
    let a = result.model.axis_point;
    let d = result.model.axis_direction;
    Ok(PyCylinderResult {
        axis_point: (a.x, a.y, a.z),
        axis_direction: (d.x, d.y, d.z),
        radius: result.model.radius,
        inliers: PyPointCloud { inner: result.inliers },
        outliers: PyPointCloud { inner: result.outliers },
    })
}

/// Symmetric Chamfer distance between two clouds (sum of mean squared
/// nearest-neighbor distances in both directions). Zero for identical clouds.
#[pyfunction]
fn chamfer_distance(a: &PyPointCloud, b: &PyPointCloud) -> PyResult<f64> {
    chamfer(&a.inner, &b.inner).map_err(to_py_err)
}

/// Symmetric Hausdorff distance between two clouds (the largest nearest-neighbor
/// distance in either direction). Captures the worst-case discrepancy.
#[pyfunction]
fn hausdorff_distance(a: &PyPointCloud, b: &PyPointCloud) -> PyResult<f64> {
    hausdorff(&a.inner, &b.inner).map_err(to_py_err)
}

/// Reads a point cloud from a file (PCD/PLY/LAS/COPC by extension).
#[pyfunction]
fn read(path: &str) -> PyResult<PyPointCloud> {
    let inner = read_point_cloud_file(path).map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// Writes a point cloud to a file (format chosen by extension).
#[pyfunction]
fn write(path: &str, cloud: &PyPointCloud) -> PyResult<()> {
    write_point_cloud_file(path, &cloud.inner).map_err(to_py_err)
}

/// Voxel-grid downsamples a cloud. `policy` is one of "auto", "cpu", "cpu-single".
#[pyfunction]
#[pyo3(signature = (cloud, leaf_size, policy="auto"))]
fn voxel_downsample(cloud: &PyPointCloud, leaf_size: f32, policy: &str) -> PyResult<PyPointCloud> {
    let config = VoxelGridDownsampleConfig::centroid(leaf_size);
    let filter = VoxelGridDownsample::new(config);
    let inner =
        filter.filter_with_policy(&cloud.inner, parse_policy(policy)?).map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// Crop to an axis-aligned box. Keeps points inside `[min, max]`, or drops them
/// when `invert=True`. `min`/`max` are `(x, y, z)` tuples.
#[pyfunction]
#[pyo3(signature = (cloud, min, max, invert=false))]
fn crop_box(
    cloud: &PyPointCloud,
    min: (f32, f32, f32),
    max: (f32, f32, f32),
    invert: bool,
) -> PyResult<PyPointCloud> {
    let bounds = Aabb::new([min.0, min.1, min.2], [max.0, max.1, max.2]);
    let filter = if invert { CropBox::inverted(bounds) } else { CropBox::new(bounds) };
    let inner = filter.filter(&cloud.inner).map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// Keep points whose value in `field` lies within `[min, max]` (e.g. a height
/// slice on "z" or an intensity threshold), or drop them when `invert=True`.
#[pyfunction]
#[pyo3(signature = (cloud, field, min, max, invert=false))]
fn pass_through(
    cloud: &PyPointCloud,
    field: &str,
    min: f32,
    max: f32,
    invert: bool,
) -> PyResult<PyPointCloud> {
    let filter = if invert {
        PassThrough::inverted(field, min, max)
    } else {
        PassThrough::new(field, min, max)
    };
    let inner = filter.filter(&cloud.inner).map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// Farthest Point Sampling: keeps `sample_size` points spread as evenly as
/// possible over the cloud — the standard downsampling for learned models.
#[pyfunction]
#[pyo3(signature = (cloud, sample_size, seed_index=0))]
fn farthest_point_sampling(
    cloud: &PyPointCloud,
    sample_size: usize,
    seed_index: usize,
) -> PyResult<PyPointCloud> {
    let config = FarthestPointSamplingConfig { sample_size, seed_index };
    let inner = FarthestPointSampling::new(config).filter(&cloud.inner).map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// Detects boundary / edge points (hole rims, scan edges): estimates normals,
/// then flags points whose tangent-plane neighbors leave a large angular gap.
/// Returns a sparse sub-cloud of the boundary points.
#[pyfunction]
#[pyo3(signature = (cloud, search_radius=0.1, angle_threshold=std::f32::consts::FRAC_PI_2, min_neighbors=5, k_neighbors=20))]
fn detect_boundary(
    cloud: &PyPointCloud,
    search_radius: f32,
    angle_threshold: f32,
    min_neighbors: usize,
    k_neighbors: usize,
) -> PyResult<PyPointCloud> {
    let with_normals = NormalEstimator::new(NormalEstimationConfig::k_neighbors(k_neighbors))
        .estimate(&cloud.inner)
        .map_err(to_py_err)?;
    let config = BoundaryConfig { search_radius, angle_threshold, min_neighbors };
    let result = BoundaryDetector::new(config).detect(&with_normals).map_err(to_py_err)?;
    Ok(PyPointCloud { inner: result.boundary })
}

/// Moving Least Squares smoothing: projects each point onto a local polynomial
/// surface fit to its neighborhood, removing scanner noise while preserving
/// curvature. `polynomial_order` is 1 (plane) or 2 (quadratic).
#[pyfunction]
#[pyo3(signature = (cloud, search_radius=0.1, polynomial_order=2, min_neighbors=6))]
fn mls_smooth(
    cloud: &PyPointCloud,
    search_radius: f32,
    polynomial_order: u8,
    min_neighbors: usize,
) -> PyResult<PyPointCloud> {
    let config = MlsConfig { search_radius, polynomial_order, min_neighbors };
    let inner = MlsSmoothing::new(config).filter(&cloud.inner).map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// Estimates normals, then propagates a single consistent orientation across a
/// k-NN graph (MST) so neighboring normals agree in sign. Returns a cloud
/// carrying the oriented normals (`normal_x/y/z` fields).
#[pyfunction]
#[pyo3(signature = (cloud, k_neighbors=15))]
fn orient_normals(cloud: &PyPointCloud, k_neighbors: usize) -> PyResult<PyPointCloud> {
    let with_normals = NormalEstimator::new(NormalEstimationConfig::k_neighbors(k_neighbors))
        .estimate(&cloud.inner)
        .map_err(to_py_err)?;
    let oriented =
        orient_normals_consistent(&with_normals, NormalOrientationConfig::new(k_neighbors))
            .map_err(to_py_err)?;
    Ok(PyPointCloud { inner: oriented })
}

/// Intrinsic Shape Signatures (ISS) keypoints: returns a sparse sub-cloud of
/// geometrically salient points (corners), useful as a front-end for
/// feature-based registration.
#[pyfunction]
#[pyo3(signature = (cloud, salient_radius=0.2, non_max_radius=0.15, gamma_21=0.975, gamma_32=0.975, min_neighbors=5))]
fn iss_keypoints(
    cloud: &PyPointCloud,
    salient_radius: f32,
    non_max_radius: f32,
    gamma_21: f32,
    gamma_32: f32,
    min_neighbors: usize,
) -> PyResult<PyPointCloud> {
    let config =
        IssKeypointConfig { salient_radius, non_max_radius, gamma_21, gamma_32, min_neighbors };
    let result = IssKeypointDetector::new(config).detect(&cloud.inner).map_err(to_py_err)?;
    Ok(PyPointCloud { inner: result.keypoints })
}

/// Statistical Outlier Removal: drops points whose mean distance to their `k`
/// nearest neighbors is more than `std_mul` standard deviations above the mean.
#[pyfunction]
#[pyo3(signature = (cloud, k_neighbors=16, std_mul=1.0))]
fn statistical_outlier_removal(
    cloud: &PyPointCloud,
    k_neighbors: usize,
    std_mul: f32,
) -> PyResult<PyPointCloud> {
    let filter =
        StatisticalOutlierRemoval::new(StatisticalOutlierConfig::new(k_neighbors, std_mul));
    let inner = filter.filter(&cloud.inner).map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// Radius Outlier Removal: drops points with fewer than `min_neighbors` other
/// points within `radius`.
#[pyfunction]
#[pyo3(signature = (cloud, radius=0.5, min_neighbors=4))]
fn radius_outlier_removal(
    cloud: &PyPointCloud,
    radius: f32,
    min_neighbors: usize,
) -> PyResult<PyPointCloud> {
    let filter = RadiusOutlierRemoval::new(RadiusOutlierConfig::new(radius, min_neighbors));
    let inner = filter.filter(&cloud.inner).map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// Runs the MVP pipeline: voxel downsample → normals → RANSAC plane → Euclidean clustering.
#[pyfunction]
#[pyo3(signature = (cloud, leaf_size=0.05, cluster_tolerance=None, min_cluster_size=None, plane_distance=None, policy="auto"))]
fn run_pipeline(
    cloud: &PyPointCloud,
    leaf_size: f32,
    cluster_tolerance: Option<f32>,
    min_cluster_size: Option<usize>,
    plane_distance: Option<f32>,
    policy: &str,
) -> PyResult<PyPipelineResult> {
    let mut config = MvpPipelineConfig::with_voxel_leaf_size(leaf_size);
    config.voxel_policy = parse_policy(policy)?;
    if let Some(tol) = cluster_tolerance {
        config.cluster.cluster_tolerance = tol;
    }
    if let Some(min) = min_cluster_size {
        config.cluster.min_cluster_size = min;
    }
    if let Some(dist) = plane_distance {
        config.plane.distance_threshold = dist;
    }

    let result = MvpPipeline::new(config).run(&cloud.inner).map_err(to_py_err)?;

    let normal = result.plane.model.normal;
    Ok(PyPipelineResult {
        output: PyPointCloud { inner: result.output },
        downsampled: PyPointCloud { inner: result.downsampled },
        cluster_count: result.clusters.cluster_count,
        cluster_sizes: result.clusters.cluster_sizes,
        plane_inliers: result.plane.inlier_count,
        plane_normal: (normal.x, normal.y, normal.z),
    })
}

/// Normal-based region growing: estimates normals, then grows smooth regions.
///
/// `smoothness_deg` is the maximum angle (degrees) between neighboring normals
/// for them to join the same region.
#[pyfunction]
#[pyo3(signature = (cloud, k_neighbors=30, smoothness_deg=3.0, min_region_size=10))]
fn region_growing(
    cloud: &PyPointCloud,
    k_neighbors: usize,
    smoothness_deg: f32,
    min_region_size: usize,
) -> PyResult<PyRegionResult> {
    let normals_config = NormalEstimationConfig::k_neighbors(k_neighbors);
    let with_normals =
        NormalEstimator::new(normals_config).estimate(&cloud.inner).map_err(to_py_err)?;

    let mut config = RegionGrowingConfig::with_smoothness(smoothness_deg.to_radians(), k_neighbors);
    config.min_cluster_size = min_region_size.max(1);
    let result = RegionGrowingSegmenter::new(config).segment(&with_normals).map_err(to_py_err)?;

    Ok(PyRegionResult {
        output: PyPointCloud { inner: result.cloud },
        cluster_count: result.cluster_count,
        cluster_sizes: result.cluster_sizes,
    })
}

/// Result of a registration (alignment) run.
#[pyclass(name = "RegistrationResult")]
pub struct PyRegistrationResult {
    matrix: [[f32; 4]; 4],
    /// Final alignment fitness (lower is better).
    #[pyo3(get)]
    fitness: f64,
    /// Number of iterations performed.
    #[pyo3(get)]
    iterations: usize,
    /// Whether the algorithm reached its convergence criterion.
    #[pyo3(get)]
    converged: bool,
}

impl PyRegistrationResult {
    fn from_result(result: &RegistrationResult) -> Self {
        Self {
            matrix: result.transform.to_mat4().m,
            fitness: result.fitness,
            iterations: result.iterations,
            converged: result.converged,
        }
    }
}

#[pymethods]
impl PyRegistrationResult {
    /// Returns the 4x4 transform mapping source into the target frame.
    fn transform<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data: Vec<f32> = self.matrix.iter().flatten().copied().collect();
        let arr = Array2::from_shape_vec((4, 4), data).map_err(to_py_err)?;
        Ok(arr.into_pyarray_bound(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "RegistrationResult(fitness={:.3e}, iterations={}, converged={})",
            self.fitness, self.iterations, self.converged
        )
    }
}

/// Point-to-point ICP aligning `source` onto `target`.
#[pyfunction]
#[pyo3(signature = (source, target, max_correspondence_distance=1.0, max_iterations=50))]
fn register_icp(
    source: &PyPointCloud,
    target: &PyPointCloud,
    max_correspondence_distance: f32,
    max_iterations: usize,
) -> PyResult<PyRegistrationResult> {
    let config = IcpConfig { max_correspondence_distance, max_iterations, ..IcpConfig::default() };
    let result =
        IcpRegistration::new(config).align(&source.inner, &target.inner).map_err(to_py_err)?;
    Ok(PyRegistrationResult::from_result(&result))
}

/// Point-to-plane ICP. Normals are estimated on `target` from k-nearest neighbors.
#[pyfunction]
#[pyo3(signature = (source, target, max_correspondence_distance=1.0, max_iterations=50, k_neighbors=20))]
fn register_point_to_plane(
    source: &PyPointCloud,
    target: &PyPointCloud,
    max_correspondence_distance: f32,
    max_iterations: usize,
    k_neighbors: usize,
) -> PyResult<PyRegistrationResult> {
    let target_with_normals =
        NormalEstimator::new(NormalEstimationConfig::k_neighbors(k_neighbors))
            .estimate(&target.inner)
            .map_err(to_py_err)?;
    let config = PointToPlaneIcpConfig {
        max_correspondence_distance,
        max_iterations,
        ..PointToPlaneIcpConfig::default()
    };
    let result = PointToPlaneIcp::new(config)
        .align(&source.inner, &target_with_normals)
        .map_err(to_py_err)?;
    Ok(PyRegistrationResult::from_result(&result))
}

/// Generalized ICP (plane-to-plane). Covariances are estimated from k-nearest neighbors.
#[pyfunction]
#[pyo3(signature = (source, target, max_correspondence_distance=1.0, max_iterations=50, k_neighbors=20))]
fn register_gicp(
    source: &PyPointCloud,
    target: &PyPointCloud,
    max_correspondence_distance: f32,
    max_iterations: usize,
    k_neighbors: usize,
) -> PyResult<PyRegistrationResult> {
    let config = GicpConfig {
        max_correspondence_distance,
        max_iterations,
        k_neighbors,
        ..GicpConfig::default()
    };
    let result =
        GicpRegistration::new(config).align(&source.inner, &target.inner).map_err(to_py_err)?;
    Ok(PyRegistrationResult::from_result(&result))
}

/// NDT (Normal Distributions Transform) registration. `resolution` is the target
/// voxel size used to build per-cell Gaussians.
#[pyfunction]
#[pyo3(signature = (source, target, resolution=1.0, max_iterations=35))]
fn register_ndt(
    source: &PyPointCloud,
    target: &PyPointCloud,
    resolution: f32,
    max_iterations: usize,
) -> PyResult<PyRegistrationResult> {
    let config = NdtConfig { resolution, max_iterations, ..NdtConfig::default() };
    let result =
        NdtRegistration::new(config).align(&source.inner, &target.inner).map_err(to_py_err)?;
    Ok(PyRegistrationResult::from_result(&result))
}

/// FPFH + RANSAC global registration: recovers a coarse alignment with no
/// initial guess (typically refined afterwards with ICP/GICP). Normals are
/// estimated on both clouds from k-nearest neighbors.
#[pyfunction]
#[pyo3(signature = (source, target, feature_radius=0.25, max_correspondence_distance=0.075, ransac_iterations=4000, k_neighbors=20))]
fn register_fpfh_ransac(
    source: &PyPointCloud,
    target: &PyPointCloud,
    feature_radius: f32,
    max_correspondence_distance: f32,
    ransac_iterations: usize,
    k_neighbors: usize,
) -> PyResult<PyRegistrationResult> {
    let normals = NormalEstimationConfig::k_neighbors(k_neighbors);
    let source_with_normals =
        NormalEstimator::new(normals).estimate(&source.inner).map_err(to_py_err)?;
    let target_with_normals =
        NormalEstimator::new(normals).estimate(&target.inner).map_err(to_py_err)?;
    let config = FpfhRansacConfig {
        feature_radius,
        max_correspondence_distance,
        ransac_iterations,
        ..FpfhRansacConfig::default()
    };
    let result = FpfhRansacRegistration::new(config)
        .align(&source_with_normals, &target_with_normals)
        .map_err(to_py_err)?;
    Ok(PyRegistrationResult::from_result(&result))
}

/// Keypoint-based FPFH + RANSAC global registration: estimates normals, detects
/// ISS keypoints, and runs FPFH matching only on those keypoints — the standard
/// keypoint → descriptor → registration flow, far faster than describing every
/// point. Returns the coarse alignment (refine with ICP/GICP afterwards).
#[pyfunction]
#[pyo3(signature = (source, target, salient_radius=0.1, feature_radius=0.25, max_correspondence_distance=0.075, ransac_iterations=4000, k_neighbors=20))]
fn register_fpfh_keypoints(
    source: &PyPointCloud,
    target: &PyPointCloud,
    salient_radius: f32,
    feature_radius: f32,
    max_correspondence_distance: f32,
    ransac_iterations: usize,
    k_neighbors: usize,
) -> PyResult<PyRegistrationResult> {
    let normals = NormalEstimationConfig::k_neighbors(k_neighbors);
    let iss = IssKeypointConfig {
        salient_radius,
        non_max_radius: salient_radius * 0.7,
        ..IssKeypointConfig::default()
    };

    // Estimate normals on the full cloud, then keep only ISS keypoints (which
    // carry the normals through), so FPFH is computed on a sparse salient set.
    let keypoints = |cloud: &PointCloud| -> PyResult<PointCloud> {
        let with_normals = NormalEstimator::new(normals).estimate(cloud).map_err(to_py_err)?;
        Ok(IssKeypointDetector::new(iss).detect(&with_normals).map_err(to_py_err)?.keypoints)
    };
    let source_keypoints = keypoints(&source.inner)?;
    let target_keypoints = keypoints(&target.inner)?;

    let config = FpfhRansacConfig {
        feature_radius,
        max_correspondence_distance,
        ransac_iterations,
        ..FpfhRansacConfig::default()
    };
    let result = FpfhRansacRegistration::new(config)
        .align(&source_keypoints, &target_keypoints)
        .map_err(to_py_err)?;
    Ok(PyRegistrationResult::from_result(&result))
}

/// Applies a 4x4 affine transform (NumPy array) to a cloud's positions and
/// normals.
#[pyfunction]
fn apply_transform(
    cloud: &PyPointCloud,
    matrix: PyReadonlyArray2<'_, f32>,
) -> PyResult<PyPointCloud> {
    let m = matrix.as_array();
    if m.shape() != [4, 4] {
        return Err(PyValueError::new_err("transform must be a (4, 4) float32 matrix"));
    }
    let mat = Mat4::from_rows(
        [m[[0, 0]], m[[0, 1]], m[[0, 2]], m[[0, 3]]],
        [m[[1, 0]], m[[1, 1]], m[[1, 2]], m[[1, 3]]],
        [m[[2, 0]], m[[2, 1]], m[[2, 2]], m[[2, 3]]],
        [m[[3, 0]], m[[3, 1]], m[[3, 2]], m[[3, 3]]],
    );
    Ok(PyPointCloud { inner: apply_tf(&cloud.inner, mat).map_err(to_py_err)? })
}

/// Translates a cloud so its centroid is at the origin.
#[pyfunction]
fn recenter(cloud: &PyPointCloud) -> PyResult<PyPointCloud> {
    Ok(PyPointCloud { inner: recenter_op(&cloud.inner).map_err(to_py_err)? })
}

/// Uniformly scales a cloud about the origin by `factor`.
#[pyfunction]
fn scale(cloud: &PyPointCloud, factor: f32) -> PyResult<PyPointCloud> {
    Ok(PyPointCloud { inner: scale_cloud(&cloud.inner, factor).map_err(to_py_err)? })
}

/// Recenters and scales a cloud so its farthest point is at unit distance.
#[pyfunction]
fn normalize_unit_sphere(cloud: &PyPointCloud) -> PyResult<PyPointCloud> {
    Ok(PyPointCloud { inner: normalize_unit(&cloud.inner).map_err(to_py_err)? })
}

/// Concatenates clouds sharing the same schema into one.
#[pyfunction]
fn merge(clouds: Vec<PyPointCloud>) -> PyResult<PyPointCloud> {
    let refs: Vec<&PointCloud> = clouds.iter().map(|c| &c.inner).collect();
    Ok(PyPointCloud { inner: merge_clouds(&refs).map_err(to_py_err)? })
}

/// Centroid (mean position) as `(x, y, z)`.
#[pyfunction]
fn centroid(cloud: &PyPointCloud) -> PyResult<(f32, f32, f32)> {
    let c = cloud_centroid(&cloud.inner).map_err(to_py_err)?;
    Ok((c.x, c.y, c.z))
}

/// Axis-aligned bounding box as `(min_xyz, max_xyz)`.
#[pyfunction]
fn bounding_box(cloud: &PyPointCloud) -> PyResult<((f32, f32, f32), (f32, f32, f32))> {
    let b = bbox(&cloud.inner).map_err(to_py_err)?;
    Ok(((b.min.x, b.min.y, b.min.z), (b.max.x, b.max.y, b.max.z)))
}

/// Oriented (PCA) bounding box as `(center, half_extents, axes_3x3)`. The axes
/// are returned principal-first; column `k` of `axes_3x3` is the k-th box axis.
#[pyfunction]
fn oriented_bounding_box(cloud: &PyPointCloud) -> PyResult<OrientedBoundingBoxTuple> {
    let o = obb(&cloud.inner).map_err(to_py_err)?;
    let axis = |k: usize| (o.axes.m[0][k], o.axes.m[1][k], o.axes.m[2][k]);
    Ok((
        (o.center.x, o.center.y, o.center.z),
        (o.half_extents.x, o.half_extents.y, o.half_extents.z),
        vec![axis(0), axis(1), axis(2)],
    ))
}

/// Voxelizes a cloud into a dense 3D grid `(nz, ny, nx)` for learned models.
/// `mode` is "occupancy" (1/0) or "count" (points per voxel). Returns
/// `(grid, origin_xyz, voxel_size)`; the grid is indexed `[z, y, x]`.
#[pyfunction]
#[pyo3(signature = (cloud, voxel_size=0.1, mode="occupancy"))]
fn voxelize<'py>(
    py: Python<'py>,
    cloud: &PyPointCloud,
    voxel_size: f32,
    mode: &str,
) -> PyResult<(Bound<'py, PyArray3<f32>>, (f32, f32, f32), f32)> {
    let fill = match mode.to_lowercase().as_str() {
        "occupancy" => VoxelFill::Occupancy,
        "count" => VoxelFill::Count,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown voxelize mode `{other}` (expected: occupancy, count)"
            )))
        }
    };
    let config = VoxelGridConfig { voxel_size, fill, ..VoxelGridConfig::default() };
    let grid = voxelize_grid(&cloud.inner, config).map_err(to_py_err)?;
    let [nx, ny, nz] = grid.dims;
    // Data is stored z-major; reshape to (nz, ny, nx) so axis order matches.
    let arr = Array3::from_shape_vec((nz, ny, nx), grid.data).map_err(to_py_err)?;
    Ok((
        arr.into_pyarray_bound(py),
        (grid.origin[0], grid.origin[1], grid.origin[2]),
        grid.voxel_size,
    ))
}

/// Builds an `edge_index` of shape `(2, E)` (PyG convention) for a neighborhood
/// graph: row 0 is the source node, row 1 the target. Each `graph_edge_index`
/// helper assembles directed edges from the cloud's k-NN or radius neighbors.
fn graph_to_edge_index<'py>(
    py: Python<'py>,
    graph: &NeighborGraph,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let e = graph.edges.len();
    let mut data = Vec::with_capacity(e * 2);
    for edge in &graph.edges {
        data.push(edge[0] as i32);
    }
    for edge in &graph.edges {
        data.push(edge[1] as i32);
    }
    let arr = Array2::from_shape_vec((2, e), data).map_err(to_py_err)?;
    Ok(arr.into_pyarray_bound(py))
}

/// Directed k-nearest-neighbor graph as a `(2, E)` `edge_index` (PyG style):
/// an edge from each point to each of its `k` nearest neighbors.
#[pyfunction]
fn knn_graph<'py>(
    py: Python<'py>,
    cloud: &PyPointCloud,
    k: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let graph = knn_graph_build(&cloud.inner, k).map_err(to_py_err)?;
    graph_to_edge_index(py, &graph)
}

/// Directed radius graph as a `(2, E)` `edge_index` (PyG style): an edge from
/// each point to every other point within `radius`.
#[pyfunction]
fn radius_graph<'py>(
    py: Python<'py>,
    cloud: &PyPointCloud,
    radius: f32,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let graph = radius_graph_build(&cloud.inner, radius).map_err(to_py_err)?;
    graph_to_edge_index(py, &graph)
}

/// Projects a rotating-LiDAR cloud into a 2D range image `(height, width)`,
/// keeping the nearest range per pixel (empty pixels are 0). Returns the range
/// image as a NumPy array.
#[pyfunction]
#[pyo3(signature = (cloud, width=1024, height=64, fov_up_deg=3.0, fov_down_deg=-25.0))]
fn range_image<'py>(
    py: Python<'py>,
    cloud: &PyPointCloud,
    width: usize,
    height: usize,
    fov_up_deg: f32,
    fov_down_deg: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let config = RangeImageConfig { width, height, fov_up_deg, fov_down_deg };
    let img = range_image_proj(&cloud.inner, config).map_err(to_py_err)?;
    let arr = Array2::from_shape_vec((img.height, img.width), img.data).map_err(to_py_err)?;
    Ok(arr.into_pyarray_bound(py))
}

/// Converts aligned `(H, W)` float32 depth and `(H, W, 3)` uint8 RGB images
/// into a colored point cloud. `depth_scale` converts stored values to meters.
#[pyfunction]
#[pyo3(signature = (
    depth,
    color,
    fx,
    fy,
    cx,
    cy,
    depth_scale=1.0,
    min_depth=f32::EPSILON,
    max_depth=f32::INFINITY,
    distortion=None
))]
#[allow(clippy::too_many_arguments)]
fn rgbd_to_point_cloud(
    depth: PyReadonlyArray2<'_, f32>,
    color: PyReadonlyArray3<'_, u8>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    depth_scale: f32,
    min_depth: f32,
    max_depth: f32,
    distortion: Option<(f64, f64, f64, f64, f64)>,
) -> PyResult<PyPointCloud> {
    let depth_view = depth.as_array();
    let color_view = color.as_array();
    let depth_shape = depth_view.shape();
    let color_shape = color_view.shape();
    let height = depth_shape[0];
    let width = depth_shape[1];
    if color_shape != [height, width, 3] {
        return Err(PyValueError::new_err(format!(
            "expected color shape ({height}, {width}, 3), found {:?}",
            color_shape
        )));
    }

    // Iteration follows logical ndarray order, so non-contiguous NumPy views
    // are packed explicitly at the Python/native boundary.
    let depth_image = Image::<f32, 1>::try_new(width, height, depth_view.iter().copied().collect())
        .map_err(to_py_err)?;
    let color_image = Image::<u8, 3>::try_new(width, height, color_view.iter().copied().collect())
        .map_err(to_py_err)?;
    let intrinsics = CameraIntrinsics::try_new(fx, fy, cx, cy, width, height).map_err(to_py_err)?;
    let mut camera = PinholeCamera::new(intrinsics);
    if let Some((k1, k2, p1, p2, k3)) = distortion {
        camera = camera.with_distortion(BrownConrady { k1, k2, p1, p2, k3 });
    }
    let options = DepthConversionOptions { depth_scale, min_depth, max_depth };
    let inner = rgbd_to_cloud(depth_image.view(), color_image.view(), &camera, options)
        .map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// Resizes an `(H, W, 3)` uint8 RGB image.
#[pyfunction]
#[pyo3(signature = (image, width, height, interpolation="bilinear"))]
fn resize_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    width: usize,
    height: usize,
    interpolation: &str,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = rgb_image_from_numpy(image)?;
    let output = resize_op(image.view(), width, height, parse_interpolation(interpolation)?)
        .map_err(to_py_err)?;
    let array = Array3::from_shape_vec((height, width, 3), output.into_vec()).map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Letterboxes an RGB image and returns `(image, transform)`, where transform
/// is `(scale, pad_left, pad_top, content_width, content_height)`.
#[pyfunction]
#[pyo3(signature = (image, width, height, interpolation="bilinear", fill=None))]
fn letterbox_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    width: usize,
    height: usize,
    interpolation: &str,
    fill: Option<(u8, u8, u8)>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, (f64, usize, usize, usize, usize))> {
    let image = rgb_image_from_numpy(image)?;
    let (output, transform) = letterbox_op(
        image.view(),
        width,
        height,
        parse_interpolation(interpolation)?,
        fill.map_or([114; 3], |(r, g, b)| [r, g, b]),
    )
    .map_err(to_py_err)?;
    let array = Array3::from_shape_vec((height, width, 3), output.into_vec()).map_err(to_py_err)?;
    Ok((
        array.into_pyarray_bound(py),
        (
            transform.scale,
            transform.pad_left,
            transform.pad_top,
            transform.content_width,
            transform.content_height,
        ),
    ))
}

/// Normalizes RGB and packs it into a float32 `(3, H, W)` CHW tensor.
#[pyfunction]
#[pyo3(signature = (image, scale=1.0/255.0, mean=None, std=None))]
fn normalize_image_chw<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    scale: f32,
    mean: Option<(f32, f32, f32)>,
    std: Option<(f32, f32, f32)>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let image = rgb_image_from_numpy(image)?;
    let mean = mean.map_or([0.0; 3], |(r, g, b)| [r, g, b]);
    let std = std.map_or([1.0; 3], |(r, g, b)| [r, g, b]);
    let output = pack_chw_op(image.view(), scale, mean, std).map_err(to_py_err)?;
    let array = Array3::from_shape_vec((3, image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Converts an RGB image to an `(H, W)` grayscale image.
#[pyfunction]
fn rgb_to_gray_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let image = rgb_image_from_numpy(image)?;
    let output = rgb_to_gray_op(image.view()).map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Converts RGB to OpenCV-style uint8 HSV.
#[pyfunction]
fn rgb_to_hsv_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = rgb_image_from_numpy(image)?;
    let output = rgb_to_hsv_op(image.view()).map_err(to_py_err)?;
    let array = Array3::from_shape_vec((image.height(), image.width(), 3), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Remaps an RGB image with absolute float32 source-coordinate maps.
#[pyfunction]
#[pyo3(signature = (image, map_x, map_y, interpolation="bilinear", fill=None))]
fn remap_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    map_x: PyReadonlyArray2<'_, f32>,
    map_y: PyReadonlyArray2<'_, f32>,
    interpolation: &str,
    fill: Option<(u8, u8, u8)>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = rgb_image_from_numpy(image)?;
    let mx = map_x.as_array();
    let my = map_y.as_array();
    if mx.shape() != my.shape() {
        return Err(PyValueError::new_err("map_x and map_y shapes must match"));
    }
    let height = mx.shape()[0];
    let width = mx.shape()[1];
    let map_x =
        Image::<f32, 1>::try_new(width, height, mx.iter().copied().collect()).map_err(to_py_err)?;
    let map_y =
        Image::<f32, 1>::try_new(width, height, my.iter().copied().collect()).map_err(to_py_err)?;
    let output = remap_op(
        image.view(),
        map_x.view(),
        map_y.view(),
        parse_interpolation(interpolation)?,
        BorderMode::Constant(fill.map_or([0; 3], |(r, g, b)| [r, g, b])),
    )
    .map_err(to_py_err)?;
    let array = Array3::from_shape_vec((height, width, 3), output.into_vec()).map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Greedy non-maximum suppression over `(N, 4)` xyxy boxes.
#[pyfunction]
#[pyo3(signature = (boxes, scores, score_threshold=0.0, iou_threshold=0.5))]
fn nms<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    scores: PyReadonlyArray1<'_, f32>,
    score_threshold: f32,
    iou_threshold: f32,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let boxes_view = boxes.as_array();
    if boxes_view.shape().len() != 2 || boxes_view.shape()[1] != 4 {
        return Err(PyValueError::new_err("expected boxes with shape (N, 4)"));
    }
    let mut native_boxes = Vec::with_capacity(boxes_view.shape()[0]);
    for row in boxes_view.rows() {
        native_boxes
            .push(BoundingBox2::try_new(row[0], row[1], row[2], row[3]).map_err(to_py_err)?);
    }
    let scores: Vec<f32> = scores.as_array().iter().copied().collect();
    let indices = nms_op(&native_boxes, &scores, score_threshold, iou_threshold)
        .map_err(to_py_err)?
        .into_iter()
        .map(|index| index as i64)
        .collect::<Vec<_>>();
    Ok(indices.into_pyarray_bound(py))
}

/// Soft-NMS returning `(indices, updated_scores)`.
#[pyfunction]
#[pyo3(signature = (boxes, scores, score_threshold=0.001, iou_threshold=0.5, method="linear", sigma=0.5))]
fn soft_nms(
    boxes: PyReadonlyArray2<'_, f32>,
    scores: PyReadonlyArray1<'_, f32>,
    score_threshold: f32,
    iou_threshold: f32,
    method: &str,
    sigma: f32,
) -> PyResult<(Vec<usize>, Vec<f32>)> {
    let boxes_view = boxes.as_array();
    if boxes_view.shape().len() != 2 || boxes_view.shape()[1] != 4 {
        return Err(PyValueError::new_err("expected boxes with shape (N, 4)"));
    }
    let mut native_boxes = Vec::with_capacity(boxes_view.shape()[0]);
    for row in boxes_view.rows() {
        native_boxes
            .push(BoundingBox2::try_new(row[0], row[1], row[2], row[3]).map_err(to_py_err)?);
    }
    let method = match method.to_lowercase().as_str() {
        "hard" => SoftNmsMethod::Hard,
        "linear" => SoftNmsMethod::Linear,
        "gaussian" => SoftNmsMethod::Gaussian { sigma },
        other => return Err(PyValueError::new_err(format!("unknown Soft-NMS method `{other}`"))),
    };
    let scores: Vec<f32> = scores.as_array().iter().copied().collect();
    let result = soft_nms_op(&native_boxes, &scores, score_threshold, iou_threshold, method)
        .map_err(to_py_err)?;
    Ok((
        result.iter().map(|value| value.index).collect(),
        result.iter().map(|value| value.score).collect(),
    ))
}

/// Labels connected foreground regions in a uint8 binary mask.
#[pyfunction]
#[pyo3(signature = (mask, connectivity=8))]
fn connected_components_image<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray2<'_, u8>,
    connectivity: u8,
) -> PyResult<(Bound<'py, PyArray2<u32>>, ComponentStats)> {
    let image = gray_u8_image_from_numpy(mask)?;
    let mask =
        BinaryMask::try_new(image.width(), image.height(), image.into_vec()).map_err(to_py_err)?;
    let connectivity = match connectivity {
        4 => Connectivity::Four,
        8 => Connectivity::Eight,
        _ => return Err(PyValueError::new_err("connectivity must be 4 or 8")),
    };
    let result = label_components(&mask, connectivity).map_err(to_py_err)?;
    let stats = result
        .components
        .iter()
        .map(|component| {
            (
                component.label,
                component.area,
                (
                    component.bbox.x_min,
                    component.bbox.y_min,
                    component.bbox.x_max,
                    component.bbox.y_max,
                ),
            )
        })
        .collect();
    let labels = Array2::from_shape_vec(
        (result.labels.height(), result.labels.width()),
        result.labels.as_slice().to_vec(),
    )
    .map_err(to_py_err)?;
    Ok((labels.into_pyarray_bound(py), stats))
}

/// Extracts and optionally simplifies mask contours.
#[pyfunction]
#[pyo3(signature = (mask, epsilon=0.0))]
fn find_mask_contours(
    mask: PyReadonlyArray2<'_, u8>,
    epsilon: f64,
) -> PyResult<Vec<Vec<(i32, i32)>>> {
    let image = gray_u8_image_from_numpy(mask)?;
    let mask =
        BinaryMask::try_new(image.width(), image.height(), image.into_vec()).map_err(to_py_err)?;
    trace_contours(&mask)
        .into_iter()
        .map(|contour| {
            let contour = if epsilon > 0.0 {
                approximate_contour(&contour, epsilon).map_err(to_py_err)?
            } else {
                contour
            };
            Ok(contour.points.into_iter().map(|[x, y]| (x, y)).collect())
        })
        .collect()
}

/// Encodes a binary mask into alternating run lengths.
#[pyfunction]
#[pyo3(signature = (mask, coco=true))]
fn encode_mask_rle(mask: PyReadonlyArray2<'_, u8>, coco: bool) -> PyResult<Vec<usize>> {
    let image = gray_u8_image_from_numpy(mask)?;
    let mask =
        BinaryMask::try_new(image.width(), image.height(), image.into_vec()).map_err(to_py_err)?;
    Ok(encode_mask_runs(&mask, if coco { RleOrder::CocoColumnMajor } else { RleOrder::RowMajor })
        .counts)
}

/// Decodes alternating mask run lengths.
#[pyfunction]
#[pyo3(signature = (width, height, counts, coco=true))]
fn decode_mask_rle<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    counts: Vec<usize>,
    coco: bool,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let rle = MaskRle {
        width,
        height,
        order: if coco { RleOrder::CocoColumnMajor } else { RleOrder::RowMajor },
        counts,
    };
    let mask = decode_mask_runs(&rle).map_err(to_py_err)?;
    let array =
        Array2::from_shape_vec((height, width), mask.into_image().into_vec()).map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Converts an `(H, W, 3)` float32 point map into a cloud.
#[pyfunction]
#[pyo3(signature = (points, confidence=None, min_confidence=0.0))]
fn point_map_to_point_cloud(
    points: PyReadonlyArray3<'_, f32>,
    confidence: Option<PyReadonlyArray2<'_, f32>>,
    min_confidence: f32,
) -> PyResult<PyPointCloud> {
    let view = points.as_array();
    let shape = view.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("expected point map shape (H, W, 3)"));
    }
    let point_map =
        PointMap::try_new(shape[1], shape[0], view.iter().copied().collect()).map_err(to_py_err)?;
    let confidence_map = if let Some(confidence) = confidence {
        let confidence = confidence.as_array();
        Some(
            ConfidenceMap::try_new(
                confidence.shape()[1],
                confidence.shape()[0],
                confidence.iter().copied().collect(),
            )
            .map_err(to_py_err)?,
        )
    } else {
        None
    };
    let inner = point_map_to_cloud(&point_map, confidence_map.as_ref(), min_confidence)
        .map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// SpatialRust Python bindings.
#[pymodule]
#[pyo3(name = "spatialrust")]
fn spatialrust_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyPipelineResult>()?;
    m.add_class::<PyRegionResult>()?;
    m.add_class::<PyDbscanResult>()?;
    m.add_class::<PyGroundResult>()?;
    m.add_class::<PyMultiPlaneResult>()?;
    m.add_class::<PySphereResult>()?;
    m.add_class::<PyCylinderResult>()?;
    m.add_class::<PyRegistrationResult>()?;
    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(write, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_downsample, m)?)?;
    m.add_function(wrap_pyfunction!(crop_box, m)?)?;
    m.add_function(wrap_pyfunction!(pass_through, m)?)?;
    m.add_function(wrap_pyfunction!(iss_keypoints, m)?)?;
    m.add_function(wrap_pyfunction!(orient_normals, m)?)?;
    m.add_function(wrap_pyfunction!(detect_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(mls_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(farthest_point_sampling, m)?)?;
    m.add_function(wrap_pyfunction!(statistical_outlier_removal, m)?)?;
    m.add_function(wrap_pyfunction!(radius_outlier_removal, m)?)?;
    m.add_function(wrap_pyfunction!(run_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(region_growing, m)?)?;
    m.add_function(wrap_pyfunction!(dbscan, m)?)?;
    m.add_function(wrap_pyfunction!(ground_segmentation, m)?)?;
    m.add_function(wrap_pyfunction!(segment_multi_plane, m)?)?;
    m.add_function(wrap_pyfunction!(ransac_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(ransac_cylinder, m)?)?;
    m.add_function(wrap_pyfunction!(chamfer_distance, m)?)?;
    m.add_function(wrap_pyfunction!(hausdorff_distance, m)?)?;
    m.add_function(wrap_pyfunction!(apply_transform, m)?)?;
    m.add_function(wrap_pyfunction!(recenter, m)?)?;
    m.add_function(wrap_pyfunction!(scale, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_unit_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    m.add_function(wrap_pyfunction!(centroid, m)?)?;
    m.add_function(wrap_pyfunction!(bounding_box, m)?)?;
    m.add_function(wrap_pyfunction!(oriented_bounding_box, m)?)?;
    m.add_function(wrap_pyfunction!(voxelize, m)?)?;
    m.add_function(wrap_pyfunction!(range_image, m)?)?;
    m.add_function(wrap_pyfunction!(rgbd_to_point_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image, m)?)?;
    m.add_function(wrap_pyfunction!(letterbox_image, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_image_chw, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_gray_image, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_hsv_image, m)?)?;
    m.add_function(wrap_pyfunction!(remap_image, m)?)?;
    m.add_function(wrap_pyfunction!(nms, m)?)?;
    m.add_function(wrap_pyfunction!(soft_nms, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components_image, m)?)?;
    m.add_function(wrap_pyfunction!(find_mask_contours, m)?)?;
    m.add_function(wrap_pyfunction!(encode_mask_rle, m)?)?;
    m.add_function(wrap_pyfunction!(decode_mask_rle, m)?)?;
    m.add_function(wrap_pyfunction!(point_map_to_point_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(knn_graph, m)?)?;
    m.add_function(wrap_pyfunction!(radius_graph, m)?)?;
    m.add_function(wrap_pyfunction!(register_icp, m)?)?;
    m.add_function(wrap_pyfunction!(register_point_to_plane, m)?)?;
    m.add_function(wrap_pyfunction!(register_gicp, m)?)?;
    m.add_function(wrap_pyfunction!(register_ndt, m)?)?;
    m.add_function(wrap_pyfunction!(register_fpfh_ransac, m)?)?;
    m.add_function(wrap_pyfunction!(register_fpfh_keypoints, m)?)?;
    Ok(())
}
