//! Python bindings for SpatialRust.
//!
//! Exposes the native-Rust point cloud pipeline (IO, voxel downsampling, RANSAC
//! plane segmentation, Euclidean clustering) to Python with zero-copy-friendly
//! NumPy interop. Build with `maturin develop`.

// PyO3's `#[pyfunction]` expansion emits `.into()` on already-`PyErr` results.
#![allow(clippy::useless_conversion)]

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2};
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
    DbscanConfig, DbscanSegmenter, GroundConfig, GroundSegmenter, RansacCylinderSegmenter,
    RansacPrimitiveConfig, RansacSphereSegmenter, RegionGrowingConfig, RegionGrowingSegmenter,
};
use spatialrust::transform::{
    apply_transform as apply_tf, bounding_box as bbox, centroid as cloud_centroid, merge_clouds,
    normalize_unit_sphere as normalize_unit, oriented_bounding_box as obb, recenter as recenter_op,
    scale_cloud,
};
use spatialrust::voxelize::{voxelize as voxelize_grid, VoxelFill, VoxelGridConfig};
use spatialrust::{
    read_point_cloud_file, write_point_cloud_file, ExecutionPolicy, HasPositions3, PointCloud,
    StandardSchemas,
};

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
#[pyo3(signature = (cloud, search_radius=0.1, angle_threshold=1.5708, min_neighbors=5, k_neighbors=20))]
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
fn oriented_bounding_box(
    cloud: &PyPointCloud,
) -> PyResult<((f32, f32, f32), (f32, f32, f32), Vec<(f32, f32, f32)>)> {
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

/// SpatialRust — PyTorch for Spatial Computing.
#[pymodule]
#[pyo3(name = "spatialrust")]
fn spatialrust_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyPipelineResult>()?;
    m.add_class::<PyRegionResult>()?;
    m.add_class::<PyDbscanResult>()?;
    m.add_class::<PyGroundResult>()?;
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
    m.add_function(wrap_pyfunction!(register_icp, m)?)?;
    m.add_function(wrap_pyfunction!(register_point_to_plane, m)?)?;
    m.add_function(wrap_pyfunction!(register_gicp, m)?)?;
    m.add_function(wrap_pyfunction!(register_ndt, m)?)?;
    m.add_function(wrap_pyfunction!(register_fpfh_ransac, m)?)?;
    m.add_function(wrap_pyfunction!(register_fpfh_keypoints, m)?)?;
    Ok(())
}
