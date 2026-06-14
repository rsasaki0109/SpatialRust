//! Python bindings for SpatialRust.
//!
//! Exposes the native-Rust point cloud pipeline (IO, voxel downsampling, RANSAC
//! plane segmentation, Euclidean clustering) to Python with zero-copy-friendly
//! NumPy interop. Build with `maturin develop`.

// PyO3's `#[pyfunction]` expansion emits `.into()` on already-`PyErr` results.
#![allow(clippy::useless_conversion)]

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use spatialrust::core::{PointBuffer, PointBufferSet, SpatialMetadata};
use spatialrust::features::{FeatureEstimator, NormalEstimationConfig, NormalEstimator};
use spatialrust::filtering::{VoxelGridDownsample, VoxelGridDownsampleConfig};
use spatialrust::pipeline::{MvpPipeline, MvpPipelineConfig};
use spatialrust::registration::{
    GicpConfig, GicpRegistration, IcpConfig, IcpRegistration, NdtConfig, NdtRegistration,
    PointCloudRegistration, PointToPlaneIcp, PointToPlaneIcpConfig, RegistrationResult,
};
use spatialrust::segmentation::{RegionGrowingConfig, RegionGrowingSegmenter};
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

/// SpatialRust — PyTorch for Spatial Computing.
#[pymodule]
#[pyo3(name = "spatialrust")]
fn spatialrust_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyPipelineResult>()?;
    m.add_class::<PyRegionResult>()?;
    m.add_class::<PyRegistrationResult>()?;
    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(write, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_downsample, m)?)?;
    m.add_function(wrap_pyfunction!(run_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(region_growing, m)?)?;
    m.add_function(wrap_pyfunction!(register_icp, m)?)?;
    m.add_function(wrap_pyfunction!(register_point_to_plane, m)?)?;
    m.add_function(wrap_pyfunction!(register_gicp, m)?)?;
    m.add_function(wrap_pyfunction!(register_ndt, m)?)?;
    Ok(())
}
