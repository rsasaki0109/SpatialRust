//! Python bindings for SpatialRust.
//!
//! Exposes the native-Rust point cloud pipeline (IO, voxel downsampling, RANSAC
//! plane segmentation, Euclidean clustering) to Python with zero-copy-friendly
//! NumPy interop. Build with `maturin develop`.

// PyO3's `#[pyfunction]` expansion emits `.into()` on already-`PyErr` results.
#![allow(clippy::useless_conversion)]
#![deny(unsafe_code)]

#[allow(unsafe_code)]
mod dlpack_capsule;

use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyBufferError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[cfg(feature = "onnxruntime")]
use spatialrust::ai::{
    CopyPolicy as AiCopyPolicy, InferenceBackend, IoBinding as AiIoBinding, ModelSession,
    ModelSource, NamedTensors as AiNamedTensors, OnnxRuntimeBackend, OutputBinding,
    RunOptions as AiRunOptions, SessionOptions as AiSessionOptions,
};

use spatialrust::camera::{
    calibrate_fisheye as calibrate_fisheye_native, calibrate_pinhole as calibrate_pinhole_native,
    CalibrationOptions, FisheyeObservation, PinholeObservation,
};
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
use spatialrust::image_io::{
    decode_path as decode_image_path, encode_path as encode_image_path, DecodeOptions,
    DecodedMetadata, DecodedPixels, EncodeOptions, ImageFileFormat,
};
use spatialrust::math::{Mat3, Mat4, Vec2, Vec3};
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
use spatialrust::tensor::{
    DataType as TensorDataType, Device as TensorDevice, DlpackImport, TensorBuffer,
    TensorDescriptor,
};
use spatialrust::transform::{
    apply_transform as apply_tf, bounding_box as bbox, centroid as cloud_centroid, merge_clouds,
    normalize_unit_sphere as normalize_unit, oriented_bounding_box as obb, recenter as recenter_op,
    scale_cloud,
};
use spatialrust::vision::{
    adaptive_threshold as adaptive_threshold_op, approximate_polygon as approximate_contour,
    batched_nms as batched_nms_op, bilateral_filter as bilateral_filter_op,
    canny_into as canny_into_op, clahe as clahe_op, connected_components_u8 as label_components_u8,
    decode_rle as decode_mask_runs, detect_and_describe_orb as detect_and_describe_orb_op,
    detect_fast as detect_fast_op, detect_harris as detect_harris_op,
    detect_shi_tomasi as detect_shi_tomasi_op, dilate_rect_u8_into as dilate_rect_u8_into_op,
    distance_transform_edt_u8_into as distance_transform_edt_u8_into_op,
    distance_transform_edt_with_spacing as distance_transform_edt_op,
    encode_rle as encode_mask_runs, equalize_histogram as equalize_histogram_op,
    erode_rect_u8_into as erode_rect_u8_into_op,
    estimate_homography_ransac as estimate_homography_ransac_op,
    estimate_rgbd_odometry as estimate_rgbd_odometry_op, filter2d as filter2d_op,
    find_contours as trace_contours,
    gaussian_blur_u8_into as gaussian_blur_u8_into_op,
    gray_world_white_balance as gray_world_white_balance_op, histogram_u8 as histogram_u8_op,
    integral_image as integral_image_op, laplacian as laplacian_op, letterbox as letterbox_op,
    match_descriptors as match_descriptors_op, median_blur as median_blur_op,
    morphology_ex as morphology_ex_op, morphology_rect_u8_into as morphology_rect_u8_into_op,
    nms as nms_op, otsu_threshold_u8 as otsu_threshold_u8_op, pack_chw as pack_chw_op,
    pack_chw_into as pack_chw_into_op, point_map_to_point_cloud as point_map_to_cloud,
    pyr_down as pyr_down_op, pyr_up as pyr_up_op, remap as remap_op, resize as resize_op,
    resize_into as resize_into_op, resize_pack_chw as resize_pack_chw_op,
    resize_pack_chw_into as resize_pack_chw_into_op, resize_rgb_to_gray as resize_rgb_to_gray_op,
    resize_rgb_to_gray_into as resize_rgb_to_gray_into_op, rgb_to_gray as rgb_to_gray_op,
    rgb_to_gray_into as rgb_to_gray_into_op, rgb_to_hsv as rgb_to_hsv_op, scharr as scharr_op,
    sobel as sobel_op, sobel_3x3_u8 as sobel_3x3_u8_op, sobel_3x3_u8_into as sobel_3x3_u8_into_op,
    sobel_abs_3x3_u8 as sobel_abs_3x3_u8_op, sobel_abs_3x3_u8_into as sobel_abs_3x3_u8_into_op,
    sobel_l1_magnitude_u8 as sobel_l1_magnitude_u8_op,
    sobel_l1_magnitude_u8_into as sobel_l1_magnitude_u8_into_op,
    sobel_threshold_3x3_u8 as sobel_threshold_3x3_u8_op,
    sobel_threshold_3x3_u8_into as sobel_threshold_3x3_u8_into_op, soft_nms as soft_nms_op,
    solve_pnp as solve_pnp_op, spatial_gradient_u8 as spatial_gradient_u8_op,
    spatial_gradient_u8_into as spatial_gradient_u8_into_op,
    stereo_block_match as stereo_block_match_op, stitch_panorama_pair as stitch_panorama_pair_op,
    threshold as threshold_op, AbsolutePose, AdaptiveThresholdMethod, BilinearResizeU8Plan,
    BinaryMask, BorderMode, BoundingBox2, CameraMatrix3, CannyOptions, CannyWorkspace,
    ConfidenceMap, Connectivity, CornerSelectionOptions, DescriptorBuffer, Detection,
    DistanceTransformWorkspace, FastOptions, GaussianBlurU8Workspace, HarrisOptions, Interpolation,
    Kernel2D, Keypoint2, MaskRle, MatchOptions, MorphologyOperation, MorphologyShape,
    ObjectImageCorrespondence, OrbOptions, OrbScoreType, PanoramaOptions, PerspectiveTransform,
    PointCorrespondence2, PointMap, RectMorphologyWorkspace, RgbdOdometryOptions, RleOrder,
    RobustEstimationOptions, ShiTomasiOptions, SoftNmsMethod, StereoBmOptions, StructuringElement,
    ThresholdType, TrackState,
};
use spatialrust::vision::{
    dense_flow_block_match as dense_flow_native, DenseFlowOptions,
    MultiObjectTracker as NativeMultiObjectTracker, MultiObjectTrackerOptions,
};
use spatialrust::voxelize::{
    range_image as range_image_proj, voxelize as voxelize_grid, RangeImageConfig, VoxelFill,
    VoxelGridConfig,
};
use spatialrust::{
    depth_to_xyz_dense as depth_to_xyz_native, depth_to_xyz_dense_into,
    rgbd_to_point_cloud as rgbd_to_cloud, BrownConrady, CameraIntrinsics, DepthConversionOptions,
    Image, ImageView, ImageViewMut, PinholeCamera,
};
use spatialrust::{
    knn_graph as knn_graph_build, radius_graph as radius_graph_build, NeighborGraph,
};
use spatialrust::{
    read_point_cloud_file, write_point_cloud_file, ExecutionPolicy, HasPositions3, PointCloud,
    StandardSchemas,
};

type Vec3Tuple = (f32, f32, f32);
type OrientedBoundingBoxTuple = (Vec3Tuple, Vec3Tuple, Vec<Vec3Tuple>);
type ObjectTrackTuple = (u64, f32, f32, f32, f32, i64, f32, u32, u32, u32, bool);
type ComponentStats = Vec<(u32, usize, (f32, f32, f32, f32))>;

fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyclass(name = "Tensor")]
#[derive(Clone)]
struct PyTensor {
    inner: TensorBuffer,
}

#[pyclass(name = "Keypoint2", frozen)]
#[derive(Clone, Copy)]
struct PyKeypoint2 {
    inner: Keypoint2,
}

#[pymethods]
impl PyKeypoint2 {
    #[getter]
    fn x(&self) -> f32 {
        self.inner.x()
    }

    #[getter]
    fn y(&self) -> f32 {
        self.inner.y()
    }

    #[getter]
    fn size(&self) -> f32 {
        self.inner.size()
    }

    #[getter]
    fn angle_degrees(&self) -> Option<f32> {
        self.inner.angle_degrees()
    }

    #[getter]
    fn response(&self) -> f32 {
        self.inner.response()
    }

    #[getter]
    fn octave(&self) -> i32 {
        self.inner.octave()
    }

    #[getter]
    fn class_id(&self) -> Option<i32> {
        self.inner.class_id()
    }

    fn __repr__(&self) -> String {
        format!(
            "Keypoint2(x={}, y={}, size={}, angle_degrees={:?}, response={})",
            self.x(),
            self.y(),
            self.size(),
            self.angle_degrees(),
            self.response()
        )
    }
}

#[pyclass(name = "OnnxRuntimeSession", unsendable)]
struct PyOnnxRuntimeSession {
    #[cfg(feature = "onnxruntime")]
    inner: Box<dyn ModelSession>,
}

#[pymethods]
impl PyOnnxRuntimeSession {
    #[new]
    #[pyo3(signature = (path, *, intra_threads=None, inter_threads=None, deterministic=false))]
    fn new(
        path: String,
        intra_threads: Option<usize>,
        inter_threads: Option<usize>,
        deterministic: bool,
    ) -> PyResult<Self> {
        #[cfg(feature = "onnxruntime")]
        {
            let options = AiSessionOptions {
                intra_threads,
                inter_threads,
                deterministic,
                ..AiSessionOptions::default()
            };
            let inner = OnnxRuntimeBackend
                .create_session(&ModelSource::Path(path.into()), &options)
                .map_err(to_py_err)?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "onnxruntime"))]
        {
            let _ = (path, intra_threads, inter_threads, deterministic);
            Err(PyRuntimeError::new_err(
                "this SpatialRust Python module was built without the `onnxruntime` feature",
            ))
        }
    }

    /// Returns `(name, dtype, dimensions)` metadata for model inputs.
    #[getter]
    fn inputs(&self) -> Vec<(String, String, Vec<String>)> {
        #[cfg(feature = "onnxruntime")]
        {
            self.inner.model_info().inputs.iter().map(python_tensor_spec).collect()
        }
        #[cfg(not(feature = "onnxruntime"))]
        {
            Vec::new()
        }
    }

    /// Returns `(name, dtype, dimensions)` metadata for model outputs.
    #[getter]
    fn outputs(&self) -> Vec<(String, String, Vec<String>)> {
        #[cfg(feature = "onnxruntime")]
        {
            self.inner.model_info().outputs.iter().map(python_tensor_spec).collect()
        }
        #[cfg(not(feature = "onnxruntime"))]
        {
            Vec::new()
        }
    }

    /// Runs named tensors; zero-copy CPU I/O binding is the default.
    #[pyo3(signature = (inputs, *, copy=false))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        copy: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        #[cfg(feature = "onnxruntime")]
        {
            let mut named = AiNamedTensors::new();
            for (name, value) in inputs.iter() {
                let name = name.extract::<String>()?;
                let tensor = value.extract::<PyRef<'_, PyTensor>>()?;
                named.insert(name, tensor.inner.clone()).map_err(to_py_err)?;
            }
            let outputs = if copy {
                self.inner
                    .run_with_options(
                        named,
                        AiRunOptions {
                            input_copy: AiCopyPolicy::Allow,
                            output_copy: AiCopyPolicy::Allow,
                        },
                    )
                    .map_err(to_py_err)?
            } else {
                let destinations = self
                    .inner
                    .model_info()
                    .outputs
                    .iter()
                    .map(|spec| OutputBinding::Allocate {
                        name: spec.name.clone(),
                        device: TensorDevice::CPU,
                    })
                    .collect();
                let mut binding = AiIoBinding::try_new(named, destinations).map_err(to_py_err)?;
                self.inner.run_with_binding(&mut binding).map_err(to_py_err)?;
                binding.into_results().ok_or_else(|| {
                    PyRuntimeError::new_err("ONNX Runtime completed without bound results")
                })?
            };
            let result = PyDict::new_bound(py);
            for (name, tensor) in outputs.into_values() {
                result.set_item(name, Py::new(py, PyTensor { inner: tensor })?)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "onnxruntime"))]
        {
            let _ = (py, inputs, copy);
            Err(PyRuntimeError::new_err(
                "this SpatialRust Python module was built without the `onnxruntime` feature",
            ))
        }
    }
}

#[cfg(feature = "onnxruntime")]
fn python_tensor_spec(spec: &spatialrust::ai::TensorSpec) -> (String, String, Vec<String>) {
    let dimensions = spec
        .shape
        .iter()
        .map(|dimension| match dimension {
            spatialrust::ai::Dimension::Fixed(value) => value.to_string(),
            spatialrust::ai::Dimension::Dynamic => "?".into(),
            spatialrust::ai::Dimension::Symbol(value) => value.clone(),
        })
        .collect();
    (spec.name.clone(), tensor_dtype_name(spec.dtype), dimensions)
}

#[pyclass(name = "DLPackTensorView", unsendable)]
struct PyDlpackTensorView {
    inner: DlpackImport,
}

#[pymethods]
impl PyDlpackTensorView {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.descriptor().shape().to_vec()
    }

    #[getter]
    fn dtype(&self) -> String {
        tensor_dtype_name(self.inner.descriptor().dtype())
    }

    #[getter]
    fn version(&self) -> (u32, u32) {
        self.inner.version()
    }

    /// Makes ownership independent of the DLPack producer with an explicit copy.
    fn copy(&self) -> PyResult<PyTensor> {
        Ok(PyTensor { inner: self.inner.view().map_err(to_py_err)?.to_owned_copy() })
    }

    fn __repr__(&self) -> String {
        format!(
            "DLPackTensorView(shape={:?}, dtype='{}', device='cpu', version={:?})",
            self.shape(),
            self.dtype(),
            self.version()
        )
    }
}

fn tensor_dtype_name(dtype: TensorDataType) -> String {
    match dtype {
        TensorDataType::U8 => "uint8".into(),
        TensorDataType::U16 => "uint16".into(),
        TensorDataType::F32 => "float32".into(),
        _ => format!("{:?}{}x{}", dtype.code(), dtype.bits(), dtype.lanes()),
    }
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.descriptor().shape().to_vec()
    }

    #[getter]
    fn dtype(&self) -> String {
        tensor_dtype_name(self.inner.descriptor().dtype())
    }

    /// Returns the DLPack CPU device tuple.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (1, 0)
    }

    /// Exports a read-only, zero-copy DLPack major-version 1 capsule.
    #[pyo3(signature = (stream=None, *, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &self,
        py: Python<'_>,
        stream: Option<Py<PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if stream.is_some() {
            return Err(PyBufferError::new_err("CPU DLPack export requires stream=None"));
        }
        if max_version.is_some_and(|version| version.0 < 1) {
            return Err(PyBufferError::new_err(
                "consumer does not support the DLPack versioned ABI",
            ));
        }
        if dl_device.is_some_and(|device| device != (1, 0)) {
            return Err(PyBufferError::new_err(
                "Tensor is CPU-resident; explicit device transfer is required",
            ));
        }
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "implicit DLPack copies are disabled; call Tensor.copy() explicitly",
            ));
        }
        dlpack_capsule::export_tensor(py, &self.inner).map_err(to_py_err)
    }

    /// Makes an explicit host-to-host allocation copy.
    fn copy(&self) -> Self {
        Self { inner: self.inner.to_owned_copy() }
    }

    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, dtype='{}', device='cpu')", self.shape(), self.dtype())
    }
}

/// Copies a NumPy uint8, uint16, or float32 array into packed CPU tensor storage.
#[pyfunction]
fn tensor_copy_from_numpy(array: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    if let Ok(array) = array.extract::<PyReadonlyArrayDyn<'_, u8>>() {
        let shape = array.shape().to_vec();
        let bytes = array.as_array().iter().copied().collect::<Vec<_>>();
        let descriptor = TensorDescriptor::contiguous(TensorDataType::U8, shape, TensorDevice::CPU);
        return Ok(PyTensor {
            inner: TensorBuffer::try_new(bytes, descriptor).map_err(to_py_err)?,
        });
    }
    if let Ok(array) = array.extract::<PyReadonlyArrayDyn<'_, u16>>() {
        let shape = array.shape().to_vec();
        let values = array.as_array().iter().copied().collect::<Vec<_>>();
        let descriptor =
            TensorDescriptor::contiguous(TensorDataType::U16, shape, TensorDevice::CPU);
        return Ok(PyTensor {
            inner: TensorBuffer::try_from_u16(values, descriptor).map_err(to_py_err)?,
        });
    }
    if let Ok(array) = array.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        let shape = array.shape().to_vec();
        let values = array.as_array().iter().copied().collect::<Vec<_>>();
        let descriptor =
            TensorDescriptor::contiguous(TensorDataType::F32, shape, TensorDevice::CPU);
        return Ok(PyTensor {
            inner: TensorBuffer::try_from_f32(values, descriptor).map_err(to_py_err)?,
        });
    }
    Err(PyTypeError::new_err("expected a NumPy uint8, uint16, or float32 array"))
}

/// Takes ownership of a producer's CPU DLPack capsule without copying its allocation.
#[pyfunction]
fn tensor_view_from_dlpack(producer: &Bound<'_, PyAny>) -> PyResult<PyDlpackTensorView> {
    let inner = dlpack_capsule::import_tensor(producer)?;
    Ok(PyDlpackTensorView { inner })
}

fn parse_image_format(format: &str) -> PyResult<ImageFileFormat> {
    match format.to_ascii_lowercase().as_str() {
        "png" => Ok(ImageFileFormat::Png),
        "jpg" | "jpeg" => Ok(ImageFileFormat::Jpeg),
        "pnm" | "pbm" | "pgm" | "ppm" => Ok(ImageFileFormat::Pnm),
        other => Err(PyValueError::new_err(format!(
            "unsupported image format `{other}` (expected: png, jpeg, or pnm)"
        ))),
    }
}

/// Source metadata returned alongside a decoded NumPy image.
#[pyclass(name = "ImageMetadata", frozen)]
#[derive(Clone)]
struct PyImageMetadata {
    inner: DecodedMetadata,
}

#[pymethods]
impl PyImageMetadata {
    #[getter]
    fn format(&self) -> String {
        self.inner.format.to_string()
    }

    #[getter]
    fn color_type(&self) -> String {
        format!("{:?}", self.inner.source_color_type)
    }

    #[getter]
    fn orientation(&self) -> u8 {
        self.inner.orientation as u8
    }

    #[getter]
    fn orientation_applied(&self) -> bool {
        self.inner.orientation_applied
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageMetadata(format='{}', color_type='{}', orientation={}, orientation_applied={})",
            self.format(),
            self.color_type(),
            self.orientation(),
            self.orientation_applied()
        )
    }
}

/// Decodes PNG, JPEG, or PNM into an owned NumPy array and source metadata.
#[pyfunction]
#[pyo3(signature = (path, apply_orientation=true))]
fn read_image<'py>(
    py: Python<'py>,
    path: &str,
    apply_orientation: bool,
) -> PyResult<(Py<PyAny>, PyImageMetadata)> {
    let decoded =
        decode_image_path(path, DecodeOptions { apply_orientation, ..Default::default() })
            .map_err(to_py_err)?;
    let metadata = PyImageMetadata { inner: decoded.metadata() };
    let (height, width) = (decoded.height(), decoded.width());
    macro_rules! array2 {
        ($image:expr) => {
            Array2::from_shape_vec((height, width), $image.into_vec())
                .map_err(to_py_err)?
                .into_pyarray_bound(py)
                .into_any()
                .unbind()
        };
    }
    macro_rules! array3 {
        ($image:expr, $channels:expr) => {
            Array3::from_shape_vec((height, width, $channels), $image.into_vec())
                .map_err(to_py_err)?
                .into_pyarray_bound(py)
                .into_any()
                .unbind()
        };
    }
    let array = match decoded.into_pixels() {
        DecodedPixels::Gray8(image) => array2!(image),
        DecodedPixels::GrayAlpha8(image) => array3!(image, 2),
        DecodedPixels::Rgb8(image) => array3!(image, 3),
        DecodedPixels::Rgba8(image) => array3!(image, 4),
        DecodedPixels::Gray16(image) => array2!(image),
        DecodedPixels::GrayAlpha16(image) => array3!(image, 2),
        DecodedPixels::Rgb16(image) => array3!(image, 3),
        DecodedPixels::Rgba16(image) => array3!(image, 4),
        DecodedPixels::Rgb32Float(image) => array3!(image, 3),
        DecodedPixels::Rgba32Float(image) => array3!(image, 4),
    };
    Ok((array, metadata))
}

/// Encodes a uint8/uint16 NumPy image as PNG, JPEG, or PNM.
#[pyfunction]
#[pyo3(signature = (path, image, format, jpeg_quality=90))]
fn write_image(
    path: &str,
    image: &Bound<'_, PyAny>,
    format: &str,
    jpeg_quality: u8,
) -> PyResult<()> {
    let format = parse_image_format(format)?;
    let pixels = if let Ok(array) = image.extract::<PyReadonlyArray2<'_, u8>>() {
        let view = array.as_array();
        let shape = view.shape();
        DecodedPixels::Gray8(
            Image::try_new(shape[1], shape[0], view.iter().copied().collect())
                .map_err(to_py_err)?,
        )
    } else if let Ok(array) = image.extract::<PyReadonlyArray2<'_, u16>>() {
        let view = array.as_array();
        let shape = view.shape();
        DecodedPixels::Gray16(
            Image::try_new(shape[1], shape[0], view.iter().copied().collect())
                .map_err(to_py_err)?,
        )
    } else if let Ok(array) = image.extract::<PyReadonlyArray3<'_, u8>>() {
        let view = array.as_array();
        let shape = view.shape();
        let packed = view.iter().copied().collect();
        match shape[2] {
            2 => DecodedPixels::GrayAlpha8(
                Image::try_new(shape[1], shape[0], packed).map_err(to_py_err)?,
            ),
            3 => {
                DecodedPixels::Rgb8(Image::try_new(shape[1], shape[0], packed).map_err(to_py_err)?)
            }
            4 => {
                DecodedPixels::Rgba8(Image::try_new(shape[1], shape[0], packed).map_err(to_py_err)?)
            }
            channels => {
                return Err(PyValueError::new_err(format!(
                    "expected 2, 3, or 4 channels, found {channels}"
                )))
            }
        }
    } else if let Ok(array) = image.extract::<PyReadonlyArray3<'_, u16>>() {
        let view = array.as_array();
        let shape = view.shape();
        let packed = view.iter().copied().collect();
        match shape[2] {
            2 => DecodedPixels::GrayAlpha16(
                Image::try_new(shape[1], shape[0], packed).map_err(to_py_err)?,
            ),
            3 => {
                DecodedPixels::Rgb16(Image::try_new(shape[1], shape[0], packed).map_err(to_py_err)?)
            }
            4 => DecodedPixels::Rgba16(
                Image::try_new(shape[1], shape[0], packed).map_err(to_py_err)?,
            ),
            channels => {
                return Err(PyValueError::new_err(format!(
                    "expected 2, 3, or 4 channels, found {channels}"
                )))
            }
        }
    } else {
        return Err(PyValueError::new_err(
            "expected a uint8 or uint16 NumPy array shaped (H, W) or (H, W, C)",
        ));
    };
    encode_image_path(path, &pixels, EncodeOptions { format, jpeg_quality }).map_err(to_py_err)
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

fn parse_threshold_type(value: &str) -> PyResult<ThresholdType> {
    match value.to_ascii_lowercase().as_str() {
        "binary" => Ok(ThresholdType::Binary),
        "binary_inv" | "binary-inv" => Ok(ThresholdType::BinaryInv),
        "truncate" | "trunc" => Ok(ThresholdType::Truncate),
        "to_zero" | "to-zero" => Ok(ThresholdType::ToZero),
        "to_zero_inv" | "to-zero-inv" => Ok(ThresholdType::ToZeroInv),
        other => Err(PyValueError::new_err(format!("unknown threshold type `{other}`"))),
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

fn rgb_image_view_from_numpy<'a, 'py>(
    array: &'a PyReadonlyArray3<'py, u8>,
    packed: &'a mut Vec<u8>,
) -> PyResult<ImageView<'a, u8, 3>> {
    let shape = array.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("expected an (H, W, 3) uint8 RGB array"));
    }
    let (height, width) = (shape[0], shape[1]);
    let data = if let Ok(slice) = array.as_slice() {
        slice
    } else {
        packed.extend(array.as_array().iter().copied());
        packed.as_slice()
    };
    ImageView::new(width, height, width * 3, data).map_err(to_py_err)
}

fn gray_u8_image_from_numpy(array: PyReadonlyArray2<'_, u8>) -> PyResult<Image<u8, 1>> {
    let view = array.as_array();
    let shape = view.shape();
    Image::try_new(shape[1], shape[0], view.iter().copied().collect()).map_err(to_py_err)
}

fn gray_u8_image_view_from_numpy<'a, 'py>(
    array: &'a PyReadonlyArray2<'py, u8>,
    packed: &'a mut Vec<u8>,
) -> PyResult<ImageView<'a, u8, 1>> {
    let shape = array.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected an (H, W) uint8 grayscale array"));
    }
    let (height, width) = (shape[0], shape[1]);
    let data = if let Ok(slice) = array.as_slice() {
        slice
    } else {
        packed.extend(array.as_array().iter().copied());
        packed.as_slice()
    };
    ImageView::new(width, height, width, data).map_err(to_py_err)
}

/// Fits pinhole intrinsics from known camera-space points and image pixels.
#[pyfunction]
#[pyo3(signature = (camera_points, pixels, width, height, huber_delta=2.0, max_iterations=12))]
fn calibrate_pinhole_camera(
    camera_points: PyReadonlyArray2<'_, f64>,
    pixels: PyReadonlyArray2<'_, f64>,
    width: usize,
    height: usize,
    huber_delta: f64,
    max_iterations: usize,
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let points = camera_points.as_array();
    let pixels = pixels.as_array();
    if points.shape().len() != 2 || points.shape()[1] != 3 {
        return Err(PyValueError::new_err("camera_points must have shape (N, 3)"));
    }
    if pixels.shape() != [points.shape()[0], 2] {
        return Err(PyValueError::new_err("pixels must have shape (N, 2)"));
    }
    let observations = (0..points.shape()[0])
        .map(|index| PinholeObservation {
            camera_point: Vec3::new(points[[index, 0]], points[[index, 1]], points[[index, 2]]),
            pixel: spatialrust::Vec2 { x: pixels[[index, 0]], y: pixels[[index, 1]] },
        })
        .collect::<Vec<_>>();
    let (camera, report) = calibrate_pinhole_native(
        &observations,
        width,
        height,
        CalibrationOptions { max_iterations, huber_delta, ..CalibrationOptions::default() },
    )
    .map_err(to_py_err)?;
    Ok((
        camera.intrinsics.fx,
        camera.intrinsics.fy,
        camera.intrinsics.cx,
        camera.intrinsics.cy,
        report.rms_residual,
        report.max_residual,
    ))
}

/// Fits Kannala–Brandt4 coefficients from incident angles and distorted radii.
#[pyfunction]
fn calibrate_fisheye_angles(
    theta: PyReadonlyArray1<'_, f64>,
    distorted_radius: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let theta = theta.as_array();
    let radii = distorted_radius.as_array();
    if theta.len() != radii.len() {
        return Err(PyValueError::new_err("theta and distorted_radius lengths must match"));
    }
    let observations = theta
        .iter()
        .copied()
        .zip(radii.iter().copied())
        .map(|(theta, distorted_radius)| FisheyeObservation { theta, distorted_radius })
        .collect::<Vec<_>>();
    let (model, report) = calibrate_fisheye_native(&observations).map_err(to_py_err)?;
    Ok((model.k1, model.k2, model.k3, model.k4, report.rms_residual))
}

/// Computes dense integer grayscale flow as an `(H, W, 2)` float32 array.
#[pyfunction]
#[pyo3(signature = (previous, next, block_radius=2, search_radius=4))]
fn dense_flow_image<'py>(
    py: Python<'py>,
    previous: PyReadonlyArray2<'_, u8>,
    next: PyReadonlyArray2<'_, u8>,
    block_radius: usize,
    search_radius: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let previous = gray_u8_image_from_numpy(previous)?;
    let next = gray_u8_image_from_numpy(next)?;
    let flow = dense_flow_native(
        previous.view(),
        next.view(),
        DenseFlowOptions { block_radius, search_radius, minimum_improvement: 1 },
    )
    .map_err(to_py_err)?;
    let array = Array3::from_shape_vec(
        (previous.height(), previous.width(), 2),
        flow.image().as_slice().to_vec(),
    )
    .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Stateful deterministic same-class IoU tracker.
#[pyclass(name = "MultiObjectTracker")]
struct PyMultiObjectTracker {
    inner: NativeMultiObjectTracker,
}

#[pymethods]
impl PyMultiObjectTracker {
    /// Creates an empty tracker.
    #[new]
    #[pyo3(signature = (iou_threshold=0.3, max_missed=3, min_confirmed_hits=2))]
    fn new(iou_threshold: f32, max_missed: u32, min_confirmed_hits: u32) -> PyResult<Self> {
        let inner = NativeMultiObjectTracker::try_new(MultiObjectTrackerOptions {
            iou_threshold,
            max_missed,
            min_confirmed_hits,
        })
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Associates detections and returns rows ending in a confirmed 0/1 field.
    fn update(
        &mut self,
        boxes: PyReadonlyArray2<'_, f32>,
        scores: PyReadonlyArray1<'_, f32>,
        class_ids: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Vec<ObjectTrackTuple>> {
        let boxes = boxes.as_array();
        let scores = scores.as_array();
        let class_ids = class_ids.as_array();
        if boxes.ndim() != 2
            || boxes.shape()[1] != 4
            || scores.len() != boxes.shape()[0]
            || class_ids.len() != boxes.shape()[0]
        {
            return Err(PyValueError::new_err(
                "boxes must be Nx4 with matching scores and class_ids",
            ));
        }
        let detections = boxes
            .outer_iter()
            .zip(scores.iter())
            .zip(class_ids.iter())
            .map(|((row, &score), &class_id)| {
                Ok(Detection {
                    bbox: BoundingBox2::try_new(row[0], row[1], row[2], row[3])
                        .map_err(to_py_err)?,
                    score,
                    class_id,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let tracks = self.inner.update(&detections).map_err(to_py_err)?;
        Ok(tracks
            .iter()
            .map(|track| {
                (
                    track.id,
                    track.bbox.x_min,
                    track.bbox.y_min,
                    track.bbox.x_max,
                    track.bbox.y_max,
                    track.class_id,
                    track.score,
                    track.age,
                    track.hits,
                    track.missed,
                    track.state == TrackState::Confirmed,
                )
            })
            .collect())
    }
}

/// Applies gray-world white balance to an RGB image.
#[pyfunction]
fn gray_world_white_balance_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = rgb_image_from_numpy(image)?;
    let output = gray_world_white_balance_op(image.view()).map_err(to_py_err)?;
    let array = Array3::from_shape_vec((output.height(), output.width(), 3), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Stitches a source RGB image into target coordinates with a 3x3 homography.
#[pyfunction]
#[pyo3(signature = (source, target, homography, max_output_pixels=67108864))]
fn stitch_panorama_pair<'py>(
    py: Python<'py>,
    source: PyReadonlyArray3<'_, u8>,
    target: PyReadonlyArray3<'_, u8>,
    homography: PyReadonlyArray2<'_, f64>,
    max_output_pixels: usize,
) -> PyResult<(Bound<'py, PyArray3<u8>>, i32, i32)> {
    let source = rgb_image_from_numpy(source)?;
    let target = rgb_image_from_numpy(target)?;
    let matrix = homography.as_array();
    if matrix.shape() != [3, 3] {
        return Err(PyValueError::new_err("homography must have shape (3, 3)"));
    }
    let panorama = stitch_panorama_pair_op(
        source.view(),
        target.view(),
        PerspectiveTransform {
            matrix: [
                [matrix[[0, 0]], matrix[[0, 1]], matrix[[0, 2]]],
                [matrix[[1, 0]], matrix[[1, 1]], matrix[[1, 2]]],
                [matrix[[2, 0]], matrix[[2, 1]], matrix[[2, 2]]],
            ],
        },
        PanoramaOptions { max_output_pixels },
    )
    .map_err(to_py_err)?;
    let (origin_x, origin_y) = (panorama.origin_x(), panorama.origin_y());
    let image = panorama.image();
    let array =
        Array3::from_shape_vec((image.height(), image.width(), 3), image.as_slice().to_vec())
            .map_err(to_py_err)?;
    Ok((array.into_pyarray_bound(py), origin_x, origin_y))
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

/// Converts depth into a dense `(H, W, 3)` XYZ image (invalid depths → NaN).
///
/// Pass a contiguous `out` buffer of shape `(H, W, 3)` to fill in place and avoid
/// per-frame allocation (typical streaming RGB-D).
#[pyfunction]
#[pyo3(signature = (
    depth,
    fx,
    fy,
    cx,
    cy,
    depth_scale=1.0,
    min_depth=f32::EPSILON,
    max_depth=f32::INFINITY,
    distortion=None,
    out=None
))]
#[allow(clippy::too_many_arguments)]
fn depth_to_xyz<'py>(
    py: Python<'py>,
    depth: PyReadonlyArray2<'_, f32>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    depth_scale: f32,
    min_depth: f32,
    max_depth: f32,
    distortion: Option<(f64, f64, f64, f64, f64)>,
    out: Option<Bound<'py, PyArray3<f32>>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let depth_view = depth.as_array();
    let shape = depth_view.shape();
    let height = shape[0];
    let width = shape[1];
    let packed;
    let depth_slice: &[f32] = match depth_view.as_slice() {
        Some(slice) => slice,
        None => {
            packed = depth_view.iter().copied().collect::<Vec<_>>();
            packed.as_slice()
        }
    };
    let depth_image =
        ImageView::<f32, 1>::new(width, height, width, depth_slice).map_err(to_py_err)?;
    let intrinsics = CameraIntrinsics::try_new(fx, fy, cx, cy, width, height).map_err(to_py_err)?;
    let mut camera = PinholeCamera::new(intrinsics);
    if let Some((k1, k2, p1, p2, k3)) = distortion {
        camera = camera.with_distortion(BrownConrady { k1, k2, p1, p2, k3 });
    }
    let options = DepthConversionOptions { depth_scale, min_depth, max_depth };
    if let Some(out) = out {
        {
            let mut out_rw = out.readwrite();
            let mut out_view = out_rw.as_array_mut();
            if out_view.shape() != [height, width, 3] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({height}, {width}, 3), found {:?}",
                    out_view.shape()
                )));
            }
            let Some(out_slice) = out_view.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous float32 array of shape (H, W, 3)",
                ));
            };
            depth_to_xyz_dense_into(depth_image, &camera, options, out_slice).map_err(to_py_err)?;
        }
        return Ok(out);
    }
    let xyz = depth_to_xyz_native(depth_image, &camera, options).map_err(to_py_err)?;
    let array = Array3::from_shape_vec((height, width, 3), xyz).map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
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

    // Prefer contiguous NumPy buffers; only pack when strides force it.
    let depth_owned;
    let depth_slice: &[f32] = match depth_view.as_slice() {
        Some(slice) => slice,
        None => {
            depth_owned = depth_view.iter().copied().collect::<Vec<_>>();
            depth_owned.as_slice()
        }
    };
    let color_owned;
    let color_slice: &[u8] = match color_view.as_slice() {
        Some(slice) => slice,
        None => {
            color_owned = color_view.iter().copied().collect::<Vec<_>>();
            color_owned.as_slice()
        }
    };
    let depth_image =
        ImageView::<f32, 1>::new(width, height, width, depth_slice).map_err(to_py_err)?;
    let color_image =
        ImageView::<u8, 3>::new(width, height, width * 3, color_slice).map_err(to_py_err)?;
    let intrinsics = CameraIntrinsics::try_new(fx, fy, cx, cy, width, height).map_err(to_py_err)?;
    let mut camera = PinholeCamera::new(intrinsics);
    if let Some((k1, k2, p1, p2, k3)) = distortion {
        camera = camera.with_distortion(BrownConrady { k1, k2, p1, p2, k3 });
    }
    let options = DepthConversionOptions { depth_scale, min_depth, max_depth };
    let inner = rgbd_to_cloud(depth_image, color_image, &camera, options).map_err(to_py_err)?;
    Ok(PyPointCloud { inner })
}

/// Correlates an RGB image with a 2D float64 kernel using Reflect101 borders.
#[pyfunction]
#[pyo3(signature = (image, kernel, delta=0.0))]
fn filter2d_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    kernel: PyReadonlyArray2<'_, f64>,
    delta: f64,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = rgb_image_from_numpy(image)?;
    let kernel_view = kernel.as_array();
    let shape = kernel_view.shape();
    let kernel = Kernel2D::try_new(shape[1], shape[0], kernel_view.iter().copied().collect())
        .map_err(to_py_err)?;
    let output =
        filter2d_op(image.view(), &kernel, delta, BorderMode::Reflect101).map_err(to_py_err)?;
    let array = Array3::from_shape_vec((image.height(), image.width(), 3), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

thread_local! {
    /// Reused across every `gaussian_blur_image` call, since the Python entry point has
    /// no way to pass a persisted workspace; without pooling, the horizontal intermediate
    /// buffer would be re-allocated from scratch on every call, allocate or reuse alike.
    static POOLED_GAUSSIAN_BLUR_WORKSPACE: std::cell::RefCell<GaussianBlurU8Workspace> =
        std::cell::RefCell::new(GaussianBlurU8Workspace::new());
}

/// Applies a normalized Gaussian blur to an RGB image using Reflect101 borders.
#[pyfunction]
#[pyo3(signature = (image, kernel_width, kernel_height, sigma_x, sigma_y=None, out=None))]
fn gaussian_blur_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    kernel_width: usize,
    kernel_height: usize,
    sigma_x: f64,
    sigma_y: Option<f64>,
    out: Option<Bound<'py, PyArray3<u8>>>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let mut packed = Vec::new();
    let image = rgb_image_view_from_numpy(&image, &mut packed)?;
    let sigma_y = sigma_y.unwrap_or(sigma_x);
    POOLED_GAUSSIAN_BLUR_WORKSPACE.with(|cell| {
        let mut workspace = cell.borrow_mut();
        if let Some(out) = out {
            {
                let mut out_rw = out.try_readwrite().map_err(|_| {
                    PyValueError::new_err("out must not overlap the Gaussian input")
                })?;
                let mut out_array = out_rw.as_array_mut();
                if out_array.shape() != [image.height(), image.width(), 3] {
                    return Err(PyValueError::new_err(format!(
                        "out shape must be ({}, {}, 3), found {:?}",
                        image.height(),
                        image.width(),
                        out_array.shape()
                    )));
                }
                let Some(out_slice) = out_array.as_slice_mut() else {
                    return Err(PyValueError::new_err(
                        "out must be a contiguous uint8 array of shape (H, W, 3)",
                    ));
                };
                gaussian_blur_u8_into_op(
                    image,
                    kernel_width,
                    kernel_height,
                    sigma_x,
                    sigma_y,
                    BorderMode::Reflect101,
                    out_slice,
                    &mut workspace,
                )
                .map_err(to_py_err)?;
            }
            return Ok(out);
        }
        let mut output = vec![0_u8; image.width() * image.height() * 3];
        gaussian_blur_u8_into_op(
            image,
            kernel_width,
            kernel_height,
            sigma_x,
            sigma_y,
            BorderMode::Reflect101,
            &mut output,
            &mut workspace,
        )
        .map_err(to_py_err)?;
        let array = Array3::from_shape_vec((image.height(), image.width(), 3), output)
            .map_err(to_py_err)?;
        Ok(array.into_pyarray_bound(py))
    })
}

/// Applies an odd-aperture median filter to an RGB image.
#[pyfunction]
fn median_blur_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    kernel_size: usize,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = rgb_image_from_numpy(image)?;
    let output =
        median_blur_op(image.view(), kernel_size, BorderMode::Replicate).map_err(to_py_err)?;
    let array = Array3::from_shape_vec((image.height(), image.width(), 3), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Applies an RGB bilateral filter using Reflect101 borders.
#[pyfunction]
fn bilateral_filter_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    diameter: usize,
    sigma_color: f64,
    sigma_space: f64,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = rgb_image_from_numpy(image)?;
    let output = bilateral_filter_op(
        image.view(),
        diameter,
        sigma_color,
        sigma_space,
        BorderMode::Reflect101,
    )
    .map_err(to_py_err)?;
    let array = Array3::from_shape_vec((image.height(), image.width(), 3), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Computes a signed float32 Sobel derivative from a grayscale uint8 image.
#[pyfunction]
#[pyo3(signature = (image, dx, dy, kernel_size=3, scale=1.0, delta=0.0, out=None))]
fn sobel_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    dx: usize,
    dy: usize,
    kernel_size: usize,
    scale: f64,
    delta: f64,
    out: Option<Bound<'py, PyArray2<f32>>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut packed = Vec::new();
    let image = gray_u8_image_view_from_numpy(&image, &mut packed)?;
    if let Some(out) = out {
        {
            let mut out_rw = out
                .try_readwrite()
                .map_err(|_| PyValueError::new_err("out must not overlap the Sobel input"))?;
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [image.height(), image.width()] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({}, {}), found {:?}",
                    image.height(),
                    image.width(),
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous float32 array of shape (H, W)",
                ));
            };
            if kernel_size == 3 && dx + dy == 1 {
                sobel_3x3_u8_into_op(
                    image,
                    dx,
                    dy,
                    scale,
                    delta,
                    BorderMode::Reflect101,
                    out_slice,
                )
                .map_err(to_py_err)?;
            } else {
                let output =
                    sobel_op(image, dx, dy, kernel_size, scale, delta, BorderMode::Reflect101)
                        .map_err(to_py_err)?;
                out_slice.copy_from_slice(output.as_slice());
            }
        }
        return Ok(out);
    }
    if kernel_size == 3 && dx + dy == 1 {
        let output = sobel_3x3_u8_op(image, dx, dy, scale, delta, BorderMode::Reflect101)
            .map_err(to_py_err)?;
        let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
            .map_err(to_py_err)?;
        return Ok(array.into_pyarray_bound(py));
    }
    let output = sobel_op(image, dx, dy, kernel_size, scale, delta, BorderMode::Reflect101)
        .map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Computes fused absolute 3x3 Sobel response as saturated uint8.
#[pyfunction]
#[pyo3(signature = (image, dx, dy, out=None))]
fn sobel_abs_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    dx: usize,
    dy: usize,
    out: Option<Bound<'py, PyArray2<u8>>>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let mut packed = Vec::new();
    let image = gray_u8_image_view_from_numpy(&image, &mut packed)?;
    if let Some(out) = out {
        {
            let mut out_rw = out
                .try_readwrite()
                .map_err(|_| PyValueError::new_err("out must not overlap the Sobel input"))?;
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [image.height(), image.width()] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({}, {}), found {:?}",
                    image.height(),
                    image.width(),
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous uint8 array of shape (H, W)",
                ));
            };
            sobel_abs_3x3_u8_into_op(image, dx, dy, BorderMode::Reflect101, out_slice)
                .map_err(to_py_err)?;
        }
        return Ok(out);
    }
    let output = sobel_abs_3x3_u8_op(image, dx, dy, BorderMode::Reflect101).map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Computes a fused binary mask from absolute 3x3 Sobel response.
#[pyfunction]
#[pyo3(signature = (image, dx, dy, threshold, out=None))]
fn sobel_threshold_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    dx: usize,
    dy: usize,
    threshold: u8,
    out: Option<Bound<'py, PyArray2<u8>>>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let mut packed = Vec::new();
    let image = gray_u8_image_view_from_numpy(&image, &mut packed)?;
    if let Some(out) = out {
        {
            let mut out_rw = out
                .try_readwrite()
                .map_err(|_| PyValueError::new_err("out must not overlap the Sobel input"))?;
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [image.height(), image.width()] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({}, {}), found {:?}",
                    image.height(),
                    image.width(),
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous uint8 array of shape (H, W)",
                ));
            };
            sobel_threshold_3x3_u8_into_op(
                image,
                dx,
                dy,
                threshold,
                BorderMode::Reflect101,
                out_slice,
            )
            .map_err(to_py_err)?;
        }
        return Ok(out);
    }
    let output = sobel_threshold_3x3_u8_op(image, dx, dy, threshold, BorderMode::Reflect101)
        .map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Computes exact paired 3x3 Sobel X/Y gradients from grayscale uint8 input.
#[pyfunction]
#[pyo3(signature = (image, out_dx=None, out_dy=None))]
fn spatial_gradient_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    out_dx: Option<Bound<'py, PyArray2<i16>>>,
    out_dy: Option<Bound<'py, PyArray2<i16>>>,
) -> PyResult<(Bound<'py, PyArray2<i16>>, Bound<'py, PyArray2<i16>>)> {
    let mut packed = Vec::new();
    let image = gray_u8_image_view_from_numpy(&image, &mut packed)?;
    match (out_dx, out_dy) {
        (None, None) => {
            let (gradient_x, gradient_y) =
                spatial_gradient_u8_op(image, BorderMode::Reflect101).map_err(to_py_err)?;
            let gradient_x =
                Array2::from_shape_vec((image.height(), image.width()), gradient_x.into_vec())
                    .map_err(to_py_err)?
                    .into_pyarray_bound(py);
            let gradient_y =
                Array2::from_shape_vec((image.height(), image.width()), gradient_y.into_vec())
                    .map_err(to_py_err)?
                    .into_pyarray_bound(py);
            Ok((gradient_x, gradient_y))
        }
        (Some(out_dx), Some(out_dy)) => {
            {
                let mut dx_rw = out_dx.try_readwrite().map_err(|_| {
                    PyValueError::new_err("out_dx must not overlap the gradient input")
                })?;
                let mut dy_rw = out_dy.try_readwrite().map_err(|_| {
                    PyValueError::new_err("out_dy must not overlap out_dx or the gradient input")
                })?;
                let mut dx_array = dx_rw.as_array_mut();
                let mut dy_array = dy_rw.as_array_mut();
                let expected = [image.height(), image.width()];
                if dx_array.shape() != expected {
                    return Err(PyValueError::new_err(format!(
                        "out_dx shape must be ({}, {}), found {:?}",
                        image.height(),
                        image.width(),
                        dx_array.shape()
                    )));
                }
                if dy_array.shape() != expected {
                    return Err(PyValueError::new_err(format!(
                        "out_dy shape must be ({}, {}), found {:?}",
                        image.height(),
                        image.width(),
                        dy_array.shape()
                    )));
                }
                let Some(dx_slice) = dx_array.as_slice_mut() else {
                    return Err(PyValueError::new_err(
                        "out_dx must be a contiguous int16 array of shape (H, W)",
                    ));
                };
                let Some(dy_slice) = dy_array.as_slice_mut() else {
                    return Err(PyValueError::new_err(
                        "out_dy must be a contiguous int16 array of shape (H, W)",
                    ));
                };
                spatial_gradient_u8_into_op(image, BorderMode::Reflect101, dx_slice, dy_slice)
                    .map_err(to_py_err)?;
            }
            Ok((out_dx, out_dy))
        }
        _ => Err(PyValueError::new_err("out_dx and out_dy must be provided together")),
    }
}

/// Computes exact fused 3x3 Sobel L1 magnitude from grayscale uint8 input.
#[pyfunction]
#[pyo3(signature = (image, out=None))]
fn sobel_l1_magnitude_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    out: Option<Bound<'py, PyArray2<i16>>>,
) -> PyResult<Bound<'py, PyArray2<i16>>> {
    let mut packed = Vec::new();
    let image = gray_u8_image_view_from_numpy(&image, &mut packed)?;
    if let Some(out) = out {
        {
            let mut out_rw = out.try_readwrite().map_err(|_| {
                PyValueError::new_err("out must not overlap the Sobel magnitude input")
            })?;
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [image.height(), image.width()] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({}, {}), found {:?}",
                    image.height(),
                    image.width(),
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous int16 array of shape (H, W)",
                ));
            };
            sobel_l1_magnitude_u8_into_op(image, BorderMode::Reflect101, out_slice)
                .map_err(to_py_err)?;
        }
        return Ok(out);
    }
    let magnitude = sobel_l1_magnitude_u8_op(image, BorderMode::Reflect101).map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), magnitude.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Computes a signed float32 Scharr derivative from a grayscale uint8 image.
#[pyfunction]
#[pyo3(signature = (image, dx, dy, scale=1.0, delta=0.0))]
fn scharr_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    dx: usize,
    dy: usize,
    scale: f64,
    delta: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let image = gray_u8_image_from_numpy(image)?;
    let output =
        scharr_op(image.view(), dx, dy, scale, delta, BorderMode::Reflect101).map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Computes a signed float32 Laplacian from a grayscale uint8 image.
#[pyfunction]
#[pyo3(signature = (image, kernel_size=1, scale=1.0, delta=0.0))]
fn laplacian_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    kernel_size: usize,
    scale: f64,
    delta: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let image = gray_u8_image_from_numpy(image)?;
    let output = laplacian_op(image.view(), kernel_size, scale, delta, BorderMode::Reflect101)
        .map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Reduces an RGB image with the canonical Gaussian pyramid kernel.
#[pyfunction]
fn pyr_down_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = rgb_image_from_numpy(image)?;
    let output = pyr_down_op(image.view(), BorderMode::Reflect101).map_err(to_py_err)?;
    let array = Array3::from_shape_vec((output.height(), output.width(), 3), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Doubles an RGB image with the canonical Gaussian pyramid kernel.
#[pyfunction]
fn pyr_up_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = rgb_image_from_numpy(image)?;
    let output = pyr_up_op(image.view(), BorderMode::Reflect101).map_err(to_py_err)?;
    let array = Array3::from_shape_vec((output.height(), output.width(), 3), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Applies grayscale erosion, dilation, or a composite morphology operation.
#[pyclass(name = "MorphologyWorkspace")]
struct PyMorphologyWorkspace {
    inner: RectMorphologyWorkspace,
    element: Option<(usize, usize, StructuringElement)>,
}

#[pymethods]
impl PyMorphologyWorkspace {
    #[new]
    fn new() -> Self {
        Self { inner: RectMorphologyWorkspace::new(), element: None }
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[getter]
    fn worker_capacity(&self) -> usize {
        self.inner.worker_capacity()
    }

    #[getter]
    fn line_capacity(&self) -> usize {
        self.inner.line_capacity()
    }
}

thread_local! {
    /// Reused across `morphology_image` calls that omit an explicit workspace, so the
    /// convenience Python entry point does not re-allocate every scratch buffer on each call.
    static POOLED_RECT_MORPHOLOGY_WORKSPACE: std::cell::RefCell<RectMorphologyWorkspace> =
        std::cell::RefCell::new(RectMorphologyWorkspace::new());
}

fn morphology_rect_dispatch_into(
    image: ImageView<'_, u8, 1>,
    operation: &str,
    element: &StructuringElement,
    iterations: usize,
    output: &mut [u8],
    workspace: &mut RectMorphologyWorkspace,
) -> spatialrust::vision::VisionResult<()> {
    match operation {
        "erode" => erode_rect_u8_into_op(
            image,
            element,
            iterations,
            BorderMode::Replicate,
            output,
            workspace,
        ),
        "dilate" => dilate_rect_u8_into_op(
            image,
            element,
            iterations,
            BorderMode::Replicate,
            output,
            workspace,
        ),
        "open" => morphology_rect_u8_into_op(
            image,
            MorphologyOperation::Open,
            element,
            iterations,
            BorderMode::Replicate,
            output,
            workspace,
        ),
        "close" => morphology_rect_u8_into_op(
            image,
            MorphologyOperation::Close,
            element,
            iterations,
            BorderMode::Replicate,
            output,
            workspace,
        ),
        "gradient" => morphology_rect_u8_into_op(
            image,
            MorphologyOperation::Gradient,
            element,
            iterations,
            BorderMode::Replicate,
            output,
            workspace,
        ),
        "tophat" | "top-hat" => morphology_rect_u8_into_op(
            image,
            MorphologyOperation::TopHat,
            element,
            iterations,
            BorderMode::Replicate,
            output,
            workspace,
        ),
        "blackhat" | "black-hat" => morphology_rect_u8_into_op(
            image,
            MorphologyOperation::BlackHat,
            element,
            iterations,
            BorderMode::Replicate,
            output,
            workspace,
        ),
        other => Err(spatialrust::vision::VisionError::InvalidParameter(format!(
            "unknown morphology operation `{other}`"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn morphology_rect_python<'py>(
    py: Python<'py>,
    image: ImageView<'_, u8, 1>,
    operation: &str,
    element: &StructuringElement,
    iterations: usize,
    out: Option<Bound<'py, PyArray2<u8>>>,
    workspace: &mut RectMorphologyWorkspace,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    if let Some(out) = out {
        {
            let mut out_rw = out
                .try_readwrite()
                .map_err(|_| PyValueError::new_err("out must not overlap the morphology input"))?;
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [image.height(), image.width()] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({}, {}), found {:?}",
                    image.height(),
                    image.width(),
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous uint8 array of shape (H, W)",
                ));
            };
            morphology_rect_dispatch_into(
                image, operation, element, iterations, out_slice, workspace,
            )
            .map_err(to_py_err)?;
        }
        return Ok(out);
    }
    let mut output = vec![0; image.width() * image.height()];
    morphology_rect_dispatch_into(image, operation, element, iterations, &mut output, workspace)
        .map_err(to_py_err)?;
    let array =
        Array2::from_shape_vec((image.height(), image.width()), output).map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (image, operation, kernel_width, kernel_height, shape="rect", iterations=1, out=None, workspace=None))]
fn morphology_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    operation: &str,
    kernel_width: usize,
    kernel_height: usize,
    shape: &str,
    iterations: usize,
    out: Option<Bound<'py, PyArray2<u8>>>,
    mut workspace: Option<PyRefMut<'_, PyMorphologyWorkspace>>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let mut packed = Vec::new();
    let image = gray_u8_image_view_from_numpy(&image, &mut packed)?;
    let shape = match shape.to_ascii_lowercase().as_str() {
        "rect" | "rectangle" => MorphologyShape::Rect,
        "cross" => MorphologyShape::Cross,
        "ellipse" | "elliptical" => MorphologyShape::Ellipse,
        "diamond" => MorphologyShape::Diamond,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown morphology shape `{other}` (expected: rect, cross, ellipse, diamond)"
            )))
        }
    };
    let operation = operation.to_ascii_lowercase();
    let rect = shape == MorphologyShape::Rect;
    if workspace.is_some() && !rect {
        return Err(PyValueError::new_err(
            "workspace reuse is available only for rectangular morphology",
        ));
    }
    if rect {
        if let Some(workspace) = workspace.as_deref_mut() {
            let stale = match workspace.element.as_ref() {
                Some((width, height, _)) => *width != kernel_width || *height != kernel_height,
                None => true,
            };
            if stale {
                workspace.element = Some((
                    kernel_width,
                    kernel_height,
                    StructuringElement::try_new(MorphologyShape::Rect, kernel_width, kernel_height)
                        .map_err(to_py_err)?,
                ));
            }
            let PyMorphologyWorkspace { inner, element } = workspace;
            let element = &element.as_ref().expect("cached rectangular element").2;
            return morphology_rect_python(py, image, &operation, element, iterations, out, inner);
        }
        let element =
            StructuringElement::try_new(shape, kernel_width, kernel_height).map_err(to_py_err)?;
        return POOLED_RECT_MORPHOLOGY_WORKSPACE.with(|cell| {
            morphology_rect_python(
                py,
                image,
                &operation,
                &element,
                iterations,
                out,
                &mut cell.borrow_mut(),
            )
        });
    }
    let element =
        StructuringElement::try_new(shape, kernel_width, kernel_height).map_err(to_py_err)?;
    let output = match operation.as_str() {
        "erode" => spatialrust::vision::erode(image, &element, iterations, BorderMode::Replicate),
        "dilate" => spatialrust::vision::dilate(image, &element, iterations, BorderMode::Replicate),
        "open" => morphology_ex_op(
            image,
            MorphologyOperation::Open,
            &element,
            iterations,
            BorderMode::Replicate,
        ),
        "close" => morphology_ex_op(
            image,
            MorphologyOperation::Close,
            &element,
            iterations,
            BorderMode::Replicate,
        ),
        "gradient" => morphology_ex_op(
            image,
            MorphologyOperation::Gradient,
            &element,
            iterations,
            BorderMode::Replicate,
        ),
        "tophat" | "top-hat" => morphology_ex_op(
            image,
            MorphologyOperation::TopHat,
            &element,
            iterations,
            BorderMode::Replicate,
        ),
        "blackhat" | "black-hat" => morphology_ex_op(
            image,
            MorphologyOperation::BlackHat,
            &element,
            iterations,
            BorderMode::Replicate,
        ),
        other => {
            return Err(PyValueError::new_err(format!("unknown morphology operation `{other}`")))
        }
    }
    .map_err(to_py_err)?;
    if let Some(out) = out {
        {
            let mut out_rw = out
                .try_readwrite()
                .map_err(|_| PyValueError::new_err("out must not overlap the morphology input"))?;
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [image.height(), image.width()] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({}, {}), found {:?}",
                    image.height(),
                    image.width(),
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous uint8 array of shape (H, W)",
                ));
            };
            out_slice.copy_from_slice(output.as_slice());
        }
        return Ok(out);
    }
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Applies a fixed threshold to a grayscale uint8 image.
#[pyfunction]
#[pyo3(signature = (image, threshold, max_value=255, threshold_type="binary"))]
fn threshold_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    threshold: f64,
    max_value: u8,
    threshold_type: &str,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let image = gray_u8_image_from_numpy(image)?;
    let output = threshold_op(
        image.view(),
        threshold,
        f64::from(max_value),
        parse_threshold_type(threshold_type)?,
    )
    .map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Selects and applies an Otsu threshold, returning `(threshold, image)`.
#[pyfunction]
#[pyo3(signature = (image, max_value=255, threshold_type="binary"))]
fn otsu_threshold_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    max_value: u8,
    threshold_type: &str,
) -> PyResult<(u8, Bound<'py, PyArray2<u8>>)> {
    let image = gray_u8_image_from_numpy(image)?;
    let (selected, output) =
        otsu_threshold_u8_op(image.view(), max_value, parse_threshold_type(threshold_type)?)
            .map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok((selected, array.into_pyarray_bound(py)))
}

/// Applies mean or Gaussian adaptive thresholding.
#[pyfunction]
#[pyo3(signature = (image, block_size, c, method="mean", max_value=255, threshold_type="binary"))]
fn adaptive_threshold_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    block_size: usize,
    c: f64,
    method: &str,
    max_value: u8,
    threshold_type: &str,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let image = gray_u8_image_from_numpy(image)?;
    let method = match method.to_ascii_lowercase().as_str() {
        "mean" => AdaptiveThresholdMethod::Mean,
        "gaussian" => AdaptiveThresholdMethod::Gaussian,
        other => return Err(PyValueError::new_err(format!("unknown adaptive method `{other}`"))),
    };
    let output = adaptive_threshold_op(
        image.view(),
        max_value,
        method,
        parse_threshold_type(threshold_type)?,
        block_size,
        c,
        BorderMode::Replicate,
    )
    .map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Returns the exact 256-bin grayscale histogram.
#[pyfunction]
fn histogram_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let image = gray_u8_image_from_numpy(image)?;
    Ok(histogram_u8_op(image.view()).into_pyarray_bound(py))
}

/// Equalizes a grayscale uint8 histogram.
#[pyfunction]
fn equalize_histogram_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let image = gray_u8_image_from_numpy(image)?;
    let output = equalize_histogram_op(image.view()).map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Applies contrast-limited adaptive histogram equalization.
#[pyfunction]
#[pyo3(signature = (image, clip_limit=2.0, tiles_x=8, tiles_y=8))]
fn clahe_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    clip_limit: f64,
    tiles_x: usize,
    tiles_y: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let image = gray_u8_image_from_numpy(image)?;
    let output = clahe_op(image.view(), clip_limit, tiles_x, tiles_y).map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Computes the `(H + 1, W + 1)` float64 summed-area table.
#[pyfunction]
fn integral_image_u8<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let image = gray_u8_image_from_numpy(image)?;
    let integral = integral_image_op(image.view(), 0).map_err(to_py_err)?;
    let array =
        Array2::from_shape_vec((integral.height(), integral.width()), integral.as_slice().to_vec())
            .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Reusable host scratch storage for allocation-light Canny edge detection.
#[pyclass(name = "CannyWorkspace")]
struct PyCannyWorkspace {
    inner: CannyWorkspace,
}

#[pymethods]
impl PyCannyWorkspace {
    #[new]
    fn new() -> Self {
        Self { inner: CannyWorkspace::new() }
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[getter]
    fn allocated_bytes(&self) -> usize {
        self.inner.allocated_bytes()
    }
}

thread_local! {
    /// Reused across `canny_image` calls that omit an explicit workspace, so the
    /// convenience Python entry point does not re-allocate every scratch buffer on each call.
    static POOLED_CANNY_WORKSPACE: std::cell::RefCell<CannyWorkspace> =
        std::cell::RefCell::new(CannyWorkspace::new());
}

/// Detects edges in a grayscale uint8 image with Canny hysteresis.
#[pyfunction]
#[pyo3(signature = (image, low_threshold, high_threshold, aperture_size=3, l2_gradient=false, out=None, workspace=None))]
fn canny_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    low_threshold: f64,
    high_threshold: f64,
    aperture_size: usize,
    l2_gradient: bool,
    out: Option<Bound<'py, PyArray2<u8>>>,
    mut workspace: Option<PyRefMut<'_, PyCannyWorkspace>>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let mut packed = Vec::new();
    let image = gray_u8_image_view_from_numpy(&image, &mut packed)?;
    let options = CannyOptions { low_threshold, high_threshold, aperture_size, l2_gradient };

    let run = move |workspace: &mut CannyWorkspace| -> PyResult<Bound<'py, PyArray2<u8>>> {
        if let Some(out) = out {
            {
                let mut out_rw = out
                    .try_readwrite()
                    .map_err(|_| PyValueError::new_err("out must not overlap the Canny input"))?;
                let mut out_array = out_rw.as_array_mut();
                if out_array.shape() != [image.height(), image.width()] {
                    return Err(PyValueError::new_err(format!(
                        "out shape must be ({}, {}), found {:?}",
                        image.height(),
                        image.width(),
                        out_array.shape()
                    )));
                }
                let Some(out_slice) = out_array.as_slice_mut() else {
                    return Err(PyValueError::new_err(
                        "out must be a contiguous uint8 array of shape (H, W)",
                    ));
                };
                let output =
                    ImageViewMut::new(image.width(), image.height(), image.width(), out_slice)
                        .map_err(to_py_err)?;
                canny_into_op(image, options, output, workspace).map_err(to_py_err)?;
            }
            return Ok(out);
        }
        let mut output = vec![0_u8; image.width() * image.height()];
        let output_view =
            ImageViewMut::new(image.width(), image.height(), image.width(), &mut output)
                .map_err(to_py_err)?;
        canny_into_op(image, options, output_view, workspace).map_err(to_py_err)?;
        let array =
            Array2::from_shape_vec((image.height(), image.width()), output).map_err(to_py_err)?;
        Ok(array.into_pyarray_bound(py))
    };

    if let Some(workspace) = workspace.as_deref_mut() {
        return run(&mut workspace.inner);
    }
    POOLED_CANNY_WORKSPACE.with(|cell| run(&mut cell.borrow_mut()))
}

fn corner_selection_options(
    max_corners: usize,
    quality_level: f32,
    min_distance: f32,
    block_size: usize,
    gradient_size: usize,
) -> CornerSelectionOptions {
    CornerSelectionOptions {
        max_corners,
        quality_level,
        min_distance,
        block_size,
        gradient_size,
        border: BorderMode::Reflect101,
    }
}

/// Detects strongest-first Harris keypoints in a grayscale uint8 image.
#[pyfunction]
#[pyo3(signature = (image, max_corners=0, quality_level=0.01, min_distance=1.0, block_size=3, gradient_size=3, k=0.04))]
#[allow(clippy::too_many_arguments)]
fn harris_keypoints(
    image: PyReadonlyArray2<'_, u8>,
    max_corners: usize,
    quality_level: f32,
    min_distance: f32,
    block_size: usize,
    gradient_size: usize,
    k: f32,
) -> PyResult<Vec<PyKeypoint2>> {
    let image = gray_u8_image_from_numpy(image)?;
    let options = HarrisOptions {
        selection: corner_selection_options(
            max_corners,
            quality_level,
            min_distance,
            block_size,
            gradient_size,
        ),
        k,
    };
    Ok(detect_harris_op(image.view(), options)
        .map_err(to_py_err)?
        .into_iter()
        .map(|inner| PyKeypoint2 { inner })
        .collect())
}

/// Detects strongest-first Shi–Tomasi keypoints in a grayscale uint8 image.
#[pyfunction]
#[pyo3(signature = (image, max_corners=0, quality_level=0.01, min_distance=1.0, block_size=3, gradient_size=3))]
fn shi_tomasi_keypoints(
    image: PyReadonlyArray2<'_, u8>,
    max_corners: usize,
    quality_level: f32,
    min_distance: f32,
    block_size: usize,
    gradient_size: usize,
) -> PyResult<Vec<PyKeypoint2>> {
    let image = gray_u8_image_from_numpy(image)?;
    let options = ShiTomasiOptions {
        selection: corner_selection_options(
            max_corners,
            quality_level,
            min_distance,
            block_size,
            gradient_size,
        ),
    };
    Ok(detect_shi_tomasi_op(image.view(), options)
        .map_err(to_py_err)?
        .into_iter()
        .map(|inner| PyKeypoint2 { inner })
        .collect())
}

/// Detects scan-ordered FAST-9/16 keypoints in a grayscale uint8 image.
#[pyfunction]
#[pyo3(signature = (image, threshold=10, nonmax_suppression=true))]
fn fast_keypoints(
    image: PyReadonlyArray2<'_, u8>,
    threshold: u8,
    nonmax_suppression: bool,
) -> PyResult<Vec<PyKeypoint2>> {
    let image = gray_u8_image_from_numpy(image)?;
    Ok(detect_fast_op(image.view(), FastOptions { threshold, nonmax_suppression })
        .map_err(to_py_err)?
        .into_iter()
        .map(|inner| PyKeypoint2 { inner })
        .collect())
}

/// Detects ORB keypoints and returns `(keypoints, uint8[N, 32] descriptors)`.
#[pyfunction]
#[pyo3(signature = (image, max_features=500, scale_factor=1.2, levels=8, edge_threshold=31, fast_threshold=20, patch_size=31, score_type="harris"))]
#[allow(clippy::too_many_arguments)]
fn orb_features<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'_, u8>,
    max_features: usize,
    scale_factor: f32,
    levels: usize,
    edge_threshold: usize,
    fast_threshold: u8,
    patch_size: usize,
    score_type: &str,
) -> PyResult<(Vec<PyKeypoint2>, Bound<'py, PyArray2<u8>>)> {
    let image = gray_u8_image_from_numpy(image)?;
    let score_type = match score_type {
        "harris" => OrbScoreType::Harris,
        "fast" => OrbScoreType::Fast,
        _ => return Err(PyValueError::new_err("score_type must be 'harris' or 'fast'")),
    };
    let features = detect_and_describe_orb_op(
        image.view(),
        OrbOptions {
            max_features,
            scale_factor,
            levels,
            edge_threshold,
            fast_threshold,
            patch_size,
            score_type,
        },
    )
    .map_err(to_py_err)?;
    let keypoints =
        features.keypoints().iter().copied().map(|inner| PyKeypoint2 { inner }).collect();
    let descriptors = Array2::from_shape_vec(
        (features.descriptors().len(), features.descriptors().width()),
        features.descriptors().binary_data().expect("ORB descriptors are binary").to_vec(),
    )
    .map_err(to_py_err)?
    .into_pyarray_bound(py);
    Ok((keypoints, descriptors))
}

fn correspondences_from_numpy(
    source: PyReadonlyArray2<'_, f64>,
    target: PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<PointCorrespondence2>> {
    let source = source.as_array();
    let target = target.as_array();
    if source.shape() != target.shape() || source.ndim() != 2 || source.shape()[1] != 2 {
        return Err(PyValueError::new_err("source/target must be Nx2 float64 arrays"));
    }
    source
        .outer_iter()
        .zip(target.outer_iter())
        .map(|(src, dst)| {
            PointCorrespondence2::try_new(
                Vec2 { x: src[0], y: src[1] },
                Vec2 { x: dst[0], y: dst[1] },
            )
            .map_err(to_py_err)
        })
        .collect()
}

fn mat3_to_numpy<'py>(py: Python<'py>, matrix: Mat3<f64>) -> Bound<'py, PyArray2<f64>> {
    let mut values = Vec::with_capacity(9);
    for row in &matrix.m {
        values.extend_from_slice(row);
    }
    Array2::from_shape_vec((3, 3), values).expect("3x3").into_pyarray_bound(py)
}

/// Estimates a homography with deterministic RANSAC.
///
/// Returns `(matrix[3,3], inliers[N], residuals[N])`.
#[pyfunction]
#[pyo3(signature = (source, target, threshold=1.0, confidence=0.99, max_iterations=2000, seed=0))]
fn estimate_homography_ransac<'py>(
    py: Python<'py>,
    source: PyReadonlyArray2<'_, f64>,
    target: PyReadonlyArray2<'_, f64>,
    threshold: f64,
    confidence: f64,
    max_iterations: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<f64>>)> {
    let pairs = correspondences_from_numpy(source, target)?;
    let estimate = estimate_homography_ransac_op(
        &pairs,
        RobustEstimationOptions { threshold, confidence, max_iterations, seed },
    )
    .map_err(to_py_err)?;
    Ok((
        mat3_to_numpy(py, estimate.model().matrix()),
        numpy::PyArray1::from_vec_bound(py, estimate.inliers().to_vec()),
        numpy::PyArray1::from_vec_bound(py, estimate.residuals().to_vec()),
    ))
}

/// Solves PnP for object points `Nx3` and image points `Nx2`.
///
/// Camera intrinsics are `fx, fy, cx, cy`. Returns `(rotation[3,3], translation[3])`.
#[pyfunction]
#[pyo3(signature = (object_points, image_points, fx, fy, cx, cy, width=640, height=480))]
#[allow(clippy::too_many_arguments)]
fn solve_pnp<'py>(
    py: Python<'py>,
    object_points: PyReadonlyArray2<'_, f64>,
    image_points: PyReadonlyArray2<'_, f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    width: usize,
    height: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let objects = object_points.as_array();
    let images = image_points.as_array();
    if objects.ndim() != 2
        || images.ndim() != 2
        || objects.shape()[1] != 3
        || images.shape()[1] != 2
        || objects.shape()[0] != images.shape()[0]
    {
        return Err(PyValueError::new_err(
            "object_points must be Nx3 and image_points Nx2 with matching N",
        ));
    }
    let pairs = objects
        .outer_iter()
        .zip(images.outer_iter())
        .map(|(object, image)| {
            ObjectImageCorrespondence::try_new(
                Vec3::new(object[0], object[1], object[2]),
                Vec2 { x: image[0], y: image[1] },
            )
            .map_err(to_py_err)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let camera = CameraMatrix3::from_intrinsics(
        CameraIntrinsics::try_new(fx, fy, cx, cy, width, height).map_err(to_py_err)?,
    );
    let pose: AbsolutePose = solve_pnp_op(&pairs, camera).map_err(to_py_err)?;
    let translation = numpy::PyArray1::from_vec_bound(
        py,
        vec![pose.translation().x, pose.translation().y, pose.translation().z],
    );
    Ok((mat3_to_numpy(py, pose.rotation()), translation))
}

/// Estimates metric RGB-D odometry from source depth and pixel tracks.
#[pyfunction]
#[pyo3(signature = (depth, source, target, fx, fy, cx, cy, depth_scale=1.0, threshold=1.0))]
#[allow(clippy::too_many_arguments)]
fn estimate_rgbd_odometry<'py>(
    py: Python<'py>,
    depth: PyReadonlyArray2<'_, f32>,
    source: PyReadonlyArray2<'_, f64>,
    target: PyReadonlyArray2<'_, f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    depth_scale: f64,
    threshold: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<bool>>,
    usize,
)> {
    let depth_view = depth.as_array();
    let (height, width) = (depth_view.shape()[0], depth_view.shape()[1]);
    let depth_image = Image::<f32, 1>::try_new(width, height, depth_view.iter().copied().collect())
        .map_err(to_py_err)?;
    let pairs = correspondences_from_numpy(source, target)?;
    let camera = CameraMatrix3::from_intrinsics(
        CameraIntrinsics::try_new(fx, fy, cx, cy, width, height).map_err(to_py_err)?,
    );
    let estimate = estimate_rgbd_odometry_op(
        depth_image.view(),
        &pairs,
        camera,
        RgbdOdometryOptions {
            depth_scale,
            robust: RobustEstimationOptions { threshold, ..Default::default() },
            ..Default::default()
        },
    )
    .map_err(to_py_err)?;
    Ok((
        mat3_to_numpy(py, estimate.pose.rotation()),
        numpy::PyArray1::from_vec_bound(
            py,
            vec![
                estimate.pose.translation().x,
                estimate.pose.translation().y,
                estimate.pose.translation().z,
            ],
        ),
        numpy::PyArray1::from_vec_bound(py, estimate.inliers),
        estimate.rejected_depth_count,
    ))
}

/// Dense SAD stereo block matching on rectified grayscale images.
#[pyfunction]
#[pyo3(signature = (left, right, window_size=15, min_disparity=0, num_disparities=64, uniqueness_ratio=15.0))]
fn stereo_block_match<'py>(
    py: Python<'py>,
    left: PyReadonlyArray2<'_, u8>,
    right: PyReadonlyArray2<'_, u8>,
    window_size: usize,
    min_disparity: i32,
    num_disparities: i32,
    uniqueness_ratio: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let left = gray_u8_image_from_numpy(left)?;
    let right = gray_u8_image_from_numpy(right)?;
    let disparity = stereo_block_match_op(
        left.view(),
        right.view(),
        StereoBmOptions { window_size, min_disparity, num_disparities, uniqueness_ratio },
    )
    .map_err(to_py_err)?;
    let array = Array2::from_shape_vec(
        (disparity.height(), disparity.width()),
        disparity.as_slice().to_vec(),
    )
    .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

fn descriptor_match_tuples(
    query: DescriptorBuffer,
    train: DescriptorBuffer,
    cross_check: bool,
    ratio: Option<f32>,
    max_distance: Option<f32>,
) -> PyResult<Vec<(usize, usize, f32)>> {
    Ok(match_descriptors_op(&query, &train, MatchOptions { cross_check, ratio, max_distance })
        .map_err(to_py_err)?
        .into_iter()
        .map(|feature_match| {
            (feature_match.query_index(), feature_match.train_index(), feature_match.distance())
        })
        .collect())
}

/// Brute-force Hamming matching for two `uint8[N, D]` descriptor matrices.
#[pyfunction]
#[pyo3(signature = (query, train, cross_check=false, ratio=None, max_distance=None))]
fn match_binary_descriptors(
    query: PyReadonlyArray2<'_, u8>,
    train: PyReadonlyArray2<'_, u8>,
    cross_check: bool,
    ratio: Option<f32>,
    max_distance: Option<f32>,
) -> PyResult<Vec<(usize, usize, f32)>> {
    let query_shape = query.shape();
    let train_shape = train.shape();
    let query = DescriptorBuffer::try_binary(
        query_shape[0],
        query_shape[1],
        query.as_array().iter().copied().collect(),
    )
    .map_err(to_py_err)?;
    let train = DescriptorBuffer::try_binary(
        train_shape[0],
        train_shape[1],
        train.as_array().iter().copied().collect(),
    )
    .map_err(to_py_err)?;
    descriptor_match_tuples(query, train, cross_check, ratio, max_distance)
}

/// Brute-force Euclidean matching for two `float32[N, D]` descriptor matrices.
#[pyfunction]
#[pyo3(signature = (query, train, cross_check=false, ratio=None, max_distance=None))]
fn match_float_descriptors(
    query: PyReadonlyArray2<'_, f32>,
    train: PyReadonlyArray2<'_, f32>,
    cross_check: bool,
    ratio: Option<f32>,
    max_distance: Option<f32>,
) -> PyResult<Vec<(usize, usize, f32)>> {
    let query_shape = query.shape();
    let train_shape = train.shape();
    let query = DescriptorBuffer::try_float32(
        query_shape[0],
        query_shape[1],
        query.as_array().iter().copied().collect(),
    )
    .map_err(to_py_err)?;
    let train = DescriptorBuffer::try_float32(
        train_shape[0],
        train_shape[1],
        train.as_array().iter().copied().collect(),
    )
    .map_err(to_py_err)?;
    descriptor_match_tuples(query, train, cross_check, ratio, max_distance)
}

/// Resizes an `(H, W, 3)` uint8 RGB image.
#[pyfunction]
#[pyo3(signature = (image, width, height, interpolation="bilinear", out=None))]
fn resize_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    width: usize,
    height: usize,
    interpolation: &str,
    out: Option<Bound<'py, PyArray3<u8>>>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let mut packed = Vec::new();
    let image = rgb_image_view_from_numpy(&image, &mut packed)?;
    let interpolation = parse_interpolation(interpolation)?;
    let bilinear_plan = (interpolation == Interpolation::Bilinear)
        .then(|| BilinearResizeU8Plan::new(image.width(), image.height(), width, height))
        .transpose()
        .map_err(to_py_err)?;
    if let Some(out) = out {
        {
            let mut out_rw = out.readwrite();
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [height, width, 3] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({height}, {width}, 3), found {:?}",
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous uint8 array of shape (H, W, 3)",
                ));
            };
            let output = ImageViewMut::<u8, 3>::new(width, height, width * 3, out_slice)
                .map_err(to_py_err)?;
            if let Some(plan) = &bilinear_plan {
                plan.resize_into(image, output).map_err(to_py_err)?;
            } else {
                resize_into_op(image, output, interpolation).map_err(to_py_err)?;
            }
        }
        return Ok(out);
    }
    let output = if let Some(plan) = &bilinear_plan {
        plan.resize(image).map_err(to_py_err)?
    } else {
        resize_op(image, width, height, interpolation).map_err(to_py_err)?
    };
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
#[pyo3(signature = (image, scale=1.0/255.0, mean=None, std=None, out=None))]
fn normalize_image_chw<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    scale: f32,
    mean: Option<(f32, f32, f32)>,
    std: Option<(f32, f32, f32)>,
    out: Option<Bound<'py, PyArray3<f32>>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let mut packed = Vec::new();
    let image = rgb_image_view_from_numpy(&image, &mut packed)?;
    let mean = mean.map_or([0.0; 3], |(r, g, b)| [r, g, b]);
    let std = std.map_or([1.0; 3], |(r, g, b)| [r, g, b]);
    if let Some(out) = out {
        {
            let mut out_rw = out.readwrite();
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [3, image.height(), image.width()] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be (3, {}, {}), found {:?}",
                    image.height(),
                    image.width(),
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous float32 array of shape (3, H, W)",
                ));
            };
            pack_chw_into_op(image, scale, mean, std, out_slice).map_err(to_py_err)?;
        }
        return Ok(out);
    }
    let output = pack_chw_op(image, scale, mean, std).map_err(to_py_err)?;
    let array = Array3::from_shape_vec((3, image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Fuses bilinear RGB resize, normalization, and CHW packing.
#[pyfunction]
#[pyo3(signature = (image, width, height, scale=1.0/255.0, mean=None, std=None, out=None))]
#[allow(clippy::too_many_arguments)]
fn resize_normalize_image_chw<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    width: usize,
    height: usize,
    scale: f32,
    mean: Option<(f32, f32, f32)>,
    std: Option<(f32, f32, f32)>,
    out: Option<Bound<'py, PyArray3<f32>>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let mut packed = Vec::new();
    let image = rgb_image_view_from_numpy(&image, &mut packed)?;
    let mean = mean.map_or([0.0; 3], |(r, g, b)| [r, g, b]);
    let std = std.map_or([1.0; 3], |(r, g, b)| [r, g, b]);
    if let Some(out) = out {
        {
            let mut out_rw = out.readwrite();
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [3, height, width] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be (3, {height}, {width}), found {:?}",
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous float32 array of shape (3, H, W)",
                ));
            };
            resize_pack_chw_into_op(image, width, height, scale, mean, std, out_slice)
                .map_err(to_py_err)?;
        }
        return Ok(out);
    }
    let output = resize_pack_chw_op(image, width, height, scale, mean, std).map_err(to_py_err)?;
    let array = Array3::from_shape_vec((3, height, width), output.into_vec()).map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Converts an RGB image to an `(H, W)` grayscale image.
#[pyfunction]
#[pyo3(signature = (image, out=None))]
fn rgb_to_gray_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    out: Option<Bound<'py, PyArray2<u8>>>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let mut packed = Vec::new();
    let image = rgb_image_view_from_numpy(&image, &mut packed)?;
    if let Some(out) = out {
        {
            let mut out_rw = out.readwrite();
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [image.height(), image.width()] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({}, {}), found {:?}",
                    image.height(),
                    image.width(),
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous uint8 array of shape (H, W)",
                ));
            };
            let output =
                ImageViewMut::<u8, 1>::new(image.width(), image.height(), image.width(), out_slice)
                    .map_err(to_py_err)?;
            rgb_to_gray_into_op(image, output).map_err(to_py_err)?;
        }
        return Ok(out);
    }
    let output = rgb_to_gray_op(image).map_err(to_py_err)?;
    let array = Array2::from_shape_vec((image.height(), image.width()), output.into_vec())
        .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
}

/// Fuses bilinear resize and RGB-to-gray conversion into an `(H, W)` image.
#[pyfunction]
#[pyo3(signature = (image, width, height, out=None))]
fn resize_rgb_to_gray_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    width: usize,
    height: usize,
    out: Option<Bound<'py, PyArray2<u8>>>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let mut packed = Vec::new();
    let image = rgb_image_view_from_numpy(&image, &mut packed)?;
    if let Some(out) = out {
        {
            let mut out_rw = out.readwrite();
            let mut out_array = out_rw.as_array_mut();
            if out_array.shape() != [height, width] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({height}, {width}), found {:?}",
                    out_array.shape()
                )));
            }
            let Some(out_slice) = out_array.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous uint8 array of shape (H, W)",
                ));
            };
            let output =
                ImageViewMut::<u8, 1>::new(width, height, width, out_slice).map_err(to_py_err)?;
            resize_rgb_to_gray_into_op(image, output).map_err(to_py_err)?;
        }
        return Ok(out);
    }
    let output = resize_rgb_to_gray_op(image, width, height).map_err(to_py_err)?;
    let array = Array2::from_shape_vec((height, width), output.into_vec()).map_err(to_py_err)?;
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
    let scores_view = scores.as_array();
    let packed_scores;
    let scores = match scores_view.as_slice() {
        Some(scores) => scores,
        None => {
            packed_scores = scores_view.iter().copied().collect::<Vec<_>>();
            packed_scores.as_slice()
        }
    };
    let indices = nms_op(&native_boxes, scores, score_threshold, iou_threshold)
        .map_err(to_py_err)?
        .into_iter()
        .map(|index| index as i64)
        .collect::<Vec<_>>();
    Ok(indices.into_pyarray_bound(py))
}

/// Class-aware greedy NMS over `(N, 4)` xyxy boxes.
#[pyfunction]
#[pyo3(signature = (boxes, scores, class_ids, score_threshold=0.0, iou_threshold=0.5))]
fn batched_nms<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    scores: PyReadonlyArray1<'_, f32>,
    class_ids: PyReadonlyArray1<'_, i64>,
    score_threshold: f32,
    iou_threshold: f32,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let boxes_view = boxes.as_array();
    if boxes_view.shape().len() != 2 || boxes_view.shape()[1] != 4 {
        return Err(PyValueError::new_err("expected boxes with shape (N, 4)"));
    }
    let scores_view = scores.as_array();
    let packed_scores;
    let scores = match scores_view.as_slice() {
        Some(scores) => scores,
        None => {
            packed_scores = scores_view.iter().copied().collect::<Vec<_>>();
            packed_scores.as_slice()
        }
    };
    let class_ids_view = class_ids.as_array();
    let packed_class_ids;
    let class_ids = match class_ids_view.as_slice() {
        Some(class_ids) => class_ids,
        None => {
            packed_class_ids = class_ids_view.iter().copied().collect::<Vec<_>>();
            packed_class_ids.as_slice()
        }
    };
    let count = boxes_view.shape()[0];
    if scores.len() != count || class_ids.len() != count {
        return Err(PyValueError::new_err("boxes, scores, and class_ids must have equal lengths"));
    }
    let mut detections = Vec::with_capacity(count);
    for (index, row) in boxes_view.rows().into_iter().enumerate() {
        detections.push(Detection {
            bbox: BoundingBox2::try_new(row[0], row[1], row[2], row[3]).map_err(to_py_err)?,
            score: scores[index],
            class_id: class_ids[index],
        });
    }
    let indices = batched_nms_op(&detections, score_threshold, iou_threshold)
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
    let scores_view = scores.as_array();
    let packed_scores;
    let scores = match scores_view.as_slice() {
        Some(scores) => scores,
        None => {
            packed_scores = scores_view.iter().copied().collect::<Vec<_>>();
            packed_scores.as_slice()
        }
    };
    let result = soft_nms_op(&native_boxes, scores, score_threshold, iou_threshold, method)
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
    let view = mask.as_array();
    let shape = view.shape();
    let (height, width) = (shape[0], shape[1]);
    let packed;
    let pixels = match mask.as_slice() {
        Ok(slice) => slice,
        Err(_) => {
            packed = view.iter().copied().collect::<Vec<_>>();
            packed.as_slice()
        }
    };
    let connectivity = match connectivity {
        4 => Connectivity::Four,
        8 => Connectivity::Eight,
        _ => return Err(PyValueError::new_err("connectivity must be 4 or 8")),
    };
    let result = label_components_u8(width, height, pixels, connectivity).map_err(to_py_err)?;
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
    let labels = Array2::from_shape_vec((height, width), result.labels.into_image().into_vec())
        .map_err(to_py_err)?;
    Ok((labels.into_pyarray_bound(py), stats))
}

/// Reusable host scratch storage for exact unit-spacing distance transforms.
#[pyclass(name = "DistanceTransformWorkspace")]
struct PyDistanceTransformWorkspace {
    inner: DistanceTransformWorkspace,
}

#[pymethods]
impl PyDistanceTransformWorkspace {
    #[new]
    fn new() -> Self {
        Self { inner: DistanceTransformWorkspace::new() }
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

thread_local! {
    /// Reused across `distance_transform_edt` calls that omit an explicit workspace, so the
    /// convenience Python entry point does not re-allocate every scratch buffer on each call.
    static POOLED_EDT_WORKSPACE: std::cell::RefCell<DistanceTransformWorkspace> =
        std::cell::RefCell::new(DistanceTransformWorkspace::new());
}

/// Computes the exact Euclidean distance to the nearest zero-valued mask pixel.
#[pyfunction]
#[pyo3(signature = (mask, spacing=(1.0, 1.0), out=None, workspace=None))]
fn distance_transform_edt<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray2<'_, u8>,
    spacing: (f32, f32),
    out: Option<Bound<'py, PyArray2<f32>>>,
    workspace: Option<PyRefMut<'_, PyDistanceTransformWorkspace>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if workspace.is_some() && spacing != (1.0, 1.0) {
        return Err(PyValueError::new_err("workspace reuse currently requires spacing=(1.0, 1.0)"));
    }
    let mask_view = mask.as_array();
    let (height, width) = (mask_view.shape()[0], mask_view.shape()[1]);
    if let Some(mut workspace) = workspace {
        let packed;
        let input = match mask_view.as_slice() {
            Some(slice) => slice,
            None => {
                packed = mask_view.iter().copied().collect::<Vec<_>>();
                packed.as_slice()
            }
        };
        if let Some(out) = out {
            {
                let mut out_rw = out.readwrite();
                let mut out_view = out_rw.as_array_mut();
                if out_view.shape() != [height, width] {
                    return Err(PyValueError::new_err(format!(
                        "out shape must be ({height}, {width}), found {:?}",
                        out_view.shape()
                    )));
                }
                let Some(out_slice) = out_view.as_slice_mut() else {
                    return Err(PyValueError::new_err(
                        "out must be a contiguous float32 array of shape (H, W)",
                    ));
                };
                distance_transform_edt_u8_into_op(
                    input,
                    width,
                    height,
                    out_slice,
                    &mut workspace.inner,
                )
                .map_err(to_py_err)?;
            }
            return Ok(out);
        }
        let mut output = vec![0.0_f32; width * height];
        distance_transform_edt_u8_into_op(input, width, height, &mut output, &mut workspace.inner)
            .map_err(to_py_err)?;
        let array = Array2::from_shape_vec((height, width), output).map_err(to_py_err)?;
        return Ok(array.into_pyarray_bound(py));
    }

    if spacing == (1.0, 1.0) && width <= u16::MAX as usize && height <= u16::MAX as usize {
        let packed;
        let input = match mask_view.as_slice() {
            Some(slice) => slice,
            None => {
                packed = mask_view.iter().copied().collect::<Vec<_>>();
                packed.as_slice()
            }
        };
        return POOLED_EDT_WORKSPACE.with(|cell| {
            let mut pooled = cell.borrow_mut();
            if let Some(out) = out {
                {
                    let mut out_rw = out.readwrite();
                    let mut out_view = out_rw.as_array_mut();
                    if out_view.shape() != [height, width] {
                        return Err(PyValueError::new_err(format!(
                            "out shape must be ({height}, {width}), found {:?}",
                            out_view.shape()
                        )));
                    }
                    let Some(out_slice) = out_view.as_slice_mut() else {
                        return Err(PyValueError::new_err(
                            "out must be a contiguous float32 array of shape (H, W)",
                        ));
                    };
                    distance_transform_edt_u8_into_op(input, width, height, out_slice, &mut pooled)
                        .map_err(to_py_err)?;
                }
                return Ok(out);
            }
            let mut output = vec![0.0_f32; width * height];
            distance_transform_edt_u8_into_op(input, width, height, &mut output, &mut pooled)
                .map_err(to_py_err)?;
            let array = Array2::from_shape_vec((height, width), output).map_err(to_py_err)?;
            Ok(array.into_pyarray_bound(py))
        });
    }

    let image = gray_u8_image_from_numpy(mask)?;
    let binary = image.into_vec().into_iter().map(|value| u8::from(value != 0)).collect();
    let mask = BinaryMask::try_new(width, height, binary).map_err(to_py_err)?;
    if let Some(out) = out {
        {
            let mut out_rw = out.readwrite();
            let mut out_view = out_rw.as_array_mut();
            if out_view.shape() != [height, width] {
                return Err(PyValueError::new_err(format!(
                    "out shape must be ({height}, {width}), found {:?}",
                    out_view.shape()
                )));
            }
            let Some(out_slice) = out_view.as_slice_mut() else {
                return Err(PyValueError::new_err(
                    "out must be a contiguous float32 array of shape (H, W)",
                ));
            };
            let distances =
                distance_transform_edt_op(&mask, spacing.0, spacing.1).map_err(to_py_err)?;
            out_slice.copy_from_slice(distances.as_slice());
        }
        return Ok(out);
    }
    let distances = distance_transform_edt_op(&mask, spacing.0, spacing.1).map_err(to_py_err)?;
    let array =
        Array2::from_shape_vec((distances.height(), distances.width()), distances.into_vec())
            .map_err(to_py_err)?;
    Ok(array.into_pyarray_bound(py))
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
    m.add_class::<PyImageMetadata>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyKeypoint2>()?;
    m.add_class::<PyDistanceTransformWorkspace>()?;
    m.add_class::<PyMorphologyWorkspace>()?;
    m.add_class::<PyCannyWorkspace>()?;
    m.add_class::<PyMultiObjectTracker>()?;
    m.add_class::<PyOnnxRuntimeSession>()?;
    m.add_class::<PyDlpackTensorView>()?;
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyPipelineResult>()?;
    m.add_class::<PyRegionResult>()?;
    m.add_class::<PyDbscanResult>()?;
    m.add_class::<PyGroundResult>()?;
    m.add_class::<PyMultiPlaneResult>()?;
    m.add_class::<PySphereResult>()?;
    m.add_class::<PyCylinderResult>()?;
    m.add_class::<PyRegistrationResult>()?;
    m.add_function(wrap_pyfunction!(read_image, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_copy_from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_view_from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(harris_keypoints, m)?)?;
    m.add_function(wrap_pyfunction!(shi_tomasi_keypoints, m)?)?;
    m.add_function(wrap_pyfunction!(fast_keypoints, m)?)?;
    m.add_function(wrap_pyfunction!(orb_features, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_homography_ransac, m)?)?;
    m.add_function(wrap_pyfunction!(solve_pnp, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_rgbd_odometry, m)?)?;
    m.add_function(wrap_pyfunction!(gray_world_white_balance_image, m)?)?;
    m.add_function(wrap_pyfunction!(stitch_panorama_pair, m)?)?;
    m.add_function(wrap_pyfunction!(stereo_block_match, m)?)?;
    m.add_function(wrap_pyfunction!(match_binary_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(match_float_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(write_image, m)?)?;
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
    m.add_function(wrap_pyfunction!(depth_to_xyz, m)?)?;
    m.add_function(wrap_pyfunction!(rgbd_to_point_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_pinhole_camera, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_fisheye_angles, m)?)?;
    m.add_function(wrap_pyfunction!(dense_flow_image, m)?)?;
    m.add_function(wrap_pyfunction!(filter2d_image, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_blur_image, m)?)?;
    m.add_function(wrap_pyfunction!(median_blur_image, m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter_image, m)?)?;
    m.add_function(wrap_pyfunction!(sobel_image, m)?)?;
    m.add_function(wrap_pyfunction!(sobel_abs_image, m)?)?;
    m.add_function(wrap_pyfunction!(sobel_threshold_image, m)?)?;
    m.add_function(wrap_pyfunction!(spatial_gradient_image, m)?)?;
    m.add_function(wrap_pyfunction!(sobel_l1_magnitude_image, m)?)?;
    m.add_function(wrap_pyfunction!(scharr_image, m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_image, m)?)?;
    m.add_function(wrap_pyfunction!(pyr_down_image, m)?)?;
    m.add_function(wrap_pyfunction!(pyr_up_image, m)?)?;
    m.add_function(wrap_pyfunction!(morphology_image, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_image, m)?)?;
    m.add_function(wrap_pyfunction!(otsu_threshold_image, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_threshold_image, m)?)?;
    m.add_function(wrap_pyfunction!(histogram_image, m)?)?;
    m.add_function(wrap_pyfunction!(equalize_histogram_image, m)?)?;
    m.add_function(wrap_pyfunction!(clahe_image, m)?)?;
    m.add_function(wrap_pyfunction!(integral_image_u8, m)?)?;
    m.add_function(wrap_pyfunction!(canny_image, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image, m)?)?;
    m.add_function(wrap_pyfunction!(letterbox_image, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_image_chw, m)?)?;
    m.add_function(wrap_pyfunction!(resize_normalize_image_chw, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_gray_image, m)?)?;
    m.add_function(wrap_pyfunction!(resize_rgb_to_gray_image, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_hsv_image, m)?)?;
    m.add_function(wrap_pyfunction!(remap_image, m)?)?;
    m.add_function(wrap_pyfunction!(nms, m)?)?;
    m.add_function(wrap_pyfunction!(batched_nms, m)?)?;
    m.add_function(wrap_pyfunction!(soft_nms, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components_image, m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform_edt, m)?)?;
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
