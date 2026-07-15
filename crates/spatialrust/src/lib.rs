//! SpatialRust meta crate.
//!
//! Re-exports the stable public API surface. Application code should depend on
//! this crate unless it needs direct access to a specific sub-crate.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub use spatialrust_core as core;
pub use spatialrust_features as features;
pub use spatialrust_filtering as filtering;
pub use spatialrust_gpu as gpu;
pub use spatialrust_io as io;
pub use spatialrust_math as math;
pub use spatialrust_metrics as metrics;
pub use spatialrust_pipeline as pipeline;
pub use spatialrust_registration as registration;
pub use spatialrust_search as search;
pub use spatialrust_segmentation as segmentation;
pub use spatialrust_transform as transform;
pub use spatialrust_voxelize as voxelize;

#[cfg(feature = "ai")]
pub use spatialrust_ai as ai;
#[cfg(feature = "camera")]
pub use spatialrust_camera as camera;
#[cfg(feature = "image")]
pub use spatialrust_image as image;
#[cfg(feature = "image-io")]
pub use spatialrust_image_io as image_io;
#[cfg(feature = "tensor")]
pub use spatialrust_tensor as tensor;
#[cfg(feature = "vision")]
pub use spatialrust_vision as vision;
#[cfg(feature = "records")]
pub use spatialrust_records as records;
#[cfg(any(
    feature = "arrow-c-data",
    feature = "arrow-c-stream",
    feature = "arrow-c-device"
))]
pub use spatialrust_arrow as arrow;
#[cfg(feature = "sync")]
pub use spatialrust_sync as sync;
#[cfg(feature = "mapping")]
pub use spatialrust_mapping as mapping;

pub use spatialrust_core::{
    CpuDevice, DType, Device, DeviceKind, ExecutionPolicy, FieldSemantic, FrameId, HasIntensity,
    HasNormals3, HasPositions3, PointBuffer, PointCloud, PointCloudBuilder, PointField,
    PointSchema, SpatialAlgorithm, SpatialError, SpatialMetadata, SpatialResult, SpatialTensor,
    SpatialTensorChunk, StandardSchemas, Timestamp, DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE,
};

#[cfg(feature = "tensor-aoso")]
pub use spatialrust_core::{AoSoAAttributeChunk, AoSoAAttributeLayout, AoSoAXyzChunk};
pub use spatialrust_io::IoError;
pub use spatialrust_math::{
    approx_eq, approx_eq_f64, f32_eps, f64_eps, smallest_eigenvector, solve_linear_system,
    symmetric_eigen3, CauchyKernel, Cov3, CovarianceAccumulator3, HuberKernel, Isometry3,
    LeastSquaresResult, Mat3, Mat4, Pose3, Quat, Real, RobustKernel, Scalar, SymmetricEigen3,
    Transform3, TransformPoint, TukeyKernel, Vec2, Vec3, Vec4,
};

#[cfg(feature = "io-pcd")]
pub use spatialrust_io::{read_pcd, read_pcd_file, write_pcd, write_pcd_file, PcdWriteFormat};

#[cfg(feature = "io-ply")]
pub use spatialrust_io::{read_ply, read_ply_file, write_ply, write_ply_file, PlyWriteFormat};

#[cfg(feature = "io-las")]
pub use spatialrust_io::{read_las, read_las_file, write_las, write_las_file, LasWriteFormat};

#[cfg(feature = "io-e57")]
pub use spatialrust_io::{read_e57, read_e57_file, write_e57, write_e57_file};

#[cfg(feature = "io-copc")]
pub use spatialrust_io::{
    copc_level_for_resolution, read_copc, read_copc_file, read_copc_file_in_bounds,
    read_copc_file_info, read_copc_file_with_query, write_copc, write_copc_file,
    write_copc_file_with_params, CopcBounds, CopcFileInfo, CopcQuery, CopcWriterParams,
};

#[cfg(feature = "io-copc-http")]
pub use spatialrust_io::{
    read_copc_url, read_copc_url_info, read_copc_url_with_query, HttpByteSource,
};

pub use spatialrust_io::{
    detect_point_cloud_format, read_point_cloud_file, read_point_cloud_file_with_format,
    write_point_cloud_file, write_point_cloud_file_with_format, PointCloudFileFormat,
};

#[cfg(feature = "search-kdtree")]
pub use spatialrust_search::{
    brute_force_knn, brute_force_radius, nearest_k_spatial_tensor, parallel_index_for_each,
    parallel_index_ranges, parallel_worker_count, radius_search_spatial_tensor, BruteForceIndex,
    ChunkQueryRange, ChunkedNearestNeighborIndex, ChunkedRadiusSearchIndex, KdTree,
    NearestNeighborIndex, Neighbor, RadiusSearchIndex, SpatialIndex, PARALLEL_STAGING_MIN_POINTS,
};

#[cfg(feature = "search-parallel")]
pub use spatialrust_search::{
    nearest_k_spatial_tensor_parallel, nearest_k_spatial_tensor_parallel_into,
    radius_search_spatial_tensor_parallel, radius_search_spatial_tensor_parallel_into,
    PARALLEL_CHUNK_QUERY_MIN_POINTS,
};

#[cfg(feature = "search-graph")]
pub use spatialrust_search::{knn_graph, radius_graph, NeighborGraph};

#[cfg(feature = "filter-voxel")]
pub use spatialrust_filtering::{
    AttributeAggregation, PointCloudFilter, VoxelAggregationMode, VoxelGridDownsample,
    VoxelGridDownsampleConfig,
};

#[cfg(feature = "filter-outlier")]
pub use spatialrust_filtering::{
    RadiusOutlierConfig, RadiusOutlierRemoval, StatisticalOutlierConfig, StatisticalOutlierRemoval,
};

#[cfg(feature = "filter-crop")]
pub use spatialrust_filtering::{Aabb, CropBox, PassThrough};

#[cfg(feature = "filter-fps")]
pub use spatialrust_filtering::{FarthestPointSampling, FarthestPointSamplingConfig};

#[cfg(feature = "filter-mls")]
pub use spatialrust_filtering::{MlsConfig, MlsSmoothing};

#[cfg(feature = "feature-normal")]
pub use spatialrust_features::{
    orient_normal_towards_viewpoint, FeatureEstimator, KdTreeNeighborhood, NeighborhoodProvider,
    NormalEstimationConfig, NormalEstimationResult, NormalEstimator,
};

#[cfg(feature = "feature-iss")]
pub use spatialrust_features::{IssKeypointConfig, IssKeypointDetector, IssKeypointResult};

#[cfg(feature = "feature-normal-orient")]
pub use spatialrust_features::{orient_normals_consistent, NormalOrientationConfig};

#[cfg(feature = "feature-boundary")]
pub use spatialrust_features::{BoundaryConfig, BoundaryDetector, BoundaryResult};

#[cfg(feature = "feature-normal-gpu")]
pub use spatialrust_features::GpuNormalEstimator;

#[cfg(feature = "segment-ransac-plane")]
pub use spatialrust_segmentation::{
    extract_indices, extract_mask, with_labels, PlaneModel, PointCloudSegmenter, RansacPlaneConfig,
    RansacPlaneSegmentation, RansacPlaneSegmenter, DEFAULT_GPU_MIN_POINTS_PLANE,
};

#[cfg(feature = "segment-ransac-plane-gpu")]
pub use spatialrust_segmentation::GpuRansacPlaneSegmenter;

#[cfg(feature = "segment-multi-plane")]
pub use spatialrust_segmentation::{MultiPlaneConfig, MultiPlaneSegmentation, MultiPlaneSegmenter};

#[cfg(feature = "segment-euclidean")]
pub use spatialrust_segmentation::{
    EuclideanClusterConfig, EuclideanClusterExtractor, EuclideanClusterResult,
    DEFAULT_GPU_MIN_POINTS_EUCLIDEAN,
};

#[cfg(feature = "segment-euclidean-gpu")]
pub use spatialrust_segmentation::GpuEuclideanClusterExtractor;

#[cfg(feature = "segment-dbscan")]
pub use spatialrust_segmentation::{DbscanConfig, DbscanResult, DbscanSegmenter};

#[cfg(feature = "segment-ransac-primitives")]
pub use spatialrust_segmentation::{
    CylinderModel, PrimitiveSegmentation, RansacCylinderSegmenter, RansacPrimitiveConfig,
    RansacSphereSegmenter, SphereModel,
};

#[cfg(feature = "segment-ground")]
pub use spatialrust_segmentation::{GroundConfig, GroundSegmentation, GroundSegmenter, UpAxis};

#[cfg(feature = "segment-region-growing")]
pub use spatialrust_segmentation::{
    RegionGrowingConfig, RegionGrowingResult, RegionGrowingSegmenter,
};

#[cfg(feature = "register-icp")]
pub use spatialrust_registration::{
    estimate_rigid_transform, transform_point_cloud, IcpConfig, IcpRegistration,
    PointCloudRegistration, RegistrationResult,
};

#[cfg(feature = "register-icp-point-to-plane")]
pub use spatialrust_registration::{PointToPlaneIcp, PointToPlaneIcpConfig};

#[cfg(feature = "register-gicp")]
pub use spatialrust_registration::{GicpConfig, GicpRegistration};

#[cfg(feature = "register-ndt")]
pub use spatialrust_registration::{NdtConfig, NdtRegistration};

#[cfg(feature = "register-fpfh")]
pub use spatialrust_registration::{
    fpfh_descriptors, FpfhDescriptor, FpfhRansacConfig, FpfhRansacRegistration, FPFH_DESCRIPTOR_LEN,
};

#[cfg(feature = "metrics-distance")]
pub use spatialrust_metrics::{
    chamfer_distance, cloud_distances, hausdorff_distance, CloudDistances,
};

#[cfg(feature = "transform-ops")]
pub use spatialrust_transform::{
    apply_transform, bounding_box, centroid, merge_clouds, normalize_unit_sphere,
    oriented_bounding_box, recenter, scale_cloud, Aabb as TransformAabb, Obb,
};

#[cfg(feature = "voxelize-occupancy")]
pub use spatialrust_voxelize::{voxelize, OccupancyGrid, VoxelFill, VoxelGridConfig};

#[cfg(feature = "voxelize-range-image")]
pub use spatialrust_voxelize::{range_image, RangeImage, RangeImageConfig};

#[cfg(feature = "image")]
pub use spatialrust_image::{
    AlphaMode, ColorRange, ColorSpace, GrayImage, Image, ImageError, ImageLayout, ImageMetadata,
    ImageRegion, ImageView, ImageViewMut, PlanarImage, PlanarImageView, RgbImage,
};

#[cfg(feature = "image-io")]
pub use spatialrust_image_io::{
    decode_bytes, decode_path, decode_reader, encode_bytes, encode_path, encode_writer,
    DecodeLimits, DecodeOptions, DecodedImage, DecodedMetadata, DecodedPixels, EncodeOptions,
    ImageFileFormat, ImageIoError, Orientation, SourceColorType,
};

#[cfg(feature = "camera-rgbd")]
pub use spatialrust_camera::{
    depth_to_point_cloud, rgbd_to_point_cloud, BrownConrady, CameraError, CameraIntrinsics,
    DepthConversionOptions, PinholeCamera, RgbdError,
};

#[cfg(feature = "vision")]
pub use spatialrust_vision::*;

#[cfg(feature = "pipeline-mvp")]
pub use spatialrust_pipeline::{
    MvpIcpConfig, MvpPipeline, MvpPipelineConfig, MvpPipelineResult, MvpRegistrationMethod,
};
