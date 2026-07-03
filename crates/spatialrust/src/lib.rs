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

pub use spatialrust_core::{
    CpuDevice, DType, Device, DeviceKind, ExecutionPolicy, FieldSemantic, FrameId, HasIntensity,
    HasNormals3, HasPositions3, PointBuffer, PointCloud, PointCloudBuilder, PointField,
    PointSchema, SpatialAlgorithm, SpatialError, SpatialMetadata, SpatialResult, SpatialTensor,
    SpatialTensorChunk, StandardSchemas, Timestamp, DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE,
};
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
    brute_force_knn, brute_force_radius, BruteForceIndex, KdTree, NearestNeighborIndex, Neighbor,
    RadiusSearchIndex, SpatialIndex,
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

#[cfg(feature = "pipeline-mvp")]
pub use spatialrust_pipeline::{
    MvpIcpConfig, MvpPipeline, MvpPipelineConfig, MvpPipelineResult, MvpRegistrationMethod,
};
