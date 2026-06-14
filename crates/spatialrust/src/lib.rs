//! SpatialRust meta crate.
//!
//! Re-exports the stable public API surface. Application code should depend on
//! this crate unless it needs direct access to a specific sub-crate.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub use spatialrust_core as core;
pub use spatialrust_gpu as gpu;
pub use spatialrust_io as io;
pub use spatialrust_math as math;
pub use spatialrust_search as search;
pub use spatialrust_filtering as filtering;
pub use spatialrust_features as features;
pub use spatialrust_segmentation as segmentation;
pub use spatialrust_registration as registration;
pub use spatialrust_pipeline as pipeline;

pub use spatialrust_core::{
    CpuDevice, DType, Device, DeviceKind, ExecutionPolicy, FieldSemantic, FrameId, HasIntensity,
    HasNormals3, HasPositions3, PointBuffer, PointCloud, PointCloudBuilder, PointField,
    PointSchema, SpatialAlgorithm, SpatialError, SpatialMetadata, SpatialResult, StandardSchemas,
    Timestamp,
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
pub use spatialrust_io::{
    read_las, read_las_file, write_las, write_las_file, LasWriteFormat,
};

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

#[cfg(feature = "filter-voxel")]
pub use spatialrust_filtering::{
    AttributeAggregation, PointCloudFilter, VoxelAggregationMode, VoxelGridDownsample,
    VoxelGridDownsampleConfig,
};

#[cfg(feature = "feature-normal")]
pub use spatialrust_features::{
    FeatureEstimator, KdTreeNeighborhood, NeighborhoodProvider, NormalEstimationConfig,
    NormalEstimationResult, NormalEstimator, orient_normal_towards_viewpoint,
};

#[cfg(feature = "segment-ransac-plane")]
pub use spatialrust_segmentation::{
    extract_indices, extract_mask, with_labels, PlaneModel, PointCloudSegmenter,
    RansacPlaneConfig, RansacPlaneSegmentation, RansacPlaneSegmenter,
};

#[cfg(feature = "segment-euclidean")]
pub use spatialrust_segmentation::{
    EuclideanClusterConfig, EuclideanClusterResult, EuclideanClusterExtractor,
};

#[cfg(feature = "segment-region-growing")]
pub use spatialrust_segmentation::{
    RegionGrowingConfig, RegionGrowingResult, RegionGrowingSegmenter,
};

#[cfg(feature = "register-icp")]
pub use spatialrust_registration::{
    estimate_rigid_transform, transform_point_cloud, IcpConfig, IcpRegistration,
    PointCloudRegistration, RegistrationResult,
};

#[cfg(feature = "pipeline-mvp")]
pub use spatialrust_pipeline::{
    MvpIcpConfig, MvpPipeline, MvpPipelineConfig, MvpPipelineResult,
};
