#[cfg(all(feature = "io-pcd", feature = "filter-voxel"))]
#[test]
fn mvp_load_voxel_downsample() {
    use spatialrust::{
        read_pcd, write_pcd, HasPositions3, PcdWriteFormat, PointCloudBuilder, PointCloudFilter,
        VoxelGridDownsample, VoxelGridDownsampleConfig,
    };
    use std::io::Cursor;

    let mut builder = PointCloudBuilder::xyz();
    for i in 0..10 {
        builder.push_point([i as f32 * 0.1, 0.0, 0.0]).unwrap();
    }
    let cloud = builder.build().unwrap();

    let mut bytes = Vec::new();
    write_pcd(&mut bytes, &cloud, PcdWriteFormat::Ascii).unwrap();
    let loaded = read_pcd(&mut Cursor::new(bytes)).unwrap();

    let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.5));
    let downsampled = filter.filter(&loaded).unwrap();
    assert_eq!(downsampled.len(), 2);
    let (x, _, _) = downsampled.positions3().unwrap();
    assert!((x[0] - 0.2).abs() < 1e-5);
    assert!((x[1] - 0.7).abs() < 1e-5);
}

#[cfg(all(feature = "io-pcd", feature = "filter-voxel", feature = "feature-normal", feature = "search-kdtree"))]
#[test]
fn mvp_load_voxel_normals() {
    use spatialrust::{
        FeatureEstimator, HasNormals3, NormalEstimationConfig, NormalEstimator, PointCloudBuilder,
        PointCloudFilter, VoxelGridDownsample, VoxelGridDownsampleConfig, read_pcd, write_pcd,
        PcdWriteFormat,
    };
    use std::io::Cursor;

    let mut builder = PointCloudBuilder::xyz();
    for x in 0..6 {
        for y in 0..6 {
            builder.push_point([x as f32, y as f32, 0.0]).unwrap();
        }
    }
    let cloud = builder.build().unwrap();

    let mut bytes = Vec::new();
    write_pcd(&mut bytes, &cloud, PcdWriteFormat::Ascii).unwrap();
    let loaded = read_pcd(&mut Cursor::new(bytes)).unwrap();

    let downsampled =
        VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(1.0)).filter(&loaded).unwrap();
    assert!(!downsampled.is_empty());

    let estimator = NormalEstimator::new(NormalEstimationConfig {
        k_neighbors: 8,
        min_neighbors: 3,
        viewpoint: Some(spatialrust::Vec3::new(0.0, 0.0, 10.0)),
        ..NormalEstimationConfig::default()
    });
    let with_normals = estimator.estimate(&downsampled).unwrap();
    let (_, _, nz) = with_normals.normals3().unwrap();
    assert!(nz.iter().all(|value| value.is_finite()));
}

#[cfg(all(
    feature = "io-pcd",
    feature = "filter-voxel",
    feature = "feature-normal",
    feature = "search-kdtree",
    feature = "segment-ransac-plane",
    feature = "segment-euclidean"
))]
#[test]
fn mvp_load_voxel_normals_plane_cluster() {
    use spatialrust::{
        EuclideanClusterConfig, EuclideanClusterExtractor, FeatureEstimator, NormalEstimationConfig,
        NormalEstimator, PointCloudBuilder, PointCloudFilter, RansacPlaneConfig, RansacPlaneSegmenter,
        VoxelGridDownsample, VoxelGridDownsampleConfig, read_pcd, write_pcd, PcdWriteFormat,
    };
    use std::io::Cursor;

    let mut builder = PointCloudBuilder::xyz();
    for x in 0..8 {
        for y in 0..8 {
            builder.push_point([x as f32, y as f32, 0.0]).unwrap();
        }
    }
    builder.push_point([0.0, 0.0, 5.0]).unwrap();
    builder.push_point([1.0, 0.0, 5.0]).unwrap();
    builder.push_point([0.0, 1.0, 5.0]).unwrap();
    let cloud = builder.build().unwrap();

    let mut bytes = Vec::new();
    write_pcd(&mut bytes, &cloud, PcdWriteFormat::Ascii).unwrap();
    let loaded = read_pcd(&mut Cursor::new(bytes)).unwrap();

    let downsampled =
        VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(1.0)).filter(&loaded).unwrap();
    let with_normals = NormalEstimator::new(NormalEstimationConfig {
        k_neighbors: 8,
        min_neighbors: 3,
        viewpoint: Some(spatialrust::Vec3::new(0.0, 0.0, 10.0)),
        ..NormalEstimationConfig::default()
    })
    .estimate(&downsampled)
    .unwrap();

    let plane = RansacPlaneSegmenter::new(RansacPlaneConfig {
        distance_threshold: 0.1,
        max_iterations: 500,
        min_inliers: 10,
        seed: 11,
    })
    .segment(&with_normals)
    .unwrap();
    assert!(plane.inlier_count >= 10);
    assert!(!plane.outliers.is_empty());

    let clusters = EuclideanClusterExtractor::new(EuclideanClusterConfig {
        cluster_tolerance: 1.5,
        min_cluster_size: 1,
        max_cluster_size: usize::MAX,
    })
    .extract(&plane.outliers)
    .unwrap();
    assert!(clusters.cluster_count >= 1);
    assert!(clusters.cloud.field("label").is_ok());
}

#[cfg(all(feature = "io-pcd", feature = "filter-voxel", feature = "register-icp", feature = "search-kdtree"))]
#[test]
fn mvp_load_voxel_icp() {
    use spatialrust::{
        IcpConfig, IcpRegistration, PointCloudBuilder, PointCloudFilter, PointCloudRegistration,
        VoxelGridDownsample, VoxelGridDownsampleConfig, read_pcd, transform_point_cloud, write_pcd,
        Isometry3, PcdWriteFormat, Quat, Vec3,
    };
    use std::io::Cursor;

    let mut builder = PointCloudBuilder::xyz();
    for x in 0..10 {
        for y in 0..10 {
            builder.push_point([x as f32 * 0.1, y as f32 * 0.1, 0.0]).unwrap();
        }
    }
    let target = builder.build().unwrap();

    let mut bytes = Vec::new();
    write_pcd(&mut bytes, &target, PcdWriteFormat::Ascii).unwrap();
    let loaded = read_pcd(&mut Cursor::new(bytes)).unwrap();
    let downsampled =
        VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.2)).filter(&loaded).unwrap();

    let shift = Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.05, -0.02, 0.0));
    let source = transform_point_cloud(&downsampled, shift).unwrap();

    let result = IcpRegistration::new(IcpConfig {
        max_correspondence_distance: 0.3,
        max_iterations: 30,
        ..IcpConfig::default()
    })
    .align(&source, &downsampled)
    .unwrap();

    assert!(result.fitness < 1e-3);
    assert!(result.converged);
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_full_pipeline_roundtrip() {
    use spatialrust::{
        HasPositions3, MvpIcpConfig, MvpPipeline, MvpPipelineConfig, NormalEstimationConfig,
        PcdWriteFormat, PointCloudBuilder, RansacPlaneConfig, read_pcd, write_pcd, Isometry3,
        Quat, Vec3,
    };
    use spatialrust::{EuclideanClusterConfig, IcpConfig};
    use std::io::Cursor;

    let mut builder = PointCloudBuilder::xyz();
    for x in 0..10 {
        for y in 0..10 {
            builder.push_point([x as f32 * 0.1, y as f32 * 0.1, 0.0]).unwrap();
        }
    }
    for point in [(0.0, 0.0, 0.5), (0.1, 0.0, 0.5), (0.0, 0.1, 0.5)] {
        builder.push_point([point.0, point.1, point.2]).unwrap();
    }
    let cloud = builder.build().unwrap();

    let mut input_bytes = Vec::new();
    write_pcd(&mut input_bytes, &cloud, PcdWriteFormat::Ascii).unwrap();
    let loaded = read_pcd(&mut Cursor::new(input_bytes)).unwrap();

    let result = MvpPipeline::new(MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(0.2),
        normals: NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 500,
            min_inliers: 10,
            seed: 17,
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 0.3,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        },
        icp: Some(MvpIcpConfig {
            icp: IcpConfig {
                max_correspondence_distance: 0.2,
                max_iterations: 30,
                ..IcpConfig::default()
            },
            source_transform: Some(Isometry3::new(
                Quat::<f32>::identity(),
                Vec3::new(0.03, -0.01, 0.0),
            )),
        }),
        ..MvpPipelineConfig::default()
    })
    .run(&loaded)
    .unwrap();

    assert!(!result.downsampled.is_empty());
    assert!(result.with_normals.field("normal_x").is_ok());
    assert!(result.plane.inlier_count >= 10);
    assert!(result.clusters.cluster_count >= 1);
    assert!(result.registration.expect("icp").converged);

    let mut output_bytes = Vec::new();
    write_pcd(&mut output_bytes, &result.output, PcdWriteFormat::Binary).unwrap();
    let saved = read_pcd(&mut Cursor::new(output_bytes)).unwrap();

    assert_eq!(saved.len(), result.output.len());
    assert!(saved.field("label").is_ok());
    let (x, y, z) = saved.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_las_pipeline_roundtrip() {
    use spatialrust::{
        HasPositions3, MvpPipeline, MvpPipelineConfig, NormalEstimationConfig, PointCloudBuilder,
        RansacPlaneConfig, read_point_cloud_file, write_point_cloud_file, Isometry3, Quat, Vec3,
    };
    use spatialrust::{EuclideanClusterConfig, IcpConfig, MvpIcpConfig};

    let mut builder = PointCloudBuilder::xyz();
    for x in 0..10 {
        for y in 0..10 {
            builder.push_point([x as f32 * 0.1, y as f32 * 0.1, 0.0]).unwrap();
        }
    }
    builder.push_point([0.0, 0.0, 0.5]).unwrap();
    let cloud = builder.build().unwrap();

    let input_path =
        std::env::temp_dir().join(format!("spatialrust_mvp_in_{}.las", std::process::id()));
    write_point_cloud_file(&input_path, &cloud).unwrap();
    let loaded = read_point_cloud_file(&input_path).unwrap();

    let result = MvpPipeline::new(MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(0.2),
        normals: NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 500,
            min_inliers: 10,
            seed: 17,
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 0.3,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        },
        icp: Some(MvpIcpConfig {
            icp: IcpConfig {
                max_correspondence_distance: 0.2,
                max_iterations: 30,
                ..IcpConfig::default()
            },
            source_transform: Some(Isometry3::new(
                Quat::<f32>::identity(),
                Vec3::new(0.03, -0.01, 0.0),
            )),
        }),
        ..MvpPipelineConfig::default()
    })
    .run(&loaded)
    .unwrap();

    let output_path =
        std::env::temp_dir().join(format!("spatialrust_mvp_out_{}.las", std::process::id()));
    write_point_cloud_file(&output_path, &result.output).unwrap();
    let saved = read_point_cloud_file(&output_path).unwrap();

    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(output_path);

    assert_eq!(saved.len(), result.output.len());
    assert!(saved.field("classification").is_ok());
    let (x, y, z) = saved.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_copc_pipeline_roundtrip() {
    use spatialrust::{
        HasPositions3, MvpPipeline, MvpPipelineConfig, NormalEstimationConfig, PointCloudBuilder,
        RansacPlaneConfig, read_copc_file, write_copc_file, Isometry3, Quat, Vec3,
    };
    use spatialrust::{EuclideanClusterConfig, IcpConfig, MvpIcpConfig};

    let mut builder = PointCloudBuilder::xyz();
    for x in 0..10 {
        for y in 0..10 {
            builder.push_point([x as f32 * 0.1, y as f32 * 0.1, 0.0]).unwrap();
        }
    }
    builder.push_point([0.0, 0.0, 0.5]).unwrap();
    builder.push_point([0.1, 0.0, 0.5]).unwrap();
    let cloud = builder.build().unwrap();

    let input_path =
        std::env::temp_dir().join(format!("spatialrust_mvp_copc_in_{}.copc.laz", std::process::id()));
    write_copc_file(&input_path, &cloud).unwrap();
    let loaded = read_copc_file(&input_path).unwrap();

    let result = MvpPipeline::new(MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(0.2),
        normals: NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 500,
            min_inliers: 10,
            seed: 17,
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 0.3,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        },
        icp: Some(MvpIcpConfig {
            icp: IcpConfig {
                max_correspondence_distance: 0.2,
                max_iterations: 30,
                ..IcpConfig::default()
            },
            source_transform: Some(Isometry3::new(
                Quat::<f32>::identity(),
                Vec3::new(0.03, -0.01, 0.0),
            )),
        }),
        ..MvpPipelineConfig::default()
    })
    .run(&loaded)
    .unwrap();

    assert!(!result.downsampled.is_empty());
    assert!(result.plane.inlier_count >= 10);
    assert!(result.clusters.cluster_count >= 1);
    assert!(result.registration.expect("icp").converged);
    assert!(result.output.field("label").is_ok());

    let output_path =
        std::env::temp_dir().join(format!("spatialrust_mvp_copc_out_{}.copc.laz", std::process::id()));
    write_copc_file(&output_path, &result.output).unwrap();
    let saved = read_copc_file(&output_path).unwrap();

    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(output_path);

    assert_eq!(saved.len(), result.output.len());
    assert!(saved.field("classification").is_ok());
    let (x, y, z) = saved.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_copc_query_pipeline() {
    use spatialrust::{
        CopcBounds, CopcQuery, ExecutionPolicy, HasPositions3, MvpPipeline, MvpPipelineConfig,
        NormalEstimationConfig, PointCloudBuilder, RansacPlaneConfig, read_copc_file_with_query,
        write_copc_file, Vec3,
    };
    use spatialrust::EuclideanClusterConfig;

    let mut builder = PointCloudBuilder::xyz();
    for x in 0..10 {
        for y in 0..10 {
            builder.push_point([x as f32 * 0.1, y as f32 * 0.1, 0.0]).unwrap();
        }
    }
    builder.push_point([0.0, 0.0, 0.5]).unwrap();
    builder.push_point([0.1, 0.0, 0.5]).unwrap();
    let cloud = builder.build().unwrap();

    let path =
        std::env::temp_dir().join(format!("spatialrust_mvp_copc_query_{}.copc.laz", std::process::id()));
    write_copc_file(&path, &cloud).unwrap();

    let bounds = CopcBounds::from_ranges((0.0, 0.85), (0.0, 0.85), (-0.01, 0.01));
    let loaded = read_copc_file_with_query(&path, &CopcQuery::bounds(bounds)).unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(loaded.len() < cloud.len());
    assert!(loaded.len() >= 10);

    let result = MvpPipeline::new(MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(0.2),
        voxel_policy: ExecutionPolicy::CpuSingle,
        normals: NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 500,
            min_inliers: 10,
            seed: 17,
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 0.3,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        },
        icp: None,
        ..MvpPipelineConfig::default()
    })
    .run(&loaded)
    .expect("mvp pipeline on queried copc subset");

    assert!(!result.downsampled.is_empty());
    assert!(result.plane.inlier_count >= 10);
    assert_eq!(result.clusters.cluster_count, 0);
    assert!(result.output.is_empty());
}
