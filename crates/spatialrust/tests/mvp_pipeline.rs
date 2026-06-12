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

#[cfg(feature = "mvp")]
#[test]
fn mvp_copc_resolution_query_pipeline() {
    use spatialrust::{
        CopcQuery, CopcWriterParams, ExecutionPolicy, MvpPipeline, MvpPipelineConfig,
        NormalEstimationConfig, PointCloudBuilder, RansacPlaneConfig, read_copc_file,
        read_copc_file_info, read_copc_file_with_query, write_copc_file_with_params, Vec3,
    };
    use spatialrust::EuclideanClusterConfig;

    let mut builder = PointCloudBuilder::xyz();
    for index in 0..7_000 {
        let x = (index % 31) as f32 - 15.0;
        let y = ((index / 31) % 29) as f32 - 14.0;
        let z = ((index / (31 * 29)) % 23) as f32 - 11.0;
        builder.push_point([x, y, z]).unwrap();
    }
    let cloud = builder.build().unwrap();

    let path = std::env::temp_dir().join(format!(
        "spatialrust_mvp_copc_resolution_{}.copc.laz",
        std::process::id()
    ));
    write_copc_file_with_params(
        &path,
        &cloud,
        &CopcWriterParams {
            max_points_per_node: 96,
            max_depth: 8,
        },
    )
    .unwrap();

    let info = read_copc_file_info(&path).unwrap();
    let full = read_copc_file(&path).unwrap();
    let query = CopcQuery::with_resolution(info.root_bounds, info.spacing * 4.0);
    let loaded = read_copc_file_with_query(&path, &query).unwrap();

    assert_eq!(full.len(), cloud.len());
    assert!(loaded.len() < full.len());

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
            min_inliers: 1,
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
    .expect("mvp pipeline on resolution-limited copc read");

    assert!(!result.downsampled.is_empty());
    assert!(result.plane.inlier_count >= 1);

    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_cli_copc_resolution_reduces_input_points() {
    use std::process::Command;

    use spatialrust::{
        read_copc_file, read_copc_file_info, write_copc_file_with_params, CopcWriterParams,
        PointCloudBuilder,
    };

    let mut builder = PointCloudBuilder::xyz();
    for index in 0..7_000 {
        let x = (index % 31) as f32 - 15.0;
        let y = ((index / 31) % 29) as f32 - 14.0;
        let z = ((index / (31 * 29)) % 23) as f32 - 11.0;
        builder.push_point([x, y, z]).unwrap();
    }
    let cloud = builder.build().unwrap();

    let input_path = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_copc_in_{}.copc.laz",
        std::process::id()
    ));
    let coarse_output = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_copc_coarse_{}.copc.laz",
        std::process::id()
    ));
    let full_output = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_copc_full_{}.copc.laz",
        std::process::id()
    ));
    write_copc_file_with_params(
        &input_path,
        &cloud,
        &CopcWriterParams {
            max_points_per_node: 96,
            max_depth: 8,
        },
    )
    .unwrap();

    let info = read_copc_file_info(&input_path).unwrap();
    let full_count = read_copc_file(&input_path).unwrap().len();
    let coarse_resolution = info.spacing * 4.0;
    let bin = env!("CARGO_BIN_EXE_spatialrust-mvp");

    let coarse = Command::new(bin)
        .args([
            "--resolution",
            &format!("{coarse_resolution}"),
            input_path.to_str().unwrap(),
            coarse_output.to_str().unwrap(),
        ])
        .output()
        .expect("run coarse resolution CLI");
    assert!(
        coarse.status.success(),
        "coarse CLI failed: {}",
        String::from_utf8_lossy(&coarse.stderr)
    );

    let full = Command::new(bin)
        .args([
            input_path.to_str().unwrap(),
            full_output.to_str().unwrap(),
        ])
        .output()
        .expect("run full detail CLI");
    assert!(
        full.status.success(),
        "full CLI failed: {}",
        String::from_utf8_lossy(&full.stderr)
    );

    let coarse_loaded = parse_cli_input_points(&coarse.stderr);
    let full_loaded = parse_cli_input_points(&full.stderr);
    assert_eq!(full_loaded, full_count);
    assert!(coarse_loaded < full_loaded);

    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(coarse_output);
    let _ = std::fs::remove_file(full_output);
}

#[cfg(feature = "mvp")]
fn parse_cli_input_points(stderr: &[u8]) -> usize {
    let text = String::from_utf8_lossy(stderr);
    text.lines()
        .find_map(|line| line.strip_prefix("input points: "))
        .and_then(|value| value.trim().parse().ok())
        .expect("CLI stderr should report input points")
}

#[cfg(feature = "mvp")]
fn sample_xyzinormal_plane_cloud() -> spatialrust::PointCloud {
    use spatialrust::{PointCloudBuilder, StandardSchemas};

    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
    for x in 0..10 {
        for y in 0..10 {
            builder
                .push_point([x as f32 * 0.1, y as f32 * 0.1, 0.0, 0.2, 0.0, 0.0, 1.0])
                .unwrap();
        }
    }
    builder
        .push_point([0.0, 0.0, 0.5, 0.9, 0.0, 0.0, 1.0])
        .unwrap();
    builder
        .push_point([0.1, 0.0, 0.5, 0.8, 0.0, 0.0, 1.0])
        .unwrap();
    builder.build().unwrap()
}

#[cfg(feature = "mvp")]
fn mvp_xyzinormal_base_config() -> spatialrust::MvpPipelineConfig {
    use spatialrust::{
        EuclideanClusterConfig, MvpPipelineConfig, NormalEstimationConfig, RansacPlaneConfig,
        Vec3,
    };

    MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(0.2).without_gpu_min_points(),
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
    }
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_xyzinormal_pcd_pipeline_roundtrip() {
    use spatialrust::{
        HasIntensity, HasNormals3, HasPositions3, MvpPipeline, PcdWriteFormat, read_pcd,
        write_pcd,
    };
    use std::io::Cursor;

    let cloud = sample_xyzinormal_plane_cloud();
    let mut input_bytes = Vec::new();
    write_pcd(&mut input_bytes, &cloud, PcdWriteFormat::Ascii).unwrap();
    let loaded = read_pcd(&mut Cursor::new(input_bytes)).unwrap();

    assert!(loaded.field("normal_x").is_ok());
    assert!(loaded.intensity().is_ok());

    let result = MvpPipeline::new(mvp_xyzinormal_base_config())
        .run(&loaded)
        .unwrap();

    assert!(!result.downsampled.is_empty());
    assert!(result.downsampled.field("normal_x").is_ok());
    assert!(result.downsampled.intensity().is_ok());
    assert!(result.with_normals.field("normal_x").is_ok());
    assert!(result.plane.inlier_count >= 10);
    assert!(result.output.field("label").is_ok());

    let (_, _, down_nz) = result.downsampled.normals3().unwrap();
    assert!(down_nz.iter().all(|value| value.is_finite()));

    let mut output_bytes = Vec::new();
    write_pcd(&mut output_bytes, &result.output, PcdWriteFormat::Binary).unwrap();
    let saved = read_pcd(&mut Cursor::new(output_bytes)).unwrap();
    assert_eq!(saved.len(), result.output.len());
    let (x, y, z) = saved.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));
}

#[cfg(all(feature = "mvp", feature = "pipeline-mvp-gpu"))]
#[test]
fn mvp_xyzinormal_gpu_voxel_matches_cpu() {
    use spatialrust::{DeviceKind, ExecutionPolicy, HasIntensity, HasNormals3, HasPositions3, MvpPipeline};

    let cloud = sample_xyzinormal_plane_cloud();
    let mut cpu_config = mvp_xyzinormal_base_config();
    cpu_config.voxel_policy = ExecutionPolicy::CpuSingle;
    let mut gpu_config = mvp_xyzinormal_base_config();
    gpu_config.voxel_policy = ExecutionPolicy::Gpu(DeviceKind::Wgpu);

    let cpu = MvpPipeline::new(cpu_config).run(&cloud).unwrap();
    let gpu = MvpPipeline::new(gpu_config).run(&cloud).unwrap();

    assert_eq!(cpu.downsampled.len(), gpu.downsampled.len());
    let (cpu_x, cpu_y, cpu_z) = cpu.downsampled.positions3().unwrap();
    let (gpu_x, gpu_y, gpu_z) = gpu.downsampled.positions3().unwrap();
    for index in 0..cpu.downsampled.len() {
        assert!((cpu_x[index] - gpu_x[index]).abs() < 1e-4);
        assert!((cpu_y[index] - gpu_y[index]).abs() < 1e-4);
        assert!((cpu_z[index] - gpu_z[index]).abs() < 1e-4);
    }

    let cpu_i = cpu.downsampled.intensity().unwrap();
    let gpu_i = gpu.downsampled.intensity().unwrap();
    for (left, right) in cpu_i.iter().zip(gpu_i) {
        assert!((left - right).abs() < 1e-4);
    }

    let (_, _, cpu_nz) = cpu.downsampled.normals3().unwrap();
    let (_, _, gpu_nz) = gpu.downsampled.normals3().unwrap();
    for (left, right) in cpu_nz.iter().zip(gpu_nz) {
        assert!((left - right).abs() < 1e-4);
    }
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_approximate_voxel_mode_pipeline() {
    use spatialrust::{
        HasPositions3, MvpPipeline, MvpPipelineConfig, NormalEstimationConfig, PointCloudBuilder,
        RansacPlaneConfig, VoxelAggregationMode, Vec3,
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

    let result = MvpPipeline::new(MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::approximate(0.2),
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
    .run(&cloud)
    .expect("mvp pipeline with approximate voxel mode");

    assert_eq!(
        spatialrust::VoxelGridDownsampleConfig::approximate(0.2).mode,
        VoxelAggregationMode::ApproximateFirst
    );
    assert!(!result.downsampled.is_empty());
    assert!(result.plane.inlier_count >= 10);
    let (x, _, _) = result.downsampled.positions3().unwrap();
    assert!(x.iter().all(|value| value.is_finite()));
}
