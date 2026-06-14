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
fn parse_cli_elapsed_ms(stderr: &[u8]) -> f64 {
    let text = String::from_utf8_lossy(stderr);
    let value = text
        .lines()
        .find_map(|line| line.strip_prefix("elapsed: "))
        .expect("CLI stderr should report elapsed time");
    parse_duration_debug_ms(value.trim())
}

#[cfg(feature = "mvp")]
fn parse_duration_debug_ms(value: &str) -> f64 {
    if let Some(ms) = value.strip_suffix("ms") {
        return ms.parse().expect("elapsed milliseconds");
    }
    if let Some(us) = value.strip_suffix("µs") {
        return us.parse::<f64>().expect("elapsed microseconds") / 1_000.0;
    }
    if let Some(us) = value.strip_suffix("us") {
        return us.parse::<f64>().expect("elapsed microseconds") / 1_000.0;
    }
    if let Some(s) = value.strip_suffix('s') {
        return s.parse::<f64>().expect("elapsed seconds") * 1_000.0;
    }
    panic!("unsupported elapsed format `{value}`");
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_cli_scan_like_copc_resolution_reduces_input_points() {
    use std::process::Command;

    use spatialrust::{
        read_copc_file, read_copc_file_info, read_copc_file_with_query, write_copc_file_with_params,
        CopcQuery, CopcWriterParams,
    };

    const POINT_COUNT: usize = 50_000;
    let cloud = sample_scan_like_xyzi(POINT_COUNT);
    let input_path = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_scan_copc_in_{}.copc.laz",
        std::process::id()
    ));
    let coarse_output = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_scan_copc_coarse_{}.copc.laz",
        std::process::id()
    ));
    let full_output = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_scan_copc_full_{}.copc.laz",
        std::process::id()
    ));
    write_copc_file_with_params(
        &input_path,
        &cloud,
        &CopcWriterParams {
            max_points_per_node: 512,
            max_depth: 10,
        },
    )
    .unwrap();

    let info = read_copc_file_info(&input_path).unwrap();
    let full_count = read_copc_file(&input_path).unwrap().len();
    let coarse_resolution = info.spacing * 4.0;
    let coarse_query_count = read_copc_file_with_query(
        &input_path,
        &CopcQuery::with_resolution(info.root_bounds, coarse_resolution),
    )
    .unwrap()
    .len();
    let bin = env!("CARGO_BIN_EXE_spatialrust-mvp");
    let cli_args = [
        "--leaf-size",
        "4.0",
        "--voxel-policy",
        "cpu",
    ];

    let coarse = Command::new(bin)
        .args(cli_args)
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
        .args(cli_args)
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
    assert_eq!(coarse_loaded, coarse_query_count);
    assert!(coarse_loaded < full_loaded);
    assert!(
        coarse_loaded * 10 < full_loaded,
        "expected >90% reduction at spacing×4, got {coarse_loaded}/{full_loaded}"
    );

    let coarse_elapsed_ms = parse_cli_elapsed_ms(&coarse.stderr);
    let full_elapsed_ms = parse_cli_elapsed_ms(&full.stderr);
    assert!(
        coarse_elapsed_ms < full_elapsed_ms,
        "coarse CLI should be faster: {coarse_elapsed_ms}ms vs {full_elapsed_ms}ms"
    );

    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(coarse_output);
    let _ = std::fs::remove_file(full_output);
}

#[cfg(feature = "mvp")]
fn write_scan_like_copc_fixture(
    path: &std::path::Path,
    point_count: usize,
) -> spatialrust::PointCloud {
    use spatialrust::{write_copc_file_with_params, CopcWriterParams};

    let cloud = sample_scan_like_xyzi(point_count);
    write_copc_file_with_params(
        path,
        &cloud,
        &CopcWriterParams {
            max_points_per_node: 512,
            max_depth: 10,
        },
    )
    .unwrap();
    cloud
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_cli_scan_like_copc_bounds_resolution_reduces_input_points() {
    use std::process::Command;

    use spatialrust::{
        read_copc_file, read_copc_file_info, read_copc_file_with_query, CopcBounds, CopcQuery,
    };

    const POINT_COUNT: usize = 50_000;
    let input_path = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_scan_bounds_res_in_{}.copc.laz",
        std::process::id()
    ));
    write_scan_like_copc_fixture(&input_path, POINT_COUNT);

    let info = read_copc_file_info(&input_path).unwrap();
    let full_count = read_copc_file(&input_path).unwrap().len();
    let roi_bounds = CopcBounds::from_ranges((0.0, 40.0), (0.0, 20.0), (-0.01, 0.5));
    let coarse_resolution = info.spacing * 4.0;
    let bounds_only_count = read_copc_file_with_query(
        &input_path,
        &CopcQuery::bounds(roi_bounds),
    )
    .unwrap()
    .len();
    let combined_count = read_copc_file_with_query(
        &input_path,
        &CopcQuery::with_resolution(roi_bounds, coarse_resolution),
    )
    .unwrap()
    .len();

    assert!(bounds_only_count < full_count);
    assert!(combined_count <= bounds_only_count);
    assert!(combined_count < full_count);

    let bounds_arg = "0,0,-0.01,40,20,0.5";
    let resolution_arg = format!("{coarse_resolution}");
    let bin = env!("CARGO_BIN_EXE_spatialrust-mvp");
    let cli_args = ["--leaf-size", "4.0", "--voxel-policy", "cpu"];
    let combined_output = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_scan_bounds_res_out_{}.copc.laz",
        std::process::id()
    ));
    let bounds_output = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_scan_bounds_out_{}.copc.laz",
        std::process::id()
    ));
    let full_output = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_scan_bounds_res_full_{}.copc.laz",
        std::process::id()
    ));

    let combined = Command::new(bin)
        .args(cli_args)
        .args([
            "--bounds",
            bounds_arg,
            "--resolution",
            &resolution_arg,
            input_path.to_str().unwrap(),
            combined_output.to_str().unwrap(),
        ])
        .output()
        .expect("run bounds+resolution CLI");
    assert!(
        combined.status.success(),
        "bounds+resolution CLI failed: {}",
        String::from_utf8_lossy(&combined.stderr)
    );

    let bounds_only = Command::new(bin)
        .args(cli_args)
        .args([
            "--bounds",
            bounds_arg,
            input_path.to_str().unwrap(),
            bounds_output.to_str().unwrap(),
        ])
        .output()
        .expect("run bounds-only CLI");
    assert!(
        bounds_only.status.success(),
        "bounds-only CLI failed: {}",
        String::from_utf8_lossy(&bounds_only.stderr)
    );

    let full = Command::new(bin)
        .args(cli_args)
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

    let combined_loaded = parse_cli_input_points(&combined.stderr);
    let bounds_loaded = parse_cli_input_points(&bounds_only.stderr);
    let full_loaded = parse_cli_input_points(&full.stderr);
    assert_eq!(full_loaded, full_count);
    assert_eq!(bounds_loaded, bounds_only_count);
    assert_eq!(combined_loaded, combined_count);
    assert!(combined_loaded < bounds_loaded);
    assert!(bounds_loaded < full_loaded);

    let combined_elapsed_ms = parse_cli_elapsed_ms(&combined.stderr);
    let full_elapsed_ms = parse_cli_elapsed_ms(&full.stderr);
    assert!(
        combined_elapsed_ms < full_elapsed_ms,
        "combined CLI should be faster than full: {combined_elapsed_ms}ms vs {full_elapsed_ms}ms"
    );

    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(combined_output);
    let _ = std::fs::remove_file(bounds_output);
    let _ = std::fs::remove_file(full_output);
}

#[cfg(feature = "mvp")]
fn scan_like_roi_bounds() -> spatialrust::CopcBounds {
    spatialrust::CopcBounds::from_ranges((0.0, 40.0), (0.0, 20.0), (-0.01, 0.5))
}

#[cfg(feature = "mvp")]
fn scan_like_roi_bounds_arg() -> &'static str {
    "0,0,-0.01,40,20,0.5"
}

#[cfg(feature = "mvp")]
fn scan_like_resolution_multipliers() -> [f64; 5] {
    [1.0, 2.0, 4.0, 8.0, 16.0]
}

#[cfg(feature = "mvp")]
fn resolution_curve_counts(
    input_path: &std::path::Path,
    bounds: spatialrust::CopcBounds,
) -> Vec<(f64, usize)> {
    use spatialrust::{read_copc_file_info, read_copc_file_with_query, CopcQuery};

    let info = read_copc_file_info(input_path).expect("copc info");
    scan_like_resolution_multipliers()
        .into_iter()
        .map(|multiplier| {
            let resolution = info.spacing * multiplier;
            let count = read_copc_file_with_query(
                input_path,
                &CopcQuery::with_resolution(bounds, resolution),
            )
            .expect("resolution query")
            .len();
            (multiplier, count)
        })
        .collect()
}

#[cfg(feature = "mvp")]
fn assert_monotonic_non_increasing(counts: &[usize], label: &str) {
    for window in counts.windows(2) {
        assert!(
            window[0] >= window[1],
            "{label} point counts should be non-increasing with coarser resolution: {:?}",
            counts
        );
    }
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_scan_like_copc_bounds_resolution_curve_monotonic() {
    use spatialrust::{read_copc_file, read_copc_file_info, read_copc_file_with_query, CopcQuery};

    const POINT_COUNT: usize = 50_000;
    let input_path = std::env::temp_dir().join(format!(
        "spatialrust_mvp_scan_curve_lib_{}.copc.laz",
        std::process::id()
    ));
    write_scan_like_copc_fixture(&input_path, POINT_COUNT);

    let info = read_copc_file_info(&input_path).unwrap();
    let full_count = read_copc_file(&input_path).unwrap().len();
    let roi_bounds = scan_like_roi_bounds();
    let bounds_only_count = read_copc_file_with_query(&input_path, &CopcQuery::bounds(roi_bounds))
        .unwrap()
        .len();

    let root_counts: Vec<usize> = resolution_curve_counts(&input_path, info.root_bounds)
        .into_iter()
        .map(|(_, count)| count)
        .collect();
    let roi_counts: Vec<usize> = resolution_curve_counts(&input_path, roi_bounds)
        .into_iter()
        .map(|(_, count)| count)
        .collect();

    assert_monotonic_non_increasing(&root_counts, "root bounds");
    assert_monotonic_non_increasing(&roi_counts, "roi bounds");
    assert!(root_counts[0] <= full_count);
    assert!(roi_counts[0] <= bounds_only_count);
    assert!(
        *roi_counts.last().expect("curve") < bounds_only_count,
        "coarsest roi+resolution should beat bounds-only"
    );
    assert!(
        root_counts.last().copied().unwrap_or(full_count) < full_count,
        "coarsest root+resolution should beat full load"
    );

    let _ = std::fs::remove_file(input_path);
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_cli_scan_like_copc_bounds_resolution_curve() {
    use std::process::Command;

    const POINT_COUNT: usize = 50_000;
    let input_path = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_scan_curve_in_{}.copc.laz",
        std::process::id()
    ));
    write_scan_like_copc_fixture(&input_path, POINT_COUNT);

    let info = spatialrust::read_copc_file_info(&input_path).unwrap();
    let root_curve = resolution_curve_counts(&input_path, info.root_bounds);
    let roi_curve = resolution_curve_counts(&input_path, scan_like_roi_bounds());
    let root_counts: Vec<usize> = root_curve.iter().map(|(_, count)| *count).collect();
    let roi_counts: Vec<usize> = roi_curve.iter().map(|(_, count)| *count).collect();
    assert_monotonic_non_increasing(&root_counts, "root bounds library");
    assert_monotonic_non_increasing(&roi_counts, "roi bounds library");

    let bin = env!("CARGO_BIN_EXE_spatialrust-mvp");
    let cli_args = ["--leaf-size", "4.0", "--voxel-policy", "cpu"];
    let mut root_cli_counts = Vec::new();
    let mut roi_cli_counts = Vec::new();

    for (multiplier, expected_count) in root_curve {
        let resolution = format!("{}", info.spacing * multiplier);
        let output_path = std::env::temp_dir().join(format!(
            "spatialrust_mvp_cli_scan_curve_root_{multiplier}_{}.copc.laz",
            std::process::id()
        ));
        let output = Command::new(bin)
            .args(cli_args)
            .args([
                "--resolution",
                &resolution,
                input_path.to_str().unwrap(),
                output_path.to_str().unwrap(),
            ])
            .output()
            .expect("run root resolution CLI");
        assert!(
            output.status.success(),
            "root resolution x{multiplier} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let loaded = parse_cli_input_points(&output.stderr);
        assert_eq!(loaded, expected_count);
        root_cli_counts.push(loaded);
        let _ = std::fs::remove_file(output_path);
    }

    for (multiplier, expected_count) in roi_curve {
        let resolution = format!("{}", info.spacing * multiplier);
        let output_path = std::env::temp_dir().join(format!(
            "spatialrust_mvp_cli_scan_curve_roi_{multiplier}_{}.copc.laz",
            std::process::id()
        ));
        let output = Command::new(bin)
            .args(cli_args)
            .args([
                "--bounds",
                scan_like_roi_bounds_arg(),
                "--resolution",
                &resolution,
                input_path.to_str().unwrap(),
                output_path.to_str().unwrap(),
            ])
            .output()
            .expect("run roi resolution CLI");
        assert!(
            output.status.success(),
            "roi resolution x{multiplier} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let loaded = parse_cli_input_points(&output.stderr);
        assert_eq!(loaded, expected_count);
        roi_cli_counts.push(loaded);
        let _ = std::fs::remove_file(output_path);
    }

    assert_monotonic_non_increasing(&root_cli_counts, "root bounds CLI");
    assert_monotonic_non_increasing(&roi_cli_counts, "roi bounds CLI");

    let _ = std::fs::remove_file(input_path);
}

#[cfg(feature = "mvp")]
#[test]
#[ignore = "manual probe for resolution curve counts"]
fn probe_scan_like_copc_resolution_curve_counts() {
    const POINT_COUNT: usize = 50_000;
    let input_path = std::env::temp_dir().join("spatialrust_probe_scan_curve.copc.laz");
    write_scan_like_copc_fixture(&input_path, POINT_COUNT);
    let info = spatialrust::read_copc_file_info(&input_path).unwrap();
    let spacing = info.spacing;
    let full_count = spatialrust::read_copc_file(&input_path).unwrap().len();
    let bounds_only = spatialrust::read_copc_file_with_query(
        &input_path,
        &spatialrust::CopcQuery::bounds(scan_like_roi_bounds()),
    )
    .unwrap()
    .len();
    eprintln!("spacing={spacing} full={full_count} roi_only={bounds_only}");
    eprintln!("multiplier | root+resolution | roi+resolution");
    for (multiplier, root_count) in resolution_curve_counts(&input_path, info.root_bounds) {
        let roi_count = resolution_curve_counts(&input_path, scan_like_roi_bounds())
            .into_iter()
            .find(|(m, _)| (*m - multiplier).abs() < f64::EPSILON)
            .map(|(_, c)| c)
            .expect("roi curve");
        eprintln!("x{multiplier:<4} | {root_count:>15} | {roi_count:>14}");
    }
    if std::env::var("SPATIALRUST_PROBE_RELEASE").is_ok() {
        use std::process::Command;

        let bin = env!("CARGO_BIN_EXE_spatialrust-mvp");
        let cli_args = ["--leaf-size", "4.0", "--voxel-policy", "cpu"];
        eprintln!("release CLI elapsed (ms):");
        let cases: [(&str, Vec<String>); 3] = [
            ("full", Vec::new()),
            (
                "roi",
                vec!["--bounds".to_string(), scan_like_roi_bounds_arg().to_string()],
            ),
            (
                "roi+x4",
                vec![
                    "--bounds".to_string(),
                    scan_like_roi_bounds_arg().to_string(),
                    "--resolution".to_string(),
                    format!("{}", spacing * 4.0),
                ],
            ),
        ];
        for (label, extra_args) in cases {
            let out = std::env::temp_dir().join(format!("spatialrust_probe_out_{label}.copc.laz"));
            let mut cmd = Command::new(bin);
            cmd.args(cli_args);
            for arg in &extra_args {
                cmd.arg(arg);
            }
            let output = cmd
                .arg(&input_path)
                .arg(&out)
                .output()
                .expect("release CLI");
            assert!(
                output.status.success(),
                "{label}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let elapsed = parse_cli_elapsed_ms(&output.stderr);
            let points = parse_cli_input_points(&output.stderr);
            eprintln!("{label}: points={points} elapsed={elapsed:.3}ms");
            let _ = std::fs::remove_file(out);
        }
    }
    let _ = std::fs::remove_file(input_path);
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
fn sample_xyzinormal_plane_grid(point_count: usize) -> spatialrust::PointCloud {
    use spatialrust::{PointCloudBuilder, StandardSchemas};

    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
    for index in 0..point_count {
        let x = (index % 256) as f32 * 0.1;
        let y = ((index / 256) % 256) as f32 * 0.1;
        let intensity = (index % 256) as f32;
        builder
            .push_point([x, y, 0.0, intensity, 0.0, 0.0, 1.0])
            .unwrap();
    }
    for x in 0..10 {
        for y in 0..10 {
            builder
                .push_point([90.0 + x as f32 * 0.02, y as f32 * 0.02, 2.5, 200.0, 0.0, 0.0, 1.0])
                .unwrap();
        }
    }
    builder.build().unwrap()
}

#[cfg(feature = "mvp")]
fn mvp_xyzinormal_approximate_auto_config() -> spatialrust::MvpPipelineConfig {
    use spatialrust::{
        EuclideanClusterConfig, MvpPipelineConfig, NormalEstimationConfig, RansacPlaneConfig,
        Vec3,
    };

    MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::approximate(4.0),
        voxel_policy: spatialrust::ExecutionPolicy::Auto,
        normals: NormalEstimationConfig {
            k_neighbors: 16,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 100.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.2,
            max_iterations: 200,
            min_inliers: 10,
            seed: 42,
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 1.0,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        },
        icp: None,
        ..MvpPipelineConfig::default()
    }
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
fn sample_xyzirgb_plane_cloud() -> spatialrust::PointCloud {
    use spatialrust::{PointCloudBuilder, StandardSchemas};

    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzirgb());
    for x in 0..10 {
        for y in 0..10 {
            builder
                .push_point([
                    x as f32 * 0.1,
                    y as f32 * 0.1,
                    0.0,
                    0.2,
                    200.0,
                    40.0,
                    40.0,
                ])
                .unwrap();
        }
    }
    builder
        .push_point([0.0, 0.0, 0.5, 0.9, 255.0, 128.0, 32.0])
        .unwrap();
    builder
        .push_point([0.1, 0.0, 0.5, 0.8, 64.0, 192.0, 96.0])
        .unwrap();
    builder.build().unwrap()
}

#[cfg(feature = "mvp")]
fn mvp_composite_base_config() -> spatialrust::MvpPipelineConfig {
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
fn mvp_composite_xyzirgb_pcd_pipeline_roundtrip() {
    use spatialrust::{
        HasIntensity, HasPositions3, MvpPipeline, PcdWriteFormat, PointBuffer, read_pcd, write_pcd,
    };
    use std::io::Cursor;

    let cloud = sample_xyzirgb_plane_cloud();
    let mut input_bytes = Vec::new();
    write_pcd(&mut input_bytes, &cloud, PcdWriteFormat::Ascii).unwrap();
    let loaded = read_pcd(&mut Cursor::new(input_bytes)).unwrap();

    assert!(loaded.intensity().is_ok());
    assert!(loaded.field("r").is_ok());

    let result = MvpPipeline::new(mvp_composite_base_config())
        .run(&loaded)
        .unwrap();

    assert!(!result.downsampled.is_empty());
    assert!(result.downsampled.intensity().is_ok());
    assert!(result.downsampled.field("r").is_ok());
    assert!(result.with_normals.field("normal_x").is_ok());
    assert!(result.plane.inlier_count >= 10);
    assert!(result.output.field("label").is_ok());

    let PointBuffer::U8(down_r) = result.downsampled.field("r").unwrap() else {
        panic!("expected u8 rgb channel");
    };
    assert!(down_r.iter().all(|value| *value > 0));

    let mut output_bytes = Vec::new();
    write_pcd(&mut output_bytes, &result.output, PcdWriteFormat::Binary).unwrap();
    let saved = read_pcd(&mut Cursor::new(output_bytes)).unwrap();
    assert_eq!(saved.len(), result.output.len());
    let (x, y, z) = saved.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_composite_xyzirgb_las_pipeline_roundtrip() {
    use spatialrust::{
        FieldSemantic, HasIntensity, HasPositions3, MvpPipeline, read_point_cloud_file,
        write_point_cloud_file,
    };

    let cloud = sample_xyzirgb_plane_cloud();
    let input_path =
        std::env::temp_dir().join(format!("spatialrust_mvp_xyzirgb_in_{}.las", std::process::id()));
    write_point_cloud_file(&input_path, &cloud).unwrap();
    let loaded = read_point_cloud_file(&input_path).unwrap();

    assert!(loaded.intensity().is_ok());
    assert!(loaded.schema().find_semantic(FieldSemantic::ColorR).is_some());

    let result = MvpPipeline::new(mvp_composite_base_config())
        .run(&loaded)
        .unwrap();

    let output_path =
        std::env::temp_dir().join(format!("spatialrust_mvp_xyzirgb_out_{}.las", std::process::id()));
    write_point_cloud_file(&output_path, &result.output).unwrap();
    let saved = read_point_cloud_file(&output_path).unwrap();

    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(output_path);

    assert_eq!(saved.len(), result.output.len());
    assert!(saved.schema().find_semantic(FieldSemantic::Label).is_some());
    assert!(saved.intensity().is_ok());
    assert!(saved.schema().find_semantic(FieldSemantic::ColorR).is_some());
    let (x, y, z) = saved.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));
}

#[cfg(all(feature = "mvp", feature = "pipeline-mvp-gpu"))]
#[test]
fn mvp_composite_xyzirgb_gpu_voxel_matches_cpu() {
    use spatialrust::{
        DeviceKind, ExecutionPolicy, HasIntensity, HasPositions3, MvpPipeline, PointBuffer,
    };

    let cloud = sample_xyzirgb_plane_cloud();
    let mut cpu_config = mvp_composite_base_config();
    cpu_config.voxel_policy = ExecutionPolicy::CpuSingle;
    let mut gpu_config = mvp_composite_base_config();
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

    for channel in ["r", "g", "b"] {
        let PointBuffer::U8(cpu_values) = cpu.downsampled.field(channel).unwrap() else {
            panic!("expected u8 channel");
        };
        let PointBuffer::U8(gpu_values) = gpu.downsampled.field(channel).unwrap() else {
            panic!("expected u8 channel");
        };
        assert_eq!(cpu_values, gpu_values);
    }
}

#[cfg(feature = "mvp")]
fn sample_large_scale_xyzi_grid(point_count: usize) -> spatialrust::PointCloud {
    use spatialrust::{PointCloudBuilder, StandardSchemas};

    let side = (point_count as f64).sqrt().ceil() as usize;
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
    for index in 0..point_count {
        let x = (index % side) as f32 * 0.1;
        let y = (index / side) as f32 * 0.1;
        let intensity = (index % 256) as f32;
        builder.push_point([x, y, 0.0, intensity]).unwrap();
    }
    for x in 0..10 {
        for y in 0..10 {
            builder
                .push_point([
                    side as f32 * 0.1 + 10.0 + x as f32 * 0.01,
                    y as f32 * 0.01,
                    0.5,
                    128.0,
                ])
                .unwrap();
        }
    }
    builder.build().unwrap()
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_large_scale_xyzi_pipeline_smoke() {
    use spatialrust::{
        EuclideanClusterConfig, HasIntensity, HasPositions3, MvpPipeline, MvpPipelineConfig,
        NormalEstimationConfig, RansacPlaneConfig, Vec3,
    };

    let cloud = sample_large_scale_xyzi_grid(100_000);
    let result = MvpPipeline::new(MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(4.0).without_gpu_min_points(),
        normals: NormalEstimationConfig {
            k_neighbors: 16,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 100.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.2,
            max_iterations: 200,
            min_inliers: 10,
            seed: 42,
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 1.0,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        },
        icp: None,
        ..MvpPipelineConfig::default()
    })
    .run(&cloud)
    .expect("large-scale xyzi MVP smoke");

    assert!(result.downsampled.len() < cloud.len());
    assert!(result.downsampled.intensity().is_ok());
    assert!(result.plane.inlier_count >= 10);
    assert!(result.output.field("label").is_ok());
    let (x, y, z) = result.output.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));
}

#[cfg(feature = "mvp")]
fn sample_scan_like_xyzi(point_count: usize) -> spatialrust::PointCloud {
    use spatialrust::{
        DType, FieldSemantic, PointCloudBuilder, PointField, StandardSchemas,
    };

    let schema = StandardSchemas::point_xyzi().with_field(PointField::scalar(
        "classification",
        FieldSemantic::Label,
        DType::U8,
    ));
    let mut builder = PointCloudBuilder::new(schema);
    for index in 0..point_count {
        let t = index as f32;
        let x = (t * 0.013).fract() * 80.0;
        let y = ((index % 97) as f32) * 0.41;
        let z = ((index % 53) as f32) * 0.023;
        let intensity = (index % 256) as f32;
        let classification = if z < 0.5 { 2.0 } else { 1.0 };
        builder
            .push_point([x, y, z, intensity, classification])
            .unwrap();
    }
    for x in 0..10 {
        for y in 0..10 {
            builder
                .push_point([90.0 + x as f32 * 0.02, y as f32 * 0.02, 2.5, 200.0, 6.0])
                .unwrap();
        }
    }
    builder.build().unwrap()
}

#[cfg(feature = "mvp")]
fn mvp_scan_like_base_config() -> spatialrust::MvpPipelineConfig {
    use spatialrust::{
        EuclideanClusterConfig, MvpPipelineConfig, NormalEstimationConfig, RansacPlaneConfig,
        Vec3,
    };

    MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(4.0).without_gpu_min_points(),
        voxel_policy: spatialrust::ExecutionPolicy::CpuSingle,
        normals: NormalEstimationConfig {
            k_neighbors: 16,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 100.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.2,
            max_iterations: 200,
            min_inliers: 10,
            seed: 42,
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 1.0,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        },
        icp: None,
        ..MvpPipelineConfig::default()
    }
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_scan_like_las_file_pipeline() {
    use spatialrust::{
        FieldSemantic, HasIntensity, HasPositions3, MvpPipeline, read_point_cloud_file,
        write_point_cloud_file,
    };

    const POINT_COUNT: usize = 50_000;
    let cloud = sample_scan_like_xyzi(POINT_COUNT);
    let input_path =
        std::env::temp_dir().join(format!("spatialrust_mvp_scan_las_in_{}.las", std::process::id()));
    write_point_cloud_file(&input_path, &cloud).unwrap();
    let loaded = read_point_cloud_file(&input_path).unwrap();

    assert_eq!(loaded.len(), cloud.len());
    assert!(loaded.intensity().is_ok());
    assert!(loaded.schema().find_semantic(FieldSemantic::Label).is_some());

    let result = MvpPipeline::new(mvp_scan_like_base_config())
        .run(&loaded)
        .expect("scan-like LAS MVP");

    let output_path =
        std::env::temp_dir().join(format!("spatialrust_mvp_scan_las_out_{}.las", std::process::id()));
    write_point_cloud_file(&output_path, &result.output).unwrap();
    let saved = read_point_cloud_file(&output_path).unwrap();

    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(output_path);

    assert!(result.downsampled.len() < loaded.len());
    assert!(result.plane.inlier_count >= 10);
    assert!(result.output.field("label").is_ok());
    assert!(saved.schema().find_semantic(FieldSemantic::Label).is_some());
    let (x, y, z) = saved.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_scan_like_copc_file_pipeline() {
    use spatialrust::{
        FieldSemantic, HasIntensity, HasPositions3, MvpPipeline, read_copc_file, write_copc_file_with_params,
        CopcWriterParams,
    };

    const POINT_COUNT: usize = 50_000;
    let cloud = sample_scan_like_xyzi(POINT_COUNT);
    let path = std::env::temp_dir().join(format!(
        "spatialrust_mvp_scan_copc_{}.copc.laz",
        std::process::id()
    ));
    write_copc_file_with_params(
        &path,
        &cloud,
        &CopcWriterParams {
            max_points_per_node: 512,
            max_depth: 10,
        },
    )
    .unwrap();

    let loaded = read_copc_file(&path).unwrap();
    assert_eq!(loaded.len(), cloud.len());
    assert!(loaded.intensity().is_ok());
    assert!(loaded.schema().find_semantic(FieldSemantic::Label).is_some());

    let result = MvpPipeline::new(mvp_scan_like_base_config())
        .run(&loaded)
        .expect("scan-like COPC MVP");

    let _ = std::fs::remove_file(&path);

    assert!(result.downsampled.len() < loaded.len());
    assert!(result.plane.inlier_count >= 10);
    assert!(result.output.field("label").is_ok());
    let (x, y, z) = result.output.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_scan_like_copc_resolution_file_pipeline() {
    use spatialrust::{
        CopcQuery, HasIntensity, MvpPipeline, read_copc_file, read_copc_file_info,
        read_copc_file_with_query, write_copc_file_with_params, CopcWriterParams,
    };

    const POINT_COUNT: usize = 50_000;
    let cloud = sample_scan_like_xyzi(POINT_COUNT);
    let path = std::env::temp_dir().join(format!(
        "spatialrust_mvp_scan_copc_res_{}.copc.laz",
        std::process::id()
    ));
    write_copc_file_with_params(
        &path,
        &cloud,
        &CopcWriterParams {
            max_points_per_node: 512,
            max_depth: 10,
        },
    )
    .unwrap();

    let info = read_copc_file_info(&path).unwrap();
    let full = read_copc_file(&path).unwrap();
    let loaded = read_copc_file_with_query(
        &path,
        &CopcQuery::with_resolution(info.root_bounds, info.spacing * 4.0),
    )
    .unwrap();

    assert_eq!(full.len(), cloud.len());
    assert!(loaded.len() < full.len());
    assert!(loaded.intensity().is_ok());

    let result = MvpPipeline::new(mvp_scan_like_base_config())
        .run(&loaded)
        .expect("scan-like COPC resolution MVP");

    let _ = std::fs::remove_file(&path);

    assert!(!result.downsampled.is_empty());
    assert!(result.plane.inlier_count >= 10);
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

#[cfg(all(feature = "mvp", feature = "pipeline-mvp-gpu"))]
fn parse_cli_plane_inliers(stderr: &[u8]) -> usize {
    let text = String::from_utf8_lossy(stderr);
    text.lines()
        .find_map(|line| line.strip_prefix("plane inliers: "))
        .and_then(|value| value.trim().parse().ok())
        .expect("CLI stderr should report plane inliers")
}

#[cfg(all(feature = "mvp", feature = "pipeline-mvp-gpu"))]
#[test]
fn mvp_cli_xyzinormal_approximate_auto_1m() {
    use std::process::Command;

    use spatialrust::{
        FieldSemantic, HasPositions3, read_point_cloud_file, write_point_cloud_file,
    };

    const POINT_COUNT: usize = 1_000_000;
    let cloud = sample_xyzinormal_plane_grid(POINT_COUNT);
    let input_path = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_xyzinormal_approx_auto_in_{}.las",
        std::process::id()
    ));
    let output_path = std::env::temp_dir().join(format!(
        "spatialrust_mvp_cli_xyzinormal_approx_auto_out_{}.las",
        std::process::id()
    ));
    write_point_cloud_file(&input_path, &cloud).unwrap();

    let bin = env!("CARGO_BIN_EXE_spatialrust-mvp");
    let output = Command::new(bin)
        .args([
            "--leaf-size",
            "4.0",
            "--voxel-mode",
            "approximate",
            "--voxel-policy",
            "auto",
            input_path.to_str().unwrap(),
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("run xyzinormal approximate Auto CLI");
    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let loaded_points = parse_cli_input_points(&output.stderr);
    assert_eq!(loaded_points, cloud.len());
    assert!(parse_cli_plane_inliers(&output.stderr) >= 10);

    let saved = read_point_cloud_file(&output_path).unwrap();
    assert!(saved.schema().find_semantic(FieldSemantic::Label).is_some());
    let (x, y, z) = saved.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));

    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(output_path);
}

#[cfg(all(feature = "mvp", feature = "pipeline-mvp-gpu"))]
#[test]
#[ignore = "manual release probe for xyzinormal approximate Auto CLI"]
fn probe_xyzinormal_approximate_auto_cli_release() {
    use std::process::Command;

    use spatialrust::write_point_cloud_file;

    const POINT_COUNT: usize = 1_000_000;
    let cloud = sample_xyzinormal_plane_grid(POINT_COUNT);
    let input_path = std::env::temp_dir().join(format!(
        "spatialrust_probe_xyzinormal_approx_auto_in_{}.las",
        std::process::id()
    ));
    write_point_cloud_file(&input_path, &cloud).unwrap();

    if std::env::var("SPATIALRUST_PROBE_RELEASE").is_ok() {
        let bin = env!("CARGO_BIN_EXE_spatialrust-mvp");
        let base_args = [
            "--leaf-size",
            "4.0",
            "--voxel-mode",
            "approximate",
        ];
        let policies = [("auto", "auto"), ("cpu", "cpu"), ("gpu", "gpu")];
        eprintln!("release xyzinormal approximate CLI (1M LAS, IO included):");
        for (label, policy) in policies {
            let output_path = std::env::temp_dir().join(format!(
                "spatialrust_probe_xyzinormal_approx_auto_{label}_{}.las",
                std::process::id()
            ));
            let output = Command::new(bin)
                .args(base_args)
                .args(["--voxel-policy", policy])
                .arg(input_path.to_str().unwrap())
                .arg(output_path.to_str().unwrap())
                .output()
                .expect("release CLI");
            assert!(
                output.status.success(),
                "{label}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let points = parse_cli_input_points(&output.stderr);
            let elapsed = parse_cli_elapsed_ms(&output.stderr);
            eprintln!("{label}: points={points} elapsed={elapsed:.3}ms");
            let _ = std::fs::remove_file(output_path);
        }
    }

    let _ = std::fs::remove_file(input_path);
}

#[cfg(all(feature = "mvp", feature = "pipeline-mvp-gpu"))]
fn elapsed_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(all(feature = "mvp", feature = "pipeline-mvp-gpu"))]
#[test]
#[ignore = "manual release probe for in-process GPU warmup"]
fn probe_xyzinormal_approximate_auto_gpu_warmup() {
    use std::time::Instant;

    use spatialrust::{DeviceKind, ExecutionPolicy, MvpPipeline};

    if std::env::var("SPATIALRUST_PROBE_RELEASE").is_err() {
        return;
    }

    const POINT_COUNT: usize = 1_000_000;
    let cloud = sample_xyzinormal_plane_grid(POINT_COUNT);

    let mut gpu_config = mvp_xyzinormal_approximate_auto_config();
    gpu_config.voxel_policy = ExecutionPolicy::Gpu(DeviceKind::Wgpu);
    let gpu_pipeline = MvpPipeline::new(gpu_config);

    let gpu_cold_started = Instant::now();
    gpu_pipeline
        .run(&cloud)
        .expect("gpu approximate Auto MVP cold");
    let gpu_cold_ms = elapsed_ms(gpu_cold_started.elapsed());

    let gpu_warm_started = Instant::now();
    gpu_pipeline
        .run(&cloud)
        .expect("gpu approximate Auto MVP warm");
    let gpu_warm_ms = elapsed_ms(gpu_warm_started.elapsed());

    let mut cpu_config = mvp_xyzinormal_approximate_auto_config();
    cpu_config.voxel_policy = ExecutionPolicy::CpuSingle;
    let cpu_started = Instant::now();
    MvpPipeline::new(cpu_config)
        .run(&cloud)
        .expect("cpu approximate Auto MVP");
    let cpu_ms = elapsed_ms(cpu_started.elapsed());

    let auto_started = Instant::now();
    MvpPipeline::new(mvp_xyzinormal_approximate_auto_config())
        .run(&cloud)
        .expect("auto approximate Auto MVP");
    let auto_ms = elapsed_ms(auto_started.elapsed());

    eprintln!("in-process xyzinormal approximate @1M (custom MVP config, no LAS IO):");
    eprintln!("cpu: {cpu_ms:.3}ms");
    eprintln!("auto (warm shared runtime): {auto_ms:.3}ms");
    eprintln!("gpu cold (includes WgpuRuntime::shared init): {gpu_cold_ms:.3}ms");
    eprintln!("gpu warm (shared runtime reused): {gpu_warm_ms:.3}ms");
}

#[cfg(all(feature = "mvp", feature = "pipeline-mvp-gpu"))]
#[test]
fn mvp_xyzinormal_approximate_auto_1m_smoke() {
    use spatialrust::{HasIntensity, HasNormals3, HasPositions3, MvpPipeline};

    const POINT_COUNT: usize = 1_000_000;
    let cloud = sample_xyzinormal_plane_grid(POINT_COUNT);
    let result = MvpPipeline::new(mvp_xyzinormal_approximate_auto_config())
        .run(&cloud)
        .expect("xyzinormal approximate Auto MVP @1M");

    assert!(result.downsampled.len() < cloud.len());
    assert!(result.downsampled.intensity().is_ok());
    assert!(result.downsampled.field("normal_x").is_ok());
    assert!(result.plane.inlier_count >= 10);
    assert!(result.output.field("label").is_ok());
    let (x, y, z) = result.output.positions3().unwrap();
    assert!(x.iter().chain(y).chain(z).all(|value| value.is_finite()));
    let (_, _, nz) = result.downsampled.normals3().unwrap();
    assert!(nz.iter().all(|value| value.is_finite()));
}
