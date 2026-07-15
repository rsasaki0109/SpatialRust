//! Image → mock AI → depth → point cloud end-to-end pipeline (Epic 90).
//!
//! Also retains a dense PointMap → MVP smoke path that skips inference.

#![cfg(all(feature = "ai-vision-pipeline", feature = "mvp"))]

use spatialrust::ai::{
    CopyPolicy, InferenceBackend, MockInferenceBackend, MockProfile, ModelSession as _,
    ModelSource, NamedTensors, RunOptions, SessionOptions,
};
use spatialrust::{
    depth_map_to_point_cloud, depth_tensor_to_depth_map, point_map_to_point_cloud,
    rgb_u8_to_nchw_f32, CameraIntrinsics, ConfidenceMap, DepthConversionOptions,
    EuclideanClusterConfig, Image, Interpolation, MvpPipeline, MvpPipelineConfig,
    NormalEstimationConfig, PinholeCamera, PointMap, RansacPlaneConfig, Vec3,
    VoxelGridDownsampleConfig,
};

#[test]
fn image_mock_depth_runs_through_spatial_pipeline() {
    let width = 32usize;
    let height = 32usize;
    let mut pixels = vec![40_u8; width * height * 3];
    for y in 12..20 {
        for x in 12..20 {
            let index = (y * width + x) * 3;
            pixels[index..index + 3].copy_from_slice(&[220, 220, 220]);
        }
    }
    let image = Image::<u8, 3>::try_new(width, height, pixels).unwrap();
    let (input, _mapping) = rgb_u8_to_nchw_f32(
        image.view(),
        width,
        height,
        Interpolation::Nearest,
        [0, 0, 0],
        1.0 / 255.0,
        [0.0; 3],
        [1.0; 3],
    )
    .unwrap();

    let backend = MockInferenceBackend;
    let mut session = backend
        .create_session(
            &ModelSource::Mock(MockProfile::SyntheticDepth),
            &SessionOptions::default(),
        )
        .unwrap();
    let mut inputs = NamedTensors::new();
    inputs.insert("images", input).unwrap();
    let outputs = session
        .run_with_options(
            inputs,
            RunOptions {
                input_copy: CopyPolicy::Forbid,
                output_copy: CopyPolicy::Allow,
            },
        )
        .unwrap();
    let depth = depth_tensor_to_depth_map(outputs.get("depth").unwrap()).unwrap();
    assert_eq!(depth.image().width(), width);
    assert_eq!(depth.image().height(), height);

    let camera = PinholeCamera::new(
        CameraIntrinsics::try_new(
            width as f64,
            height as f64,
            (width as f64 - 1.0) * 0.5,
            (height as f64 - 1.0) * 0.5,
            width,
            height,
        )
        .unwrap(),
    );
    let cloud =
        depth_map_to_point_cloud(&depth, &camera, DepthConversionOptions::default()).unwrap();
    assert_eq!(cloud.len(), width * height);

    let pipeline = MvpPipeline::new(MvpPipelineConfig {
        voxel: VoxelGridDownsampleConfig::centroid(0.05),
        normals: NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 0.0)),
            ..Default::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 200,
            min_inliers: 40,
            seed: 90,
            ..Default::default()
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 0.15,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
            ..Default::default()
        },
        icp: None,
        ..Default::default()
    });
    let result = pipeline.run(&cloud).unwrap();
    assert!(result.plane.inlier_count >= 40);
    assert!(!result.output.is_empty());
}

#[test]
fn dense_ai_point_map_runs_through_spatial_pipeline() {
    let width = 16;
    let height = 16;
    let mut points = Vec::with_capacity(width * height * 3);
    let mut confidence = vec![1.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let z = if (6..10).contains(&x) && (6..10).contains(&y) { 0.7 } else { 1.0 };
            points.extend_from_slice(&[
                (x as f32 - 7.5) * z / 20.0,
                (y as f32 - 7.5) * z / 20.0,
                z,
            ]);
        }
    }
    confidence[0] = 0.1;
    let point_map = PointMap::try_new(width, height, points).unwrap();
    let confidence = ConfidenceMap::try_new(width, height, confidence).unwrap();
    let cloud = point_map_to_point_cloud(&point_map, Some(&confidence), 0.5).unwrap();

    let pipeline = MvpPipeline::new(MvpPipelineConfig {
        voxel: VoxelGridDownsampleConfig::centroid(0.025),
        normals: NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 0.0)),
            ..Default::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.02,
            max_iterations: 200,
            min_inliers: 50,
            seed: 75,
            ..Default::default()
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 0.1,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
            ..Default::default()
        },
        icp: None,
        ..Default::default()
    });
    let result = pipeline.run(&cloud).unwrap();
    assert_eq!(cloud.len(), width * height - 1);
    assert!(result.plane.inlier_count >= 50);
    assert!(!result.output.is_empty());
}
