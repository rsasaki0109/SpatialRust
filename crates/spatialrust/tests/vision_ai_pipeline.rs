#![cfg(all(feature = "vision-full", feature = "mvp"))]

use spatialrust::{
    point_map_to_point_cloud, ConfidenceMap, EuclideanClusterConfig, MvpPipeline,
    MvpPipelineConfig, NormalEstimationConfig, PointMap, RansacPlaneConfig, Vec3,
    VoxelGridDownsampleConfig,
};

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
