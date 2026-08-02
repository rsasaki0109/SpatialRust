#![cfg(all(feature = "camera-rgbd", feature = "mvp"))]

use spatialrust::{
    rgbd_to_point_cloud, CameraIntrinsics, DepthConversionOptions, EuclideanClusterConfig, Image,
    MvpPipeline, MvpPipelineConfig, NormalEstimationConfig, PinholeCamera, RansacPlaneConfig, Vec3,
    VoxelGridDownsampleConfig,
};

#[test]
fn rgbd_cloud_runs_through_mvp_pipeline() {
    let width = 16;
    let height = 16;
    let mut depths = vec![1.0_f32; width * height];
    for y in 6..10 {
        for x in 6..10 {
            depths[y * width + x] = 0.7;
        }
    }
    let depth = Image::<f32, 1>::try_new(width, height, depths).unwrap();
    let color = Image::<u8, 3>::try_new(width, height, vec![128; width * height * 3]).unwrap();
    let camera =
        PinholeCamera::new(CameraIntrinsics::try_new(20.0, 20.0, 7.5, 7.5, width, height).unwrap());
    let cloud =
        rgbd_to_point_cloud(depth.view(), color.view(), &camera, DepthConversionOptions::default())
            .unwrap();

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
            seed: 42,
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
    assert_eq!(cloud.len(), width * height);
    assert!(result.plane.inlier_count >= 50);
    assert!(!result.output.is_empty());
    assert!(result.output.schema().find_semantic(spatialrust::FieldSemantic::ColorR).is_some());
}
