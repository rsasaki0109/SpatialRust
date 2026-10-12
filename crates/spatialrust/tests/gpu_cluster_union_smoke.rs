#![cfg(feature = "segment-euclidean-gpu")]

use spatialrust::core::{
    DeviceKind, ExecutionPolicy, PointBuffer, PointCloudBuilder, StandardSchemas,
};
use spatialrust::segmentation::{EuclideanClusterConfig, EuclideanClusterExtractor};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_gpu_component_labels_match_cpu_and_record_gpu_stage() {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    for group in 0..3 {
        for i in 0..521 {
            builder.push_point([group as f32 * 1000.0 + i as f32 * 0.5, 0.0, 0.0]).unwrap();
        }
    }
    let input = builder.build().unwrap();
    let extractor = EuclideanClusterExtractor::new(EuclideanClusterConfig {
        cluster_tolerance: 0.75,
        min_cluster_size: 1,
        max_cluster_size: usize::MAX,
        gpu_min_points: Some(2000),
    });
    let gpu = extractor
        .extract_with_policy_and_receipt(&input, ExecutionPolicy::Gpu(DeviceKind::Wgpu))
        .unwrap();
    let cpu = extractor.extract_with_policy_and_receipt(&input, ExecutionPolicy::Auto).unwrap();
    assert_eq!(gpu.output().cluster_sizes, [521; 3]);
    assert_eq!(
        gpu.output().cloud.field("label").unwrap(),
        &PointBuffer::I32((0..1563).map(|i| i / 521).collect())
    );
    assert_eq!(
        gpu.output().cloud.field("label").unwrap(),
        cpu.output().cloud.field("label").unwrap()
    );
    assert_eq!(cpu.receipt().resolved_policy(), ExecutionPolicy::CpuSingle);
    assert_eq!(cpu.receipt().device_to_host_bytes(), 0);
    assert_eq!(
        gpu.receipt().stages(),
        &["euclidean-clustering", "gpu-sparse-grid", "gpu-component-labeling"]
    );
    assert_eq!(gpu.receipt().host_to_device_bytes(), 1563 * 12);
    assert_eq!(gpu.receipt().device_to_host_bytes(), 1563 * 4 + 4);
}
