#![cfg(all(feature = "register-gicp-gpu", not(target_arch = "wasm32")))]
use spatialrust::core::{PointCloudBuilder, StandardSchemas};
use spatialrust::gpu::WgpuRuntime;
use spatialrust::registration::{GicpConfig, GicpRegistration};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_gpu_gicp_combines_matching_and_radius_covariance_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    for i in 0..257 {
        builder
            .push_point([(i % 9) as f32 * 0.5, (i / 9 % 9) as f32 * 0.5, (i / 81) as f32 * 0.5])
            .unwrap();
    }
    let cloud = builder.build().unwrap();
    let registration = GicpRegistration::new(GicpConfig {
        covariance_radius: Some(0.75),
        ..GicpConfig::default()
    });
    let actual = registration.align_with_gpu_correspondences(&runtime, &cloud, &cloud).unwrap();
    assert_eq!(actual.result, registration.align_with_diagnostics(&cloud, &cloud).unwrap());
    assert_eq!(actual.covariance_batches, 2);
    assert_eq!(actual.correspondence_batches, 2);
    assert_eq!(actual.transfers.device_to_host_bytes(), 257 * (32 * 2 + 8 * 2));
}
