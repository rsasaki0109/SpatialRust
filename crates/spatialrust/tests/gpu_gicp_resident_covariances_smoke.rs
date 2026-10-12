#![cfg(all(feature = "register-gicp-gpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_exposes_resident_gicp_covariance_preparation() {
    use spatialrust::{
        gpu::WgpuRuntime, GicpConfig, GicpRegistration, GpuGicpReductionDiagnostics,
        PointCloudBuilder,
    };
    let mut b = PointCloudBuilder::xyz();
    for i in 0..65 {
        b.push_point([(i % 5) as f32 * 0.25, (i / 5 % 5) as f32 * 0.25, (i / 25) as f32 * 0.25])
            .unwrap();
    }
    let cloud = b.build().unwrap();
    let runtime = WgpuRuntime::new_headless().unwrap();
    let actual: GpuGicpReductionDiagnostics = GicpRegistration::new(GicpConfig::default())
        .align_with_gpu_resident_covariances(&runtime, &cloud, &cloud)
        .unwrap();
    assert!(actual.result.converged);
    assert_eq!(actual.result.fitness, 0.0);
    assert_eq!(actual.evaluation_batches, 2);
    assert_eq!(actual.covariance_batches, 2);
    assert_eq!(actual.transfers.device_to_host_bytes(), 512);
}
