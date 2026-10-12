#![cfg(all(feature = "register-gicp-gpu", not(target_arch = "wasm32")))]
use spatialrust::{
    gpu::WgpuRuntime, GicpConfig, GicpRegistration, GpuGicpReductionDiagnostics, PointCloudBuilder,
};
#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_gicp_reduction_exposes_equations_receipts_and_convergence() {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..65 {
        builder
            .push_point([(i % 5) as f32 * 0.5, (i / 5 % 5) as f32 * 0.5, (i / 25) as f32 * 0.5])
            .unwrap();
    }
    let cloud = builder.build().unwrap();
    let runtime = WgpuRuntime::new_headless().unwrap();
    let result: GpuGicpReductionDiagnostics = GicpRegistration::new(GicpConfig::default())
        .align_with_gpu_reduction(&runtime, &cloud, &cloud)
        .unwrap();
    assert!(result.result.converged);
    assert_eq!(result.evaluation_batches, 2);
    assert_eq!(result.transfers.device_to_host_bytes(), 512);
    assert_eq!(result.covariance_batches, 0);
}
