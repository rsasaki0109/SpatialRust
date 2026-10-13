#![cfg(all(feature = "register-icp-gpu", not(target_arch = "wasm32")))]
use spatialrust::{
    gpu::WgpuRuntime, GpuIcpReductionDiagnostics, IcpConfig, IcpRegistration, PointCloudBuilder,
};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_resident_icp_returns_fixed_size_receipts_and_convergence() {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..65 {
        builder.push_point([(i % 5) as f32, (i / 5 % 5) as f32, (i / 25) as f32]).unwrap();
    }
    let cloud = builder.build().unwrap();
    let runtime = WgpuRuntime::new_headless().unwrap();
    let result: GpuIcpReductionDiagnostics = IcpRegistration::new(IcpConfig::default())
        .align_with_gpu_reduction(&runtime, &cloud, &cloud)
        .unwrap();
    assert!(result.diagnostics.result.converged);
    assert_eq!(result.evaluation_batches, 2);
    assert_eq!(result.transfers.device_to_host_bytes(), 352);
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_resident_trimmed_icp_exposes_retained_and_all_gated_counts() {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..65 {
        builder.push_point([(i % 5) as f32, (i / 5 % 5) as f32, (i / 25) as f32]).unwrap();
    }
    let cloud = builder.build().unwrap();
    let runtime = WgpuRuntime::new_headless().unwrap();
    let result: GpuIcpReductionDiagnostics = IcpRegistration::new(IcpConfig::default())
        .with_trim_fraction(0.8)
        .align_with_gpu_trimmed_reduction(&runtime, &cloud, &cloud)
        .unwrap();
    assert!(result.diagnostics.result.converged);
    assert_eq!(result.diagnostics.history[0].correspondences, 52);
    assert_eq!(result.diagnostics.history[0].evaluated_correspondences, 65);
    assert_eq!(result.evaluation_batches, 2);
    assert_eq!(result.transfers.device_to_host_bytes(), 352);
}
