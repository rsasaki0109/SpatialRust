#![cfg(all(feature = "register-icp-gpu", not(target_arch = "wasm32")))]

use spatialrust::core::{PointCloudBuilder, StandardSchemas};
use spatialrust::gpu::WgpuRuntime;
use spatialrust::registration::{IcpConfig, IcpRegistration};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_explicit_gpu_icp_preserves_cpu_trace_and_reports_transfers() {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    for i in 0..257 {
        builder.push_point([(i % 9) as f32, (i / 9 % 9) as f32, (i / 81) as f32]).unwrap();
    }
    let cloud = builder.build().unwrap();
    let registration = IcpRegistration::new(IcpConfig::default());
    let cpu = registration.align_with_trace(&cloud, &cloud).unwrap();
    let runtime = WgpuRuntime::new_headless().unwrap();
    let gpu = registration.align_with_gpu_correspondences(&runtime, &cloud, &cloud).unwrap();
    assert_eq!(gpu.diagnostics, cpu);
    assert_eq!(gpu.correspondence_batches, 3);
    assert!(gpu.transfers.host_to_device_bytes() > 257 * 16 * 3);
    assert_eq!(gpu.transfers.device_to_host_bytes(), 257 * 8 * 3);
}
