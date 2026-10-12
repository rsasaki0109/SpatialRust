#![cfg(all(feature = "filter-mls-gpu", not(target_arch = "wasm32")))]

use spatialrust::{HasPositions3, MlsConfig, MlsSmoothing, PointCloudBuilder};
use spatialrust_gpu::WgpuRuntime;

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_gpu_mls_exposes_quadratic_surface_and_complete_receipts() {
    let mut builder = PointCloudBuilder::xyz();
    for x in -2..=2 {
        for y in -2..=2 {
            let x = x as f32 * 0.05;
            let y = y as f32 * 0.05;
            let z = if x == 0.0 && y == 0.0 { 0.02 } else { 0.5 * (x * x + y * y) };
            builder.push_point([x, y, z]).unwrap();
        }
    }
    let cloud = builder.build().unwrap();
    let runtime = WgpuRuntime::new_headless().unwrap();
    let smoother = MlsSmoothing::new(MlsConfig::with_radius(0.3));
    let output = smoother.filter_with_gpu(&runtime, &cloud, 25).unwrap();
    let (_, _, z) = output.cloud.positions3().unwrap();
    assert!(z[12].abs() < 1e-6);
    assert_eq!(output.cloud.schema(), cloud.schema());
    assert_eq!(output.cloud.metadata(), cloud.metadata());
    assert_eq!(output.modes.len(), 25);
    assert!(output.transfers.host_to_device_bytes() >= 25 * 32);
    assert_eq!(output.transfers.device_to_host_bytes(), 25 * 16 + 32);
    assert!(smoother.filter_with_gpu(&runtime, &cloud, 1).is_err());
}
