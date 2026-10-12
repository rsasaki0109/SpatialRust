#![cfg(all(feature = "feature-iss-gpu", not(target_arch = "wasm32")))]

use spatialrust::{IssKeypointConfig, IssKeypointDetector, PointCloudBuilder};
use spatialrust_gpu::WgpuRuntime;

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_gpu_iss_keeps_analytic_ties_and_reports_explicit_transfers() {
    let mut builder = PointCloudBuilder::xyz();
    for x in [-0.06, 0.06] {
        for y in [-0.04, 0.04] {
            for z in [-0.02, 0.02] {
                builder.push_point([x, y, z]).unwrap();
            }
        }
    }
    builder.push_point([100.0, 100.0, 100.0]).unwrap();
    let input = builder.build().unwrap();
    let detector = IssKeypointDetector::new(IssKeypointConfig::with_radii(0.2, 1.0));
    let runtime = WgpuRuntime::new_headless().unwrap();
    let gpu = detector.detect_with_gpu(&runtime, &input, 9, 9).unwrap();
    assert_eq!(gpu.result, detector.detect(&input).unwrap());
    assert_eq!(gpu.result.indices, (0..8).collect::<Vec<_>>());
    assert!(gpu.transfers.host_to_device_bytes() >= 9 * 32);
    assert_eq!(gpu.transfers.device_to_host_bytes(), 9 * 4 + 32);
    assert!(detector.detect_with_gpu(&runtime, &input, 1, 9).is_err());
}
