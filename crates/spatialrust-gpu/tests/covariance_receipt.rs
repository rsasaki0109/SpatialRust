#![cfg(all(feature = "gpu-wgpu", not(target_arch = "wasm32")))]
use spatialrust_core::TransferStats;
use spatialrust_gpu::{
    estimate_plane_covariances_grid_gpu, estimate_plane_covariances_grid_gpu_with_receipt,
    WgpuRuntime,
};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn covariance_receipt_counts_indices_offsets_and_padded_readback() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for (x, radius, upload) in [
        (vec![0.0], 1.0, 24),
        (vec![0.0, 1.0, 2.0], 1.0, 64),
        (vec![0.0, 1.0, 2.0], f32::INFINITY, 56),
    ] {
        let yz = vec![0.0; x.len()];
        let old =
            estimate_plane_covariances_grid_gpu(&runtime, &x, &yz, &yz, radius, 0.001).unwrap();
        let (new, receipt) =
            estimate_plane_covariances_grid_gpu_with_receipt(&runtime, &x, &yz, &yz, radius, 0.001)
                .unwrap();
        assert_eq!(new, old);
        assert_eq!(receipt.host_to_device_bytes(), upload);
        assert_eq!(receipt.device_to_host_bytes(), x.len() as u64 * 32);
    }
    let (values, receipt) =
        estimate_plane_covariances_grid_gpu_with_receipt(&runtime, &[], &[], &[], 0.0, f32::NAN)
            .unwrap();
    assert!(values.is_empty());
    assert_eq!(receipt, TransferStats::default());
    assert!(estimate_plane_covariances_grid_gpu_with_receipt(
        &runtime,
        &[0.0],
        &[],
        &[0.0],
        1.0,
        0.001
    )
    .is_err());
}
