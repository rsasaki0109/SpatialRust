#![cfg(feature = "gpu-wgpu")]

use spatialrust::gpu::{estimate_plane_covariances_grid_gpu, WgpuRuntime};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly on GPU-capable hosts"]
fn facade_covariance_fallback_clears_previous_plane_cross_terms() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let n = 257;
    let x: Vec<_> = (0..n).map(|i| (i % 32) as f32 * 0.25).collect();
    let y: Vec<_> = (0..n).map(|i| (i / 32) as f32 * 0.25).collect();
    let z: Vec<_> = x.iter().zip(&y).map(|(&x, &y)| x * 0.25 + y * 0.5).collect();
    let plane = estimate_plane_covariances_grid_gpu(&runtime, &x, &y, &z, 1.0, 0.001).unwrap();
    assert!(plane.iter().any(|v| v[4].abs() > 1e-3 && v[5].abs() > 1e-3));
    let before = runtime.scratch_buffer_pool_stats();
    let sparse: Vec<_> = (0..n).map(|i| i as f32 * 4.0).collect();
    let zero = vec![0.0; n];
    let fallback =
        estimate_plane_covariances_grid_gpu(&runtime, &sparse, &zero, &zero, 1.0, 0.001).unwrap();
    assert_eq!(fallback, vec![[0.001, 0.001, 0.001, 0.0, 0.0, 0.0]; n]);
    assert!(runtime.scratch_buffer_pool_stats().reuse_count >= before.reuse_count + 5);
    // The prior host result remains independently owned after storage reuse.
    assert!(plane.iter().any(|v| v[4].abs() > 1e-3 && v[5].abs() > 1e-3));
}
