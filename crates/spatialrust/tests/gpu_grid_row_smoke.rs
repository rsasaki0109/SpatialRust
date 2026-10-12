#![cfg(feature = "gpu-wgpu")]

use spatialrust::gpu::{
    estimate_normals_grid_gpu, estimate_plane_covariances_grid_gpu, WgpuRuntime,
};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly on GPU-capable hosts"]
fn facade_grid_rows_preserve_planar_normals_and_covariances() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let x: Vec<_> = (0..257).map(|i| (i % 32) as f32 * 0.5 - 8.0).collect();
    let y: Vec<_> = (0..257).map(|i| (i / 32) as f32 * 0.5 - 2.0).collect();
    let z = vec![0.0; 257];
    let normals = estimate_normals_grid_gpu(&runtime, &x, &y, &z, 1.0).unwrap();
    let covariances =
        estimate_plane_covariances_grid_gpu(&runtime, &x, &y, &z, 1.0, 0.001).unwrap();
    assert_eq!(normals.len(), 257);
    assert_eq!(covariances.len(), 257);
    for normal in normals {
        assert!(normal.normal[2].abs() > 0.999);
        assert!(normal.curvature.abs() < 1e-6);
    }
    for covariance in covariances {
        for (actual, expected) in covariance.into_iter().zip([1.0, 1.0, 0.001, 0.0, 0.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
