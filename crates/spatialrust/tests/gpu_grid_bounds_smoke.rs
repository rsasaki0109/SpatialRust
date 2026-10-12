#![cfg(feature = "gpu-wgpu")]

use spatialrust::gpu::{
    estimate_normals_grid_gpu, estimate_plane_covariances_grid_gpu, WgpuRuntime,
};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly on GPU-capable hosts"]
fn facade_gpu_grid_bounds_reject_before_scratch_and_keep_unbounded_radius() {
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let before = runtime.scratch_buffer_pool_stats();
    for x in [[0.0, 4_294_967_296.0], [f32::MIN, f32::MAX], [0.0, f32::NAN]] {
        assert!(estimate_normals_grid_gpu(&runtime, &x, &[0.0; 2], &[0.0; 2], 1.0).is_err());
        assert!(estimate_plane_covariances_grid_gpu(
            &runtime, &x, &[0.0; 2], &[0.0; 2], 1.0, 0.001
        )
        .is_err());
        assert_eq!(runtime.scratch_buffer_pool_stats(), before);
    }
    let x = [f32::MIN, f32::MAX];
    let normals =
        estimate_normals_grid_gpu(&runtime, &x, &[0.0; 2], &[0.0; 2], f32::INFINITY).unwrap();
    assert_eq!(normals.len(), 2);
    assert!(normals.iter().all(|n| n.normal == [0.0, 0.0, 1.0] && n.curvature == 0.0));
    let covariances = estimate_plane_covariances_grid_gpu(
        &runtime,
        &x,
        &[0.0; 2],
        &[0.0; 2],
        f32::INFINITY,
        0.001,
    )
    .unwrap();
    assert_eq!(covariances, vec![[0.001, 0.001, 0.001, 0.0, 0.0, 0.0]; 2]);
}
