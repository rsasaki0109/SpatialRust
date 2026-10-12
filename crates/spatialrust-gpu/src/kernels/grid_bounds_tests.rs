use super::*;
use crate::kernels::{estimate_normals_grid_gpu, estimate_plane_covariances_grid_gpu};

#[test]
fn invalid_grid_bounds_do_not_allocate_scratch_or_initialize_pipelines() {
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let mut cases = vec![
        ([vec![0.0, 4_294_967_296.0], vec![0.0; 2], vec![0.0; 2]], 1.0),
        ([vec![f32::MIN, f32::MAX], vec![0.0; 2], vec![0.0; 2]], 1.0),
        ([vec![0.0, 1.0e20], vec![0.0; 2], vec![0.0; 2]], 1.0),
        ([vec![0.0, 1.0], vec![0.0; 2], vec![0.0; 2]], f32::from_bits(1)),
        ([vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]], f32::from_bits(0x00200000)),
    ];
    for axis in 0..3 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for radius in [1.0, f32::INFINITY] {
                let mut xyz = [vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]];
                xyz[axis][1] = value;
                cases.push((xyz, radius));
            }
        }
    }
    for warm in [false, true] {
        if warm {
            estimate_normals_grid_gpu(&runtime, &[0.0, 0.5], &[0.0; 2], &[0.0; 2], 1.0).unwrap();
            estimate_plane_covariances_grid_gpu(
                &runtime,
                &[0.0, 0.5],
                &[0.0; 2],
                &[0.0; 2],
                1.0,
                0.001,
            )
            .unwrap();
        }
        let before = runtime.scratch_buffer_pool_stats();
        runtime.device().push_error_scope(wgpu::ErrorFilter::Validation);
        for (xyz, radius) in &cases {
            assert!(matches!(
                estimate_normals_grid_gpu(&runtime, &xyz[0], &xyz[1], &xyz[2], *radius),
                Err(SpatialError::InvalidArgument(_))
            ));
            assert!(matches!(
                estimate_plane_covariances_grid_gpu(
                    &runtime, &xyz[0], &xyz[1], &xyz[2], *radius, 0.001
                ),
                Err(SpatialError::InvalidArgument(_))
            ));
            assert_eq!(runtime.scratch_buffer_pool_stats(), before);
            if !warm {
                assert!(runtime.point_estimator_pipelines.get().is_none());
            }
        }
        assert!(pollster::block_on(runtime.device().pop_error_scope()).is_none());
    }
}
