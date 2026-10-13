//! Lossless unrounded horizontal packing and buffer-footprint references.
use super::*;

#[test]
fn unrounded_horizontal_extrema_match_both_frozen_layouts_and_integer_reference() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_horizontal_pipelines(&runtime);
    let original = previous_pipelines(&runtime);
    for (size, derivative_max, smooth_max) in [(5, 765, 4080), (7, 2550, 16320)] {
        for profile in [-1, 0, 1] {
            let data = (0..size * 3)
                .map(|i| {
                    let tap = i % size;
                    let coefficient =
                        choose(size - 2, tap as isize - 1) - choose(size - 2, tap as isize);
                    if profile == 0 || coefficient * profile > 0 {
                        255
                    } else {
                        0
                    }
                })
                .collect();
            let image = Image::<u8, 1>::try_new(size, 3, data).unwrap();
            for l2_gradient in [false, true] {
                let o = GpuCannyOptions {
                    aperture_size: size as u32,
                    l2_gradient,
                    ..Default::default()
                };
                let observed = snapshot(&runtime, &image, o, None, true);
                let old =
                    snapshot_with_cache(&runtime, &image, o, None, true, Some((&previous, false)));
                let uncompressed =
                    snapshot_with_cache(&runtime, &image, o, None, true, Some((&original, true)));
                let expected = reference_horizontal(&image, o);
                assert_eq!(observed.horizontal, expected);
                assert_eq!(old.horizontal, expected);
                assert_eq!(uncompressed.horizontal, expected);
                let center = observed.horizontal[size + size / 2];
                assert_eq!(center[0], derivative_max * profile as i32);
                if profile == 0 {
                    assert_eq!(center[1], smooth_max);
                }
                assert_eq!(observed.gradients, old.gradients);
                assert_eq!(observed.gradients, uncompressed.gradients);
                assert_eq!(observed.edges, old.edges);
            }
        }
    }
}

#[test]
fn packed_horizontal_footprints_keep_exact_gradient_binding_limits() {
    let limits = wgpu::Limits::default();
    for aperture_size in [3, 5, 7] {
        let o = GpuCannyOptions { aperture_size, ..Default::default() };
        let p = plan(3840, 2160, 1, o, &limits).unwrap();
        assert_eq!(p.gradient_bytes, 66_355_200);
        assert_eq!(p.horizontal_bytes, if aperture_size == 3 { 8 } else { 33_177_600 });
        assert_eq!(
            p.gradient_bytes + p.horizontal_bytes + 3 * p.pixel_bytes,
            if aperture_size == 3 { 165_888_008 } else { 199_065_600 }
        );
        let mut smaller = limits.clone();
        smaller.max_storage_buffer_binding_size = 66_355_199;
        assert!(plan(3840, 2160, 1, o, &smaller).is_err());
    }
}

#[path = "canny_horizontal_benchmark.rs"]
mod benchmark;
