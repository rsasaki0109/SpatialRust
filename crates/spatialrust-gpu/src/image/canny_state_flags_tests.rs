//! Exact shared classification/root flags versus the frozen five-buffer route.
use super::*;

#[test]
fn shared_state_words_match_frozen_classification_roots_and_flags() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_state_flags_pipelines(&runtime);
    let mut words = 0;
    for (w, h) in [(1, 1), (1, 513), (513, 1), (17, 19), (257, 65)] {
        for profile in 0..4 {
            let image = fixture(w, h, profile);
            for aperture_size in [3, 5, 7] {
                for l2_gradient in [false, true] {
                    let o = GpuCannyOptions {
                        aperture_size,
                        l2_gradient,
                        low_threshold: 40.25,
                        high_threshold: 120.75,
                    };
                    let observed = snapshot(&runtime, &image, o, None, true);
                    let old = snapshot_with_cache(
                        &runtime,
                        &image,
                        o,
                        None,
                        true,
                        Some((&previous, FrozenLayout::PackedHorizontal)),
                    );
                    assert_eq!(observed.gradients, old.gradients);
                    assert_eq!(observed.horizontal, old.horizontal);
                    assert_eq!(observed.states, old.states);
                    assert_eq!(observed.roots, old.roots);
                    assert_eq!(observed.flags, old.flags);
                    assert_eq!(observed.edges, old.edges);
                    assert_eq!(observed.edges, canny(image.view(), cpu_options(o)).unwrap());
                    for i in 0..w * h {
                        assert_eq!(observed.state_words[i], old.states[i] | (old.flags[i] << 2));
                        if old.flags[i] != 0 {
                            assert_eq!(old.states[i], 1);
                            assert_eq!(old.roots[i], i as u32);
                        }
                        words += 1;
                    }
                }
            }
        }
    }
    eprintln!("GPU_CANNY_STATE_FLAGS exact_words={words}");
}

#[test]
fn concurrent_root_markers_preserve_low_bits_and_all_seeded_components() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_state_flags_pipelines(&runtime);
    for (w, h) in [(1025, 1), (1, 1025), (257, 129)] {
        for profile in 0..4 {
            let states: Vec<u32> = (0..w * h)
                .map(|i| match profile {
                    0 => {
                        if i == w * h - 1 {
                            2
                        } else {
                            1
                        }
                    }
                    1 => {
                        if i % 13 == 0 {
                            2
                        } else {
                            1
                        }
                    }
                    2 => {
                        if (i / w) % 5 == 0 {
                            0
                        } else if i % 17 == 0 {
                            2
                        } else {
                            1
                        }
                    }
                    _ => {
                        if i % 3 == 0 {
                            1
                        } else {
                            0
                        }
                    }
                })
                .collect();
            let image = fixture(w, h, 0);
            let observed = snapshot(&runtime, &image, Default::default(), Some(&states), false);
            let old = snapshot_with_cache(
                &runtime,
                &image,
                Default::default(),
                Some(&states),
                false,
                Some((&previous, FrozenLayout::PackedHorizontal)),
            );
            let (roots, flags, edges) = reference_components(w, h, &states);
            assert_eq!(observed.states, states);
            assert_eq!(observed.roots, roots);
            assert_eq!(observed.flags, flags);
            assert_eq!(observed.edges.as_slice(), edges);
            assert_eq!(observed.states, old.states);
            assert_eq!(observed.roots, old.roots);
            assert_eq!(observed.flags, old.flags);
            assert_eq!(observed.edges, old.edges);
            for i in 0..w * h {
                assert_eq!(observed.state_words[i], states[i] | (flags[i] << 2));
            }
        }
    }
}

#[test]
fn three_storage_buffers_and_reused_working_footprint_are_exact() {
    let mut limits = wgpu::Limits { max_storage_buffers_per_shader_stage: 3, ..Default::default() };
    for aperture_size in [3, 5, 7] {
        let p =
            plan(3840, 2160, 1, GpuCannyOptions { aperture_size, ..Default::default() }, &limits)
                .unwrap();
        let n = 3840 * 2160;
        assert_eq!(p.gradient_bytes + 2 * p.pixel_bytes, 16 * n);
        assert_eq!(p.pixel_bytes, 33_177_600);
        assert_eq!(pass_indices(aperture_size).len(), if aperture_size == 3 { 5 } else { 6 });
    }
    limits.max_storage_buffers_per_shader_stage = 2;
    assert!(plan(1, 1, 1, Default::default(), &limits).is_err());
    assert!(!SHADER.contains("binding(6)"));
    assert!(SHADER.contains("states:array<atomic<u32>>"));
}

#[path = "canny_state_flags_benchmark.rs"]
mod benchmark;
