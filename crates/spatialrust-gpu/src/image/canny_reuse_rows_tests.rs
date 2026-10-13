//! Observe exact row values before reuse, then exact classification/root flags.
use super::*;

#[test]
fn reused_rows_and_final_state_words_match_frozen_four_buffer_execution() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_reuse_rows_pipelines(&runtime);
    let mut words = 0;
    for (w, h) in [(1, 1), (1, 257), (257, 1), (17, 19), (257, 65), (513, 129)] {
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
                    for one_group in [false, true] {
                        let observed = snapshot(&runtime, &image, o, None, one_group);
                        let old = snapshot_with_cache(
                            &runtime,
                            &image,
                            o,
                            None,
                            one_group,
                            Some((&previous, FrozenLayout::SharedFlags)),
                        );
                        assert_eq!(observed.horizontal, old.horizontal);
                        assert_eq!(observed.horizontal, reference_horizontal(&image, o));
                        assert_eq!(observed.gradients, old.gradients);
                        assert_eq!(observed.state_words, old.state_words);
                        assert_eq!(observed.states, old.states);
                        assert_eq!(observed.roots, old.roots);
                        assert_eq!(observed.flags, old.flags);
                        assert_eq!(observed.edges, old.edges);
                        assert_eq!(observed.edges, canny(image.view(), cpu_options(o)).unwrap());
                        words += w * h;
                    }
                }
            }
        }
    }
    eprintln!("GPU_CANNY_REUSE_ROWS exact_state_words={words}");
}

#[test]
fn three_actual_scratch_leases_match_exact_working_bytes_without_dummy_rows() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for (w, h) in [(1, 1), (17, 19), (3840, 2160)] {
        for aperture_size in [3, 5, 7] {
            runtime.clear_scratch_buffer_pool();
            let p = plan(
                w,
                h,
                1,
                GpuCannyOptions { aperture_size, ..Default::default() },
                &runtime.device().limits(),
            )
            .unwrap();
            let before = runtime.scratch_buffer_pool_stats().allocation_count;
            let mut scratch = ScratchBuffers::new(&runtime);
            let work = Work::acquire(&mut scratch, &p);
            assert_eq!(
                work.gradients.size() + work.states.size() + work.parent.size(),
                16 * u64::from(w) * u64::from(h)
            );
            assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count - before, 3);
            assert_ne!(work.gradients, work.states);
            assert_ne!(work.states, work.parent);
            scratch
                .submit_and_recycle(runtime.device().create_command_encoder(&Default::default()));
            runtime.wait_idle();
        }
    }
}

#[path = "canny_reuse_rows_benchmark.rs"]
mod benchmark;
