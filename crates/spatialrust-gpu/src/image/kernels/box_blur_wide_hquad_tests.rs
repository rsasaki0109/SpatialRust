use super::*;

#[path = "box_blur_wide_hquad_reference.rs"]
mod parent;

fn populated(cache: &BlurPipeline) -> Vec<usize> {
    cache.variants.iter().enumerate().filter_map(|(i, v)| v.get().map(|_| i)).collect()
}

#[test]
fn box_wide_hquad_owners_tails_threshold_and_maximum_indices_are_bounded() {
    let mut checked = 0usize;
    let mut crossings = 0usize;
    for (w, h) in [
        (1u32, 1u32),
        (1, 7),
        (2, 7),
        (3, 257),
        (4, 3),
        (5, 7),
        (17, 19),
        (255, 257),
        (256, 256),
        (1, 65_537),
        (257, 259),
    ] {
        let pixels = w * h;
        let windows = pixels / 4 + u32::from(pixels % 4 != 0);
        assert_eq!(u64::from(windows), u64::from(pixels).div_ceil(4));
        let mut owners = vec![0u8; pixels as usize];
        for window in 0..windows {
            let first = window.checked_mul(4).unwrap();
            for lane in 0..4u32 {
                let index = first.checked_add(lane).unwrap();
                if pixels - first <= lane {
                    assert!(index >= pixels);
                    break;
                }
                owners[index as usize] += 1;
                if lane > 0 && index % w == 0 {
                    crossings += 1;
                }
            }
        }
        assert!(owners.iter().all(|&n| n == 1));
        checked += pixels as usize;
        for stride in [256u32, 1024, 4096 * 256] {
            let mut covered = 0;
            for gid in 0..stride.min(windows) {
                let mut window = gid;
                loop {
                    covered += 1;
                    if windows - window <= stride {
                        break;
                    }
                    window = window.checked_add(stride).unwrap();
                }
            }
            assert_eq!(covered, windows);
        }
    }
    for pixels in [65_535u32, 65_536, 65_537, u32::MAX - 2, u32::MAX - 1, u32::MAX] {
        let windows = pixels / 4 + u32::from(pixels % 4 != 0);
        for window in [0, 1, windows - 1] {
            let first = window.checked_mul(4).unwrap();
            assert!(first < pixels);
            for lane in 0..4 {
                let index = first.checked_add(lane).unwrap();
                assert_eq!(index < pixels, pixels - first > lane);
            }
        }
    }
    assert!(crossings > 0);
    println!("BOX_WIDE_QUAD_PLAN verified {checked} pixel owners, {crossings} row-crossing fallbacks, forced strides, one/two/three-pixel tails, 65535/65536/65537 and maximum-u32 indices.");
}

#[test]
fn box_wide_hquad_lazy_cache_boundaries_profiles_and_parent_bytes_match() {
    for (w, h, kw, kh, expected_variants) in [
        (17, 19, 259, 31, &[1, 2][..]),
        (1, 521, 4099, 257, &[1, 2]),
        (521, 1, 4099, 257, &[1, 2]),
        (2, 521, 259, 81, &[1, 2]),
        (255, 257, 259, 31, &[1, 2]),
        (256, 256, 259, 31, &[1, 2]),
        (257, 257, 511, 31, &[1, 2]),
        (258, 257, 511, 31, &[1, 2]),
        (255, 257, 4099, 31, &[1, 2]),
        (256, 256, 4097, 31, &[1, 2]),
        (256, 256, 4099, 31, &[1, 2]),
        (257, 257, 4099, 31, &[1, 2]),
        (258, 257, 4101, 31, &[1, 2]),
        (257, 259, 259, 1, &[1, 2]),
    ] {
        let r =
            WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
                .unwrap();
        let previous = parent::ParentBlurPipeline::new();
        let image = fixture(w, h, false);
        let source = GpuImage::upload_u8(&r, image.view()).unwrap();
        assert!(r.image_blur_pipeline.get().is_none());
        for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
            let expected = prefix_oracle(&image, kw, kh, border);
            let mut before =
                parent::previous(&previous, &r, &source, kw, kh, border, None).unwrap();
            let mut profiler = GpuStageProfiler::new(&r, 2).unwrap();
            let mut after =
                box_blur_gpu_with_profiler(&r, &source, kw, kh, border, Some(&mut profiler))
                    .unwrap();
            assert_eq!(before.receipt(), after.receipt());
            assert_eq!(before.metadata(), after.metadata());
            assert_eq!(populated(r.image_blur_pipeline.get().unwrap()), expected_variants);
            let report = profiler.collect(&r).unwrap();
            assert_eq!(report.device_to_host_bytes, 32);
            assert_eq!(
                report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
                ["box_blur_horizontal_gpu", "box_blur_gpu"]
            );
            assert_eq!(before.readback_u8::<1>(&r).unwrap().as_slice(), expected);
            assert_eq!(after.readback_u8::<1>(&r).unwrap().as_slice(), expected);
            before.recycle(&r);
            after.recycle(&r);
        }
        source.recycle(&r);
    }
    let r = WgpuRuntime::new_headless().unwrap();
    let current = BlurPipeline::new();
    let previous = old::ParentBlurPipeline::new();
    let mut words = 0;
    for hot in [false, true] {
        for kw in [259, 511, 4097, 4099, 4101, max_height(1)] {
            for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                words += raw(&r, &current, &previous, &fixture(258, 257, hot), kw, border);
            }
        }
    }
    println!("BOX_WIDE_QUAD_CACHE verified shared scalar/quad lazy cache, exact cutoff, row crossings and one/two/three-pixel tails, 28 immediate-parent/CPU outputs, {words} additional raw/u64/guard words, two stages and unchanged receipts/metadata.");
}

#[test]
fn box_wide_hquad_mixed_cache_reuses_programs_resources_and_rejects_before_initialization() {
    let r = WgpuRuntime::new_headless().unwrap();
    let source = GpuImage::upload_u8(&r, fixture(17, 19, false).view()).unwrap();
    let scratch = r.scratch_buffer_pool_stats();
    let textures = r.image_texture_pool_stats();
    for (kw, kh) in [(0, 9), (2, 9), (9, 2), (4101, 4101)] {
        assert!(box_blur_gpu(&r, &source, kw, kh, GpuImageBorder::Replicate).is_err());
    }
    assert!(r.image_blur_pipeline.get().is_none());
    assert_eq!(scratch, r.scratch_buffer_pool_stats());
    assert_eq!(textures, r.image_texture_pool_stats());
    source.recycle(&r);
    let mut identities: Vec<(usize, usize)> = Vec::new();
    for (w, h, kw, kh, expected_variants) in [
        (256, 256, 259, 31, vec![1, 2]),
        (17, 19, 259, 31, vec![1, 2]),
        (17, 19, 3, 3, vec![0, 1, 2]),
        (257, 259, 257, 31, vec![0, 1, 2, 3, 4]),
    ] {
        let image = fixture(w, h, false);
        let input = GpuImage::upload_u8(&r, image.view()).unwrap();
        let expected = prefix_oracle(&image, kw, kh, GpuImageBorder::Replicate);
        for _ in 0..2 {
            let mut output = box_blur_gpu(&r, &input, kw, kh, GpuImageBorder::Replicate).unwrap();
            assert_eq!(output.readback_u8::<1>(&r).unwrap().as_slice(), expected);
            output.recycle(&r);
        }
        let allocations = r.scratch_buffer_pool_stats().allocation_count;
        let textures = r.image_texture_pool_stats().allocation_count;
        let mut output = box_blur_gpu(&r, &input, kw, kh, GpuImageBorder::Replicate).unwrap();
        assert_eq!(allocations, r.scratch_buffer_pool_stats().allocation_count);
        assert_eq!(textures, r.image_texture_pool_stats().allocation_count);
        assert_eq!(output.readback_u8::<1>(&r).unwrap().as_slice(), expected);
        output.recycle(&r);
        let cache = r.image_blur_pipeline.get().unwrap();
        assert_eq!(populated(cache), expected_variants);
        for &(i, address) in &identities {
            assert_eq!(cache.variants[i].get().unwrap() as *const Variant as usize, address);
        }
        identities = populated(cache)
            .into_iter()
            .map(|i| (i, cache.variants[i].get().unwrap() as *const Variant as usize))
            .collect();
        for budget in [4, 0] {
            r.set_scratch_buffer_pool_budget(budget);
            let mut output = box_blur_gpu(&r, &input, kw, kh, GpuImageBorder::Replicate).unwrap();
            assert_eq!(output.readback_u8::<1>(&r).unwrap().as_slice(), expected);
            output.recycle(&r);
            assert!(r.scratch_buffer_pool_stats().cached_bytes <= budget);
        }
        r.set_scratch_buffer_pool_budget(crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES);
        input.recycle(&r);
    }
    println!("BOX_WIDE_QUAD_RESOURCES verified unchanged mixed cache growth 2/2/3/5, stable pipeline identities, warm scratch/texture reuse, tiny/zero budgets and cold rejection.");
}

#[test]
fn box_wide_hquad_concurrent_small_large_and_control_first_use_retains_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let gate = std::sync::Barrier::new(4);
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> =
            [(17, 19, 259, 31), (257, 259, 4099, 31), (257, 259, 257, 3), (16, 17, 3, 3)]
                .into_iter()
                .map(|(w, h, kw, kh)| {
                    let r = &r;
                    let gate = &gate;
                    scope.spawn(move || {
                        let image = fixture(w, h, false);
                        let source = GpuImage::upload_u8(r, image.view()).unwrap();
                        gate.wait();
                        let output =
                            box_blur_gpu(r, &source, kw, kh, GpuImageBorder::ConstantZero).unwrap();
                        source.recycle(r);
                        (output, prefix_oracle(&image, kw, kh, GpuImageBorder::ConstantZero))
                    })
                })
                .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
    });
    assert_eq!(populated(r.image_blur_pipeline.get().unwrap()), [0, 1, 2, 3, 4]);
    r.clear_scratch_buffer_pool();
    r.clear_image_texture_pool();
    for (w, h) in [(17, 19), (257, 259)] {
        let input = GpuImage::upload_u8(&r, fixture(w, h, true).view()).unwrap();
        let mut changed = box_blur_gpu(&r, &input, 65_537, 3, GpuImageBorder::Replicate).unwrap();
        assert!(changed.readback_u8::<1>(&r).unwrap().as_slice().iter().all(|&v| v == 255));
        input.recycle(&r);
        changed.recycle(&r);
    }
    for (mut output, expected) in outputs {
        assert_eq!(output.readback_u8::<1>(&r).unwrap().as_slice(), expected);
    }
    println!("BOX_WIDE_QUAD_CONCURRENT verified simultaneous scalar/paired/packed/serial initialization, independent retained outputs and changed-input pool reuse.");
}

#[path = "box_blur_wide_hquad_benchmark.rs"]
mod benchmark;
