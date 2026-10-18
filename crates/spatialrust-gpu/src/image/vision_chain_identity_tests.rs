use super::*;
use crate::{GpuStageProfiler, WgpuPowerPreference};
use spatialrust_image::{ColorSpace, Image, ImageMetadata, ImageRegion, ImageView};

#[path = "vision_chain_identity_reference.rs"]
mod parent;

fn fixture(w: usize, h: usize, hot: bool) -> Image<u8, 3> {
    Image::try_new_with_metadata(
        w,
        h,
        (0..w * h * 3)
            .map(|i| if hot { 137 } else { (i * 37 + i / 3 * 11 + i % 3 * 53) as u8 })
            .collect(),
        ImageMetadata { color_space: ColorSpace::Rgb, ..Default::default() },
    )
    .unwrap()
}

fn cpu(rgb: &[u8], sw: usize, sh: usize, o: GpuVisionChainOptions) -> Vec<f32> {
    let (w, h) = (o.width as usize, o.height as usize);
    let mut gray = vec![0u64; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * sh / h) * sw + x * sw / w) * 3;
            gray[y * w + x] = (77 * u64::from(rgb[i])
                + 150 * u64::from(rgb[i + 1])
                + 29 * u64::from(rgb[i + 2])
                + 128)
                >> 8;
        }
    }
    // Independent u64 row/column prefixes; only the final box result is rounded.
    let radius = o.blur_kernel as usize / 2;
    let mut horizontal = vec![0u64; w * h];
    let mut prefix = vec![0u64; w.max(h) + 1];
    for y in 0..h {
        prefix[0] = 0;
        for x in 0..w {
            prefix[x + 1] = prefix[x] + gray[y * w + x];
        }
        for x in 0..w {
            horizontal[y * w + x] = prefix[(x + radius + 1).min(w)]
                - prefix[x.saturating_sub(radius)]
                + radius.saturating_sub(x) as u64 * gray[y * w]
                + radius.saturating_sub(w - 1 - x) as u64 * gray[(y + 1) * w - 1];
        }
    }
    let mut blurred = vec![0u8; w * h];
    let area = u64::from(o.blur_kernel).pow(2);
    for x in 0..w {
        prefix[0] = 0;
        for y in 0..h {
            prefix[y + 1] = prefix[y] + horizontal[y * w + x];
        }
        for y in 0..h {
            let sum = prefix[(y + radius + 1).min(h)] - prefix[y.saturating_sub(radius)]
                + radius.saturating_sub(y) as u64 * horizontal[x]
                + radius.saturating_sub(h - 1 - y) as u64 * horizontal[(h - 1) * w + x];
            blurred[y * w + x] = ((sum + area / 2) / area) as u8;
        }
    }
    let mut edges = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let sample = |dx: isize, dy: isize| {
                i32::from(
                    blurred[(y as isize + dy).clamp(0, h as isize - 1) as usize * w
                        + (x as isize + dx).clamp(0, w as isize - 1) as usize],
                )
            };
            let gx = -sample(-1, -1) + sample(1, -1) - 2 * sample(-1, 0) + 2 * sample(1, 0)
                - sample(-1, 1)
                + sample(1, 1);
            let gy = -sample(-1, -1) - 2 * sample(0, -1) - sample(1, -1)
                + sample(-1, 1)
                + 2 * sample(0, 1)
                + sample(1, 1);
            edges[y * w + x] = (gx.abs() + gy.abs()).min(255) as u8;
        }
    }
    let radius = o.morphology_kernel as usize / 2;
    let initial = if o.morphology == GpuMorphology::Dilate { 0 } else { 255 };
    let merge =
        |a: u8, b: u8| if o.morphology == GpuMorphology::Dilate { a.max(b) } else { a.min(b) };
    let mut horizontal = vec![initial; w * h];
    let mut result = vec![initial; w * h];
    // Replicated edge duplicates cannot change an extremum.
    for y in 0..h {
        for x in 0..w {
            for sx in x.saturating_sub(radius)..=(x + radius).min(w - 1) {
                horizontal[y * w + x] = merge(horizontal[y * w + x], edges[y * w + sx]);
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            for sy in y.saturating_sub(radius)..=(y + radius).min(h - 1) {
                result[y * w + x] = merge(result[y * w + x], horizontal[sy * w + x]);
            }
        }
    }
    result.into_iter().map(|v| (f32::from(v) * o.scale - o.mean[0]) / o.std[0]).collect()
}

fn words(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "word {i}");
    }
}

fn passes(o: GpuVisionChainOptions, identity: bool) -> u32 {
    3 + u32::from(!identity)
        + crate::image::kernels::box_blur::blur_pass_count(
            o.width,
            o.height,
            o.blur_kernel,
            o.blur_kernel,
        )
        + crate::image::kernels::spatial::morphology_pass_count(
            o.width,
            o.height,
            o.morphology_kernel,
            o.morphology_kernel,
        )
}

fn footprint(o: GpuVisionChainOptions, identity: bool) -> u64 {
    let n = u64::from(o.width) * u64::from(o.height);
    let bytes = 4 * n;
    let blur =
        if u64::from(o.blur_kernel).pow(2) >= 81 && n * u64::from(o.blur_kernel).pow(2) >= 65_536 {
            if o.blur_kernel <= 257 {
                n.div_ceil(2) * 4
            } else {
                bytes
            }
        } else {
            0
        };
    let morphology = if u64::from(o.morphology_kernel).pow(2) >= 81
        && n * u64::from(o.morphology_kernel).pow(2) >= 1_048_576
    {
        u64::from(o.width.div_ceil(4)) * u64::from(o.height) * 4
    } else {
        0
    };
    bytes * if identity { 5 } else { 6 } + blur + morphology
}

fn receipts(source: &GpuImage, a: &GpuAiTensor, b: &GpuAiTensor, o: GpuVisionChainOptions) {
    let identity = source.width() == o.width && source.height() == o.height;
    let mut expected = a.receipt().stages().to_vec();
    if identity {
        assert_eq!(expected.remove(source.receipt().stages().len()), "resize_nearest_gpu");
    }
    assert_eq!(b.receipt().stages(), expected);
    assert_eq!(a.receipt().host_to_device_bytes(), source.receipt().host_to_device_bytes());
    assert_eq!(a.receipt().host_to_device_bytes(), b.receipt().host_to_device_bytes());
    assert_eq!(a.receipt().device_to_host_bytes(), b.receipt().device_to_host_bytes());
    assert_eq!(
        b.receipt().gpu_to_gpu_bytes(),
        source.receipt().gpu_to_gpu_bytes().saturating_add(footprint(o, identity))
    );
    assert_eq!(a.shape(), b.shape());
}

#[test]
fn identity_chain_parent_cpu_bits_and_source_history_match_for_identity_and_resize() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut count = 0;
    let mut checked = 0;
    for (w, h) in [(1, 1), (1, 17), (17, 1), (17, 19), (33, 31), (257, 259)] {
        for hot in [false, true] {
            let image = fixture(w, h, hot);
            let mut source = GpuImage::upload_u8(&r, image.view()).unwrap();
            let original = source.receipt().clone();
            for (bk, mk) in [(1, 1), (3, 3), (9, 3), (3, 9), (9, 9), (259, 31)] {
                for operation in [GpuMorphology::Erode, GpuMorphology::Dilate] {
                    for (ow, oh) in [(w, h), (w + 1, h), (w, h + 1), (h, w)] {
                        let o = GpuVisionChainOptions {
                            width: ow as u32,
                            height: oh as u32,
                            blur_kernel: bk,
                            morphology_kernel: mk,
                            morphology: operation,
                            scale: 0.25,
                            mean: [8.; 4],
                            std: [2.; 4],
                        };
                        let expected = cpu(image.as_slice(), w, h, o);
                        let mut a = parent::previous(&r, &source, o).unwrap();
                        let mut b = run_gpu_vision_chain(&r, &source, o).unwrap();
                        receipts(&source, &a, &b, o);
                        words(&a.readback_f32(&r).unwrap(), &expected);
                        words(&b.readback_f32(&r).unwrap(), &expected);
                        receipts(&source, &a, &b, o);
                        assert_eq!(source.receipt(), &original);
                        assert_eq!(source.metadata(), image.metadata());
                        a.recycle(&r);
                        b.recycle(&r);
                        count += 1;
                        checked += expected.len() * 2;
                    }
                }
            }
            assert_eq!(source.readback_u8::<3>(&r).unwrap().as_slice(), image.as_slice());
            source.recycle(&r);
        }
    }
    println!("IDENTITY_CHAIN_OUTPUT verified {checked} exact parent/CPU/current tensor words across {count} identity/width-change/height-change cases, both morphology operations, serial/separable/wide blur and original source bytes/history/metadata.");
}

#[test]
fn identity_chain_profile_capacity_query_bytes_accumulation_and_failures_are_exact() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    for (w, h, bk, mk, identity_passes) in
        [(17, 19, 3, 3, 5), (33, 31, 9, 3, 6), (257, 259, 3, 9, 6), (257, 259, 9, 9, 7)]
    {
        for identity in [true, false] {
            let image = fixture(w + usize::from(!identity), h, false);
            let source = GpuImage::upload_u8(&r, image.view()).unwrap();
            let o = GpuVisionChainOptions {
                width: w as u32,
                height: h as u32,
                blur_kernel: bk,
                morphology_kernel: mk,
                ..Default::default()
            };
            let required = identity_passes + u32::from(!identity);
            assert_eq!(passes(o, identity), required);
            let mut short = GpuStageProfiler::new(&r, required - 1).unwrap();
            let pools = r.image_texture_pool_stats();
            let buffers = r.buffer_pool().stats();
            assert!(run_gpu_vision_chain_profiled(&r, &source, o, &mut short).is_err());
            assert_eq!(pools, r.image_texture_pool_stats());
            assert_eq!(buffers, r.buffer_pool().stats());
            assert!(short.collect(&r).unwrap().stages.is_empty());
            let mut before = GpuStageProfiler::new(&r, required + u32::from(identity)).unwrap();
            let a = parent::previous_profiled(&r, &source, o, &mut before).unwrap();
            let old = before.collect(&r).unwrap();
            let mut exact = GpuStageProfiler::new(&r, required).unwrap();
            let b = run_gpu_vision_chain_profiled(&r, &source, o, &mut exact).unwrap();
            let report = exact.collect(&r).unwrap();
            assert_eq!(report.stages.len(), required as usize);
            assert_eq!(report.device_to_host_bytes, u64::from(required) * 16);
            let mut expected = old.stages.iter().map(|s| s.stage).collect::<Vec<_>>();
            if identity {
                assert_eq!(expected.remove(0), "resize_nearest_gpu");
            }
            assert_eq!(report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(), expected);
            receipts(&source, &a, &b, o);
            a.recycle(&r);
            b.recycle(&r);
            let mut accumulated = GpuStageProfiler::new(&r, required + 1).unwrap();
            crate::image::kernels::gray::rgb_to_gray_gpu_with_profiler(
                &r,
                &source,
                Some(&mut accumulated),
            )
            .unwrap()
            .recycle(&r);
            run_gpu_vision_chain_profiled(&r, &source, o, &mut accumulated).unwrap().recycle(&r);
            assert_eq!(accumulated.collect(&r).unwrap().stages.len(), required as usize + 1);
            source.recycle(&r);
        }
    }
    let source = GpuImage::upload_u8(&r, fixture(17, 19, false).view()).unwrap();
    let mut failed = GpuStageProfiler::new(&r, 8).unwrap();
    let invalid =
        GpuVisionChainOptions { width: 17, height: 19, blur_kernel: 2, ..Default::default() };
    assert!(run_gpu_vision_chain_profiled(&r, &source, invalid, &mut failed).is_err());
    assert!(run_gpu_vision_chain_profiled(
        &r,
        &source,
        GpuVisionChainOptions { width: 17, height: 19, ..Default::default() },
        &mut failed
    )
    .is_err());
    assert!(failed.collect(&r).is_err());
    println!("IDENTITY_CHAIN_PROFILE verified exact 5/6/7 identity and 6/7/8 resize capacities, one fewer stage and 16 query bytes, accumulated recordings, preflight rejection and failed-recording safety.");
}

#[test]
fn identity_chain_admission_roi_dirty_channels_saturation_and_pool_requests_are_preserved() {
    let r = WgpuRuntime::new_headless().unwrap();
    let tiny = GpuImage::upload_u8(&r, fixture(1, 1, false).view()).unwrap();
    let too_wide = r.device().limits().max_texture_dimension_2d + 1;
    let malformed = GpuImage::from_parts(
        &r,
        too_wide,
        1,
        3,
        tiny.texture().clone(),
        tiny.metadata(),
        tiny.receipt().clone(),
    )
    .unwrap();
    let o = GpuVisionChainOptions { width: too_wide, height: 1, ..Default::default() };
    let stats = r.image_texture_pool_stats();
    let a = parent::previous(&r, &malformed, o).err().unwrap().to_string();
    let b = run_gpu_vision_chain(&r, &malformed, o).err().unwrap().to_string();
    assert_eq!(a, b);
    assert_eq!(stats, r.image_texture_pool_stats());
    assert_eq!(r.initialized_image_pipeline_count(), 0);
    drop(malformed);
    let other = WgpuRuntime::new_headless().unwrap();
    let o = GpuVisionChainOptions { width: 1, height: 1, ..Default::default() };
    assert_eq!(
        parent::previous(&other, &tiny, o).err().unwrap().to_string(),
        run_gpu_vision_chain(&other, &tiny, o).err().unwrap().to_string()
    );
    assert_eq!(other.initialized_image_pipeline_count(), 0);
    let source = GpuImage::upload_u8(&r, fixture(17, 19, false).view()).unwrap();
    for invalid in [
        GpuVisionChainOptions { width: 0, ..o },
        GpuVisionChainOptions { height: 0, ..o },
        GpuVisionChainOptions { width: 17, height: 19, blur_kernel: 2, ..o },
        GpuVisionChainOptions { width: 17, height: 19, morphology_kernel: 33, ..o },
        GpuVisionChainOptions { width: 17, height: 19, scale: f32::NAN, ..o },
        GpuVisionChainOptions { width: 17, height: 19, std: [0.; 4], ..o },
    ] {
        assert_eq!(
            parent::previous(&r, &source, invalid).err().unwrap().to_string(),
            run_gpu_vision_chain(&r, &source, invalid).err().unwrap().to_string()
        );
    }
    source.recycle(&r);
    tiny.recycle(&r);
    let stride = 29 * 3 + 7;
    let data: Vec<u8> = (0..stride * 27).map(|i| (i * 37 + 11) as u8).collect();
    let view = ImageView::<u8, 3>::new(29, 27, stride, &data)
        .unwrap()
        .subview(ImageRegion::new(3, 2, 17, 19))
        .unwrap();
    let rgb: Vec<u8> = (0..19)
        .flat_map(|y| data[(y + 2) * stride + 9..(y + 2) * stride + 9 + 17 * 3].iter().copied())
        .collect();
    let uploaded = GpuImage::upload_u8(&r, view).unwrap();
    let mut source = crate::image::resize_nearest_gpu(&r, &uploaded, 17, 19).unwrap();
    uploaded.recycle(&r);
    source.readback_u8::<3>(&r).unwrap();
    let o = GpuVisionChainOptions { width: 17, height: 19, scale: 1., ..Default::default() };
    let expected = cpu(&rgb, 17, 19, o);
    for budget in [crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES, 4, 0] {
        r.set_image_texture_pool_budget(budget);
        r.set_scratch_buffer_pool_budget(budget);
        r.buffer_pool().set_budget(budget);
        let mut requests = Vec::new();
        for current in [false, true] {
            r.clear_image_texture_pool();
            let before = r.image_texture_pool_stats();
            let mut output = if current {
                run_gpu_vision_chain(&r, &source, o)
            } else {
                parent::previous(&r, &source, o)
            }
            .unwrap();
            let after = r.image_texture_pool_stats();
            requests.push(
                after.allocation_count - before.allocation_count + after.reuse_count
                    - before.reuse_count,
            );
            assert_eq!(
                output.receipt().device_to_host_bytes(),
                source.receipt().device_to_host_bytes()
            );
            words(&output.readback_f32(&r).unwrap(), &expected);
            output.recycle(&r);
            assert!(r.image_texture_pool_stats().cached_bytes <= budget);
        }
        assert_eq!(requests, [5, 4]);
    }
    r.set_image_texture_pool_budget(crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES);
    r.buffer_pool().set_budget(crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES);
    let rgba = Image::<u8, 4>::try_new(
        17,
        19,
        rgb.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 239]).collect(),
    )
    .unwrap();
    let physical = GpuImage::upload_u8(&r, rgba.view()).unwrap();
    let mut history = physical.receipt().clone();
    history.record_gpu_to_gpu(u64::MAX - 1, "resize_nearest_gpu");
    let dirty = GpuImage::from_parts(
        &r,
        17,
        19,
        3,
        physical.texture().clone(),
        ImageMetadata { color_space: ColorSpace::Rgb, ..Default::default() },
        history,
    )
    .unwrap();
    drop(physical);
    let mut a = parent::previous(&r, &dirty, o).unwrap();
    let mut b = run_gpu_vision_chain(&r, &dirty, o).unwrap();
    receipts(&dirty, &a, &b, o);
    assert_eq!(b.receipt().gpu_to_gpu_bytes(), u64::MAX);
    words(&a.readback_f32(&r).unwrap(), &expected);
    words(&b.readback_f32(&r).unwrap(), &expected);
    a.recycle(&r);
    b.recycle(&r);
    dirty.recycle(&r);
    source.recycle(&r);
    println!("IDENTITY_CHAIN_RESOURCES verified original admission/errors, foreign devices and metadata-only invalid images, ROI/history/readback preservation, dirty alpha and saturated receipts; five versus four texture requests at normal/tiny/zero budgets.");
}

#[test]
fn identity_chain_concurrent_owned_tensors_survive_source_recycle_and_changed_reuse() {
    let r = WgpuRuntime::new_headless().unwrap();
    let image = fixture(257, 259, false);
    let source = GpuImage::upload_u8(&r, image.view()).unwrap();
    let gate = std::sync::Barrier::new(4);
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = [(257, 259, 3, 3), (258, 259, 9, 3), (257, 259, 3, 9), (257, 259, 9, 9)]
            .into_iter()
            .map(|(w, h, bk, mk)| {
                let r = &r;
                let source = &source;
                let gate = &gate;
                let image = &image;
                scope.spawn(move || {
                    let o = GpuVisionChainOptions {
                        width: w,
                        height: h,
                        blur_kernel: bk,
                        morphology_kernel: mk,
                        scale: 1.,
                        ..Default::default()
                    };
                    let expected = cpu(image.as_slice(), 257, 259, o);
                    gate.wait();
                    (run_gpu_vision_chain(r, source, o).unwrap(), expected)
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect::<Vec<_>>()
    });
    source.recycle(&r);
    r.clear_image_texture_pool();
    r.clear_scratch_buffer_pool();
    let changed = GpuImage::upload_u8(&r, fixture(257, 259, true).view()).unwrap();
    for _ in 0..3 {
        run_gpu_vision_chain(
            &r,
            &changed,
            GpuVisionChainOptions { width: 257, height: 259, ..Default::default() },
        )
        .unwrap()
        .recycle(&r);
    }
    for (mut output, expected) in outputs {
        words(&output.readback_f32(&r).unwrap(), &expected);
        output.recycle(&r);
    }
    changed.recycle(&r);
    println!("IDENTITY_CHAIN_CONCURRENT verified mixed identity/resize and separable first use, shared immutable source and independently owned tensors after source recycling, pool clears and changed-data reuse.");
}

#[path = "vision_chain_identity_benchmark.rs"]
mod benchmark;
