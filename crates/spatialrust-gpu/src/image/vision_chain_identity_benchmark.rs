use super::*;
use std::{hint::black_box, time::Instant};

fn measure(r: &WgpuRuntime, image: &Image<u8, 3>, o: GpuVisionChainOptions, hot: bool) {
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let identity = source.width() == o.width && source.height() == o.height;
    let expected = cpu(image.as_slice(), image.width(), image.height(), o);
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    r.wait_idle();
    for round in 0..33 {
        for offset in 0..4 {
            let variant = offset ^ (round % 4);
            let current = variant % 2 == 1;
            let full = variant >= 2;
            let start = Instant::now();
            let uploaded =
                full.then(|| GpuImage::upload_u8(black_box(r), black_box(image.view())).unwrap());
            let input = uploaded.as_ref().unwrap_or(&source);
            let mut tensor = if current {
                run_gpu_vision_chain(black_box(r), black_box(input), o)
            } else {
                parent::previous(black_box(r), black_box(input), o)
            }
            .unwrap();
            let values = if full {
                Some(tensor.readback_f32(r).unwrap())
            } else {
                r.wait_idle();
                None
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            let values = values.unwrap_or_else(|| tensor.readback_f32(r).unwrap());
            words(&values, &expected);
            assert_eq!(tensor.receipt().gpu_to_gpu_bytes(), footprint(o, current && identity));
            tensor.recycle(r);
            if let Some(input) = uploaded {
                input.recycle(r);
            }
            if round >= 3 {
                samples[variant].push(elapsed);
            }
        }
    }
    let mut gpu: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for offset in 0..2 {
            let current = (round + offset) % 2 == 1;
            let count = passes(o, current && identity);
            let mut profiler = GpuStageProfiler::new(r, count).unwrap();
            let mut tensor = if current {
                run_gpu_vision_chain_profiled(r, &source, o, &mut profiler)
            } else {
                parent::previous_profiled(r, &source, o, &mut profiler)
            }
            .unwrap();
            let report = profiler.collect(r).unwrap();
            assert_eq!(report.stages.len(), count as usize);
            assert_eq!(report.device_to_host_bytes, u64::from(count) * 16);
            let elapsed = report.stages.iter().map(|s| s.elapsed_ns).sum::<f64>() / 1_000_000.;
            words(&tensor.readback_f32(r).unwrap(), &expected);
            tensor.recycle(r);
            if round >= 3 {
                gpu[usize::from(current)].push(elapsed);
            }
        }
    }
    println!("IDENTITY_CHAIN_RECORD {{\"source_width\":{},\"source_height\":{},\"width\":{},\"height\":{},\"blur\":{},\"morphology\":{},\"hot\":{hot},\"identity\":{identity},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"gpu_old_ms\":{:?},\"gpu_ms\":{:?}}}",image.width(),image.height(),o.width,o.height,o.blur_kernel,o.morphology_kernel,samples[0],samples[1],samples[2],samples[3],gpu[0],gpu[1]);
    source.recycle(r);
}

#[test]
#[ignore = "paired complete resident/full calls and sum of stage timestamps; run alone in release"]
fn benchmark_identity_vision_chain() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    println!("IDENTITY_CHAIN_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 wall_order=balanced gpu_order=alternating gpu_scope=sum_of_stages checks_drop_recycle_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (sw, sh, w, h, bk, mk, hot) in [
        (17, 19, 17, 19, 3, 3, false),
        (257, 259, 257, 259, 3, 3, false),
        (640, 480, 640, 480, 3, 3, false),
        (1920, 1080, 1920, 1080, 3, 3, false),
        (3840, 2160, 3840, 2160, 3, 3, false),
        (640, 480, 640, 480, 9, 9, false),
        (1920, 1080, 1920, 1080, 9, 9, false),
        (257, 259, 257, 259, 4099, 31, false),
        (1920, 1080, 1920, 1080, 3, 3, true),
        (17, 19, 18, 19, 3, 3, false),
        (640, 480, 320, 240, 3, 3, false),
        (640, 480, 480, 640, 3, 3, false),
        (320, 240, 640, 480, 9, 9, false),
        (1920, 1080, 960, 540, 3, 3, false),
        (3840, 2160, 1920, 1080, 3, 3, false),
    ] {
        let o = GpuVisionChainOptions {
            width: w,
            height: h,
            blur_kernel: bk,
            morphology_kernel: mk,
            ..Default::default()
        };
        measure(&r, &fixture(sw, sh, hot), o, hot);
    }
}

#[test]
#[ignore = "fresh application image caches on retained device; run alone in release"]
fn benchmark_identity_vision_chain_cold() {
    let mut r = WgpuRuntime::new_headless().unwrap();
    println!("IDENTITY_CHAIN_COLD_ENV backend={} device_type={} pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true",r.adapter_info().backend,r.adapter_info().device_type);
    for (name, sw, sh, w, h, bk, mk) in [
        ("small_identity", 17, 19, 17, 19, 3, 3),
        ("vga_identity", 640, 480, 640, 480, 3, 3),
        ("separable_identity", 257, 259, 257, 259, 9, 9),
        ("resize_control", 640, 480, 320, 240, 3, 3),
    ] {
        let image = fixture(sw, sh, false);
        let source = GpuImage::upload_u8(&r, image.view()).unwrap();
        let o = GpuVisionChainOptions {
            width: w,
            height: h,
            blur_kernel: bk,
            morphology_kernel: mk,
            ..Default::default()
        };
        let expected = cpu(image.as_slice(), sw, sh, o);
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..20 {
            for offset in 0..2 {
                let current = (round + offset) % 2 == 1;
                r.wait_idle();
                r.image_gray_pipeline.take();
                r.image_blur_pipeline.take();
                r.image_spatial_pipelines.take();
                r.image_ai_pack_pipeline.take();
                r.clear_image_texture_pool();
                r.clear_scratch_buffer_pool();
                r.buffer_pool().set_budget(0);
                r.buffer_pool().set_budget(crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES);
                let start = Instant::now();
                let mut tensor = if current {
                    run_gpu_vision_chain(black_box(&r), black_box(&source), o)
                } else {
                    parent::previous(black_box(&r), black_box(&source), o)
                }
                .unwrap();
                r.wait_idle();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(r.initialized_image_pipeline_count(), 4);
                words(&tensor.readback_f32(&r).unwrap(), &expected);
                tensor.recycle(&r);
                samples[usize::from(current)].push(elapsed);
            }
        }
        println!("IDENTITY_CHAIN_COLD_RECORD {{\"scenario\":\"{name}\",\"component_cache_counts\":[4,4],\"old_ms\":{:?},\"ms\":{:?}}}",samples[0],samples[1]);
        source.recycle(&r);
    }
}
