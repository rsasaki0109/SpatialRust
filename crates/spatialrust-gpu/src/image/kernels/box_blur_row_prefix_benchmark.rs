use super::*;
use std::{hint::black_box, time::Instant};

fn horizontal_passes(
    r: &WgpuRuntime,
    previous: &parent::ParentBlurPipeline,
    input: &GpuImage,
    kw: u32,
    kh: u32,
    border: GpuImageBorder,
    expected: &[u8],
) -> [Vec<f64>; 2] {
    let count = blur_pass_count(input.width(), input.height(), kw, kh);
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for offset in 0..2 {
            let variant = (round + offset) % 2;
            let mut profiler = GpuStageProfiler::new(r, count).unwrap();
            let mut output = if variant == 1 {
                box_blur_gpu_with_profiler(r, input, kw, kh, border, Some(&mut profiler))
            } else {
                parent::previous(previous, r, input, kw, kh, border, Some(&mut profiler))
            }
            .unwrap();
            let report = profiler.collect(r).unwrap();
            assert_eq!(report.device_to_host_bytes, u64::from(count) * 16);
            assert_eq!(report.stages.len(), count as usize);
            assert_eq!(
                report.stages[0].stage,
                if count == 2 { "box_blur_horizontal_gpu" } else { "box_blur_gpu" }
            );
            if round >= 3 {
                samples[variant].push(report.stages[0].elapsed_ns / 1_000_000.);
            }
            assert_eq!(output.readback_u8::<1>(r).unwrap().as_slice(), expected);
            output.recycle(r);
        }
    }
    samples
}

fn measure(
    r: &WgpuRuntime,
    previous: &parent::ParentBlurPipeline,
    w: usize,
    h: usize,
    kw: u32,
    kh: u32,
    zero: bool,
) {
    let image = fixture(w, h, false);
    let border = if zero { GpuImageBorder::ConstantZero } else { GpuImageBorder::Replicate };
    let expected = prefix_oracle(&image, kw, kh, border);
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let p = plan(w as u32, h as u32, kw, kh, border, &r.device().limits()).unwrap();
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    r.wait_idle();
    for round in 0..33 {
        for offset in 0..4 {
            let variant = offset ^ (round % 4);
            let full = variant >= 2;
            let current = variant % 2 == 1;
            let start = Instant::now();
            let uploaded =
                full.then(|| GpuImage::upload_u8(black_box(r), black_box(image.view())).unwrap());
            let input = uploaded.as_ref().unwrap_or(&source);
            let mut output = if current {
                box_blur_gpu(black_box(r), black_box(input), kw, kh, border)
            } else {
                parent::previous(previous, black_box(r), black_box(input), kw, kh, border, None)
            }
            .unwrap();
            let downloaded = if full {
                Some(output.readback_u8::<1>(r).unwrap())
            } else {
                r.wait_idle();
                None
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            let downloaded = downloaded.unwrap_or_else(|| output.readback_u8::<1>(r).unwrap());
            assert_eq!(downloaded.as_slice(), expected);
            assert_eq!(
                output.receipt().gpu_to_gpu_bytes(),
                p.bytes + if p.separable { p.scratch_bytes } else { 0 }
            );
            output.recycle(r);
            if let Some(input) = uploaded {
                input.recycle(r);
            }
            if round >= 3 {
                samples[variant].push(elapsed);
            }
        }
    }
    let pass = horizontal_passes(r, previous, &source, kw, kh, border, &expected);
    source.recycle(r);
    println!("BOX_ROW_PREFIX_RECORD {{\"width\":{w},\"height\":{h},\"kw\":{kw},\"kh\":{kh},\"zero\":{zero},\"packed\":{},\"wide\":{},\"changed\":{},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"pass_old_ms\":{:?},\"pass_ms\":{:?}}}",p.packed,p.separable && !p.packed,row_prefix_enabled(w as u32,h as u32,kw,&r.device().limits()),samples[0],samples[1],samples[2],samples[3],pass[0],pass[1]);
}

#[test]
#[ignore = "paired first-stage/resident/upload-filter-readback calls; run alone in release mode"]
fn benchmark_complete_box_row_prefix_calls() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    let previous = parent::ParentBlurPipeline::new();
    println!("BOX_ROW_PREFIX_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (w, h, kw, kh, zero) in [
        (17, 19, 3, 3, false),
        (257, 259, 257, 31, false),
        (511, 129, 259, 9, false),
        (512, 127, 259, 9, false),
        (4097, 128, 259, 9, false),
        (1025, 257, 257, 31, false),
        (512, 128, 259, 1, false),
        (513, 129, 259, 31, false),
        (1025, 257, 259, 9, false),
        (1025, 257, 259, 9, true),
        (1920, 1080, 259, 1, false),
        (1920, 1080, 259, 31, false),
        (1920, 1080, 511, 31, false),
        (1920, 1080, 4099, 9, false),
        (1920, 1080, 4099, 9, true),
        (3840, 2160, 259, 3, false),
        (3840, 2160, 259, 31, false),
        (3840, 2160, 4099, 3, false),
        (3840, 2160, 4099, 31, false),
        (4096, 128, 259, 9, false),
        (4096, 129, 4099, 31, true),
        (512, 129, 65537, 1, false),
        (512, 128, 259, 81, false),
        (1920, 1080, 259, 81, false),
    ] {
        measure(&r, &previous, w, h, kw, kh, zero);
    }
}

#[test]
#[ignore = "fresh application pipeline caches on a retained device; run alone in release mode"]
fn benchmark_cold_box_row_prefix_calls() {
    let mut r = WgpuRuntime::new_headless().unwrap();
    println!("BOX_ROW_PREFIX_COLD_ENV backend={} device_type={} pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true", r.adapter_info().backend, r.adapter_info().device_type);
    type Job = (usize, usize, u32, u32);
    let small: Job = (17, 19, 259, 31);
    let large: Job = (512, 128, 4099, 31);
    for (name, jobs) in [
        ("small", vec![small]),
        ("large", vec![large]),
        ("small_then_large", vec![small, large]),
        ("large_then_small", vec![large, small]),
        ("all_programs", vec![(16, 17, 3, 3), (257, 259, 257, 31), small, large]),
    ] {
        let inputs: Vec<_> = jobs
            .iter()
            .map(|&(w, h, _, _)| GpuImage::upload_u8(&r, fixture(w, h, false).view()).unwrap())
            .collect();
        let expected: Vec<_> = jobs
            .iter()
            .map(|&(w, h, kw, kh)| {
                prefix_oracle(&fixture(w, h, false), kw, kh, GpuImageBorder::Replicate)
            })
            .collect();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        let mut counts = [0; 2];
        for round in 0..20 {
            for offset in 0..2 {
                let current = (round + offset) % 2 == 1;
                r.wait_idle();
                r.image_blur_pipeline.take();
                r.clear_scratch_buffer_pool();
                r.clear_image_texture_pool();
                let previous = parent::ParentBlurPipeline::new();
                let mut outputs = Vec::with_capacity(jobs.len());
                let start = Instant::now();
                for (&(_, _, kw, kh), input) in jobs.iter().zip(&inputs) {
                    outputs.push(
                        if current {
                            box_blur_gpu(
                                black_box(&r),
                                black_box(input),
                                kw,
                                kh,
                                GpuImageBorder::Replicate,
                            )
                        } else {
                            parent::previous(
                                &previous,
                                black_box(&r),
                                black_box(input),
                                kw,
                                kh,
                                GpuImageBorder::Replicate,
                                None,
                            )
                        }
                        .unwrap(),
                    );
                    r.wait_idle();
                }
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                counts[usize::from(current)] = if current {
                    populated(r.image_blur_pipeline.get().unwrap()).len()
                } else {
                    previous.variants.iter().filter(|v| v.get().is_some()).count()
                };
                for (mut output, bytes) in outputs.into_iter().zip(&expected) {
                    assert_eq!(output.readback_u8::<1>(&r).unwrap().as_slice(), bytes);
                    output.recycle(&r);
                }
                samples[usize::from(current)].push(elapsed);
            }
        }
        assert_eq!(
            counts,
            match name {
                "small" | "large" => [2, 2],
                "small_then_large" | "large_then_small" => [2, 3],
                "all_programs" => [5, 6],
                _ => unreachable!(),
            }
        );
        println!("BOX_ROW_PREFIX_COLD_RECORD {{\"scenario\":\"{name}\",\"pipeline_counts\":{counts:?},\"old_ms\":{:?},\"ms\":{:?}}}", samples[0], samples[1]);
        for input in inputs {
            input.recycle(&r);
        }
    }
}
