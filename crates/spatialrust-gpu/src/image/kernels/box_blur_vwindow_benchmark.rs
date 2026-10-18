use super::*;
use std::{hint::black_box, time::Instant};

fn vertical_passes(
    r: &WgpuRuntime,
    previous: &old::ParentBlurPipeline,
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
                old::previous(previous, r, input, kw, kh, border, Some(&mut profiler))
            }
            .unwrap();
            let report = profiler.collect(r).unwrap();
            assert_eq!(report.device_to_host_bytes, u64::from(count) * 16);
            assert_eq!(report.stages.len(), count as usize);
            assert_eq!(report.stages[count as usize - 1].stage, "box_blur_gpu");
            if round >= 3 {
                samples[variant].push(report.stages[count as usize - 1].elapsed_ns / 1_000_000.);
            }
            assert_eq!(output.readback_u8::<1>(r).unwrap().as_slice(), expected);
            output.recycle(r);
        }
    }
    samples
}

fn measure(
    r: &WgpuRuntime,
    previous: &old::ParentBlurPipeline,
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
                old::previous(previous, black_box(r), black_box(input), kw, kh, border, None)
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
    let pass = vertical_passes(r, previous, &source, kw, kh, border, &expected);
    source.recycle(r);
    println!("BOX_VERTICAL_RECORD {{\"width\":{w},\"height\":{h},\"kw\":{kw},\"kh\":{kh},\"zero\":{zero},\"packed\":{},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"pass_old_ms\":{:?},\"pass_ms\":{:?}}}",p.packed,samples[0],samples[1],samples[2],samples[3],pass[0],pass[1]);
}

#[test]
#[ignore = "paired last-stage/resident/upload-filter-readback calls; run alone in release mode"]
fn benchmark_complete_box_vertical_calls() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    let previous = old::ParentBlurPipeline::new();
    println!("BOX_VERTICAL_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 routes=last_pass_resident_full wall_order=balanced last_order=alternating checks_drop_recycle_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (w, h, kw, kh, zero) in [
        (16, 17, 3, 3, false),
        (257, 259, 259, 3, false),
        (1920, 1080, 9, 9, false),
        (1920, 1080, 31, 31, false),
        (1920, 1080, 1, 81, false),
        (1920, 1080, 257, 3, false),
        (1920, 1080, 257, 1, false),
        (1920, 1080, 9, 257, false),
        (1920, 1080, 31, 31, true),
        (3840, 2160, 9, 9, false),
        (3840, 2160, 31, 31, false),
        (3840, 2160, 1, 81, false),
        (1025, 257, 9, 9, true),
        (257, 259, 257, 257, false),
        (17, 19, 31, 31, false),
        (1, 521, 257, 257, false),
        (521, 1, 257, 257, false),
        (2, 521, 9, 81, true),
    ] {
        measure(&r, &previous, w, h, kw, kh, zero);
    }
}
