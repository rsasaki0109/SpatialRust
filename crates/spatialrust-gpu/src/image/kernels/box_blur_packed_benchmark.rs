use super::*;
use std::{hint::black_box, time::Instant};

fn measure(
    r: &WgpuRuntime,
    old: &frozen::FrozenBlurPipeline,
    w: usize,
    h: usize,
    kw: u32,
    kh: u32,
) {
    let image = fixture(w, h, false);
    let border = GpuImageBorder::Replicate;
    let expected = prefix_oracle(&image, kw, kh, border);
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    r.wait_idle();
    for round in 0..33 {
        for offset in 0..4 {
            let variant = (round + offset) % 4;
            let full = variant >= 2;
            let current = variant % 2 == 1;
            let start = Instant::now();
            let uploaded =
                full.then(|| GpuImage::upload_u8(black_box(r), black_box(image.view())).unwrap());
            let input = uploaded.as_ref().unwrap_or(&source);
            let mut output = if current {
                box_blur_gpu(black_box(r), black_box(input), kw, kh, border).unwrap()
            } else {
                frozen::baseline_blur(black_box(r), black_box(input), kw, kh, border, None, old)
                    .unwrap()
            };
            let downloaded = if full {
                Some(output.readback_u8::<1>(r).unwrap())
            } else {
                r.wait_idle();
                None
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let downloaded = downloaded.unwrap_or_else(|| output.readback_u8::<1>(r).unwrap());
            assert_eq!(downloaded.as_slice(), expected);
            let p = plan(w as u32, h as u32, kw, kh, border, &r.device().limits()).unwrap();
            assert_eq!(
                output.receipt().gpu_to_gpu_bytes(),
                p.bytes
                    + if p.separable {
                        if current {
                            p.scratch_bytes
                        } else {
                            p.bytes
                        }
                    } else {
                        0
                    }
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
    println!("BLUR_PACKED_RECORD {{\"width\":{w},\"height\":{h},\"kw\":{kw},\"kh\":{kh},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3]);
    source.recycle(r);
}

#[test]
#[ignore = "paired completed box blur calls; run alone in release mode"]
fn benchmark_complete_packed_box_blur_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    let old = frozen::FrozenBlurPipeline::new();
    println!("BLUR_PACKED_ENV backend={} device_type={} warmups=3 pairs=30 routes=resident_completed_and_upload_filter_download old_cache=prewarmed output_checks_drop_recycle_excluded=true",r.adapter_info().backend,r.adapter_info().device_type);
    for (w, h, kw, kh) in [
        (16, 17, 3, 3),
        (257, 259, 9, 9),
        (1920, 1080, 9, 9),
        (1920, 1080, 31, 31),
        (257, 259, 259, 3),
        (257, 259, 1, 81),
        (3840, 2160, 9, 9),
        (3840, 2160, 31, 31),
    ] {
        measure(&r, &old, w, h, kw, kh);
    }
}
