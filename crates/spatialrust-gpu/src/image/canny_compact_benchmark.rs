//! Historical gradient comparison between frozen PR #180 and #181 shaders.
use super::*;
use std::time::Instant;

fn measure(
    runtimes: [&WgpuRuntime; 2],
    previous: &CannyPipelines,
    compact: &CannyPipelines,
    w: usize,
    h: usize,
    profile: u32,
    o: GpuCannyOptions,
) {
    // Start each case without resources retained by a different geometry.
    for runtime in runtimes {
        runtime.clear_scratch_buffer_pool();
        runtime.clear_image_texture_pool();
    }
    let image = fixture(w, h, profile);
    let expected = canny(image.view(), cpu_options(o)).unwrap();
    let sources = runtimes.map(|runtime| GpuImage::upload_u8(runtime, image.view()).unwrap());
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| vec![]);
    // Separate runtime pools keep each route's normal retention budget intact.
    // Calls still execute sequentially on the same physical backend.
    for round in 0..23 {
        for offset in 0..4 {
            let stage = (round + offset) % 4;
            let variant = stage % 2;
            let runtime = runtimes[variant];
            let full = stage < 2;
            let start = Instant::now();
            let uploaded = full.then(|| GpuImage::upload_u8(runtime, image.view()).unwrap());
            let source = uploaded.as_ref().unwrap_or(&sources[variant]);
            let mut output = if variant == 0 {
                previous_canny(runtime, source, o, previous)
            } else {
                previous_horizontal_canny(runtime, source, o, compact)
            };
            let elapsed;
            if full {
                let result = output.readback_u8::<1>(runtime).unwrap();
                let receipt = output.receipt().clone();
                uploaded.unwrap().recycle(runtime);
                output.recycle(runtime);
                elapsed = start.elapsed();
                assert_eq!(result, expected);
                assert_eq!(receipt.host_to_device_bytes(), (w * h * 4) as u64);
                let footprint =
                    if o.aperture_size == 3 { 36 } else { 44 } + if variant == 0 { 8 } else { 0 };
                assert_eq!(receipt.gpu_to_gpu_bytes(), (w * h * footprint) as u64);
                assert_eq!(
                    receipt.device_to_host_bytes(),
                    ((w * 4).div_ceil(256) * 256 * h) as u64
                );
            } else {
                runtime.wait_idle();
                elapsed = start.elapsed();
                output.receipt().validate_resident_chain((w * h * 4) as u64).unwrap();
                assert_eq!(output.readback_u8::<1>(runtime).unwrap(), expected);
                output.recycle(runtime);
            }
            if round >= 3 {
                samples[stage].push(elapsed.as_secs_f64() * 1000.0);
            }
        }
    }
    let n = (w * h) as u64;
    let old_scratch = if o.aperture_size == 3 { 28 * n + 8 } else { 36 * n };
    let new_scratch = old_scratch - 8 * n;
    let edges = expected.as_slice().iter().filter(|&&v| v != 0).count();
    let retained = runtimes.map(|runtime| runtime.scratch_buffer_pool_stats().cached_bytes);
    assert!(runtimes
        .iter()
        .all(|runtime| runtime.scratch_buffer_pool_stats().budget_bytes == 64 * 1024 * 1024));
    println!("GPU_CANNY_COMPACT width={w} height={h} profile={profile} aperture={} l2={} edges={edges} old_gradient={} new_gradient={} old_scratch={old_scratch} new_scratch={new_scratch} old_retained={} new_retained={} old_full_ms={:?} new_full_ms={:?} old_resident_ms={:?} new_resident_ms={:?}",o.aperture_size,u8::from(o.l2_gradient),16*n,8*n,retained[0],retained[1],samples[0],samples[1],samples[2],samples[3]);
    for (source, runtime) in sources.into_iter().zip(runtimes) {
        source.recycle(runtime);
    }
}

#[test]
#[ignore = "run this exact paired completed-call survey alone in release mode"]
fn completed_compact_gradient_calls() {
    let old = WgpuRuntime::new_headless().unwrap();
    let new = WgpuRuntime::new_headless().unwrap();
    let previous = previous_pipelines(&old);
    let compact = previous_horizontal_pipelines(&new);
    let info = old.adapter_info();
    assert_eq!(info.backend, new.adapter_info().backend);
    assert_eq!(info.device_type, new.adapter_info().device_type);
    println!(
        "GPU_CANNY_COMPACT_ENV backend={} device_type={} cpu_workers={}",
        info.backend,
        info.device_type,
        rayon::current_num_threads()
    );
    for (w, h, profile, aperture_size, l2_gradient) in [
        (256, 192, 2, 3, false),
        (256, 192, 1, 7, true),
        (1920, 1080, 2, 3, false),
        (1920, 1080, 1, 3, true),
        (1920, 1080, 2, 5, false),
        (1920, 1080, 1, 7, true),
        (3840, 2160, 2, 3, false),
        (3840, 2160, 1, 3, true),
        (3840, 2160, 2, 5, true),
        (3840, 2160, 3, 7, true),
    ] {
        measure(
            [&old, &new],
            &previous,
            &compact,
            w,
            h,
            profile,
            GpuCannyOptions { aperture_size, l2_gradient, ..Default::default() },
        );
    }
}
