//! Measure repeat-frame pooled allocations with retention enabled and disabled.
//!
//! `cargo run -p spatialrust-gpu --release --features gpu-image --example gpu_pool_reuse`
//! Uploads and warmup occur outside timings; timings include explicit completion.

use std::time::Instant;

use spatialrust_gpu::{
    build_voxel_segments_gpu_from_keys_buffer, run_gpu_vision_chain, GpuImage,
    GpuVisionChainOptions, WgpuRuntime, DEFAULT_GPU_POOL_BUDGET_BYTES,
};
use spatialrust_image::Image;

const RUNS: u64 = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = WgpuRuntime::new_headless()?;
    println!("Adapter: {:?}", runtime.adapter_info());
    println!("Counters cover pooled resources only; output buffers, uniforms, staging and driver overhead are excluded.");
    for count in [65_536_u32, 1_048_576] {
        let keys: Vec<[i32; 4]> = (0..count)
            .map(|i| {
                [
                    ((i * 37) % 4093) as i32 - 2046,
                    ((i * 13) % 257) as i32 - 128,
                    ((i * 7) % 17) as i32 - 8,
                    0,
                ]
            })
            .collect();
        let uploaded = runtime.upload_pod_storage("pool-example-keys", &keys)?;
        for budget in [0, DEFAULT_GPU_POOL_BUDGET_BYTES] {
            runtime.clear_scratch_buffer_pool();
            runtime.set_scratch_buffer_pool_budget(budget);
            for _ in 0..3 {
                let segments =
                    build_voxel_segments_gpu_from_keys_buffer(&runtime, &uploaded, count, count)?;
                assert_eq!(segments.construction_readback_bytes(), 4);
            }
            runtime.wait_idle();
            let before = runtime.scratch_buffer_pool_stats();
            let started = Instant::now();
            for _ in 0..RUNS {
                let segments =
                    build_voxel_segments_gpu_from_keys_buffer(&runtime, &uploaded, count, count)?;
                runtime.wait_idle();
                assert_eq!(segments.construction_readback_bytes(), 4);
            }
            let elapsed = started.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;
            let after = runtime.scratch_buffer_pool_stats();
            println!("point count={count} budget={budget}: {elapsed:.3} ms/frame; scratch allocations/frame={:.1}, reuses/frame={:.1}, retained={} bytes/{} buffers, scalar readback=4 bytes/frame",
                (after.allocation_count - before.allocation_count) as f64 / RUNS as f64,
                (after.reuse_count - before.reuse_count) as f64 / RUNS as f64,
                after.cached_bytes, after.cached_resources);
        }
    }
    let source =
        Image::<u8, 3>::try_new(640, 480, (0..640 * 480 * 3).map(|i| (i % 251) as u8).collect())?;
    let uploaded = GpuImage::upload_u8(&runtime, source.view())?;
    let options = GpuVisionChainOptions { width: 320, height: 240, ..Default::default() };
    for budget in [0, DEFAULT_GPU_POOL_BUDGET_BYTES] {
        runtime.clear_image_texture_pool();
        runtime.buffer_pool().clear();
        runtime.set_image_texture_pool_budget(budget);
        runtime.buffer_pool().set_budget(budget);
        for _ in 0..3 {
            run_gpu_vision_chain(&runtime, &uploaded, options)?.recycle(&runtime);
        }
        runtime.wait_idle();
        let textures_before = runtime.image_texture_pool_stats();
        let buffers_before = runtime.buffer_pool().stats();
        let started = Instant::now();
        for _ in 0..RUNS {
            let tensor = run_gpu_vision_chain(&runtime, &uploaded, options)?;
            tensor.receipt().validate_resident_chain(640 * 480 * 4)?;
            runtime.wait_idle();
            tensor.recycle(&runtime);
        }
        let elapsed = started.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;
        let textures = runtime.image_texture_pool_stats();
        let buffers = runtime.buffer_pool().stats();
        println!("image 640x480->320x240 budget={budget}: {elapsed:.3} ms/frame; texture allocations/frame={:.1}, CHW allocations/frame={:.1}, retained={} texture bytes + {} buffer bytes, payload readback=0 bytes/frame",
            (textures.allocation_count - textures_before.allocation_count) as f64 / RUNS as f64,
            (buffers.allocation_count - buffers_before.allocation_count) as f64 / RUNS as f64,
            textures.cached_bytes, buffers.cached_bytes);
    }
    uploaded.recycle(&runtime);
    Ok(())
}
