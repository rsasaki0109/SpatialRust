//! Run with `--features gpu-image,gpu-aoso-staging --example gpu_stage_timing`.

use std::collections::BTreeMap;
use std::time::Instant;

use spatialrust_core::{PointCloudBuilder, SpatialTensor};
use spatialrust_gpu::{
    run_aoso_voxel_normal_frame, run_aoso_voxel_normal_frame_profiled, run_gpu_vision_chain,
    run_gpu_vision_chain_profiled, GpuImage, GpuStageProfiler, GpuTimingReport,
    GpuVisionChainOptions, WgpuPowerPreference, WgpuRuntime,
};
use spatialrust_image::Image;

#[derive(Default)]
struct Measurements {
    wall_ms: f64,
    stages_ms: BTreeMap<&'static str, f64>,
    query_bytes: u64,
}

impl Measurements {
    fn add(&mut self, wall: f64, report: GpuTimingReport) {
        self.wall_ms += wall;
        self.query_bytes += report.device_to_host_bytes;
        for stage in report.stages {
            *self.stages_ms.entry(stage.stage).or_default() += stage.elapsed_ns / 1e6;
        }
    }

    fn print(&self, name: &str, runs: usize) {
        let runs = runs as f64;
        println!("{name}: completed wall {:.3} ms; GPU compute passes {:.3} ms; query readback {:.0} bytes/run",
            self.wall_ms / runs, self.stages_ms.values().sum::<f64>() / runs, self.query_bytes as f64 / runs);
        for (stage, ms) in &self.stages_ms {
            println!("  {stage}: {:.3} ms/run", ms / runs);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance)?;
    println!("Adapter: {:?}", runtime.adapter_info());
    if !runtime.timestamp_queries_enabled() {
        println!("This adapter does not support optional GPU timestamp queries.");
        return Ok(());
    }
    let source = Image::<u8, 3>::try_new(
        640,
        480,
        (0..640 * 480 * 3).map(|i| (i * 13 % 256) as u8).collect(),
    )?;
    let image = GpuImage::upload_u8(&runtime, source.view())?;
    let options = GpuVisionChainOptions { width: 320, height: 240, ..Default::default() };
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..65_536 {
        let point = [
            (i % 256) as f32 * 0.03,
            (i / 256) as f32 * 0.03,
            ((i % 17) as f32 * 0.001).sin() * 0.01,
        ];
        builder.push_point(point)?;
    }
    let cloud = builder.build()?;
    let tensor = SpatialTensor::new(&cloud, 16_384)?;
    for _ in 0..3 {
        let output = run_gpu_vision_chain(&runtime, &image, options)?;
        let frame = run_aoso_voxel_normal_frame(&runtime, &tensor, [0.0; 3], 20.0, 0.08)?;
        runtime.wait_idle();
        output.recycle(&runtime);
        frame.recycle(&runtime)?;
    }
    let runs = 10;
    let mut images = Measurements::default();
    let mut points = Measurements::default();
    for _ in 0..runs {
        let mut profiler = GpuStageProfiler::new(&runtime, 6)?;
        let started = Instant::now();
        let output = run_gpu_vision_chain_profiled(&runtime, &image, options, &mut profiler)?;
        runtime.wait_idle();
        let wall = started.elapsed().as_secs_f64() * 1e3;
        assert_eq!(output.receipt().device_to_host_bytes(), 0);
        images.add(wall, profiler.collect(&runtime)?);
        output.recycle(&runtime);

        let mut profiler = GpuStageProfiler::new(&runtime, 256)?;
        let started = Instant::now();
        let frame = run_aoso_voxel_normal_frame_profiled(
            &runtime,
            &tensor,
            [0.0; 3],
            20.0,
            0.08,
            &mut profiler,
        )?;
        runtime.wait_idle();
        let wall = started.elapsed().as_secs_f64() * 1e3;
        assert_eq!(frame.receipt().device_to_host_bytes(), 8);
        points.add(wall, profiler.collect(&runtime)?);
        frame.recycle(&runtime)?;
    }
    images.print("Image 640x480 -> 320x240 CHW (source upload excluded)", runs);
    points.print("Point frame 65,536 XYZ (upload and 8-byte metadata readback included)", runs);
    println!(
        "Wall times include completion but exclude profiler allocation and timestamp collection."
    );
    image.recycle(&runtime);
    Ok(())
}
