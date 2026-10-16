//! Exact, bounded two-pass histograms of resident u8 image channels.

mod cdf;

pub(crate) use cdf::CdfPipeline;
pub use cdf::{histogram_cdf_gpu, histogram_cdf_gpu_profiled, GpuImageCdf};

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

use super::gpu_image::runtime_device_key;
#[cfg(not(target_arch = "wasm32"))]
use super::gpu_image::with_staging_bytes;
use super::{GpuImage, GpuImageReceipt};
use crate::{scratch::ScratchBuffers, GpuStageProfiler, WgpuRuntime};

const BINS: u32 = 256;
const HISTOGRAM_BYTES: u64 = BINS as u64 * 4;
const MAX_GROUPS: u32 = 256;
const REPLICATED_MIN_PIXELS: u32 = 4 * 1024 * 1024;
const REPLICATED_MAX_PIXELS: u32 = 8 * 1024 * 1024;
const REPLICATED_WORKGROUP_BYTES: u32 = 8 * 257 * 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    channel: u32,
    groups: u32,
}

const SHADER: &str = r#"
struct Params { width: u32, height: u32, channel: u32, groups: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var pixels: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> partials: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
var<workgroup> local_bins: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn histogram_partials(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    atomicStore(&local_bins[lane], 0u);
    workgroupBarrier();
    let total = params.width * params.height;
    let stride = params.groups * 256u;
    var index = group.x * 256u + lane;
    loop {
        if (index >= total) { break; }
        let xy = vec2<i32>(i32(index % params.width), i32(index / params.width));
        let value = textureLoad(pixels, xy, 0)[params.channel];
        atomicAdd(&local_bins[value], 1u);
        // Prevent u32 wrapping even at the maximum accepted pixel count.
        if (total - index <= stride) { break; }
        index += stride;
    }
    workgroupBarrier();
    // Every entry is overwritten, including zero bins in recycled scratch.
    partials[group.x * 256u + lane] = atomicLoad(&local_bins[lane]);
}

@compute @workgroup_size(256)
fn histogram_reduce(@builtin(local_invocation_index) bin: u32) {
    var count = 0u;
    for (var group = 0u; group < params.groups; group++) {
        count += partials[group * 256u + bin];
    }
    output[bin] = count;
}
"#;

pub(crate) struct HistogramPipelines {
    layout: wgpu::BindGroupLayout,
    partials: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    replicated: OnceLock<wgpu::ComputePipeline>,
}

impl HistogramPipelines {
    fn replicated_partials(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        self.replicated.get_or_init(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu-image-histogram-replicated-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("histogram/replicated.wgsl").into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu-image-histogram-replicated-layout"),
                bind_group_layouts: &[&self.layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu-image-histogram-replicated-partials"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("histogram_partials"),
                compilation_options: Default::default(),
                cache: None,
            })
        })
    }
}

fn create_pipelines(device: &wgpu::Device) -> HistogramPipelines {
    let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu-image-histogram-layout"),
        entries: &[
            buffer(0, wgpu::BufferBindingType::Uniform),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            buffer(2, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer(3, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu-image-histogram-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu-image-histogram-pipeline-layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = |entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };
    HistogramPipelines {
        partials: pipeline("histogram_partials"),
        reduce: pipeline("histogram_reduce"),
        replicated: OnceLock::new(),
        layout,
    }
}

pub(super) struct DispatchPlan {
    pixels: u32,
    groups: u32,
    partial_bytes: u64,
}

fn use_replicated(plan: &DispatchPlan, limits: &wgpu::Limits) -> bool {
    plan.pixels >= REPLICATED_MIN_PIXELS
        && plan.pixels <= REPLICATED_MAX_PIXELS
        && limits.max_compute_workgroup_storage_size >= REPLICATED_WORKGROUP_BYTES
}

pub(super) fn dispatch_plan(
    width: u32,
    height: u32,
    limits: &wgpu::Limits,
) -> SpatialResult<DispatchPlan> {
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 || pixels > u64::from(u32::MAX) {
        return Err(invalid("GPU histogram requires 1..=u32::MAX pixels"));
    }
    if limits.max_compute_invocations_per_workgroup < BINS
        || limits.max_compute_workgroup_size_x < BINS
        || u64::from(limits.max_compute_workgroup_storage_size) < HISTOGRAM_BYTES
        || limits.max_storage_buffers_per_shader_stage < 2
    {
        return Err(invalid(
            "GPU histogram requires 256 lanes, 1 KiB workgroup storage and two storage buffers",
        ));
    }
    let binding_groups =
        limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size))
            / HISTOGRAM_BYTES;
    let groups = pixels
        .div_ceil(u64::from(BINS))
        .min(u64::from(MAX_GROUPS))
        .min(u64::from(limits.max_compute_workgroups_per_dimension))
        .min(binding_groups) as u32;
    if groups == 0 {
        return Err(invalid("GPU histogram device limits cannot hold one 256-bin histogram"));
    }
    Ok(DispatchPlan {
        pixels: pixels as u32,
        groups,
        partial_bytes: u64::from(groups) * HISTOGRAM_BYTES,
    })
}

/// Resident exact 256-bin u32 counts for one logical u8 image channel.
///
/// Construction performs two compute passes without host readback. Input is
/// limited to u32::MAX pixels so every bin and their sum are exactly representable.
/// A live result is never recycled implicitly; call `recycle` when finished.
pub struct GpuImageHistogram {
    buffer: wgpu::Buffer,
    device_key: std::sync::Arc<()>,
    pixel_count: u32,
    channel: u32,
    receipt: GpuImageReceipt,
}

impl GpuImageHistogram {
    /// Returns the source pixel count (also the sum of all bins).
    #[must_use]
    pub const fn pixel_count(&self) -> u32 {
        self.pixel_count
    }

    /// Returns the selected logical channel, without implicit luma conversion.
    #[must_use]
    pub const fn channel(&self) -> u32 {
        self.channel
    }

    /// Returns cumulative source transfers and histogram device-stage writes.
    #[must_use]
    pub const fn receipt(&self) -> &GpuImageReceipt {
        &self.receipt
    }

    /// Explicitly synchronizes and reads exactly 1,024 bin bytes on native hosts.
    /// Each successful call adds another 1,024 bytes to the transfer receipt.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn readback_u32(&mut self, runtime: &WgpuRuntime) -> SpatialResult<[u32; 256]> {
        self.validate_runtime(runtime)?;
        let result = readback_bins_u32(runtime, &self.buffer)?;
        self.receipt.record_device_to_host(HISTOGRAM_BYTES, "readback_histogram_u32");
        Ok(result)
    }

    /// Returns bin storage to its runtime's bounded buffer pool after submission.
    /// A different runtime discards this result instead of contaminating its pool.
    pub fn recycle(self, runtime: &WgpuRuntime) {
        if self.validate_runtime(runtime).is_ok() {
            runtime.recycle_storage(HISTOGRAM_BYTES, self.buffer);
        }
    }

    fn validate_runtime(&self, runtime: &WgpuRuntime) -> SpatialResult<()> {
        if !std::sync::Arc::ptr_eq(&self.device_key, &runtime_device_key(runtime)) {
            return Err(invalid("GPU histogram belongs to a different runtime device"));
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn readback_bins_u32(runtime: &WgpuRuntime, buffer: &wgpu::Buffer) -> SpatialResult<[u32; 256]> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-image-histogram-readback"),
        size: HISTOGRAM_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-image-histogram-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, HISTOGRAM_BYTES);
    runtime.queue().submit(Some(encoder.finish()));
    with_staging_bytes(runtime.device(), &staging, HISTOGRAM_BYTES as usize, |bytes| {
        let mut result = [0; 256];
        for (count, bytes) in result.iter_mut().zip(bytes.chunks_exact(4)) {
            *count = u32::from_le_bytes(bytes.try_into().expect("four-byte histogram count"));
        }
        result
    })
}

/// Counts one logical channel (0..source.channels()) without leaving the GPU.
///
/// Uses at most 256 workgroups, 256 KiB private pooled partials and a 1 KiB
/// resident output. Every scratch/output bin is overwritten before reuse.
/// Images with 4..=8 Mi pixels use eight local counter replicas when shared
/// memory permits; other calls retain the original kernel.
/// Apply `rgb_to_gray_gpu` first when a luma histogram is desired.
pub fn histogram_u8_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    channel: u32,
) -> SpatialResult<GpuImageHistogram> {
    histogram_with_profiler(runtime, source, channel, None)
}

/// Counts a resident channel while recording two optional GPU pass timestamps.
/// Requires two unused profiler stages on the same runtime before allocating work.
pub fn histogram_u8_gpu_profiled(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    channel: u32,
    profiler: &mut GpuStageProfiler,
) -> SpatialResult<GpuImageHistogram> {
    profiler.validate(runtime, 2)?;
    let result = histogram_with_profiler(runtime, source, channel, Some(profiler));
    profiler.finish_recording(&result);
    result
}

fn histogram_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    channel: u32,
    mut profiler: Option<&mut GpuStageProfiler>,
) -> SpatialResult<GpuImageHistogram> {
    source.validate_runtime(runtime)?;
    if channel >= source.channels() {
        return Err(invalid("GPU histogram channel is outside the logical image channels"));
    }
    let limits = runtime.device().limits();
    let plan = dispatch_plan(source.width(), source.height(), &limits)?;
    let params =
        Params { width: source.width(), height: source.height(), channel, groups: plan.groups };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-image-histogram-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let pipelines =
        runtime.image_histogram_pipelines.get_or_init(|| create_pipelines(runtime.device()));
    let partial_pipeline = if use_replicated(&plan, &limits) {
        pipelines.replicated_partials(runtime.device())
    } else {
        &pipelines.partials
    };
    let mut scratch = ScratchBuffers::new(runtime);
    let partials = scratch.acquire("gpu-image-histogram-partials", plan.partial_bytes);
    let output = runtime.buffer_pool().acquire_storage(
        runtime,
        "gpu-image-histogram-output",
        HISTOGRAM_BYTES,
    );
    let source_view = source.view();
    let bind_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-image-histogram-bind-group"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: partials.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-image-histogram-encoder"),
    });
    for (pipeline, stage, groups) in [
        (partial_pipeline, "histogram_u8_partials_gpu", plan.groups),
        (&pipelines.reduce, "histogram_u8_reduce_gpu", 1),
    ] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(stage),
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    scratch.submit_and_recycle(encoder);
    let mut receipt = source.receipt().clone();
    receipt.record_gpu_to_gpu(plan.partial_bytes, "histogram_u8_partials_gpu");
    receipt.record_gpu_to_gpu(HISTOGRAM_BYTES, "histogram_u8_reduce_gpu");
    Ok(GpuImageHistogram {
        buffer: output,
        device_key: runtime_device_key(runtime),
        pixel_count: plan.pixels,
        channel,
        receipt,
    })
}

fn invalid(message: &str) -> SpatialError {
    SpatialError::InvalidArgument(message.to_owned())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "histogram_experiment_execution.rs"]
mod experiment;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "histogram_replica_tests.rs"]
mod replica_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "histogram_replica_benchmark.rs"]
mod replica_benchmark;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use spatialrust_image::{Image, ImageRegion, ImageView};

    #[test]
    fn plan_checks_counter_overflow_and_bounds_scratch_and_device_dispatch() {
        let limits = wgpu::Limits::default();
        for (width, height) in [(1, 1), (257, 1), (3840, 2160), (u32::MAX, 1)] {
            let plan = dispatch_plan(width, height, &limits).unwrap();
            assert_eq!(u64::from(plan.pixels), u64::from(width) * u64::from(height));
            assert!((1..=256).contains(&plan.groups));
            assert!(plan.partial_bytes <= 256 * 1024);
        }
        for (width, height) in [(0, 1), (1, 0), (65_536, 65_536), (u32::MAX, 2)] {
            assert!(dispatch_plan(width, height, &limits).is_err());
        }
        let mut bounded = limits.clone();
        bounded.max_compute_workgroups_per_dimension = 3;
        bounded.max_storage_buffer_binding_size = 2 * 1024;
        assert_eq!(dispatch_plan(3840, 2160, &bounded).unwrap().groups, 2);
        bounded.max_storage_buffer_binding_size = 1024;
        assert_eq!(dispatch_plan(3840, 2160, &bounded).unwrap().groups, 1);
        for field in 0..6 {
            let mut invalid = limits.clone();
            match field {
                0 => invalid.max_compute_invocations_per_workgroup = 255,
                1 => invalid.max_compute_workgroup_size_x = 255,
                2 => invalid.max_compute_workgroup_storage_size = 1023,
                3 => invalid.max_storage_buffer_binding_size = 1023,
                4 => invalid.max_compute_workgroups_per_dimension = 0,
                _ => invalid.max_storage_buffers_per_shader_stage = 1,
            }
            assert!(dispatch_plan(17, 19, &invalid).is_err());
        }
    }

    #[test]
    fn packed_histograms_match_exact_counts_across_lanes_grid_stride_and_hot_bins() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        for (width, height) in
            [(1, 1), (255, 1), (256, 1), (257, 1), (1, 257), (257, 257), (1001, 1003)]
        {
            for pattern in [0, 1, 2] {
                let data: Vec<_> = (0..width * height)
                    .map(|i| match pattern {
                        0 => 255,
                        1 => 0,
                        _ => ((i * 37 + i / 257) % 256) as u8,
                    })
                    .collect();
                let mut expected = [0u32; 256];
                for &value in &data {
                    expected[value as usize] += 1;
                }
                let image = Image::<u8, 1>::try_new(width, height, data).unwrap();
                let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
                let mut result = histogram_u8_gpu(&runtime, &uploaded, 0).unwrap();
                assert_eq!(result.pixel_count(), (width * height) as u32);
                assert_eq!(result.channel(), 0);
                assert_eq!(result.receipt().device_to_host_bytes(), 0);
                assert_eq!(result.readback_u32(&runtime).unwrap(), expected);
                assert_eq!(result.receipt().device_to_host_bytes(), 1024);
                result.recycle(&runtime);
                uploaded.recycle(&runtime);
            }
        }
        assert_eq!(runtime.initialized_image_pipeline_count(), 1);
    }

    fn check_roi<const C: usize>(runtime: &WgpuRuntime) {
        let stride = 21 * C + 13;
        let mut data = vec![255; 23 * stride];
        for y in 0..23 {
            for x in 0..21 {
                for channel in 0..C {
                    data[y * stride + x * C + channel] =
                        ((x * 17 + y * 31 + channel * 53) % 251) as u8;
                }
            }
        }
        let roi = ImageView::<u8, C>::new(21, 23, stride, &data)
            .unwrap()
            .subview(ImageRegion::new(3, 2, 17, 19))
            .unwrap();
        let uploaded = GpuImage::upload_u8(runtime, roi).unwrap();
        for channel in 0..C {
            let mut expected = [0u32; 256];
            for y in 0..roi.height() {
                for pixel in roi.row(y).unwrap().chunks_exact(C) {
                    expected[pixel[channel] as usize] += 1;
                }
            }
            let mut result = histogram_u8_gpu(runtime, &uploaded, channel as u32).unwrap();
            result.receipt().validate_resident_chain(17 * 19 * 4).unwrap();
            assert_eq!(result.readback_u32(runtime).unwrap(), expected);
            assert_eq!(result.readback_u32(runtime).unwrap(), expected);
            assert_eq!(result.receipt().device_to_host_bytes(), 2048);
            assert!(result.receipt().validate_resident_chain(17 * 19 * 4).is_err());
            assert_eq!(result.receipt().gpu_to_gpu_bytes(), 2 * 1024 + 1024);
            result.recycle(runtime);
        }
        let before = runtime.buffer_pool().stats();
        assert!(histogram_u8_gpu(runtime, &uploaded, C as u32).is_err());
        assert!(histogram_u8_gpu(runtime, &uploaded, u32::MAX).is_err());
        assert_eq!(runtime.buffer_pool().stats(), before);
        uploaded.recycle(runtime);
    }

    #[test]
    fn all_logical_channels_ignore_roi_padding_and_record_only_explicit_bin_readbacks() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        check_roi::<1>(&runtime);
        check_roi::<2>(&runtime);
        check_roi::<3>(&runtime);
        check_roi::<4>(&runtime);
    }

    #[test]
    fn live_results_survive_scratch_reuse_pool_clears_and_runtime_moves() {
        let boxed = Box::new(WgpuRuntime::new_headless().unwrap());
        let image = Image::<u8, 1>::try_new(257, 257, vec![73; 257 * 257]).unwrap();
        let uploaded = GpuImage::upload_u8(&boxed, image.view()).unwrap();
        let mut first = histogram_u8_gpu(&boxed, &uploaded, 0).unwrap();
        let runtime = *boxed;
        for value in [0, 255, 42, 73] {
            let image = Image::<u8, 1>::try_new(257, 257, vec![value; 257 * 257]).unwrap();
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            let mut result = histogram_u8_gpu(&runtime, &source, 0).unwrap();
            let bins = result.readback_u32(&runtime).unwrap();
            assert_eq!(bins[value as usize], 257 * 257);
            assert_eq!(bins.iter().sum::<u32>(), 257 * 257);
            result.recycle(&runtime);
            source.recycle(&runtime);
        }
        let scratch = runtime.scratch_buffer_pool_stats();
        let buffers = runtime.buffer_pool().stats();
        for _ in 0..5 {
            histogram_u8_gpu(&runtime, &uploaded, 0).unwrap().recycle(&runtime);
        }
        assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, scratch.allocation_count);
        assert_eq!(runtime.buffer_pool().stats().allocation_count, buffers.allocation_count);
        assert_eq!(runtime.scratch_buffer_pool_stats().reuse_count - scratch.reuse_count, 5);
        runtime.set_scratch_buffer_pool_budget(0);
        runtime.buffer_pool().set_budget(0);
        let expected = {
            let mut bins = [0; 256];
            bins[73] = 257 * 257;
            bins
        };
        assert_eq!(first.readback_u32(&runtime).unwrap(), expected);
        first.recycle(&runtime);
        assert_eq!(runtime.buffer_pool().stats().cached_bytes, 0);
        uploaded.recycle(&runtime);
    }

    #[test]
    fn foreign_devices_reject_sources_results_and_recycling_without_pool_changes() {
        let first = WgpuRuntime::new_headless().unwrap();
        let other = WgpuRuntime::new_headless().unwrap();
        let image = Image::<u8, 1>::try_new(1, 1, vec![19]).unwrap();
        let source = GpuImage::upload_u8(&first, image.view()).unwrap();
        let mut result = histogram_u8_gpu(&first, &source, 0).unwrap();
        assert!(histogram_u8_gpu(&other, &source, 0).is_err());
        assert!(result.readback_u32(&other).is_err());
        assert_eq!(result.receipt().device_to_host_bytes(), 0);
        result.recycle(&other);
        assert_eq!(other.buffer_pool().stats().cached_bytes, 0);
        assert_eq!(other.scratch_buffer_pool_stats().allocation_count, 0);
        assert_eq!(other.initialized_image_pipeline_count(), 0);
        source.recycle(&first);
    }

    #[test]
    fn concurrent_histograms_keep_live_outputs_and_private_scratch_separate() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for value in [0u8, 19, 73, 255] {
                let runtime = &runtime;
                handles.push(scope.spawn(move || {
                    let image = Image::<u8, 1>::try_new(257, 257, vec![value; 257 * 257]).unwrap();
                    let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
                    let mut outputs = Vec::new();
                    for _ in 0..3 {
                        outputs.push(histogram_u8_gpu(runtime, &source, 0).unwrap());
                    }
                    for mut output in outputs {
                        let bins = output.readback_u32(runtime).unwrap();
                        assert_eq!(bins[value as usize], 257 * 257);
                        assert_eq!(bins.iter().sum::<u32>(), 257 * 257);
                        output.recycle(runtime);
                    }
                    source.recycle(runtime);
                }));
            }
            for handle in handles {
                handle.join().unwrap();
            }
        });
        assert!(runtime.scratch_buffer_pool_stats().cached_bytes <= 64 * 1024 * 1024);
        assert_eq!(runtime.initialized_image_pipeline_count(), 1);
    }
}
