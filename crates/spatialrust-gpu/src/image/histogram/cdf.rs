//! Exact inclusive cumulative counts from an owned resident histogram.

use spatialrust_core::{SpatialError, SpatialResult};

use super::{GpuImageHistogram, HISTOGRAM_BYTES};
use crate::image::gpu_image::runtime_device_key;
use crate::image::GpuImageReceipt;
use crate::{GpuStageProfiler, WgpuRuntime};

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> bins: array<u32>;
@group(0) @binding(1) var<storage, read_write> cumulative: array<u32>;
var<workgroup> tree: array<u32, 256>;

// The same work-efficient tree scan used by point-cloud compaction, specialized
// to exactly one full block. The original count turns exclusive into inclusive.
@compute @workgroup_size(256)
fn histogram_cdf(@builtin(local_invocation_index) lane: u32) {
    let value = bins[lane];
    tree[lane] = value;
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        let slot = (lane + 1u) * stride * 2u - 1u;
        if slot < 256u { tree[slot] += tree[slot - stride]; }
        workgroupBarrier();
    }
    if lane == 0u { tree[255u] = 0u; }
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        let slot = (lane + 1u) * stride * 2u - 1u;
        if slot < 256u {
            let left = tree[slot - stride];
            tree[slot - stride] = tree[slot];
            tree[slot] += left;
        }
        workgroupBarrier();
    }
    // All output bins are overwritten, including leading zero counts.
    cumulative[lane] = tree[lane] + value;
}
"#;

pub(crate) struct CdfPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl CdfPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(HISTOGRAM_BYTES),
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu-image-cdf-layout"),
            entries: &[entry(0, true), entry(1, false)],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-image-cdf-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-image-cdf-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu-image-cdf-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("histogram_cdf"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { layout, pipeline }
    }
}

fn validate_limits(limits: &wgpu::Limits) -> SpatialResult<()> {
    if limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || u64::from(limits.max_compute_workgroup_storage_size) < HISTOGRAM_BYTES
        || limits.max_storage_buffers_per_shader_stage < 2
        || limits.max_compute_workgroups_per_dimension == 0
        || limits.max_buffer_size < HISTOGRAM_BYTES
        || u64::from(limits.max_storage_buffer_binding_size) < HISTOGRAM_BYTES
    {
        return Err(invalid(
            "GPU CDF requires 256 lanes, 1 KiB local storage and two 1 KiB buffers",
        ));
    }
    Ok(())
}

/// Resident inclusive u32 cumulative counts for all 256 image byte values.
///
/// Entry `i` counts source pixels whose selected channel is at most `i`.
/// Counts are exact, monotonically nondecreasing, and entry 255 equals
/// `pixel_count()`. This is an unnormalized CDF; no float conversion or rounding
/// occurs. The result owns independent 1 KiB storage and never recycles implicitly.
pub struct GpuImageCdf {
    buffer: wgpu::Buffer,
    device_key: std::sync::Arc<()>,
    pixel_count: u32,
    channel: u32,
    receipt: GpuImageReceipt,
}

impl GpuImageCdf {
    pub(in crate::image) const fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Returns the source pixel count, also the final cumulative count.
    #[must_use]
    pub const fn pixel_count(&self) -> u32 {
        self.pixel_count
    }

    /// Returns the source histogram's logical channel.
    #[must_use]
    pub const fn channel(&self) -> u32 {
        self.channel
    }

    /// Returns cumulative source transfers plus the 1 KiB CDF stage write.
    #[must_use]
    pub const fn receipt(&self) -> &GpuImageReceipt {
        &self.receipt
    }

    /// Explicitly synchronizes and reads 1,024 cumulative-count bytes natively.
    /// Each successful call adds 1,024 bytes to the cumulative transfer receipt.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn readback_u32(&mut self, runtime: &WgpuRuntime) -> SpatialResult<[u32; 256]> {
        self.validate_runtime(runtime)?;
        let counts = super::readback_bins_u32(runtime, &self.buffer)?;
        self.receipt.record_device_to_host(HISTOGRAM_BYTES, "readback_cdf_u32");
        Ok(counts)
    }

    /// Returns output storage to its original runtime's bounded buffer pool.
    /// A foreign runtime discards this result instead of receiving its storage.
    pub fn recycle(self, runtime: &WgpuRuntime) {
        if self.validate_runtime(runtime).is_ok() {
            runtime.recycle_storage(HISTOGRAM_BYTES, self.buffer);
        }
    }

    fn validate_runtime(&self, runtime: &WgpuRuntime) -> SpatialResult<()> {
        if !std::sync::Arc::ptr_eq(&self.device_key, &runtime_device_key(runtime)) {
            return Err(invalid("GPU CDF belongs to a different runtime device"));
        }
        Ok(())
    }
}

/// Computes exact inclusive cumulative counts without downloading source bins.
///
/// One 256-lane pass uses 1 KiB workgroup-local memory and writes a separate
/// 1 KiB resident result. Source bins stay unchanged. The source may be recycled
/// after this call returns; its reads have already been submitted in queue order.
pub fn histogram_cdf_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImageHistogram,
) -> SpatialResult<GpuImageCdf> {
    cdf_with_profiler(runtime, source, None)
}

/// Computes cumulative counts while recording one optional GPU pass timestamp.
/// Requires one unused profiler stage on the original runtime before allocation.
pub fn histogram_cdf_gpu_profiled(
    runtime: &WgpuRuntime,
    source: &GpuImageHistogram,
    profiler: &mut GpuStageProfiler,
) -> SpatialResult<GpuImageCdf> {
    profiler.validate(runtime, 1)?;
    let result = cdf_with_profiler(runtime, source, Some(profiler));
    profiler.finish_recording(&result);
    result
}

fn cdf_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImageHistogram,
    mut profiler: Option<&mut GpuStageProfiler>,
) -> SpatialResult<GpuImageCdf> {
    source.validate_runtime(runtime)?;
    validate_limits(&runtime.device().limits())?;
    let pipelines = runtime.image_cdf_pipeline.get_or_init(|| CdfPipeline::new(runtime.device()));
    let output =
        runtime.buffer_pool().acquire_storage(runtime, "gpu-image-cdf-output", HISTOGRAM_BYTES);
    let bind_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-image-cdf-bind-group"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: source.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-image-cdf-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram_cdf_u32_gpu"),
            timestamp_writes: crate::profiling::timestamp_writes(
                &mut profiler,
                "histogram_cdf_u32_gpu",
            ),
        });
        pass.set_pipeline(&pipelines.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    let mut receipt = source.receipt().clone();
    receipt.record_gpu_to_gpu(HISTOGRAM_BYTES, "histogram_cdf_u32_gpu");
    Ok(GpuImageCdf {
        buffer: output,
        device_key: runtime_device_key(runtime),
        pixel_count: source.pixel_count(),
        channel: source.channel(),
        receipt,
    })
}

fn invalid(message: &str) -> SpatialError {
    SpatialError::InvalidArgument(message.to_owned())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "mapped_readback_tests.rs"]
mod mapped_readback_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::image::{histogram_u8_gpu, GpuImage};
    use spatialrust_image::Image;

    #[test]
    fn cdf_requires_bounded_scan_limits() {
        let limits = wgpu::Limits::default();
        validate_limits(&limits).unwrap();
        for field in 0..7 {
            let mut invalid = limits.clone();
            match field {
                0 => invalid.max_compute_invocations_per_workgroup = 255,
                1 => invalid.max_compute_workgroup_size_x = 255,
                2 => invalid.max_compute_workgroup_storage_size = 1023,
                3 => invalid.max_storage_buffers_per_shader_stage = 1,
                4 => invalid.max_compute_workgroups_per_dimension = 0,
                5 => invalid.max_buffer_size = 1023,
                _ => invalid.max_storage_buffer_binding_size = 1023,
            }
            assert!(validate_limits(&invalid).is_err());
        }
    }

    fn reference(bins: &[u32; 256]) -> [u32; 256] {
        let mut total = 0u64;
        std::array::from_fn(|i| {
            total += u64::from(bins[i]);
            u32::try_from(total).unwrap()
        })
    }

    // Test-only resident input avoids allocating a u32::MAX-pixel texture.
    fn synthetic(runtime: &WgpuRuntime, bins: &[u32; 256]) -> GpuImageHistogram {
        let pixel_count = reference(bins)[255];
        let buffer =
            runtime.buffer_pool().acquire_storage(runtime, "cdf-test-bins", HISTOGRAM_BYTES);
        runtime.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(bins));
        let mut receipt = GpuImageReceipt::default();
        receipt.record_host_to_device(HISTOGRAM_BYTES, "cdf_test_upload");
        GpuImageHistogram {
            buffer,
            device_key: runtime_device_key(runtime),
            pixel_count,
            channel: 0,
            receipt,
        }
    }

    #[test]
    fn scan_is_exact_through_u32_max_and_preserves_source_bins() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        let mut cases = vec![std::array::from_fn(|i| (i * i + 3) as u32)];
        for (slot, count) in
            [(0, 1), (127, 19), (255, 257), (0, u32::MAX), (127, u32::MAX), (255, u32::MAX)]
        {
            let mut bins = [0; 256];
            bins[slot] = count;
            cases.push(bins);
        }
        let mut split = [0; 256];
        split[0] = u32::MAX - 1;
        split[255] = 1;
        cases.push(split);
        cases.push(std::array::from_fn(|i| if i == 255 { 16_777_470 } else { 16_777_215 }));
        for bins in cases {
            let mut source = synthetic(&runtime, &bins);
            let mut result = histogram_cdf_gpu(&runtime, &source).unwrap();
            let expected = reference(&bins);
            result.receipt().validate_resident_chain(HISTOGRAM_BYTES).unwrap();
            assert_eq!(result.readback_u32(&runtime).unwrap(), expected);
            assert_eq!(result.readback_u32(&runtime).unwrap(), expected);
            assert_eq!(result.pixel_count(), expected[255]);
            assert_eq!(result.channel(), 0);
            assert_eq!(result.receipt().device_to_host_bytes(), 2048);
            assert_eq!(result.receipt().gpu_to_gpu_bytes(), 1024);
            assert_eq!(result.receipt().stages().last(), Some(&"readback_cdf_u32"));
            assert!(expected.windows(2).all(|pair| pair[0] <= pair[1]));
            assert_eq!(source.readback_u32(&runtime).unwrap(), bins);
            source.recycle(&runtime);
            result.recycle(&runtime);
        }
        assert_eq!(runtime.initialized_image_pipeline_count(), 1);
        assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, 0);
    }

    #[test]
    fn live_cdfs_survive_source_recycling_output_reuse_and_runtime_moves() {
        let boxed = Box::new(WgpuRuntime::new_headless().unwrap());
        let mut bins = [0; 256];
        bins[42] = 65_537;
        let source = synthetic(&boxed, &bins);
        let mut retained = histogram_cdf_gpu(&boxed, &source).unwrap();
        source.recycle(&boxed);
        let runtime = *boxed;
        for value in [255, 0, 127, 255] {
            let mut replacement = [0; 256];
            replacement[value] = 9;
            let source = synthetic(&runtime, &replacement);
            let mut result = histogram_cdf_gpu(&runtime, &source).unwrap();
            source.recycle(&runtime);
            assert_eq!(result.readback_u32(&runtime).unwrap(), reference(&replacement));
            result.recycle(&runtime);
        }
        let source = synthetic(&runtime, &bins);
        let before = runtime.buffer_pool().stats();
        for _ in 0..5 {
            histogram_cdf_gpu(&runtime, &source).unwrap().recycle(&runtime);
        }
        assert_eq!(runtime.buffer_pool().stats().allocation_count, before.allocation_count);
        assert_eq!(runtime.buffer_pool().stats().reuse_count - before.reuse_count, 5);
        runtime.buffer_pool().set_budget(0);
        runtime.clear_scratch_buffer_pool();
        assert_eq!(retained.readback_u32(&runtime).unwrap(), reference(&bins));
        retained.recycle(&runtime);
        source.recycle(&runtime);
        assert_eq!(runtime.buffer_pool().stats().cached_bytes, 0);
    }

    #[test]
    fn foreign_sources_readbacks_and_recycling_fail_without_work_allocation() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        let other = WgpuRuntime::new_headless().unwrap();
        let mut bins = [0; 256];
        bins[19] = 1;
        let source = synthetic(&runtime, &bins);
        let mut result = histogram_cdf_gpu(&runtime, &source).unwrap();
        assert!(histogram_cdf_gpu(&other, &source).is_err());
        assert!(result.readback_u32(&other).is_err());
        assert_eq!(result.receipt().device_to_host_bytes(), 0);
        result.recycle(&other);
        assert_eq!(other.buffer_pool().stats().allocation_count, 0);
        assert_eq!(other.buffer_pool().stats().cached_bytes, 0);
        assert_eq!(other.initialized_image_pipeline_count(), 0);
        source.recycle(&runtime);
    }

    #[test]
    fn shared_histogram_supports_concurrent_independent_cdfs() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        let image =
            Image::<u8, 1>::try_new(257, 129, (0..257 * 129).map(|i| (i % 251) as u8).collect())
                .unwrap();
        let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let mut bins = [0; 256];
        for &value in image.as_slice() {
            bins[value as usize] += 1;
        }
        let source = histogram_u8_gpu(&runtime, &uploaded, 0).unwrap();
        let outputs = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        (0..3)
                            .map(|_| histogram_cdf_gpu(&runtime, &source).unwrap())
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
        });
        source.recycle(&runtime);
        uploaded.recycle(&runtime);
        for mut output in outputs {
            assert_eq!(output.readback_u32(&runtime).unwrap(), reference(&bins));
            output.recycle(&runtime);
        }
        assert_eq!(runtime.initialized_image_pipeline_count(), 2);
        let stats = runtime.buffer_pool().stats();
        assert!(stats.cached_bytes <= stats.budget_bytes);
    }
}
