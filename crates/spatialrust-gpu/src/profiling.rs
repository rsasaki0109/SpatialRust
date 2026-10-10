//! Optional, explicitly collected compute-pass timestamp queries.

use spatialrust_core::{SpatialError, SpatialResult};

use crate::WgpuRuntime;

/// One compute pass measured by the GPU clock, in submission order.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuStageTiming {
    /// Kernel stage name; repeated dispatches retain separate samples.
    pub stage: &'static str,
    /// End minus beginning timestamp, converted to nanoseconds.
    pub elapsed_ns: f64,
}

/// Explicit timestamp readback, separate from image/point payload receipts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuTimingReport {
    /// Compute passes in submission order.
    pub stages: Vec<GpuStageTiming>,
    /// Query bytes copied to the CPU (two `u64` values per pass).
    pub device_to_host_bytes: u64,
}

/// Bounded profiler shared by GPU image chains and point-cloud frames.
///
/// Requires a runtime created with timestamp queries enabled. Recording adds
/// pass-boundary queries only; native `collect` explicitly synchronizes and
/// reads them back. Timings exclude uploads, copies, CPU setup, readbacks and
/// gaps between passes, and must not be interpreted as total frame latency.
/// Timestamp precision is backend-dependent and samples may be zero.
pub struct GpuStageProfiler {
    runtime_identity: std::sync::Arc<()>,
    queries: wgpu::QuerySet,
    max_stages: u32,
    stages: Vec<&'static str>,
    failed: bool,
}

impl GpuStageProfiler {
    /// Allocates at most `max_stages` compute-pass samples (1..=2048).
    pub fn new(runtime: &WgpuRuntime, max_stages: u32) -> SpatialResult<Self> {
        if !runtime.timestamp_queries_enabled() {
            return Err(invalid("GPU timestamp queries are not enabled on this runtime"));
        }
        if max_stages == 0 || max_stages > wgpu::QUERY_SET_MAX_QUERIES / 2 {
            return Err(invalid("GPU profiling capacity must be in 1..=2048 stages"));
        }
        Ok(Self {
            runtime_identity: runtime.profiling_identity(),
            queries: runtime.device().create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("spatialrust-stage-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: max_stages * 2,
            }),
            max_stages,
            stages: Vec::new(),
            failed: false,
        })
    }

    pub(crate) fn validate(&self, runtime: &WgpuRuntime, required: u32) -> SpatialResult<()> {
        // wgpu handle equality can collide across separately created Instances.
        // Retain a runtime identity so moving the runtime cannot invalidate it.
        if !std::sync::Arc::ptr_eq(&self.runtime_identity, &runtime.profiling_identity()) {
            return Err(invalid("GPU profiler belongs to a different runtime"));
        }
        if self.failed {
            return Err(invalid("GPU profiler contains a failed recording"));
        }
        if required > self.max_stages - self.stages.len() as u32 {
            return Err(invalid("GPU profiling capacity is insufficient for this operation"));
        }
        Ok(())
    }

    #[cfg(any(feature = "gpu-image", feature = "gpu-aoso-staging"))]
    pub(crate) fn finish_recording<T>(&mut self, result: &SpatialResult<T>) {
        // An error may abandon an encoder containing reserved query slots.
        // Never resolve those slots or expose them as successful samples.
        self.failed |= result.is_err();
    }

    /// Consumes the recording and explicitly reads back timestamps on native hosts.
    ///
    /// Call only after the profiled operations have returned successfully. The
    /// profiler can accumulate several operations on its original runtime.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn collect(self, runtime: &WgpuRuntime) -> SpatialResult<GpuTimingReport> {
        self.validate(runtime, 0)?;
        if self.stages.is_empty() {
            return Ok(GpuTimingReport::default());
        }
        let count = self.stages.len() as u32 * 2;
        let bytes = u64::from(count) * 8;
        let resolve = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-timestamp-resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-timestamp-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-timestamp-collect"),
            });
        encoder.resolve_query_set(&self.queries, 0..count, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
        runtime.queue().submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        runtime.device().poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| invalid("GPU timestamp map callback was lost"))?
            .map_err(|error| invalid(&format!("GPU timestamp map failed: {error}")))?;
        let data = slice.get_mapped_range();
        let period = f64::from(runtime.queue().get_timestamp_period());
        let stages = self
            .stages
            .into_iter()
            .zip(data.chunks_exact(16))
            .map(|(stage, pair)| GpuStageTiming { stage, elapsed_ns: elapsed_ns(pair, period) })
            .collect();
        drop(data);
        staging.unmap();
        Ok(GpuTimingReport { stages, device_to_host_bytes: bytes })
    }
}

pub(crate) fn timestamp_writes<'a>(
    profiler: &'a mut Option<&mut GpuStageProfiler>,
    stage: &'static str,
) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
    let profiler = profiler.as_deref_mut()?;
    if profiler.failed || profiler.stages.len() as u32 >= profiler.max_stages {
        profiler.failed = true;
        return None;
    }
    let index = profiler.stages.len() as u32 * 2;
    profiler.stages.push(stage);
    Some(wgpu::ComputePassTimestampWrites {
        query_set: &profiler.queries,
        beginning_of_pass_write_index: Some(index),
        end_of_pass_write_index: Some(index + 1),
    })
}

fn invalid(message: &str) -> SpatialError {
    SpatialError::InvalidArgument(message.to_owned())
}

#[cfg(not(target_arch = "wasm32"))]
fn elapsed_ns(pair: &[u8], period: f64) -> f64 {
    let begin = u64::from_le_bytes(pair[..8].try_into().expect("timestamp width"));
    let end = u64::from_le_bytes(pair[8..].try_into().expect("timestamp width"));
    end.wrapping_sub(begin) as f64 * period
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #[test]
    fn timestamp_ticks_convert_to_ns_including_zero_and_wraparound() {
        for (begin, end, expected) in
            [(100_u64, 160_u64, 30.0), (7, 7, 0.0), (u64::MAX - 3, 4, 4.0)]
        {
            let mut pair = Vec::from(begin.to_le_bytes());
            pair.extend_from_slice(&end.to_le_bytes());
            assert_eq!(super::elapsed_ns(&pair, 0.5), expected);
        }
    }
}
