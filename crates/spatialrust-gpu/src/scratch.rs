//! Scratch leases remain private and exclusive until their commands are submitted.

use crate::WgpuRuntime;

/// Owns scratch buffers referenced by an encoder. Dropping an unsubmitted batch
/// releases its handles without recycling them, so abandoned encoders cannot
/// race with future batches. Caller-visible output buffers never enter this set.
pub(crate) struct ScratchBuffers<'a> {
    runtime: &'a WgpuRuntime,
    buffers: Vec<wgpu::Buffer>,
}

impl<'a> ScratchBuffers<'a> {
    pub(crate) fn new(runtime: &'a WgpuRuntime) -> Self {
        Self { runtime, buffers: Vec::new() }
    }

    pub(crate) fn acquire(&mut self, label: &'static str, bytes: u64) -> wgpu::Buffer {
        let buffer = self.runtime.scratch_pool.acquire_storage(self.runtime, label, bytes);
        self.buffers.push(buffer.clone());
        buffer
    }

    /// Submits all users of the scratch set before making it reusable. Future
    /// submissions on this runtime's queue observe these writes in queue order.
    /// Encode any dependent copies (including scalar receipts) before calling.
    pub(crate) fn submit_and_recycle(self, encoder: wgpu::CommandEncoder) {
        self.runtime.queue().submit(Some(encoder.finish()));
        for buffer in self.buffers {
            self.runtime.scratch_pool.recycle(buffer.size(), buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abandoned_encoder_cannot_return_scratch_to_the_pool() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        let mut scratch = ScratchBuffers::new(&runtime);
        let buffer = scratch.acquire("abandoned", 16);
        encoder.clear_buffer(&buffer, 0, None);
        drop(scratch);
        assert_eq!(runtime.scratch_buffer_pool_stats().cached_resources, 0);
        let mut next = ScratchBuffers::new(&runtime);
        let next_buffer = next.acquire("next", 16);
        assert_ne!(buffer, next_buffer);
        assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, 2);
        // The old encoder may still be submitted by its owner; its resource
        // cannot alias the second batch even after the lease is abandoned.
        runtime.queue().submit(Some(encoder.finish()));
        next.submit_and_recycle(runtime.device().create_command_encoder(&Default::default()));
        assert_eq!(runtime.scratch_buffer_pool_stats().cached_resources, 1);
        runtime.wait_idle();
    }
}
