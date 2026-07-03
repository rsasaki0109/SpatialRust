//! Reusable wgpu storage-buffer pool for host→device uploads.

use std::collections::HashMap;
use std::sync::Mutex;

use bytemuck::Pod;
use spatialrust_core::SpatialResult;

use crate::runtime::WgpuRuntime;

/// Reusable GPU storage buffers keyed by byte length.
///
/// Buffers are recycled through [`WgpuRuntime::recycle_storage`] or
/// [`GpuBufferPool::recycle`]. Prefer the runtime helpers for typical use;
/// access the pool directly when implementing custom kernels.
#[derive(Default, Debug)]
pub struct GpuBufferPool {
    free: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
}

impl GpuBufferPool {
    /// Uploads a POD slice into a pooled storage buffer.
    pub fn upload_pod_storage<T: Pod>(
        &self,
        runtime: &WgpuRuntime,
        label: &'static str,
        data: &[T],
    ) -> SpatialResult<wgpu::Buffer> {
        let byte_len = std::mem::size_of_val(data) as u64;
        let buffer = self.take_storage(runtime, label, byte_len);
        runtime.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        Ok(buffer)
    }

    /// Uploads `f32` values into a pooled storage buffer.
    pub fn upload_f32_storage(
        &self,
        runtime: &WgpuRuntime,
        label: &'static str,
        data: &[f32],
    ) -> SpatialResult<wgpu::Buffer> {
        self.upload_pod_storage(runtime, label, data)
    }

    /// Uploads `u32` values into a pooled storage buffer.
    pub fn upload_u32_storage(
        &self,
        runtime: &WgpuRuntime,
        label: &'static str,
        data: &[u32],
    ) -> SpatialResult<wgpu::Buffer> {
        self.upload_pod_storage(runtime, label, data)
    }

    /// Returns a storage buffer to the pool for reuse.
    pub fn recycle(&self, byte_len: u64, buffer: wgpu::Buffer) {
        if byte_len == 0 {
            return;
        }
        if let Ok(mut free) = self.free.lock() {
            free.entry(byte_len).or_default().push(buffer);
        }
    }

    /// Discards all cached buffers without returning them to the device allocator.
    pub fn clear(&self) {
        if let Ok(mut free) = self.free.lock() {
            free.clear();
        }
    }

    /// Returns the number of buffers currently held in the pool.
    #[must_use]
    pub fn cached_buffer_count(&self) -> usize {
        self.free.lock().map(|free| free.values().map(Vec::len).sum()).unwrap_or(0)
    }

    fn take_storage(
        &self,
        runtime: &WgpuRuntime,
        label: &'static str,
        byte_len: u64,
    ) -> wgpu::Buffer {
        if byte_len == 0 {
            return runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if let Ok(mut free) = self.free.lock() {
            if let Some(buffer) = free.get_mut(&byte_len).and_then(|buffers| buffers.pop()) {
                return buffer;
            }
        }

        runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

#[cfg(all(feature = "gpu-wgpu", test))]
mod tests {
    use super::GpuBufferPool;
    use crate::runtime::WgpuRuntime;

    #[test]
    fn buffer_pool_reuses_equal_sized_buffers() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let pool = GpuBufferPool::default();
        let data = [1.0_f32, 2.0, 3.0];

        let first =
            pool.upload_f32_storage(&runtime, "upload-pool-test", &data).expect("first upload");
        assert_eq!(pool.cached_buffer_count(), 0);
        pool.recycle(first.size(), first);
        assert_eq!(pool.cached_buffer_count(), 1);

        let second =
            pool.upload_f32_storage(&runtime, "upload-pool-test", &data).expect("second upload");
        assert_eq!(second.size(), (data.len() * std::mem::size_of::<f32>()) as u64);
        assert_eq!(pool.cached_buffer_count(), 0);
    }

    #[test]
    fn runtime_buffer_pool_matches_direct_upload() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let data = [4.0_f32, 5.0];
        let buffer = runtime.upload_f32_storage("runtime-pool-test", &data).expect("upload");
        runtime.recycle_storage(buffer.size(), buffer);
        assert_eq!(runtime.buffer_pool().cached_buffer_count(), 1);
        runtime.buffer_pool().clear();
        assert_eq!(runtime.buffer_pool().cached_buffer_count(), 0);
    }
}
