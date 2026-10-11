//! Reusable wgpu storage-buffer pool for host→device uploads.

use std::sync::Mutex;

use bytemuck::Pod;
use spatialrust_core::SpatialResult;

use crate::resource_cache::{GpuResourcePoolStats, ResourceCache, DEFAULT_GPU_POOL_BUDGET_BYTES};
use crate::runtime::WgpuRuntime;

/// Reusable GPU storage buffers keyed by byte length.
///
/// Buffers are recycled through [`WgpuRuntime::recycle_storage`] or
/// [`GpuBufferPool::recycle`]. Prefer the runtime helpers for typical use;
/// access the pool directly when implementing custom kernels. Use each pool with
/// one runtime/device only. Recycle only after submitting every encoder using
/// the buffer and relinquishing all other handles to it. New acquisitions are
/// uninitialized: kernels must overwrite or explicitly clear what they read.
///
/// Retention defaults to 64 MiB with at most eight buffers per size. This budget
/// does not limit live allocations. Full pools drop incoming buffers; reducing
/// the budget below current retention clears the cache.
#[derive(Debug)]
pub struct GpuBufferPool {
    free: Mutex<ResourceCache<u64, wgpu::Buffer>>,
}

impl Default for GpuBufferPool {
    fn default() -> Self {
        Self::with_budget(DEFAULT_GPU_POOL_BUDGET_BYTES)
    }
}

impl GpuBufferPool {
    /// Creates a pool with a retained-byte budget; zero disables retention.
    #[must_use]
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self { free: Mutex::new(ResourceCache::new(budget_bytes)) }
    }

    /// Changes the retention budget, clearing cached buffers if needed.
    pub fn set_budget(&self, budget_bytes: u64) {
        if let Ok(mut free) = self.free.lock() {
            free.set_budget(budget_bytes);
        }
    }

    /// Returns retained bytes/resources and lifetime allocation/reuse counters.
    #[must_use]
    pub fn stats(&self) -> GpuResourcePoolStats {
        self.free.lock().map(|free| free.stats()).unwrap_or_default()
    }

    /// Acquires an uninitialized reusable storage buffer of the requested size.
    pub fn acquire_storage(
        &self,
        runtime: &WgpuRuntime,
        label: &'static str,
        byte_len: u64,
    ) -> wgpu::Buffer {
        self.take_storage(runtime, label, byte_len)
    }

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

    /// Returns a storage buffer after all users have submitted their work.
    ///
    /// Buffers with a mismatched size or incompatible usage are discarded.
    /// The caller must relinquish every other handle and use this pool only
    /// with the runtime/device that created the buffer.
    pub fn recycle(&self, byte_len: u64, buffer: wgpu::Buffer) {
        if let Ok(mut free) = self.free.lock() {
            if buffer.size() != byte_len || !buffer.usage().contains(storage_usage()) {
                free.record_discard();
                return;
            }
            free.put(byte_len, byte_len, buffer);
        }
    }

    /// Drops cached handles while preserving lifetime counters and the budget.
    /// Actual device memory release follows wgpu's outstanding-work lifetimes.
    pub fn clear(&self) {
        if let Ok(mut free) = self.free.lock() {
            free.clear();
        }
    }

    /// Returns the number of buffers currently held in the pool.
    #[must_use]
    pub fn cached_buffer_count(&self) -> usize {
        self.stats().cached_resources
    }

    fn take_storage(
        &self,
        runtime: &WgpuRuntime,
        label: &'static str,
        byte_len: u64,
    ) -> wgpu::Buffer {
        if let Ok(mut free) = self.free.lock() {
            if byte_len != 0 {
                if let Some(buffer) = free.take(&byte_len) {
                    return buffer;
                }
            }
            free.record_allocation();
        }
        runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: if byte_len == 0 { 4 } else { byte_len },
            usage: storage_usage(),
            mapped_at_creation: false,
        })
    }
}

fn storage_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
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

    #[test]
    fn pool_budget_rejects_incompatible_buffers_and_preserves_live_buffers() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        let pool = GpuBufferPool::with_budget(8);
        let live = pool.acquire_storage(&runtime, "live", 8);
        let extra = pool.acquire_storage(&runtime, "extra", 8);
        pool.recycle(extra.size(), extra);
        assert_eq!(pool.stats().cached_bytes, 8);
        let mismatched = pool.acquire_storage(&runtime, "mismatched", 4);
        pool.recycle(8, mismatched);
        let incompatible = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        pool.recycle(4, incompatible);
        assert_eq!(pool.stats().discarded_count, 2);
        pool.set_budget(0);
        assert_eq!(pool.stats().cached_bytes, 0);
        runtime.queue().write_buffer(&live, 0, bytemuck::cast_slice(&[1_u32, 2]));
        runtime.wait_idle();
        pool.recycle(live.size(), live);
        assert_eq!(pool.stats().cached_resources, 0);
        assert_eq!(pool.stats().allocation_count, 3);
        assert_eq!(pool.stats().discarded_count, 4);
    }
}
