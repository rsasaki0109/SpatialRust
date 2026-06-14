use std::collections::HashMap;
use std::sync::Mutex;

use spatialrust_core::SpatialResult;

use crate::runtime::WgpuRuntime;

/// Reusable GPU storage buffers keyed by byte length.
#[derive(Default)]
pub(crate) struct GpuUploadPool {
    free: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
}

impl GpuUploadPool {
    /// Uploads `f32` values into a pooled storage buffer.
    pub(crate) fn upload_f32_storage(
        &self,
        runtime: &WgpuRuntime,
        label: &'static str,
        data: &[f32],
    ) -> SpatialResult<wgpu::Buffer> {
        let byte_len = (data.len() * std::mem::size_of::<f32>()) as u64;
        let buffer = self.take_storage(runtime, label, byte_len);
        runtime
            .queue()
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        Ok(buffer)
    }

    /// Returns a storage buffer to the pool for reuse.
    pub(crate) fn recycle_storage(&self, byte_len: u64, buffer: wgpu::Buffer) {
        if byte_len == 0 {
            return;
        }
        if let Ok(mut free) = self.free.lock() {
            free.entry(byte_len).or_default().push(buffer);
        }
    }

    fn take_storage(&self, runtime: &WgpuRuntime, label: &'static str, byte_len: u64) -> wgpu::Buffer {
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
    use super::GpuUploadPool;
    use crate::runtime::WgpuRuntime;

    #[test]
    fn upload_pool_reuses_equal_sized_buffers() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let pool = GpuUploadPool::default();
        let data = [1.0_f32, 2.0, 3.0];

        let first = pool
            .upload_f32_storage(&runtime, "upload-pool-test", &data)
            .expect("first upload");
        pool.recycle_storage(first.size(), first);

        let second = pool
            .upload_f32_storage(&runtime, "upload-pool-test", &data)
            .expect("second upload");
        assert_eq!(second.size(), (data.len() * std::mem::size_of::<f32>()) as u64);
    }
}
