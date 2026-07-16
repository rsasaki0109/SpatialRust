use std::sync::{Arc, OnceLock};

#[cfg(feature = "gpu-image")]
use std::collections::HashMap;
#[cfg(feature = "gpu-image")]
use std::sync::Mutex;

use spatialrust_core::{SpatialError, SpatialResult};

use crate::pipeline_cache::ComputePipelineCache;
use crate::upload_cache::GpuBufferPool;

/// Headless wgpu runtime for compute-only workloads.
#[cfg(feature = "gpu-wgpu")]
pub struct WgpuRuntime {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: OnceLock<ComputePipelineCache>,
    max_gather_channels: u32,
    upload_pool: GpuBufferPool,
    adapter_info: WgpuAdapterInfo,
    #[cfg(feature = "gpu-image")]
    image_texture_pool: Mutex<HashMap<(u32, u32), Vec<wgpu::Texture>>>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_gray_pipeline: OnceLock<crate::image::kernels::gray::GrayPipeline>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_blur_pipeline: OnceLock<crate::image::kernels::box_blur::BlurPipeline>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_spatial_pipelines: OnceLock<crate::image::kernels::spatial::SpatialPipelines>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_ai_pack_pipeline: OnceLock<crate::image::ai_tensor::AiPackPipeline>,
}

/// Stable, serializable-friendly identity for the selected wgpu adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuAdapterInfo {
    /// Human-readable adapter name.
    pub name: String,
    /// Graphics API backend such as Vulkan, Metal, or DirectX 12.
    pub backend: String,
    /// Adapter class such as integrated, discrete, CPU, or virtual GPU.
    pub device_type: String,
    /// Driver name reported by wgpu.
    pub driver: String,
    /// Additional driver version/details reported by wgpu.
    pub driver_info: String,
}

/// Minimum storage buffers required for the 4-channel gather kernel.
#[cfg(feature = "gpu-wgpu")]
pub const MULTI_GATHER4_STORAGE_BUFFERS: u32 = 10;

/// Minimum storage buffers required for the 2-channel gather kernel.
#[cfg(feature = "gpu-wgpu")]
pub const MULTI_GATHER2_STORAGE_BUFFERS: u32 = 6;

#[cfg(feature = "gpu-wgpu")]
static SHARED_RUNTIME: OnceLock<Result<Arc<WgpuRuntime>, String>> = OnceLock::new();

#[cfg(feature = "gpu-wgpu")]
impl WgpuRuntime {
    /// Creates a headless wgpu runtime using the default adapter.
    ///
    /// Prefer [`Self::shared`] when running multiple GPU filters in one process.
    pub fn new_headless() -> SpatialResult<Self> {
        pollster::block_on(Self::new_headless_async())
    }

    /// Returns a process-wide shared headless runtime, initializing it on first use.
    pub fn shared() -> SpatialResult<Arc<Self>> {
        match SHARED_RUNTIME.get_or_init(init_shared_runtime) {
            Ok(runtime) => Ok(Arc::clone(runtime)),
            Err(message) => Err(SpatialError::InvalidArgument(message.clone())),
        }
    }

    /// Returns the underlying wgpu device.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns the underlying wgpu queue.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Returns the selected adapter/backend receipt.
    #[must_use]
    pub const fn adapter_info(&self) -> &WgpuAdapterInfo {
        &self.adapter_info
    }

    /// Blocks until all previously submitted work on this device is complete.
    ///
    /// Normal device-resident chains do not synchronize implicitly. This is an
    /// explicit profiling/testing boundary or host-coordination primitive.
    pub fn wait_idle(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    #[cfg(feature = "gpu-image")]
    pub(crate) fn acquire_image_texture(
        &self,
        width: u32,
        height: u32,
        label: &'static str,
    ) -> wgpu::Texture {
        if let Some(texture) = self
            .image_texture_pool
            .lock()
            .expect("image texture pool poisoned")
            .get_mut(&(width, height))
            .and_then(Vec::pop)
        {
            return texture;
        }
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Uint,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
    }

    #[cfg(feature = "gpu-image")]
    pub(crate) fn recycle_image_texture(&self, width: u32, height: u32, texture: wgpu::Texture) {
        let mut pool = self.image_texture_pool.lock().expect("image texture pool poisoned");
        let textures = pool.entry((width, height)).or_default();
        if textures.len() < 8 {
            textures.push(texture);
        } else {
            texture.destroy();
        }
    }

    /// Returns the number of image textures retained for steady-state reuse.
    #[cfg(feature = "gpu-image")]
    #[must_use]
    pub fn cached_image_texture_count(&self) -> usize {
        self.image_texture_pool.lock().map(|pool| pool.values().map(Vec::len).sum()).unwrap_or(0)
    }

    /// Returns the number of initialized GPU image pipeline families.
    #[cfg(feature = "gpu-image")]
    #[must_use]
    pub fn initialized_image_pipeline_count(&self) -> usize {
        usize::from(self.image_gray_pipeline.get().is_some())
            + usize::from(self.image_blur_pipeline.get().is_some())
            + usize::from(self.image_spatial_pipelines.get().is_some())
            + usize::from(self.image_ai_pack_pipeline.get().is_some())
    }

    /// Returns cached compute pipelines for this runtime's device.
    #[must_use]
    pub fn pipelines(&self) -> &ComputePipelineCache {
        self.pipelines.get_or_init(|| ComputePipelineCache::new(&self.device))
    }

    /// Returns the maximum attribute channels gatherable in one multi dispatch.
    #[must_use]
    pub fn max_gather_channels(&self) -> u32 {
        self.max_gather_channels
    }

    /// Returns the reusable storage-buffer pool owned by this runtime.
    #[must_use]
    pub fn buffer_pool(&self) -> &GpuBufferPool {
        &self.upload_pool
    }

    /// Uploads a POD slice into a reusable pooled storage buffer.
    pub fn upload_pod_storage<T: bytemuck::Pod>(
        &self,
        label: &'static str,
        data: &[T],
    ) -> SpatialResult<wgpu::Buffer> {
        self.upload_pool.upload_pod_storage(self, label, data)
    }

    /// Uploads `f32` values into a reusable pooled storage buffer.
    pub fn upload_f32_storage(
        &self,
        label: &'static str,
        data: &[f32],
    ) -> SpatialResult<wgpu::Buffer> {
        self.upload_pool.upload_f32_storage(self, label, data)
    }

    /// Uploads `u32` values into a reusable pooled storage buffer.
    pub fn upload_u32_storage(
        &self,
        label: &'static str,
        data: &[u32],
    ) -> SpatialResult<wgpu::Buffer> {
        self.upload_pool.upload_u32_storage(self, label, data)
    }

    /// Returns a storage buffer to the upload pool for reuse.
    pub fn recycle_storage(&self, byte_len: u64, buffer: wgpu::Buffer) {
        self.upload_pool.recycle(byte_len, buffer);
    }

    /// Clears all buffers cached in the upload pool.
    pub fn clear_buffer_pool(&self) {
        self.upload_pool.clear();
    }

    async fn new_headless_async() -> SpatialResult<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                SpatialError::InvalidArgument(
                    "no compatible wgpu adapter found for headless compute".to_owned(),
                )
            })?;

        let raw_adapter_info = adapter.get_info();
        let adapter_info = WgpuAdapterInfo {
            name: raw_adapter_info.name,
            backend: format!("{:?}", raw_adapter_info.backend),
            device_type: format!("{:?}", raw_adapter_info.device_type),
            driver: raw_adapter_info.driver,
            driver_info: raw_adapter_info.driver_info,
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("spatialrust-wgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|error| {
                SpatialError::InvalidArgument(format!("failed to create wgpu device: {error}"))
            })?;

        let max_gather_channels =
            max_gather_channels_for_limit(device.limits().max_storage_buffers_per_shader_stage);

        Ok(Self {
            _instance: instance,
            device,
            queue,
            pipelines: OnceLock::new(),
            max_gather_channels,
            upload_pool: GpuBufferPool::default(),
            adapter_info,
            #[cfg(feature = "gpu-image")]
            image_texture_pool: Mutex::new(HashMap::new()),
            #[cfg(feature = "gpu-image")]
            image_gray_pipeline: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_blur_pipeline: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_spatial_pipelines: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_ai_pack_pipeline: OnceLock::new(),
        })
    }
}

#[cfg(feature = "gpu-wgpu")]
fn max_gather_channels_for_limit(storage_buffers_per_stage: u32) -> u32 {
    if storage_buffers_per_stage >= MULTI_GATHER4_STORAGE_BUFFERS {
        4
    } else if storage_buffers_per_stage >= MULTI_GATHER2_STORAGE_BUFFERS {
        2
    } else {
        1
    }
}

#[cfg(feature = "gpu-wgpu")]
fn init_shared_runtime() -> Result<Arc<WgpuRuntime>, String> {
    WgpuRuntime::new_headless().map(Arc::new).map_err(|error| error.to_string())
}

#[cfg(all(feature = "gpu-wgpu", test))]
mod tests {
    use super::WgpuRuntime;
    use crate::pipeline_cache::ComputePipelineCache;
    use std::sync::Arc;

    #[test]
    fn shared_runtime_is_singleton() {
        let first = WgpuRuntime::shared().expect("shared runtime");
        let second = WgpuRuntime::shared().expect("shared runtime");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn shared_and_headless_use_same_device_type() {
        let shared = WgpuRuntime::shared().expect("shared runtime");
        let local = WgpuRuntime::new_headless().expect("local runtime");
        assert_eq!(
            shared.device().limits().max_storage_buffers_per_shader_stage,
            local.device().limits().max_storage_buffers_per_shader_stage
        );
    }

    #[test]
    fn pipeline_cache_is_initialized_once_per_runtime() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let first = runtime.pipelines() as *const ComputePipelineCache;
        let second = runtime.pipelines() as *const ComputePipelineCache;
        assert_eq!(first, second);
    }

    #[test]
    fn adapter_limits_enable_multi_channel_gather() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let limit = runtime.device().limits().max_storage_buffers_per_shader_stage;
        assert!(
            limit >= super::MULTI_GATHER2_STORAGE_BUFFERS,
            "expected at least {} storage buffers per stage, got {limit}",
            super::MULTI_GATHER2_STORAGE_BUFFERS
        );
        assert!(runtime.max_gather_channels() >= 2);
        assert_eq!(
            runtime.max_gather_channels(),
            runtime.pipelines().voxel_gather.multi_max_channels
        );
    }
}
