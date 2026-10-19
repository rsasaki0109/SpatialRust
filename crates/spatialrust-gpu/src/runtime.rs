#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::sync::OnceLock;

#[cfg(feature = "gpu-image")]
use std::sync::Mutex;

use spatialrust_core::{DeviceKind, SpatialError, SpatialResult, SpatialRuntime};

use crate::pipeline_cache::ComputePipelineCache;
use crate::resource_cache::GpuResourcePoolStats;
#[cfg(feature = "gpu-image")]
use crate::resource_cache::{ResourceCache, DEFAULT_GPU_POOL_BUDGET_BYTES};
use crate::upload_cache::GpuBufferPool;

/// Headless wgpu runtime for compute-only workloads.
#[cfg(feature = "gpu-wgpu")]
pub struct WgpuRuntime {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: OnceLock<ComputePipelineCache>,
    pub(crate) voxel_scan_pipelines: OnceLock<crate::kernels::PrefixScanPipelines>,
    pub(crate) ransac_plane_pipelines: OnceLock<crate::kernels::RansacPlanePipelines>,
    pub(crate) cluster_pipelines: OnceLock<crate::kernels::cluster_union::ClusterPipelines>,
    pub(crate) nearest_pipeline: OnceLock<crate::kernels::nearest::NearestPipeline>,
    pub(crate) tree_gather_pipeline:
        OnceLock<crate::kernels::nearest::tree_gpu_gather::TreeGatherPipeline>,
    pub(crate) neighborhood_pipelines: OnceLock<crate::kernels::nearest::NeighborhoodPipelines>,
    pub(crate) fpfh_pipelines: OnceLock<crate::kernels::fpfh::FpfhPipelines>,
    pub(crate) fpfh_sample_pipeline:
        OnceLock<crate::kernels::fpfh::sample_gather::FpfhSamplePipeline>,
    pub(crate) iss_pipelines: OnceLock<crate::kernels::iss::IssPipelines>,
    pub(crate) mls_pipelines: OnceLock<crate::kernels::mls::MlsPipelines>,
    pub(crate) icp_reduction_pipelines:
        OnceLock<crate::kernels::icp_reduction::IcpReductionPipelines>,
    pub(crate) icp_trim_pipelines: OnceLock<crate::kernels::icp_reduction::trim::TrimPipelines>,
    pub(crate) point_to_plane_reduction_pipelines:
        OnceLock<crate::kernels::point_to_plane_reduction::PointToPlaneReductionPipelines>,
    pub(crate) gicp_covariance_pipelines:
        OnceLock<crate::kernels::gicp_reduction::resident_covariances::GicpCovariancePipelines>,
    pub(crate) gicp_grid_covariance_pipelines:
        OnceLock<crate::kernels::gicp_reduction::resident_grid_covariances::GridPipelines>,
    pub(crate) gicp_reduction_pipelines:
        OnceLock<crate::kernels::gicp_reduction::GicpReductionPipelines>,
    pub(crate) point_estimator_pipelines:
        OnceLock<crate::kernels::point_estimators::PointEstimatorPipelines>,
    max_gather_channels: u32,
    upload_pool: GpuBufferPool,
    pub(crate) scratch_pool: GpuBufferPool,
    adapter_info: WgpuAdapterInfo,
    profiling_identity: std::sync::Arc<()>,
    #[cfg(feature = "gpu-image")]
    image_texture_pool: Mutex<ResourceCache<(u32, u32), wgpu::Texture>>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_gray_pipeline: OnceLock<crate::image::kernels::gray::GrayPipeline>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_blur_pipeline: OnceLock<crate::image::kernels::box_blur::BlurPipeline>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_gaussian_pipelines:
        OnceLock<crate::image::kernels::gaussian::GaussianPipelines>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_bilateral_pipeline:
        OnceLock<crate::image::kernels::bilateral::BilateralPipeline>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_median_pipelines: OnceLock<crate::image::kernels::median::MedianPipelines>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_canny_pipelines: OnceLock<crate::image::canny::CannyPipelines>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_spatial_pipelines: OnceLock<crate::image::kernels::spatial::SpatialPipelines>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_ai_pack_pipeline: OnceLock<crate::image::ai_tensor::AiPackPipeline>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_histogram_pipelines: OnceLock<crate::image::histogram::HistogramPipelines>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_cdf_pipeline: OnceLock<crate::image::histogram::CdfPipeline>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_equalize_pipelines: OnceLock<crate::image::equalize::EqualizePipelines>,
    #[cfg(feature = "gpu-image")]
    pub(crate) image_clahe_pipelines: OnceLock<crate::image::clahe::ClahePipelines>,
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

/// Adapter power preference used when creating a headless wgpu runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WgpuPowerPreference {
    /// Prefer a discrete or otherwise high-performance adapter.
    #[default]
    HighPerformance,
    /// Prefer an integrated or otherwise power-efficient adapter.
    LowPower,
}

/// Explicit backend set for a headless runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum WgpuBackendSelection {
    /// The original primary backend set used by existing constructors.
    #[default]
    Primary,
    /// GL only, requiring platform support for OpenGL or OpenGLES compute.
    Gl,
}

impl WgpuBackendSelection {
    /// Stable requested-backend label for execution receipts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Gl => "gl",
        }
    }

    const fn backends(self) -> wgpu::Backends {
        match self {
            Self::Primary => wgpu::Backends::PRIMARY,
            Self::Gl => wgpu::Backends::GL,
        }
    }
}

/// Explicit headless selection and optional profiling requests.
///
/// Defaults preserve existing primary/high-performance/no-timestamp behavior.
/// Options constructors create an owned runtime independently of [`WgpuRuntime::shared`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct WgpuRuntimeOptions {
    /// Requested backend set; GL is an explicit opt-in without backend fallback.
    pub backend: WgpuBackendSelection,
    /// Adapter power preference within the selected backend set.
    pub power_preference: WgpuPowerPreference,
    /// Request timestamp queries when available; unsupported adapters still initialize.
    pub request_timestamps: bool,
}

/// Minimum storage buffers required for the 4-channel gather kernel.
#[cfg(feature = "gpu-wgpu")]
pub const MULTI_GATHER4_STORAGE_BUFFERS: u32 = 10;

/// Minimum storage buffers required for the 2-channel gather kernel.
#[cfg(feature = "gpu-wgpu")]
pub const MULTI_GATHER2_STORAGE_BUFFERS: u32 = 6;

#[cfg(feature = "gpu-wgpu")]
#[cfg(not(target_arch = "wasm32"))]
static SHARED_RUNTIME: OnceLock<Result<Arc<WgpuRuntime>, String>> = OnceLock::new();

#[cfg(feature = "gpu-wgpu")]
impl WgpuRuntime {
    /// Creates a headless wgpu runtime preferring a high-performance adapter.
    ///
    /// Prefer [`Self::shared`] when running multiple GPU filters in one process.
    pub fn new_headless() -> SpatialResult<Self> {
        Self::new_headless_with_preference(WgpuPowerPreference::HighPerformance)
    }

    /// Creates a headless wgpu runtime with an explicit adapter power preference.
    pub fn new_headless_with_preference(preference: WgpuPowerPreference) -> SpatialResult<Self> {
        pollster::block_on(Self::new_headless_async(preference))
    }

    /// Creates a runtime enabling pass timestamps when supported by the adapter.
    ///
    /// Unsupported adapters still initialize; check [`Self::timestamp_queries_enabled`]
    /// before constructing a profiler. The default constructors request no timestamps.
    pub fn new_headless_with_timestamps(preference: WgpuPowerPreference) -> SpatialResult<Self> {
        pollster::block_on(Self::new_headless_with_timestamps_async(preference))
    }

    /// Creates an owned runtime using explicit backend and profiling options.
    ///
    /// GL support depends on the build/platform/adapter. Admission errors do not
    /// select another backend. Browser callers should use the async counterpart.
    pub fn new_headless_with_options(options: WgpuRuntimeOptions) -> SpatialResult<Self> {
        pollster::block_on(Self::new_headless_with_options_async(options))
    }

    /// Returns whether this device has optional pass timestamp queries enabled.
    #[must_use]
    pub fn timestamp_queries_enabled(&self) -> bool {
        self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    pub(crate) fn profiling_identity(&self) -> std::sync::Arc<()> {
        std::sync::Arc::clone(&self.profiling_identity)
    }

    /// Returns a process-wide shared headless runtime, initializing it on first use.
    #[cfg(not(target_arch = "wasm32"))]
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
        {
            let mut pool = self.image_texture_pool.lock().expect("image texture pool poisoned");
            if let Some(texture) = pool.take(&(width, height)) {
                return texture;
            }
            pool.record_allocation();
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
        pool.put((width, height), u64::from(width) * u64::from(height) * 4, texture);
    }

    /// Returns the number of image textures retained for steady-state reuse.
    #[cfg(feature = "gpu-image")]
    #[must_use]
    pub fn cached_image_texture_count(&self) -> usize {
        self.image_texture_pool_stats().cached_resources
    }

    /// Returns image-texture retention and lifetime allocation/reuse counters.
    /// Texel bytes exclude driver overhead and live images; the default is 64 MiB.
    #[cfg(feature = "gpu-image")]
    #[must_use]
    pub fn image_texture_pool_stats(&self) -> GpuResourcePoolStats {
        self.image_texture_pool.lock().map(|pool| pool.stats()).unwrap_or_default()
    }

    /// Changes retained RGBA8 texel bytes; zero disables texture retention.
    /// Shrinking below current retention clears cached textures. Active images
    /// remain valid and allocations are not capped by the retention budget.
    #[cfg(feature = "gpu-image")]
    pub fn set_image_texture_pool_budget(&self, budget_bytes: u64) {
        if let Ok(mut pool) = self.image_texture_pool.lock() {
            pool.set_budget(budget_bytes);
        }
    }

    /// Drops cached texture handles without affecting live images or counters.
    #[cfg(feature = "gpu-image")]
    pub fn clear_image_texture_pool(&self) {
        if let Ok(mut pool) = self.image_texture_pool.lock() {
            pool.clear();
        }
    }

    /// Returns the number of initialized GPU image pipeline families.
    #[cfg(feature = "gpu-image")]
    #[must_use]
    pub fn initialized_image_pipeline_count(&self) -> usize {
        usize::from(self.image_gray_pipeline.get().is_some())
            + usize::from(self.image_blur_pipeline.get().is_some())
            + usize::from(self.image_gaussian_pipelines.get().is_some())
            + usize::from(self.image_median_pipelines.get().is_some())
            + usize::from(self.image_bilateral_pipeline.get().is_some())
            + usize::from(self.image_canny_pipelines.get().is_some())
            + usize::from(self.image_spatial_pipelines.get().is_some())
            + usize::from(self.image_ai_pack_pipeline.get().is_some())
            + usize::from(self.image_histogram_pipelines.get().is_some())
            + usize::from(self.image_cdf_pipeline.get().is_some())
            + usize::from(self.image_equalize_pipelines.get().is_some())
            + usize::from(self.image_clahe_pipelines.get().is_some())
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

    /// Returns private point-kernel scratch retention and lifetime counters.
    /// The default budget is 64 MiB, independent of the public upload pool.
    #[must_use]
    pub fn scratch_buffer_pool_stats(&self) -> GpuResourcePoolStats {
        self.scratch_pool.stats()
    }

    /// Changes private scratch retention; zero disables reuse. Shrinking below
    /// current retention clears cached buffers without affecting submitted work.
    pub fn set_scratch_buffer_pool_budget(&self, budget_bytes: u64) {
        self.scratch_pool.set_budget(budget_bytes);
    }

    /// Drops cached scratch handles without changing the budget or counters.
    pub fn clear_scratch_buffer_pool(&self) {
        self.scratch_pool.clear();
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

    /// Creates a headless runtime asynchronously.
    ///
    /// This is the browser/WebGPU construction path because blocking a WASM
    /// main thread is not supported. Native callers may also use it from their
    /// async executor.
    pub async fn new_headless_async(preference: WgpuPowerPreference) -> SpatialResult<Self> {
        Self::new_headless_impl(WgpuRuntimeOptions {
            power_preference: preference,
            ..Default::default()
        })
        .await
    }

    /// Async counterpart of [`Self::new_headless_with_timestamps`].
    pub async fn new_headless_with_timestamps_async(
        preference: WgpuPowerPreference,
    ) -> SpatialResult<Self> {
        Self::new_headless_impl(WgpuRuntimeOptions {
            power_preference: preference,
            request_timestamps: true,
            ..Default::default()
        })
        .await
    }

    /// Async counterpart of [`Self::new_headless_with_options`].
    pub async fn new_headless_with_options_async(
        options: WgpuRuntimeOptions,
    ) -> SpatialResult<Self> {
        Self::new_headless_impl(options).await
    }

    async fn new_headless_impl(options: WgpuRuntimeOptions) -> SpatialResult<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: options.backend.backends(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: match options.power_preference {
                    WgpuPowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
                    WgpuPowerPreference::LowPower => wgpu::PowerPreference::LowPower,
                },
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
                    required_features: timestamp_features(
                        options.request_timestamps,
                        adapter.features(),
                    ),
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
            voxel_scan_pipelines: OnceLock::new(),
            ransac_plane_pipelines: OnceLock::new(),
            cluster_pipelines: OnceLock::new(),
            nearest_pipeline: OnceLock::new(),
            tree_gather_pipeline: OnceLock::new(),
            neighborhood_pipelines: OnceLock::new(),
            fpfh_pipelines: OnceLock::new(),
            fpfh_sample_pipeline: OnceLock::new(),
            iss_pipelines: OnceLock::new(),
            mls_pipelines: OnceLock::new(),
            icp_reduction_pipelines: OnceLock::new(),
            icp_trim_pipelines: OnceLock::new(),
            gicp_reduction_pipelines: OnceLock::new(),
            gicp_covariance_pipelines: OnceLock::new(),
            gicp_grid_covariance_pipelines: OnceLock::new(),
            point_to_plane_reduction_pipelines: OnceLock::new(),
            point_estimator_pipelines: OnceLock::new(),
            max_gather_channels,
            upload_pool: GpuBufferPool::default(),
            scratch_pool: GpuBufferPool::default(),
            adapter_info,
            profiling_identity: std::sync::Arc::new(()),
            #[cfg(feature = "gpu-image")]
            image_texture_pool: Mutex::new(ResourceCache::new(DEFAULT_GPU_POOL_BUDGET_BYTES)),
            #[cfg(feature = "gpu-image")]
            image_gray_pipeline: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_blur_pipeline: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_gaussian_pipelines: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_bilateral_pipeline: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_median_pipelines: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_canny_pipelines: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_spatial_pipelines: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_ai_pack_pipeline: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_histogram_pipelines: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_cdf_pipeline: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_equalize_pipelines: OnceLock::new(),
            #[cfg(feature = "gpu-image")]
            image_clahe_pipelines: OnceLock::new(),
        })
    }
}

#[cfg(feature = "gpu-wgpu")]
impl SpatialRuntime for WgpuRuntime {
    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Wgpu
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

fn timestamp_features(requested: bool, available: wgpu::Features) -> wgpu::Features {
    if requested {
        available & wgpu::Features::TIMESTAMP_QUERY
    } else {
        wgpu::Features::empty()
    }
}

#[cfg(feature = "gpu-wgpu")]
#[cfg(not(target_arch = "wasm32"))]
fn init_shared_runtime() -> Result<Arc<WgpuRuntime>, String> {
    WgpuRuntime::new_headless().map(Arc::new).map_err(|error| error.to_string())
}

#[cfg(all(feature = "gpu-wgpu", test, not(target_arch = "wasm32")))]
mod tests {
    use super::WgpuRuntime;
    use crate::pipeline_cache::ComputePipelineCache;
    use std::sync::Arc;

    #[test]
    fn runtime_options_preserve_existing_constructor_policy() {
        let options = super::WgpuRuntimeOptions::default();
        assert_eq!(options.backend, super::WgpuBackendSelection::Primary);
        assert_eq!(options.power_preference, super::WgpuPowerPreference::HighPerformance);
        assert!(!options.request_timestamps);
    }

    #[test]
    fn backend_selection_is_explicit_without_cross_backend_fallback() {
        let primary = super::WgpuBackendSelection::Primary;
        let gl = super::WgpuBackendSelection::Gl;
        assert_eq!(primary.backends(), wgpu::Backends::PRIMARY);
        assert_eq!(gl.backends(), wgpu::Backends::GL);
        assert!(!primary.backends().intersects(gl.backends()));
        assert_eq!((primary.as_str(), gl.as_str()), ("primary", "gl"));
    }

    #[test]
    fn optional_timestamp_features_handle_unsupported_adapters() {
        let unrelated = wgpu::Features::INDIRECT_FIRST_INSTANCE;
        let supported = unrelated | wgpu::Features::TIMESTAMP_QUERY;
        assert_eq!(super::timestamp_features(false, supported), wgpu::Features::empty());
        assert_eq!(super::timestamp_features(true, unrelated), wgpu::Features::empty());
        assert_eq!(super::timestamp_features(true, supported), wgpu::Features::TIMESTAMP_QUERY);
    }

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
