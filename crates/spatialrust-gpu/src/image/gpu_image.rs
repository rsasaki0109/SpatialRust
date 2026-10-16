//! Owned GPU image textures and named upload/readback.

use std::borrow::Cow;

use spatialrust_core::{SpatialError, SpatialResult};
use spatialrust_image::{Image, ImageMetadata, ImageView};

use super::transfer_packing::{pack_rgba8, unpack_rgba8};
use crate::WgpuRuntime;

const BYTES_PER_TEXEL: u64 = 4;

struct UploadPlan {
    width: u32,
    height: u32,
    row_bytes: u32,
    storage_bytes: u64,
    packed_len: usize,
}

struct ReadbackPlan {
    row_bytes: u32,
    padded_row: u32,
    staging_bytes: u64,
    staging_len: usize,
    output_len: usize,
}

fn invalid_transfer() -> SpatialError {
    SpatialError::InvalidArgument(
        "GpuImage transfer requires positive supported dimensions, 1..=4 channels and representable byte lengths".to_owned(),
    )
}

fn transfer_dimensions(
    width: u32,
    height: u32,
    channels: usize,
    limits: &wgpu::Limits,
) -> SpatialResult<u32> {
    if width == 0
        || height == 0
        || !(1..=4).contains(&channels)
        || width > limits.max_texture_dimension_2d
        || height > limits.max_texture_dimension_2d
    {
        return Err(invalid_transfer());
    }
    width.checked_mul(BYTES_PER_TEXEL as u32).ok_or_else(invalid_transfer)
}

fn host_length(bytes: u64) -> SpatialResult<usize> {
    if bytes > isize::MAX as u64 {
        return Err(invalid_transfer());
    }
    usize::try_from(bytes).map_err(|_| invalid_transfer())
}

fn upload_plan(
    width: usize,
    height: usize,
    channels: usize,
    limits: &wgpu::Limits,
) -> SpatialResult<UploadPlan> {
    let width = u32::try_from(width).map_err(|_| invalid_transfer())?;
    let height = u32::try_from(height).map_err(|_| invalid_transfer())?;
    let row_bytes = transfer_dimensions(width, height, channels, limits)?;
    let storage_bytes = texture_bytes(width, height)?;
    let packed_len = host_length(storage_bytes)?;
    Ok(UploadPlan { width, height, row_bytes, storage_bytes, packed_len })
}

fn upload_texels<const CHANNELS: usize>(
    view: ImageView<'_, u8, CHANNELS>,
    packed_len: usize,
) -> Cow<'_, [u8]> {
    if CHANNELS == 4 {
        if let Some(pixels) = view.as_packed_slice() {
            return Cow::Borrowed(pixels);
        }
    }
    Cow::Owned(pack_rgba8(view, packed_len))
}

fn readback_plan(
    width: u32,
    height: u32,
    channels: usize,
    limits: &wgpu::Limits,
) -> SpatialResult<ReadbackPlan> {
    let row_bytes = transfer_dimensions(width, height, channels, limits)?;
    let alignment = u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let padded_row = u32::try_from(u64::from(row_bytes).div_ceil(alignment) * alignment)
        .map_err(|_| invalid_transfer())?;
    let staging_bytes = u64::from(padded_row) * u64::from(height);
    if staging_bytes > limits.max_buffer_size {
        return Err(invalid_transfer());
    }
    let staging_len = host_length(staging_bytes)?;
    let output_len = host_length(u64::from(width) * u64::from(height) * channels as u64)?;
    Ok(ReadbackPlan { row_bytes, padded_row, staging_bytes, staging_len, output_len })
}

/// Transfer accounting for GPU image uploads, device copies, and readbacks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuImageReceipt {
    host_to_device_bytes: u64,
    gpu_to_gpu_bytes: u64,
    device_to_host_bytes: u64,
    stages: Vec<&'static str>,
}

impl GpuImageReceipt {
    /// Returns physical bytes explicitly uploaded from host memory.
    #[must_use]
    pub const fn host_to_device_bytes(&self) -> u64 {
        self.host_to_device_bytes
    }

    /// Returns copied bytes or destination storage footprints of device stages.
    /// Repeated in-place atomic updates and reads are not a bandwidth measurement.
    #[must_use]
    pub const fn gpu_to_gpu_bytes(&self) -> u64 {
        self.gpu_to_gpu_bytes
    }

    /// Returns physical bytes explicitly copied back to host memory.
    #[must_use]
    pub const fn device_to_host_bytes(&self) -> u64 {
        self.device_to_host_bytes
    }

    /// Returns the ordered logical transfer and kernel stage names.
    #[must_use]
    pub fn stages(&self) -> &[&'static str] {
        &self.stages
    }

    /// Verifies the transfer contract for a caller-uploaded resident chain.
    ///
    /// The expected byte count is the physical RGBA8 texture upload size.
    pub fn validate_resident_chain(&self, expected_upload_bytes: u64) -> SpatialResult<()> {
        if self.host_to_device_bytes != expected_upload_bytes {
            return Err(SpatialError::InvalidArgument(format!(
                "resident chain expected {expected_upload_bytes} host-to-device bytes, recorded {}",
                self.host_to_device_bytes
            )));
        }
        if self.device_to_host_bytes != 0 {
            return Err(SpatialError::InvalidArgument(format!(
                "resident chain forbids device-to-host transfers, recorded {} bytes",
                self.device_to_host_bytes
            )));
        }
        Ok(())
    }

    pub(crate) fn record_host_to_device(&mut self, bytes: u64, stage: &'static str) {
        self.host_to_device_bytes = self.host_to_device_bytes.saturating_add(bytes);
        self.stages.push(stage);
    }

    pub(crate) fn record_gpu_to_gpu(&mut self, bytes: u64, stage: &'static str) {
        self.gpu_to_gpu_bytes = self.gpu_to_gpu_bytes.saturating_add(bytes);
        self.stages.push(stage);
    }

    pub(crate) fn record_device_to_host(&mut self, bytes: u64, stage: &'static str) {
        self.device_to_host_bytes = self.device_to_host_bytes.saturating_add(bytes);
        self.stages.push(stage);
    }

    pub(crate) fn merge_from(&mut self, other: &Self) {
        self.host_to_device_bytes =
            self.host_to_device_bytes.saturating_add(other.host_to_device_bytes);
        self.gpu_to_gpu_bytes = self.gpu_to_gpu_bytes.saturating_add(other.gpu_to_gpu_bytes);
        self.device_to_host_bytes =
            self.device_to_host_bytes.saturating_add(other.device_to_host_bytes);
        self.stages.extend_from_slice(&other.stages);
    }
}

/// GPU-resident packed image backed by an `rgba8uint` 2D texture.
///
/// The logical channel count remains 1..=4. Unused texture components are zero,
/// so device storage is a predictable four bytes per pixel on every backend.
pub struct GpuImage {
    width: u32,
    height: u32,
    channels: u32,
    device_key: std::sync::Arc<()>,
    texture: wgpu::Texture,
    storage_bytes: u64,
    metadata: ImageMetadata,
    receipt: GpuImageReceipt,
}

impl GpuImage {
    /// Explicitly uploads a packed or strided `u8` image into a texture.
    /// Dimensions, row bytes and host capacity are checked before allocation.
    /// Packed four-channel pixels are borrowed during the queue write; other
    /// layouts use host packing. The backend still performs its own staging.
    pub fn upload_u8<const CHANNELS: usize>(
        runtime: &WgpuRuntime,
        view: ImageView<'_, u8, CHANNELS>,
    ) -> SpatialResult<Self> {
        let plan = upload_plan(view.width(), view.height(), CHANNELS, &runtime.device().limits())?;
        let (width, height) = (plan.width, plan.height);
        let texture = create_texture(runtime, width, height, "gpu-image-upload");
        let texels = upload_texels(view, plan.packed_len);
        runtime.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(plan.row_bytes),
                rows_per_image: Some(height),
            },
            extent(width, height),
        );
        let storage_bytes = plan.storage_bytes;
        let mut receipt = GpuImageReceipt::default();
        receipt.record_host_to_device(storage_bytes, "upload_u8_texture");
        Ok(Self {
            width,
            height,
            channels: CHANNELS as u32,
            device_key: runtime_device_key(runtime),
            texture,
            storage_bytes,
            metadata: view.metadata(),
            receipt,
        })
    }

    /// Explicitly reads the texture back into an owned packed host image.
    /// Row padding, staging limits and host capacity are checked before work.
    /// Pixels are unpacked directly from mapped staging into the owned output;
    /// the mapping is released before returning.
    pub fn readback_u8<const CHANNELS: usize>(
        &mut self,
        runtime: &WgpuRuntime,
    ) -> SpatialResult<Image<u8, CHANNELS>> {
        self.validate_runtime(runtime)?;
        if self.channels as usize != CHANNELS {
            return Err(SpatialError::InvalidArgument(format!(
                "GpuImage has {} channels but readback requested {CHANNELS}",
                self.channels
            )));
        }
        let plan = readback_plan(self.width, self.height, CHANNELS, &runtime.device().limits())?;
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-image-texture-readback"),
            size: plan.staging_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-image-texture-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(plan.padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            extent(self.width, self.height),
        );
        runtime.queue().submit(Some(encoder.finish()));
        let data = with_staging_bytes(runtime.device(), &staging, plan.staging_len, |rgba| {
            unpack_rgba8::<CHANNELS>(
                rgba,
                plan.row_bytes as usize / BYTES_PER_TEXEL as usize,
                self.height as usize,
                plan.padded_row as usize,
                plan.output_len,
            )
        })?;
        self.receipt.record_device_to_host(self.storage_bytes, "readback_u8_texture");
        Image::try_new_with_metadata(self.width as usize, self.height as usize, data, self.metadata)
            .map_err(|error| SpatialError::InvalidArgument(error.to_string()))
    }

    /// Returns image width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns image height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns the logical component count.
    #[must_use]
    pub const fn channels(&self) -> u32 {
        self.channels
    }

    /// Returns retained semantic metadata.
    #[must_use]
    pub const fn metadata(&self) -> ImageMetadata {
        self.metadata
    }

    /// Returns cumulative transfer and device-stage accounting.
    #[must_use]
    pub const fn receipt(&self) -> &GpuImageReceipt {
        &self.receipt
    }

    /// Drops texture storage after verifying the runtime ownership contract.
    pub fn recycle(self, runtime: &WgpuRuntime) {
        if self.validate_runtime(runtime).is_ok() {
            runtime.recycle_image_texture(self.width, self.height, self.texture);
        }
    }

    pub(crate) fn validate_runtime(&self, runtime: &WgpuRuntime) -> SpatialResult<()> {
        if !std::sync::Arc::ptr_eq(&self.device_key, &runtime_device_key(runtime)) {
            return Err(SpatialError::InvalidArgument(
                "GpuImage belongs to a different runtime device".to_owned(),
            ));
        }
        Ok(())
    }

    pub(crate) const fn storage_bytes(&self) -> u64 {
        self.storage_bytes
    }

    pub(crate) const fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub(crate) fn view(&self) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub(crate) fn from_parts(
        runtime: &WgpuRuntime,
        width: u32,
        height: u32,
        channels: u32,
        texture: wgpu::Texture,
        metadata: ImageMetadata,
        receipt: GpuImageReceipt,
    ) -> SpatialResult<Self> {
        if width == 0 || height == 0 || !(1..=4).contains(&channels) {
            return Err(SpatialError::InvalidArgument(
                "GpuImage dimensions/channels must be positive with 1..=4 channels".to_owned(),
            ));
        }
        Ok(Self {
            width,
            height,
            channels,
            device_key: runtime_device_key(runtime),
            texture,
            storage_bytes: texture_bytes(width, height)?,
            metadata,
            receipt,
        })
    }
}

pub(crate) fn create_texture(
    runtime: &WgpuRuntime,
    width: u32,
    height: u32,
    label: &'static str,
) -> wgpu::Texture {
    runtime.acquire_image_texture(width, height, label)
}

const fn extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d { width, height, depth_or_array_layers: 1 }
}

fn texture_bytes(width: u32, height: u32) -> SpatialResult<u64> {
    (u64::from(width) * u64::from(height)).checked_mul(BYTES_PER_TEXEL).ok_or_else(invalid_transfer)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "transfer_tests.rs"]
mod transfer_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "transfer_packing_gpu_tests.rs"]
mod transfer_packing_gpu_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "packed_upload_tests.rs"]
mod packed_upload_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "mapped_readback_tests.rs"]
mod mapped_readback_tests;

pub(crate) fn runtime_device_key(runtime: &WgpuRuntime) -> std::sync::Arc<()> {
    runtime.profiling_identity()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn read_staging_bytes(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<u8>> {
    with_staging_bytes(device, staging_buffer, len, <[u8]>::to_vec)
}

// Declared before the mapped view, so unwinding drops the view before unmapping.
struct StagingUnmap<'a>(&'a wgpu::Buffer);

impl Drop for StagingUnmap<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}

pub(super) fn with_staging_bytes<T>(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
    decode: impl FnOnce(&[u8]) -> T,
) -> SpatialResult<T> {
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|error| {
            SpatialError::InvalidArgument(format!("failed to map wgpu texture: {error}"))
        })?;
    let _unmap = StagingUnmap(staging_buffer);
    let data = slice.get_mapped_range();
    Ok(decode(&data[..len]))
}
