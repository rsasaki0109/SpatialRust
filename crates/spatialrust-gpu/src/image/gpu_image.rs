//! Owned GPU image textures and named upload/readback.

use spatialrust_core::{SpatialError, SpatialResult};
use spatialrust_image::{Image, ImageMetadata, ImageView};

use crate::WgpuRuntime;

const BYTES_PER_TEXEL: u64 = 4;

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

    /// Returns physical bytes copied or written by device-side stages.
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
    device_key: usize,
    texture: wgpu::Texture,
    storage_bytes: u64,
    metadata: ImageMetadata,
    receipt: GpuImageReceipt,
}

impl GpuImage {
    /// Explicitly uploads a packed or strided `u8` image into a texture.
    pub fn upload_u8<const CHANNELS: usize>(
        runtime: &WgpuRuntime,
        view: ImageView<'_, u8, CHANNELS>,
    ) -> SpatialResult<Self> {
        if CHANNELS == 0 || CHANNELS > 4 {
            return Err(SpatialError::InvalidArgument(
                "GpuImage upload supports 1..=4 channels".to_owned(),
            ));
        }
        let width = u32::try_from(view.width())
            .map_err(|_| SpatialError::InvalidArgument("GpuImage width exceeds u32".to_owned()))?;
        let height = u32::try_from(view.height())
            .map_err(|_| SpatialError::InvalidArgument("GpuImage height exceeds u32".to_owned()))?;
        if width == 0 || height == 0 {
            return Err(SpatialError::InvalidArgument(
                "GpuImage upload requires positive width and height".to_owned(),
            ));
        }
        let texture = create_texture(runtime, width, height, "gpu-image-upload");
        let texels = pack_rgba8(view);
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
                bytes_per_row: Some(width * BYTES_PER_TEXEL as u32),
                rows_per_image: Some(height),
            },
            extent(width, height),
        );
        let storage_bytes = texture_bytes(width, height);
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
        let unpadded_row = self.width * BYTES_PER_TEXEL as u32;
        let padded_row = unpadded_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging_size = u64::from(padded_row) * u64::from(self.height);
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-image-texture-readback"),
            size: staging_size,
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
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            extent(self.width, self.height),
        );
        runtime.queue().submit(Some(encoder.finish()));
        let rgba = read_staging_bytes(runtime.device(), &staging, staging_size as usize)?;
        let mut data = Vec::with_capacity(self.width as usize * self.height as usize * CHANNELS);
        for row in rgba.chunks_exact(padded_row as usize).take(self.height as usize) {
            for texel in row[..unpadded_row as usize].chunks_exact(BYTES_PER_TEXEL as usize) {
                data.extend_from_slice(&texel[..CHANNELS]);
            }
        }
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
        if self.device_key != runtime_device_key(runtime) {
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
            storage_bytes: texture_bytes(width, height),
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

fn pack_rgba8<const CHANNELS: usize>(view: ImageView<'_, u8, CHANNELS>) -> Vec<u8> {
    let mut packed = vec![0_u8; view.width() * view.height() * BYTES_PER_TEXEL as usize];
    for y in 0..view.height() {
        let source = view.row(y).expect("input row in bounds");
        let target = &mut packed[y * view.width() * 4..(y + 1) * view.width() * 4];
        for (pixel, texel) in source.chunks_exact(CHANNELS).zip(target.chunks_exact_mut(4)) {
            texel[..CHANNELS].copy_from_slice(pixel);
        }
    }
    packed
}

const fn extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d { width, height, depth_or_array_layers: 1 }
}

const fn texture_bytes(width: u32, height: u32) -> u64 {
    width as u64 * height as u64 * BYTES_PER_TEXEL
}

pub(crate) fn runtime_device_key(runtime: &WgpuRuntime) -> usize {
    runtime.device() as *const wgpu::Device as usize
}

pub(crate) fn read_staging_bytes(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<u8>> {
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
    let data = slice.get_mapped_range();
    let values = data[..len].to_vec();
    drop(data);
    staging_buffer.unmap();
    Ok(values)
}
