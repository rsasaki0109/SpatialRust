//! Owned GPU image buffers and named upload/readback.

use spatialrust_core::{SpatialError, SpatialResult};
use spatialrust_image::{Image, ImageMetadata, ImageView};

use crate::WgpuRuntime;

/// Transfer accounting for GPU image uploads, device copies, and readbacks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuImageReceipt {
    host_to_device_bytes: u64,
    gpu_to_gpu_bytes: u64,
    device_to_host_bytes: u64,
    stages: Vec<&'static str>,
}

impl GpuImageReceipt {
    /// Returns bytes uploaded from host memory.
    #[must_use]
    pub const fn host_to_device_bytes(&self) -> u64 {
        self.host_to_device_bytes
    }

    /// Returns bytes copied between GPU buffers.
    #[must_use]
    pub const fn gpu_to_gpu_bytes(&self) -> u64 {
        self.gpu_to_gpu_bytes
    }

    /// Returns bytes explicitly read back to host memory.
    #[must_use]
    pub const fn device_to_host_bytes(&self) -> u64 {
        self.device_to_host_bytes
    }

    /// Returns recorded logical stages.
    #[must_use]
    pub fn stages(&self) -> &[&'static str] {
        &self.stages
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

/// GPU-resident packed interleaved image (`u32` storage per component).
pub struct GpuImage {
    width: u32,
    height: u32,
    channels: u32,
    device_key: usize,
    buffer: wgpu::Buffer,
    storage_bytes: u64,
    metadata: ImageMetadata,
    receipt: GpuImageReceipt,
}

impl GpuImage {
    /// Uploads a packed or strided `u8` image view into a new GPU image.
    ///
    /// Strided views are packed on the host first; those host bytes are counted
    /// in the receipt together with the device upload.
    pub fn upload_u8<const CHANNELS: usize>(
        runtime: &WgpuRuntime,
        view: ImageView<'_, u8, CHANNELS>,
    ) -> SpatialResult<Self> {
        if CHANNELS == 0 || CHANNELS > 4 {
            return Err(SpatialError::InvalidArgument(
                "GpuImage upload supports 1..=4 channels".to_owned(),
            ));
        }
        if view.width() == 0 || view.height() == 0 {
            return Err(SpatialError::InvalidArgument(
                "GpuImage upload requires positive width and height".to_owned(),
            ));
        }
        let packed = pack_u8_view(view);
        let words = packed.iter().map(|&value| u32::from(value)).collect::<Vec<_>>();
        let buffer = runtime.upload_u32_storage("gpu-image-upload", &words)?;
        let storage_bytes = (words.len() * std::mem::size_of::<u32>()) as u64;
        let mut receipt = GpuImageReceipt::default();
        receipt.record_host_to_device(storage_bytes, "upload_u8");
        Ok(Self {
            width: view.width() as u32,
            height: view.height() as u32,
            channels: CHANNELS as u32,
            device_key: runtime_device_key(runtime),
            buffer,
            storage_bytes,
            metadata: view.metadata(),
            receipt,
        })
    }

    /// Reads the image back into an owned packed host `Image`.
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
        let word_count = self.element_count();
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-image-readback"),
            size: self.storage_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-image-readback-encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.storage_bytes);
        runtime.queue().submit(Some(encoder.finish()));
        let words = read_staging_u32(runtime.device(), &staging, word_count)?;
        let data = words
            .into_iter()
            .map(|value| u8::try_from(value.min(255)).unwrap_or(255))
            .collect::<Vec<_>>();
        self.receipt.record_device_to_host(self.storage_bytes, "readback_u8");
        Image::try_new_with_metadata(
            self.width as usize,
            self.height as usize,
            data,
            self.metadata,
        )
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

    /// Returns channel count.
    #[must_use]
    pub const fn channels(&self) -> u32 {
        self.channels
    }

    /// Returns retained semantic metadata.
    #[must_use]
    pub const fn metadata(&self) -> ImageMetadata {
        self.metadata
    }

    /// Returns transfer accounting for this image's lifetime so far.
    #[must_use]
    pub const fn receipt(&self) -> &GpuImageReceipt {
        &self.receipt
    }

    /// Returns mutable transfer accounting.
    pub fn receipt_mut(&mut self) -> &mut GpuImageReceipt {
        &mut self.receipt
    }

    /// Recycles the storage buffer into the runtime pool.
    pub fn recycle(self, runtime: &WgpuRuntime) {
        if self.validate_runtime(runtime).is_ok() {
            runtime.recycle_storage(self.storage_bytes, self.buffer);
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

    pub(crate) fn element_count(&self) -> usize {
        (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(self.channels as usize)
    }

    pub(crate) fn storage_bytes(&self) -> u64 {
        self.storage_bytes
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(crate) fn from_parts(
        runtime: &WgpuRuntime,
        width: u32,
        height: u32,
        channels: u32,
        buffer: wgpu::Buffer,
        storage_bytes: u64,
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
            buffer,
            storage_bytes,
            metadata,
            receipt,
        })
    }
}

fn pack_u8_view<const CHANNELS: usize>(view: ImageView<'_, u8, CHANNELS>) -> Vec<u8> {
    let mut packed = Vec::with_capacity(view.width() * view.height() * CHANNELS);
    for y in 0..view.height() {
        for x in 0..view.width() {
            packed.extend_from_slice(view.get(x, y).expect("in-bounds"));
        }
    }
    packed
}

pub(crate) fn runtime_device_key(runtime: &WgpuRuntime) -> usize {
    runtime.device() as *const wgpu::Device as usize
}

pub(crate) fn read_staging_u32(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<u32>> {
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
            SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {error}"))
        })?;
    let data = slice.get_mapped_range();
    let values: Vec<u32> = bytemuck::cast_slice(&data)[..len].to_vec();
    drop(data);
    staging_buffer.unmap();
    Ok(values)
}
