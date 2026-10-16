// Complete original GPU image transfers and row packing from merged PR 252.
use super::super::*;

impl GpuImage {
    pub(super) fn row_parent_upload_u8<const CHANNELS: usize>(
        runtime: &WgpuRuntime,
        view: ImageView<'_, u8, CHANNELS>,
    ) -> SpatialResult<Self> {
        let plan = upload_plan(view.width(), view.height(), CHANNELS, &runtime.device().limits())?;
        let (width, height) = (plan.width, plan.height);
        let texture = create_texture(runtime, width, height, "gpu-image-upload");
        let texels = row_parent_upload_texels(view, plan.packed_len);
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

    pub(super) fn row_parent_readback_u8<const CHANNELS: usize>(
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
            row_parent_unpack_rgba8::<CHANNELS>(
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
}

fn row_parent_upload_texels<const CHANNELS: usize>(
    view: ImageView<'_, u8, CHANNELS>,
    packed_len: usize,
) -> Cow<'_, [u8]> {
    if CHANNELS == 4 {
        if let Some(pixels) = view.as_packed_slice() {
            return Cow::Borrowed(pixels);
        }
    }
    Cow::Owned(row_parent_pack_rgba8(view, packed_len))
}

fn row_parent_transfer_workers(width: usize, height: usize, channels: usize) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    if channels < 4 && width * height >= 1_048_576 {
        return std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min((width * height).div_ceil(262_144))
            .min(height)
            .max(1);
    }
    let _ = (width, height, channels);
    1
}

fn row_parent_write_rows(
    output: &mut [u8],
    row_bytes: usize,
    height: usize,
    workers: usize,
    fill: impl Fn(usize, &mut [u8]) + Sync,
) {
    if output.is_empty() {
        return;
    }
    let workers = if cfg!(target_arch = "wasm32") { 1 } else { workers.max(1).min(height) };
    if workers <= 1 {
        for (y, row) in output.chunks_exact_mut(row_bytes).enumerate() {
            fill(y, row);
        }
        return;
    }
    let rows_per_worker = height.div_ceil(workers);
    std::thread::scope(|scope| {
        let fill = &fill;
        for (band, chunk) in output.chunks_mut(rows_per_worker * row_bytes).enumerate() {
            scope.spawn(move || {
                for (local, row) in chunk.chunks_exact_mut(row_bytes).enumerate() {
                    fill(band * rows_per_worker + local, row);
                }
            });
        }
    });
}

pub(super) fn row_parent_pack_rgba8<const CHANNELS: usize>(
    view: ImageView<'_, u8, CHANNELS>,
    packed_len: usize,
) -> Vec<u8> {
    row_parent_pack_with_workers(
        view,
        packed_len,
        row_parent_transfer_workers(view.width(), view.height(), CHANNELS),
    )
}

fn row_parent_pack_with_workers<const CHANNELS: usize>(
    view: ImageView<'_, u8, CHANNELS>,
    packed_len: usize,
    workers: usize,
) -> Vec<u8> {
    let mut packed = vec![0; packed_len];
    row_parent_write_rows(&mut packed, view.width() * 4, view.height(), workers, |y, target| {
        let source = view.row(y).expect("input row in bounds");
        if CHANNELS == 4 {
            target.copy_from_slice(source);
        } else {
            for (pixel, texel) in source.chunks_exact(CHANNELS).zip(target.chunks_exact_mut(4)) {
                texel[..CHANNELS].copy_from_slice(pixel);
            }
        }
    });
    packed
}

pub(super) fn row_parent_unpack_rgba8<const CHANNELS: usize>(
    rgba: &[u8],
    width: usize,
    height: usize,
    padded_row: usize,
    output_len: usize,
) -> Vec<u8> {
    if CHANNELS == 4 && padded_row == width * 4 {
        // Validated transfer plans make all logical pixels one contiguous span.
        return rgba[..output_len].to_vec();
    }
    row_parent_unpack_with_workers::<CHANNELS>(
        rgba,
        width,
        height,
        padded_row,
        output_len,
        row_parent_transfer_workers(width, height, CHANNELS),
    )
}

fn row_parent_unpack_with_workers<const CHANNELS: usize>(
    rgba: &[u8],
    width: usize,
    height: usize,
    padded_row: usize,
    output_len: usize,
    workers: usize,
) -> Vec<u8> {
    let mut data = vec![0; output_len];
    row_parent_write_rows(&mut data, width * CHANNELS, height, workers, |y, target| {
        let source = &rgba[y * padded_row..y * padded_row + width * 4];
        if CHANNELS == 4 {
            target.copy_from_slice(source);
        } else {
            for (texel, pixel) in source.chunks_exact(4).zip(target.chunks_exact_mut(CHANNELS)) {
                pixel.copy_from_slice(&texel[..CHANNELS]);
            }
        }
    });
    data
}

// Deferred safe row-copy candidate; production retains the immediate-parent path.
impl GpuImage {
    pub(super) fn row_candidate_upload_u8<const CHANNELS: usize>(
        runtime: &WgpuRuntime,
        view: ImageView<'_, u8, CHANNELS>,
    ) -> SpatialResult<Self> {
        let plan = upload_plan(view.width(), view.height(), CHANNELS, &runtime.device().limits())?;
        let (width, height) = (plan.width, plan.height);
        let texture = create_texture(runtime, width, height, "gpu-image-upload");
        let texels = row_candidate_upload_texels(view, plan.packed_len);
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

    pub(super) fn row_candidate_readback_u8<const CHANNELS: usize>(
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
            row_candidate_unpack_rgba8::<CHANNELS>(
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
}

pub(super) fn row_candidate_upload_texels<const CHANNELS: usize>(
    view: ImageView<'_, u8, CHANNELS>,
    packed_len: usize,
) -> Cow<'_, [u8]> {
    if CHANNELS == 4 {
        if let Some(pixels) = view.as_packed_slice() {
            return Cow::Borrowed(pixels);
        }
    }
    Cow::Owned(row_candidate_pack_rgba8(view, packed_len))
}

pub(super) fn row_candidate_pack_rgba8<const CHANNELS: usize>(
    view: ImageView<'_, u8, CHANNELS>,
    packed_len: usize,
) -> Vec<u8> {
    if CHANNELS == 4 {
        let mut packed = Vec::with_capacity(packed_len);
        for y in 0..view.height() {
            packed.extend_from_slice(view.row(y).expect("input row in bounds"));
        }
        return packed;
    }
    row_parent_pack_with_workers(
        view,
        packed_len,
        row_parent_transfer_workers(view.width(), view.height(), CHANNELS),
    )
}

pub(super) fn row_candidate_unpack_rgba8<const CHANNELS: usize>(
    rgba: &[u8],
    width: usize,
    height: usize,
    padded_row: usize,
    output_len: usize,
) -> Vec<u8> {
    if CHANNELS == 4 && padded_row == width * 4 {
        // Validated transfer plans make all logical pixels one contiguous span.
        return rgba[..output_len].to_vec();
    }
    if CHANNELS == 4 {
        let mut data = Vec::with_capacity(output_len);
        for y in 0..height {
            data.extend_from_slice(&rgba[y * padded_row..y * padded_row + width * 4]);
        }
        return data;
    }
    row_parent_unpack_with_workers::<CHANNELS>(
        rgba,
        width,
        height,
        padded_row,
        output_len,
        row_parent_transfer_workers(width, height, CHANNELS),
    )
}
