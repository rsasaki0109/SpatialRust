// Complete original readback and unpacking helpers from merged PR 251.
use super::super::*;

impl GpuImage {
    pub(super) fn baseline_contiguous_readback_u8<const CHANNELS: usize>(
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
            baseline_unpack_rgba8::<CHANNELS>(
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

fn baseline_transfer_workers(width: usize, height: usize, channels: usize) -> usize {
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

fn baseline_write_rows(
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

pub(super) fn baseline_unpack_rgba8<const CHANNELS: usize>(
    rgba: &[u8],
    width: usize,
    height: usize,
    padded_row: usize,
    output_len: usize,
) -> Vec<u8> {
    baseline_unpack_with_workers::<CHANNELS>(
        rgba,
        width,
        height,
        padded_row,
        output_len,
        baseline_transfer_workers(width, height, CHANNELS),
    )
}

fn baseline_unpack_with_workers<const CHANNELS: usize>(
    rgba: &[u8],
    width: usize,
    height: usize,
    padded_row: usize,
    output_len: usize,
    workers: usize,
) -> Vec<u8> {
    let mut data = vec![0; output_len];
    baseline_write_rows(&mut data, width * CHANNELS, height, workers, |y, target| {
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
