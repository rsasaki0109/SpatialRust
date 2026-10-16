use spatialrust_image::ImageView;

// Transfer plans validate all dimensions and byte products before these helpers.
fn transfer_workers(width: usize, height: usize, channels: usize) -> usize {
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

fn write_rows(
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

pub(super) fn pack_rgba8<const CHANNELS: usize>(
    view: ImageView<'_, u8, CHANNELS>,
    packed_len: usize,
) -> Vec<u8> {
    pack_with_workers(view, packed_len, transfer_workers(view.width(), view.height(), CHANNELS))
}

fn pack_with_workers<const CHANNELS: usize>(
    view: ImageView<'_, u8, CHANNELS>,
    packed_len: usize,
    workers: usize,
) -> Vec<u8> {
    let mut packed = vec![0; packed_len];
    write_rows(&mut packed, view.width() * 4, view.height(), workers, |y, target| {
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

pub(super) fn unpack_rgba8<const CHANNELS: usize>(
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
    unpack_with_workers::<CHANNELS>(
        rgba,
        width,
        height,
        padded_row,
        output_len,
        transfer_workers(width, height, CHANNELS),
    )
}

fn unpack_with_workers<const CHANNELS: usize>(
    rgba: &[u8],
    width: usize,
    height: usize,
    padded_row: usize,
    output_len: usize,
    workers: usize,
) -> Vec<u8> {
    let mut data = vec![0; output_len];
    write_rows(&mut data, width * CHANNELS, height, workers, |y, target| {
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

#[cfg(test)]
#[path = "transfer_packing_tests.rs"]
mod tests;
