pub(super) fn pack_rgba8<const CHANNELS: usize>(
    view: ImageView<'_, u8, CHANNELS>,
    packed_len: usize,
) -> Vec<u8> {
    let mut packed = vec![0_u8; packed_len];
    for y in 0..view.height() {
        let source = view.row(y).expect("input row in bounds");
        let target = &mut packed[y * view.width() * 4..(y + 1) * view.width() * 4];
        for (pixel, texel) in source.chunks_exact(CHANNELS).zip(target.chunks_exact_mut(4)) {
            texel[..CHANNELS].copy_from_slice(pixel);
        }
    }
    packed
}

pub(super) fn unpack_rgba8<const CHANNELS: usize>(
    rgba: &[u8], width: usize, height: usize, padded_row: usize, output_len: usize,
) -> Vec<u8> {
    let mut data = Vec::with_capacity(output_len);
    for row in rgba.chunks_exact(padded_row).take(height) {
        for texel in row[..width * 4].chunks_exact(4) {
            data.extend_from_slice(&texel[..CHANNELS]);
        }
    }
    data
}
