use super::*;
use crate::WgpuRuntime;
use spatialrust_image::{ColorRange, Image, ImageMetadata};

fn nearest<const C: usize>(image: &Image<u8, C>, w: usize, h: usize) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let x = (i % w) * image.width() / w;
            let y = (i / w) * image.height() / h;
            image.as_slice()[(y * image.width() + x) * C..][..C].iter().copied()
        })
        .collect()
}

fn sobel(values: &[u8], w: usize, h: usize) -> Vec<u8> {
    let sample = |x: isize, y: isize| {
        i32::from(
            values[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize],
        )
    };
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            let mut gx = 0;
            let mut gy = 0;
            for (dy, row) in [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]].iter().enumerate() {
                for (dx, &weight) in row.iter().enumerate() {
                    gx += sample(x + dx as isize - 1, y + dy as isize - 1) * weight;
                    gy += sample(x + dy as isize - 1, y + dx as isize - 1) * weight;
                }
            }
            (gx.abs() + gy.abs()).min(255) as u8
        })
        .collect()
}

fn queued_chains<const C: usize>(r: &WgpuRuntime) -> usize {
    let metadata = ImageMetadata { color_range: ColorRange::Limited, ..Default::default() };
    let mut pending = Vec::new();
    for (sw, sh, w, h) in [
        (1, 1, 17, 19),
        (1, 17, 1, 33),
        (17, 1, 33, 1),
        (15, 19, 17, 15),
        (16, 16, 16, 16),
        (17, 19, 31, 33),
        (255, 257, 257, 255),
        (256, 256, 257, 257),
    ] {
        let image = Image::<u8, C>::try_new_with_metadata(
            sw,
            sh,
            (0..sw * sh * C).map(|i| (i * 37 + i / C * 11) as u8).collect(),
            metadata,
        )
        .unwrap();
        let source = GpuImage::upload_u8(r, image.view()).unwrap();
        let expected = nearest(&image, w, h);
        let expected = if C == 1 { sobel(&expected, w, h) } else { expected };
        for (scale, mean, std) in [
            (1.0, [0.0; 4], [1.0; 4]),
            (0.25, [1.0, -2.0, 4.0, -8.0], [0.5, -2.0, 4.0, 0.25]),
            (-0.5, [-16.0, 8.0, 2.0, -4.0], [-4.0, 0.5, -2.0, 8.0]),
        ] {
            let resized = resize_nearest_gpu(r, &source, w as u32, h as u32).unwrap();
            assert_eq!(resized.metadata(), metadata);
            let last = if C == 1 {
                let edges = sobel_gpu(r, &resized).unwrap();
                assert_eq!(edges.metadata(), metadata);
                resized.recycle(r);
                edges
            } else {
                resized
            };
            let tensor = pack_ai_chw_gpu(r, &last, scale, mean, std).unwrap();
            last.recycle(r);
            assert_eq!(tensor.shape(), [C as u32, h as u32, w as u32]);
            assert_eq!(tensor.receipt().host_to_device_bytes(), (sw * sh * 4) as u64);
            assert_eq!(tensor.receipt().device_to_host_bytes(), 0);
            assert_eq!(
                tensor.receipt().gpu_to_gpu_bytes(),
                (w * h * (4 + usize::from(C == 1) * 4 + C * 4)) as u64
            );
            assert_eq!(
                tensor.receipt().stages(),
                if C == 1 {
                    vec!["upload_u8_texture", "resize_nearest_gpu", "sobel_gpu", "pack_ai_chw_gpu"]
                } else {
                    vec!["upload_u8_texture", "resize_nearest_gpu", "pack_ai_chw_gpu"]
                }
            );
            let words: Vec<u32> = (0..C)
                .flat_map(|c| {
                    let expected = &expected;
                    (0..w * h).map(move |i| {
                        ((f32::from(expected[i * C + c]) * scale - mean[c]) / std[c]).to_bits()
                    })
                })
                .collect();
            pending.push((tensor, words));
        }
        source.recycle(r);
    }
    // Keep all tensors alive while subsequent submissions recycle image storage.
    // Read in reverse order after clearing idle pools to expose shared controls.
    r.wait_idle();
    r.clear_image_texture_pool();
    r.clear_buffer_pool();
    let mut count = 0;
    for (mut tensor, words) in pending.into_iter().rev() {
        let result = tensor.readback_f32(r).unwrap();
        assert_eq!(result.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), words);
        assert_eq!(tensor.receipt().device_to_host_bytes(), words.len() as u64 * 4);
        tensor.recycle(r);
        r.clear_buffer_pool();
        assert_eq!(result.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), words);
        count += words.len();
    }
    count
}

#[test]
fn queued_image_uniforms_keep_dimensions_normalization_receipts_and_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let words = queued_chains::<1>(&r)
        + queued_chains::<2>(&r)
        + queued_chains::<3>(&r)
        + queued_chains::<4>(&r);
    eprintln!("IMAGE_UNIFORMS verified {words} exact words across 96 queued image chains");
}
