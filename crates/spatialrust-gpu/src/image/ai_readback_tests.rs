use super::*;
use spatialrust_image::Image;

#[path = "ai_readback_reference.rs"]
mod parent;

const SPECIAL: [u32; 16] = [
    0,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0001,
    0x7f80_0001,
    0xffc0_1234,
    0x7fff_ffff,
    1,
    0x007f_ffff,
    0x0080_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x3f80_0000,
    0xbf80_0000,
    0x3eaa_aaab,
];

fn words(len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| {
            if i < SPECIAL.len() {
                SPECIAL[i]
            } else {
                let x = (i as u32).wrapping_mul(0x9e37_79b9).rotate_left(13);
                x ^ x.wrapping_mul(0x85eb_ca6b).rotate_right(7)
            }
        })
        .collect()
}

fn raw_tensor(r: &WgpuRuntime, w: u32, h: u32, c: u32, words: &[u32]) -> GpuAiTensor {
    assert_eq!(words.len(), (w * h * c) as usize);
    let buffer = r.upload_u32_storage("ai-readback-bit-fixture", words).unwrap();
    let byte_len = std::mem::size_of_val(words) as u64;
    let mut receipt = GpuImageReceipt::default();
    receipt.record_host_to_device(byte_len, "fixture_tensor_upload");
    GpuAiTensor {
        buffer,
        width: w,
        height: h,
        channels: c,
        byte_len,
        device_key: runtime_device_key(r),
        receipt,
    }
}

fn assert_bits(actual: &[f32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&value, &bits)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(value.to_bits(), bits, "word {i}");
    }
}

#[test]
fn direct_ai_download_keeps_all_f32_bits_shape_receipts_and_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut count = 0;
    for c in 1..=4 {
        for (w, h) in [(1, 1), (65, 17), (1024, 257)] {
            let (mut new, mut old, expected) = {
                let mut input = words((w * h * c) as usize);
                let new = raw_tensor(&r, w, h, c, &input);
                let old = raw_tensor(&r, w, h, c, &input);
                let expected = input.clone();
                input.fill(0x1234_5678);
                (new, old, expected)
            };
            assert_eq!(new.shape(), [c, h, w]);
            let mut live = new.readback_f32(&r).unwrap();
            assert_bits(&live, &expected);
            assert_bits(&old.parent_readback_f32(&r).unwrap(), &expected);
            assert_eq!(new.receipt(), old.receipt());
            live[0] = f32::from_bits(0xdead_beef);
            let next = new.readback_f32(&r).unwrap();
            assert_bits(&next, &expected);
            assert_bits(&old.parent_readback_f32(&r).unwrap(), &expected);
            assert_eq!(new.receipt(), old.receipt());
            assert_eq!(new.receipt().device_to_host_bytes(), expected.len() as u64 * 8);
            assert_eq!(new.receipt().gpu_to_gpu_bytes(), 0);
            assert_eq!(
                new.receipt().stages(),
                &["fixture_tensor_upload", "readback_ai_chw_f32", "readback_ai_chw_f32"]
            );
            count += expected.len() * 4;
            new.recycle(&r);
            old.recycle(&r);
            r.clear_buffer_pool();
            assert_bits(&next, &expected);
            assert_eq!(live[0].to_bits(), 0xdead_beef);
        }
    }
    eprintln!("AI_DIRECT_READBACK verified {count} f32 words, including NaN payloads, infinities, subnormals and signed zeros");
}

fn real_pack<const C: usize>(r: &WgpuRuntime) -> usize {
    let (w, h) = (257, 129);
    let image = Image::<u8, C>::try_new(
        w,
        h,
        (0..w * h * C).map(|i| (i * 37 + i / C * 11) as u8).collect(),
    )
    .unwrap();
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mean = [1.0, -2.0, 3.0, -4.0];
    let std = [2.0, 4.0, 0.5, 1.0];
    let mut new = pack_ai_chw_gpu(r, &source, 0.5, mean, std).unwrap();
    let mut old = pack_ai_chw_gpu(r, &source, 0.5, mean, std).unwrap();
    assert_eq!(new.receipt().device_to_host_bytes(), 0);
    source.recycle(r);
    let expected: Vec<u32> = (0..C)
        .flat_map(|c| {
            let data = image.as_slice();
            (0..w * h)
                .map(move |i| ((f32::from(data[i * C + c]) * 0.5 - mean[c]) / std[c]).to_bits())
        })
        .collect();
    drop(image);
    assert_bits(&new.readback_f32(r).unwrap(), &expected);
    assert_bits(&old.parent_readback_f32(r).unwrap(), &expected);
    assert_eq!(new.shape(), [C as u32, h as u32, w as u32]);
    assert_eq!(new.receipt(), old.receipt());
    assert_eq!(new.receipt().host_to_device_bytes(), (w * h * 4) as u64);
    assert_eq!(new.receipt().gpu_to_gpu_bytes(), (w * h * C * 4) as u64);
    assert_eq!(new.receipt().device_to_host_bytes(), (w * h * C * 4) as u64);
    new.recycle(r);
    old.recycle(r);
    expected.len() * 2
}

#[test]
fn direct_ai_download_keeps_real_chw_planes_normalization_and_runtime_rejection() {
    let r = WgpuRuntime::new_headless().unwrap();
    let count = real_pack::<1>(&r) + real_pack::<2>(&r) + real_pack::<3>(&r) + real_pack::<4>(&r);
    let other = WgpuRuntime::new_headless().unwrap();
    let values = words(17);
    let mut tensor = raw_tensor(&r, 17, 1, 1, &values);
    let before = tensor.receipt().clone();
    let pool = other.buffer_pool().stats();
    let error = tensor.readback_f32(&other).unwrap_err();
    let mut prior = raw_tensor(&r, 17, 1, 1, &values);
    assert_eq!(error, prior.parent_readback_f32(&other).unwrap_err());
    assert_eq!(tensor.receipt(), &before);
    assert_eq!(other.buffer_pool().stats(), pool);
    assert_bits(&tensor.readback_f32(&r).unwrap(), &values);
    tensor.recycle(&r);
    prior.recycle(&r);
    eprintln!("AI_DIRECT_CHW verified {count} analytically exact dyadic normalization words");
}

#[test]
fn concurrent_direct_ai_downloads_keep_live_results_independent() {
    let r = WgpuRuntime::new_headless().unwrap();
    std::thread::scope(|scope| {
        for offset in 0..4 {
            let r = &r;
            scope.spawn(move || {
                let expected: Vec<u32> =
                    words(513 * 259 * 4).into_iter().map(|x| x.rotate_left(offset)).collect();
                let mut a = raw_tensor(r, 513, 259, 4, &expected);
                let mut b = raw_tensor(r, 513, 259, 4, &expected);
                let live = a.readback_f32(r).unwrap();
                for _ in 0..3 {
                    assert_bits(&a.readback_f32(r).unwrap(), &expected);
                    assert_bits(&b.readback_f32(r).unwrap(), &expected);
                }
                a.recycle(r);
                b.recycle(r);
                assert_bits(&live, &expected);
            });
        }
    });
}

#[path = "ai_readback_benchmark.rs"]
mod benchmark;
