use super::*;
use spatialrust_image::ImageRegion;

mod previous {
    use spatialrust_image::ImageView;
    include!("../../tests/fixtures/image_transfer_serial_packing.rs");
}

fn check<const CHANNELS: usize>(width: usize, height: usize, mode: usize) -> usize {
    let (parent_w, parent_h, ox, oy) =
        if mode == 2 { (width + 5, height + 7, 2, 3) } else { (width, height, 0, 0) };
    let stride = parent_w * CHANNELS + if mode == 0 { 0 } else { 11 };
    let value = |x: usize, y: usize, c: usize| ((x * 37 + y * 19 + c * 53) % 251) as u8;
    let mut source = vec![253; stride * parent_h];
    for y in 0..parent_h {
        for x in 0..parent_w {
            for c in 0..CHANNELS {
                source[y * stride + x * CHANNELS + c] = value(x, y, c);
            }
        }
    }
    let view = ImageView::<u8, CHANNELS>::new(parent_w, parent_h, stride, &source)
        .unwrap()
        .subview(ImageRegion::new(ox, oy, width, height))
        .unwrap();
    let mut logical = Vec::new();
    let mut expected = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for c in 0..4 {
                expected.push(if c < CHANNELS { value(x + ox, y + oy, c) } else { 0 });
                if c < CHANNELS {
                    logical.push(value(x + ox, y + oy, c));
                }
            }
        }
    }
    let padded_row = (width * 4).div_ceil(256) * 256;
    let mut padded = vec![252; padded_row * height];
    for (y, row) in expected.chunks_exact(width * 4).enumerate() {
        padded[y * padded_row..y * padded_row + width * 4].copy_from_slice(row);
    }
    assert_eq!(previous::pack_rgba8(view, expected.len()), expected);
    assert_eq!(
        previous::unpack_rgba8::<CHANNELS>(&padded, width, height, padded_row, logical.len()),
        logical
    );
    for workers in [0, 1, 2, 6] {
        assert_eq!(pack_with_workers(view, expected.len(), workers), expected);
        assert_eq!(
            unpack_with_workers::<CHANNELS>(
                &padded,
                width,
                height,
                padded_row,
                logical.len(),
                workers
            ),
            logical
        );
    }
    assert_eq!(pack_rgba8(view, expected.len()), expected);
    assert_eq!(
        unpack_rgba8::<CHANNELS>(&padded, width, height, padded_row, logical.len()),
        logical
    );
    (expected.len() + logical.len()) * 6
}

fn cases<const CHANNELS: usize>() -> usize {
    let mut bytes = 0;
    for (width, height) in [
        (1, 1),
        (1, 17),
        (257, 1),
        (63, 3),
        (64, 5),
        (65, 7),
        (1023, 11),
        (1024, 13),
        (1025, 17),
        (1024, 1023),
        (1024, 1024),
        (1024, 1025),
        (1_048_577, 1),
        (1, 1_048_577),
    ] {
        for mode in if width * height >= 262_144 { 2..3 } else { 0..3 } {
            bytes += check::<CHANNELS>(width, height, mode);
        }
    }
    bytes
}

#[test]
fn transfer_rows_match_independent_and_frozen_bytes_for_all_channel_layouts() {
    let bytes = cases::<1>() + cases::<2>() + cases::<3>() + cases::<4>();
    eprintln!("TRANSFER_PACKING verified {bytes} frozen/oracle byte comparisons including ROI, worker and cutoff boundaries.");
}

#[test]
fn empty_transforms_and_native_worker_budgets_are_bounded() {
    let empty = ImageView::<u8, 3>::new(0, 0, 0, &[]).unwrap();
    assert!(pack_with_workers(empty, 0, 6).is_empty());
    assert!(unpack_with_workers::<3>(&[], 0, 0, 0, 0, 6).is_empty());
    for (width, height, channels) in
        [(1, 1, 1), (1024, 1023, 3), (1024, 1024, 4), (1_048_577, 1, 3)]
    {
        assert_eq!(transfer_workers(width, height, channels), 1);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert_eq!(transfer_workers(1024, 1024, 3), available.min(4));
        assert_eq!(transfer_workers(1_048_576, 2, 1), available.min(2));
    }
    #[cfg(target_arch = "wasm32")]
    assert_eq!(transfer_workers(1024, 1024, 3), 1);
}

#[test]
fn concurrent_row_transforms_keep_prior_results_owned() {
    let source = vec![71; 1024 * 1025 * 3];
    let view = ImageView::<u8, 3>::new(1024, 1025, 1024 * 3, &source).unwrap();
    let expected = previous::pack_rgba8(view, 1024 * 1025 * 4);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let expected = &expected;
            scope.spawn(move || {
                let first = pack_rgba8(view, expected.len());
                let second = pack_rgba8(view, expected.len());
                assert_eq!(&first, expected);
                assert_eq!(&second, expected);
            });
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[path = "transfer_packing_benchmark.rs"]
mod benchmark;
