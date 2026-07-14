//! Cross-module properties over generated image and geometry inputs.

#![cfg(feature = "full")]

use proptest::prelude::*;
use spatialrust_image::Image;
use spatialrust_vision::{
    decode_rle, encode_rle, resize, BinaryMask, BoundingBox2, Interpolation, RleOrder,
};

proptest! {
    #[test]
    fn identity_resize_preserves_u8_rgb(
        width in 1usize..24,
        height in 1usize..24,
        seed in any::<u8>(),
    ) {
        let len = width * height * 3;
        let data = (0..len)
            .map(|index| seed.wrapping_add((index as u8).wrapping_mul(37)))
            .collect::<Vec<_>>();
        let image = Image::<u8, 3>::try_new(width, height, data.clone()).unwrap();
        for interpolation in [
            Interpolation::Nearest,
            Interpolation::Bilinear,
            Interpolation::Bicubic,
            Interpolation::Area,
        ] {
            let output = resize(image.view(), width, height, interpolation).unwrap();
            prop_assert_eq!(output.as_slice(), data.as_slice());
        }
    }

    #[test]
    fn mask_rle_round_trips_both_orders(
        width in 1usize..32,
        height in 1usize..32,
        seed in any::<u64>(),
    ) {
        let mut state = seed;
        let data = (0..width * height)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 1) as u8
            })
            .collect::<Vec<_>>();
        let mask = BinaryMask::try_new(width, height, data.clone()).unwrap();
        for order in [RleOrder::RowMajor, RleOrder::CocoColumnMajor] {
            let decoded = decode_rle(&encode_rle(&mask, order)).unwrap();
            prop_assert_eq!(decoded.image().as_slice(), data.as_slice());
        }
    }

    #[test]
    fn iou_is_symmetric_and_bounded(
        ax in -100.0f32..100.0,
        ay in -100.0f32..100.0,
        aw in 0.0f32..100.0,
        ah in 0.0f32..100.0,
        bx in -100.0f32..100.0,
        by in -100.0f32..100.0,
        bw in 0.0f32..100.0,
        bh in 0.0f32..100.0,
    ) {
        let a = BoundingBox2::try_new(ax, ay, ax + aw, ay + ah).unwrap();
        let b = BoundingBox2::try_new(bx, by, bx + bw, by + bh).unwrap();
        let ab = a.iou(b);
        let ba = b.iou(a);
        prop_assert!((ab - ba).abs() <= f32::EPSILON);
        prop_assert!((0.0..=1.0).contains(&ab));
    }
}
