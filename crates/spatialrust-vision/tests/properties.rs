//! Cross-module properties over generated image and geometry inputs.

#![cfg(feature = "full")]

use proptest::prelude::*;
use spatialrust_image::Image;
use spatialrust_vision::{
    canny, decode_rle, encode_rle, erode, filter2d, integral_image, match_descriptors, resize,
    BinaryMask, BorderMode, BoundingBox2, CannyOptions, DescriptorBuffer, Interpolation, Kernel2D,
    MatchOptions, MorphologyShape, RleOrder, StructuringElement,
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
    fn identity_filter_preserves_arbitrary_u16_roi_storage(
        width in 1usize..24,
        height in 1usize..24,
        padding in 0usize..8,
        seed in any::<u16>(),
    ) {
        let stride = width + padding;
        let data = (0..stride * height)
            .map(|index| seed.wrapping_add((index as u16).wrapping_mul(251)))
            .collect::<Vec<_>>();
        let view = spatialrust_image::ImageView::<u16, 1>::new(width, height, stride, &data).unwrap();
        let kernel = Kernel2D::try_new(1, 1, vec![1.0]).unwrap();
        let output = filter2d(view, &kernel, 0.0, BorderMode::Reflect101).unwrap();
        for y in 0..height {
            for x in 0..width {
                prop_assert_eq!(output[(x, y)][0], data[y * stride + x]);
            }
        }
    }

    #[test]
    fn morphology_preserves_constant_strided_u16_images(
        width in 1usize..20,
        height in 1usize..20,
        padding in 0usize..6,
        value in any::<u16>(),
        iterations in 0usize..4,
    ) {
        let stride = width + padding;
        let storage = vec![value; stride * height];
        let view = spatialrust_image::ImageView::<u16, 1>::new(width, height, stride, &storage).unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Ellipse, 5, 3).unwrap();
        let output = erode(view, &element, iterations, BorderMode::Replicate).unwrap();
        prop_assert!(output.as_slice().iter().all(|&actual| actual == value));
    }

    #[test]
    fn integral_total_matches_strided_u16_sum(
        width in 0usize..20,
        height in 0usize..20,
        padding in 0usize..6,
        seed in any::<u16>(),
    ) {
        let stride = width + padding;
        let storage = (0..stride * height)
            .map(|index| seed.wrapping_add(index as u16))
            .collect::<Vec<_>>();
        let view = spatialrust_image::ImageView::<u16, 1>::new(width, height, stride, &storage).unwrap();
        let expected = (0..height)
            .map(|y| (0..width).map(|x| storage[y * stride + x] as f64).sum::<f64>())
            .sum::<f64>();
        let integral = integral_image(view, 0).unwrap();
        prop_assert_eq!(integral.sum_region(0, 0, width, height).unwrap(), expected);
    }

    #[test]
    fn canny_constant_strided_images_are_empty(
        width in 0usize..24,
        height in 0usize..24,
        padding in 0usize..8,
        value in any::<u8>(),
        aperture_size in prop_oneof![Just(3usize), Just(5usize), Just(7usize)],
        l2_gradient in any::<bool>(),
    ) {
        let stride = width + padding;
        let storage = vec![value; stride * height];
        let view = spatialrust_image::ImageView::<u8, 1>::new(width, height, stride, &storage).unwrap();
        let edges = canny(
            view,
            CannyOptions {
                low_threshold: 25.0,
                high_threshold: 50.0,
                aperture_size,
                l2_gradient,
            },
        ).unwrap();
        prop_assert!(edges.as_slice().iter().all(|&pixel| pixel == 0));
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

    #[test]
    fn hamming_distance_is_symmetric_and_bounded(
        left in prop::collection::vec(any::<u8>(), 1..65),
        seed in any::<u8>(),
    ) {
        let right = left
            .iter()
            .enumerate()
            .map(|(index, value)| value ^ seed.wrapping_add(index as u8))
            .collect::<Vec<_>>();
        let left_descriptors = DescriptorBuffer::try_binary(1, left.len(), left).unwrap();
        let right_descriptors = DescriptorBuffer::try_binary(1, right.len(), right).unwrap();
        let forward = match_descriptors(
            &left_descriptors,
            &right_descriptors,
            MatchOptions::default(),
        ).unwrap()[0].distance();
        let reverse = match_descriptors(
            &right_descriptors,
            &left_descriptors,
            MatchOptions::default(),
        ).unwrap()[0].distance();
        prop_assert_eq!(forward, reverse);
        prop_assert!(forward <= (left_descriptors.width() * 8) as f32);
    }
}
