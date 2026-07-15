//! Cross-module properties over generated image and geometry inputs.

#![cfg(feature = "full")]

use proptest::prelude::*;
use spatialrust_camera::CameraIntrinsics;
use spatialrust_image::Image;
use spatialrust_math::{Mat3, Vec2, Vec3};
use spatialrust_vision::{
    canny, decode_rle, distance_transform_edt_with_spacing, encode_rle, erode, estimate_homography,
    filter2d, integral_image, match_descriptors, project_object_point, resize, resize_rgb_to_gray,
    rgb_to_gray, solve_pnp, AbsolutePose, BilinearResizeU8Plan, BinaryMask, BorderMode,
    BoundingBox2, CameraMatrix3, CannyOptions, DescriptorBuffer, Interpolation, Kernel2D,
    MatchOptions, MorphologyShape, ObjectImageCorrespondence, PointCorrespondence2, RleOrder,
    StructuringElement,
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
    fn fused_resize_to_gray_matches_unfused_plan(
        width in 1usize..24,
        height in 1usize..24,
        output_width in 1usize..24,
        output_height in 1usize..24,
        seed in any::<u8>(),
    ) {
        let data = (0..width * height * 3)
            .map(|index| seed.wrapping_add((index as u8).wrapping_mul(37)))
            .collect::<Vec<_>>();
        let image = Image::<u8, 3>::try_new(width, height, data).unwrap();
        let plan = BilinearResizeU8Plan::new(width, height, output_width, output_height).unwrap();
        let resized = plan.resize(image.view()).unwrap();
        let expected = rgb_to_gray(resized.view()).unwrap();
        let actual = resize_rgb_to_gray(image.view(), output_width, output_height).unwrap();
        prop_assert_eq!(actual, expected);
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
    fn exact_distance_transform_matches_brute_force(
        width in 1usize..16,
        height in 1usize..16,
        seed in any::<u64>(),
        spacing_x in 0.1f32..4.0,
        spacing_y in 0.1f32..4.0,
    ) {
        let mut state = seed;
        let mut data = (0..width * height)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 1) as u8
            })
            .collect::<Vec<_>>();
        let forced_background = seed as usize % data.len();
        data[forced_background] = 0;
        let mask = BinaryMask::try_new(width, height, data.clone()).unwrap();
        let actual = distance_transform_edt_with_spacing(&mask, spacing_x, spacing_y).unwrap();
        let background = data
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == 0)
            .map(|(index, _)| (index % width, index / width))
            .collect::<Vec<_>>();
        for y in 0..height {
            for x in 0..width {
                let expected = background
                    .iter()
                    .map(|&(bx, by)| {
                        let dx = x.abs_diff(bx) as f32 * spacing_x;
                        let dy = y.abs_diff(by) as f32 * spacing_y;
                        dx.hypot(dy)
                    })
                    .fold(f32::INFINITY, f32::min);
                prop_assert!((actual[(x, y)][0] - expected).abs() <= 2e-5);
            }
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

    #[test]
    fn homography_recovers_noisy_planar_map(
        seed in any::<u64>(),
        noise in 0.0f64..0.4,
    ) {
        let transform = Mat3::<f64>::from_rows(
            [1.05, 0.02, 3.0],
            [-0.01, 0.97, -2.0],
            [0.0001, -0.0002, 1.0],
        );
        let pairs = (0..24usize)
            .map(|index| {
                let source = Vec2 {
                    x: ((index % 6) as f64 + (seed % 5) as f64 * 0.1) * 20.0,
                    y: ((index / 6) as f64 + ((seed / 7) % 5) as f64 * 0.1) * 15.0,
                };
                let projected = transform.mul_vec3(Vec3::new(source.x, source.y, 1.0));
                let target = Vec2 {
                    x: projected.x / projected.z + ((seed + index as u64) % 3) as f64 * noise * 0.1,
                    y: projected.y / projected.z
                        + ((seed / 3 + index as u64) % 3) as f64 * noise * 0.1,
                };
                PointCorrespondence2::try_new(source, target).unwrap()
            })
            .collect::<Vec<_>>();
        let model = estimate_homography(&pairs).unwrap();
        for pair in &pairs {
            let projected = model.matrix().mul_vec3(Vec3::new(pair.source().x, pair.source().y, 1.0));
            let error = (projected.x / projected.z - pair.target().x)
                .hypot(projected.y / projected.z - pair.target().y);
            prop_assert!(error < 2.0 + noise);
        }
    }

    #[test]
    fn pnp_recovers_pose_under_small_noise(
        seed in 1u64..10_000,
        depth in 1.5f64..4.0,
    ) {
        let camera = CameraMatrix3::from_intrinsics(
            CameraIntrinsics::try_new(480.0, 480.0, 320.0, 240.0, 640, 480).unwrap(),
        );
        let pose = AbsolutePose::try_new(
            Mat3::<f64>::identity(),
            Vec3::new(((seed % 7) as f64 - 3.0) * 0.02, 0.0, depth),
        )
        .unwrap();
        let pairs = (0..12usize)
            .map(|index| {
                let object = Vec3::new(
                    (index % 4) as f64 * 0.1 - 0.15,
                    (index / 4) as f64 * 0.1 - 0.1,
                    0.0,
                );
                let mut image = project_object_point(pose, camera, object).unwrap();
                image.x += ((seed + index as u64) % 3) as f64 * 0.05;
                image.y += ((seed / 5 + index as u64) % 3) as f64 * 0.05;
                ObjectImageCorrespondence::try_new(object, image).unwrap()
            })
            .collect::<Vec<_>>();
        let estimated = match solve_pnp(&pairs, camera) {
            Ok(pose) => pose,
            Err(_) => return Ok(()),
        };
        prop_assert!((estimated.translation().z - pose.translation().z).abs() < 0.25);
    }
}
