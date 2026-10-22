use super::*;
use spatialrust_image::{Image, ImageRegion};

// Frozen parent call graph: each public LK call constructs both pyramids.
fn parent<T: crate::PixelComponent>(
    previous: ImageView<'_, T, 1>,
    next: ImageView<'_, T, 1>,
    points: &[Vec2<f64>],
    options: RobustTrackOptions,
) -> VisionResult<RobustTracks> {
    if !options.max_forward_backward_error.is_finite() || options.max_forward_backward_error < 0.0 {
        return Err(VisionError::InvalidParameter(
            "forward/backward threshold must be finite and non-negative".into(),
        ));
    }
    let forward = crate::track_points_lucas_kanade(previous, next, points, options.lucas_kanade)?;
    let backward = crate::track_points_lucas_kanade(
        next,
        previous,
        forward.next_points(),
        options.lucas_kanade,
    )?;
    let mut status = Vec::with_capacity(points.len());
    let mut errors = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let valid = forward.status()[index] && backward.status()[index];
        let error = if valid {
            let dx = backward.next_points()[index].x - point.x;
            let dy = backward.next_points()[index].y - point.y;
            dx.hypot(dy)
        } else {
            f64::INFINITY
        };
        status.push(valid && error <= options.max_forward_backward_error);
        errors.push(error);
    }
    Ok(RobustTracks {
        next_points: forward.next_points().to_vec(),
        status,
        forward_backward_errors: errors,
    })
}

fn compare<T: crate::PixelComponent>(
    a: ImageView<'_, T, 1>,
    b: ImageView<'_, T, 1>,
    points: &[Vec2<f64>],
    options: RobustTrackOptions,
) {
    let expected = parent(a, b, points, options);
    let actual = track_points_forward_backward(a, b, points, options);
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => {
            assert_eq!(actual.status(), expected.status());
            let bits = |r: &RobustTracks| {
                r.next_points().iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect::<Vec<_>>()
            };
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(
                actual.forward_backward_errors().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.forward_backward_errors().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(actual.accepted_count(), expected.accepted_count());
        }
        (Err(actual), Err(expected)) => assert_eq!(actual, expected),
        _ => panic!("LK pair result admission differs"),
    }
}

fn texture<T: crate::PixelComponent>(w: usize, h: usize, dx: i32, dy: i32) -> Image<T, 1> {
    Image::try_new(
        w,
        h,
        (0..w * h)
            .map(|i| {
                let x = (i % w) as i32 - dx;
                let y = (i / w) as i32 - dy;
                let hash = (x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663);
                T::from_f64(f64::from((hash ^ (hash >> 13)) & 255))
            })
            .collect(),
    )
    .unwrap()
}

fn scalar_cases<T: crate::PixelComponent>() {
    for (w, h) in [(33, 35), (65, 79), (96, 80)] {
        let a = texture::<T>(w, h, 0, 0);
        let b = texture::<T>(w, h, 1, -1);
        let points: Vec<_> = (0..12)
            .map(|i| Vec2 {
                x: 8.25 + ((i * 17) % (w - 16)) as f64,
                y: 8.5 + ((i * 13) % (h - 16)) as f64,
            })
            .collect();
        for levels in [1, 2, 3, 5] {
            for threshold in [0., 0.5, 1.] {
                compare(
                    a.view(),
                    b.view(),
                    &points,
                    RobustTrackOptions {
                        lucas_kanade: LucasKanadeOptions {
                            window_radius: 2,
                            pyramid_levels: levels,
                            max_iterations: 5,
                            ..Default::default()
                        },
                        max_forward_backward_error: threshold,
                    },
                );
            }
        }
    }
}

#[test]
fn builtin_scalar_coordinate_status_and_error_bits_match_two_calls() {
    scalar_cases::<u8>();
    scalar_cases::<u16>();
    scalar_cases::<f32>();
    scalar_cases::<f64>();
}

#[test]
fn nonfinite_float_pixels_keep_original_decisions_and_bits() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0., 1e300, 1e-300] {
        let mut data = vec![value; 33 * 35];
        data[17 * 33 + 16] = 0.;
        let a = Image::<f64, 1>::try_new(33, 35, data.clone()).unwrap();
        let b = Image::<f32, 1>::try_new(33, 35, data.iter().map(|&v| v as f32).collect()).unwrap();
        let points = [Vec2 { x: 16.25, y: 17.5 }];
        compare(a.view(), a.view(), &points, Default::default());
        compare(b.view(), b.view(), &points, Default::default());
    }
}

#[test]
fn strided_subviews_keep_logical_pixel_sampling() {
    let a = texture::<u8>(100, 90, 0, 0);
    let b = texture::<u8>(100, 90, 1, -1);
    let a = a.view().subview(ImageRegion::new(2, 3, 65, 79)).unwrap();
    let b = b.view().subview(ImageRegion::new(5, 6, 65, 79)).unwrap();
    assert!(a.as_packed_slice().is_none());
    compare(a, b, &[Vec2 { x: 32., y: 35. }, Vec2 { x: 40.25, y: 43.5 }], Default::default());
}

#[test]
fn option_shape_and_round_trip_error_priority_matches_parent() {
    let small = Image::<u8, 1>::try_new(2, 2, vec![0; 4]).unwrap();
    let image = texture::<u8>(8, 8, 0, 0);
    let lk = LucasKanadeOptions::default();
    let invalid = [
        LucasKanadeOptions { window_radius: 0, ..lk },
        LucasKanadeOptions { pyramid_levels: 0, ..lk },
        LucasKanadeOptions { pyramid_levels: 33, ..lk },
        LucasKanadeOptions { max_iterations: 0, ..lk },
        LucasKanadeOptions { epsilon: f64::NAN, ..lk },
        LucasKanadeOptions { epsilon: 0., ..lk },
        LucasKanadeOptions { min_eigenvalue: f64::NAN, ..lk },
        LucasKanadeOptions { min_eigenvalue: -1., ..lk },
        LucasKanadeOptions { window_radius: usize::MAX, ..lk },
        lk,
    ];
    for options in invalid {
        for threshold in [-1., f64::NAN, f64::INFINITY, 0., 1.] {
            let options =
                RobustTrackOptions { lucas_kanade: options, max_forward_backward_error: threshold };
            compare(small.view(), image.view(), &[], options);
            compare(small.view(), small.view(), &[], options);
            compare(image.view(), image.view(), &[], options);
        }
    }
}

#[test]
fn empty_nonfinite_duplicate_points_and_deep_pyramids_match_parent() {
    let a = texture::<u8>(33, 35, 0, 0);
    let points = [
        Vec2 { x: -0., y: 0. },
        Vec2 { x: 16.25, y: 17.5 },
        Vec2 { x: 16.25, y: 17.5 },
        Vec2 { x: f64::from_bits(0x7ff8000000000123), y: 1. },
        Vec2 { x: f64::INFINITY, y: f64::NEG_INFINITY },
    ];
    for levels in [1, 3, 32] {
        for threshold in [-0., 0., 1e-6, 1., f64::MAX] {
            let options = RobustTrackOptions {
                lucas_kanade: LucasKanadeOptions {
                    window_radius: 1,
                    pyramid_levels: levels,
                    max_iterations: 2,
                    ..Default::default()
                },
                max_forward_backward_error: threshold,
            };
            compare(a.view(), a.view(), &[], options);
            compare(a.view(), a.view(), &points, options);
        }
    }
}

#[test]
fn stationary_texture_retains_accepted_identity_tracks() {
    let a = texture::<u8>(96, 80, 0, 0);
    let points = [Vec2 { x: 32., y: 32. }, Vec2 { x: 48., y: 40. }, Vec2 { x: 64., y: 48. }];
    let result =
        track_points_forward_backward(a.view(), a.view(), &points, Default::default()).unwrap();
    assert!(result.accepted_count() > 0);
    compare(a.view(), a.view(), &points, Default::default());
}

#[test]
fn custom_conversion_observations_keep_the_original_four_image_scans() {
    use std::cell::Cell;
    thread_local! {static CALLS: Cell<usize> = const {Cell::new(0)};}
    #[derive(Clone, Copy)]
    struct Observed(u8);
    impl crate::PixelComponent for Observed {
        fn to_f64(self) -> f64 {
            CALLS.with(|c| {
                let n = c.get();
                c.set(n + 1);
                f64::from(self.0) + n as f64 * 1e-5
            })
        }
        fn from_f64(v: f64) -> Self {
            Self(v as u8)
        }
    }
    let a = texture::<Observed>(33, 35, 0, 0);
    let points = [Vec2 { x: 16., y: 17. }];
    CALLS.with(|c| c.set(0));
    let expected = parent(a.view(), a.view(), &points, Default::default()).unwrap();
    assert_eq!(CALLS.with(Cell::get), 4 * 33 * 35);
    CALLS.with(|c| c.set(0));
    let actual =
        track_points_forward_backward(a.view(), a.view(), &points, Default::default()).unwrap();
    assert_eq!(CALLS.with(Cell::get), 4 * 33 * 35);
    assert_eq!(actual.status(), expected.status());
    assert_eq!(
        actual.next_points().iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect::<Vec<_>>(),
        expected.next_points().iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect::<Vec<_>>()
    );
    assert_eq!(actual.forward_backward_errors(), expected.forward_backward_errors());
}
