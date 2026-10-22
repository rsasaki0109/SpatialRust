use super::*;
use spatialrust_image::{Image, ImageRegion};
#[path = "optical_flow_reference.rs"]
mod reference;

fn options(o: LucasKanadeOptions) -> reference::LucasKanadeOptions {
    reference::LucasKanadeOptions {
        window_radius: o.window_radius,
        pyramid_levels: o.pyramid_levels,
        max_iterations: o.max_iterations,
        epsilon: o.epsilon,
        min_eigenvalue: o.min_eigenvalue,
    }
}
fn compare<T: PixelComponent>(
    a: ImageView<'_, T, 1>,
    b: ImageView<'_, T, 1>,
    points: &[Vec2<f64>],
    o: LucasKanadeOptions,
) {
    match (
        track_points_lucas_kanade(a, b, points, o),
        reference::track_points_lucas_kanade(a, b, points, options(o)),
    ) {
        (Ok(actual), Ok(expected)) => {
            assert_eq!(actual.status(), expected.status());
            let bits = |p: &[Vec2<f64>]| {
                p.iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect::<Vec<_>>()
            };
            assert_eq!(bits(actual.next_points()), bits(expected.next_points()));
        }
        (Err(a), Err(b)) => assert_eq!(a, b),
        _ => panic!("frozen LK admission differs"),
    }
}
fn texture<T: PixelComponent>(w: usize, h: usize, dx: i32, dy: i32) -> Image<T, 1> {
    Image::try_new(
        w,
        h,
        (0..w * h)
            .map(|i| {
                let x = (i % w) as i32 - dx;
                let y = (i / w) as i32 - dy;
                let v = (x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663);
                T::from_f64(f64::from((v ^ (v >> 13)) & 255))
            })
            .collect(),
    )
    .unwrap()
}
fn scalar_cases<T: PixelComponent>() {
    for (w, h) in [(33, 35), (96, 80)] {
        for shift in [0, 1] {
            let a = texture::<T>(w, h, 0, 0);
            let b = texture::<T>(w, h, shift, -shift);
            let points = [
                Vec2 { x: 16.25, y: 17.5 },
                Vec2 { x: 8.5, y: 9.25 },
                Vec2 { x: (w - 9) as f64, y: (h - 10) as f64 },
            ];
            for radius in [1, 3, 5, 6, 9] {
                for iterations in [1, 2, 9] {
                    for levels in [1, 3] {
                        compare(
                            a.view(),
                            b.view(),
                            &points,
                            LucasKanadeOptions {
                                window_radius: radius,
                                pyramid_levels: levels,
                                max_iterations: iterations,
                                ..Default::default()
                            },
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn all_scalar_types_radii_iterations_and_pyramids_match_frozen_parent() {
    scalar_cases::<u8>();
    scalar_cases::<u16>();
    scalar_cases::<f32>();
    scalar_cases::<f64>();
}
#[test]
fn boundary_motion_and_nonfinite_points_match_frozen_parent() {
    let a = texture::<u8>(64, 65, 0, 0);
    let points = [
        Vec2 { x: 0., y: 0. },
        Vec2 { x: -0., y: 32. },
        Vec2 { x: 5., y: 5. },
        Vec2 { x: 5.25, y: 6.5 },
        Vec2 { x: 56.5, y: 57.25 },
        Vec2 { x: 32., y: 32. },
        Vec2 { x: f64::from_bits(0x7ff8000000000123), y: 1. },
        Vec2 { x: f64::INFINITY, y: f64::NEG_INFINITY },
    ];
    for shift in [-7, -2, 0, 2, 7] {
        let b = texture::<u8>(64, 65, shift, -shift);
        for radius in [1, 4, 5, 6] {
            compare(
                a.view(),
                b.view(),
                &points,
                LucasKanadeOptions {
                    window_radius: radius,
                    pyramid_levels: 1,
                    max_iterations: 30,
                    ..Default::default()
                },
            );
        }
    }
}
#[test]
fn nonfinite_float_pixels_and_extreme_magnitudes_match_frozen_parent() {
    for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0., f32::MAX, f32::from_bits(1)] {
        let mut data = vec![v; 48 * 51];
        data[25 * 48 + 24] = 1.;
        let a = Image::<f32, 1>::try_new(48, 51, data).unwrap();
        for radius in [1, 5, 6] {
            compare(
                a.view(),
                a.view(),
                &[Vec2 { x: 24.25, y: 25.5 }],
                LucasKanadeOptions {
                    window_radius: radius,
                    pyramid_levels: 1,
                    ..Default::default()
                },
            );
        }
    }
}
#[test]
fn strided_sources_repeated_calls_and_input_ownership_match_parent() {
    let a = texture::<u16>(100, 96, 0, 0);
    let b = texture::<u16>(100, 96, 2, -2);
    let before = (a.clone(), b.clone());
    let av = a.view().subview(ImageRegion::new(2, 3, 65, 79)).unwrap();
    let bv = b.view().subview(ImageRegion::new(4, 5, 65, 79)).unwrap();
    assert!(av.as_packed_slice().is_none());
    let p = [Vec2 { x: 32.25, y: 36.5 }, Vec2 { x: 32.25, y: 36.5 }];
    for _ in 0..3 {
        compare(av, bv, &p, Default::default());
        compare(bv, av, &p, Default::default());
        compare(av, bv, &[], Default::default());
    }
    assert_eq!(a, before.0);
    assert_eq!(b, before.1);
}
#[test]
fn invalid_options_shapes_and_tiny_frames_keep_error_priority() {
    let a = Image::<u8, 1>::try_new(2, 2, vec![0; 4]).unwrap();
    let b = texture::<u8>(8, 8, 0, 0);
    let o = LucasKanadeOptions::default();
    for options in [
        LucasKanadeOptions { window_radius: 0, ..o },
        LucasKanadeOptions { window_radius: usize::MAX, ..o },
        LucasKanadeOptions { pyramid_levels: 33, ..o },
        LucasKanadeOptions { max_iterations: 0, ..o },
        LucasKanadeOptions { epsilon: f64::NAN, ..o },
        LucasKanadeOptions { min_eigenvalue: 0., ..o },
        o,
    ] {
        compare(a.view(), b.view(), &[], options);
        compare(a.view(), a.view(), &[], options);
        compare(b.view(), b.view(), &[], options);
    }
}
#[test]
fn stationary_tracks_include_successful_exact_coordinates() {
    let a = texture::<u8>(96, 80, 0, 0);
    let p = [Vec2 { x: 32., y: 32. }, Vec2 { x: 48., y: 40. }, Vec2 { x: 64., y: 48. }];
    let r = track_points_lucas_kanade(a.view(), a.view(), &p, Default::default()).unwrap();
    assert!(r.status().iter().any(|&v| v));
    compare(a.view(), a.view(), &p, Default::default());
}
