//! Synthetic in-process two-call versus robust LK comparison; no device access.
use spatialrust_image::Image;
use spatialrust_math::Vec2;
use spatialrust_vision::{
    track_points_forward_backward, track_points_lucas_kanade, RobustTrackOptions, VisionError,
};
use std::{hint::black_box, time::Instant};

struct ParentTracks {
    points: Vec<Vec2<f64>>,
    status: Vec<bool>,
    errors: Vec<f64>,
}

// Frozen parent public call graph, including original output allocations.
fn parent(
    a: &Image<u8, 1>,
    b: &Image<u8, 1>,
    points: &[Vec2<f64>],
    options: RobustTrackOptions,
) -> Result<ParentTracks, VisionError> {
    if !options.max_forward_backward_error.is_finite() || options.max_forward_backward_error < 0. {
        return Err(VisionError::InvalidParameter(
            "forward/backward threshold must be finite and non-negative".into(),
        ));
    }
    let forward = track_points_lucas_kanade(a.view(), b.view(), points, options.lucas_kanade)?;
    let backward =
        track_points_lucas_kanade(b.view(), a.view(), forward.next_points(), options.lucas_kanade)?;
    let mut status = Vec::with_capacity(points.len());
    let mut errors = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let valid = forward.status()[i] && backward.status()[i];
        let error = if valid {
            (backward.next_points()[i].x - p.x).hypot(backward.next_points()[i].y - p.y)
        } else {
            f64::INFINITY
        };
        status.push(valid && error <= options.max_forward_backward_error);
        errors.push(error);
    }
    Ok(ParentTracks { points: forward.next_points().to_vec(), status, errors })
}

fn snapshot(points: &[Vec2<f64>], status: &[bool], errors: &[f64]) -> serde_json::Value {
    serde_json::json!({"next_bits":points.iter().map(|p|[p.x.to_bits(),p.y.to_bits()]).collect::<Vec<_>>(),"status":status,"error_bits":errors.iter().map(|v|v.to_bits()).collect::<Vec<_>>()})
}

fn texture(w: usize, h: usize, dx: i32, dy: i32) -> Image<u8, 1> {
    Image::try_new(
        w,
        h,
        (0..w * h)
            .map(|i| {
                let x = (i % w) as i32 - dx;
                let y = (i / w) as i32 - dy;
                let hash = (x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663);
                ((hash ^ (hash >> 13)) & 255) as u8
            })
            .collect(),
    )
    .unwrap()
}

fn main() {
    let mut rows = Vec::new();
    let options = RobustTrackOptions::default();
    for (width, height) in [(192, 144), (256, 192), (320, 240)] {
        for count in [0, 8, 80] {
            let a = texture(width, height, 0, 0);
            let b = texture(width, height, 1, -1);
            let points: Vec<_> = (0..count)
                .map(|i| Vec2 {
                    x: (24 + (i * 17) % (width - 48)) as f64,
                    y: (24 + (i * 13) % (height - 48)) as f64,
                })
                .collect();
            let expected = parent(&a, &b, &points, options).unwrap();
            let expected = snapshot(&expected.points, &expected.status, &expected.errors);
            for _ in 0..4 {
                let p = parent(&a, &b, &points, options).unwrap();
                assert_eq!(snapshot(&p.points, &p.status, &p.errors), expected);
                let c =
                    track_points_forward_backward(a.view(), b.view(), &points, options).unwrap();
                assert_eq!(
                    snapshot(c.next_points(), c.status(), c.forward_backward_errors()),
                    expected
                );
            }
            let mut samples = Vec::new();
            for round in 0..4 {
                for flavor in ["baseline", "candidate", "candidate", "baseline"] {
                    let (elapsed_ns, result) = if flavor == "baseline" {
                        let start = Instant::now();
                        let p = black_box(
                            parent(black_box(&a), black_box(&b), black_box(&points), options)
                                .unwrap(),
                        );
                        let ns = start.elapsed().as_nanos() as u64;
                        (ns, snapshot(&p.points, &p.status, &p.errors))
                    } else {
                        let start = Instant::now();
                        let c = black_box(
                            track_points_forward_backward(
                                black_box(a.view()),
                                black_box(b.view()),
                                black_box(&points),
                                options,
                            )
                            .unwrap(),
                        );
                        let ns = start.elapsed().as_nanos() as u64;
                        (ns, snapshot(c.next_points(), c.status(), c.forward_backward_errors()))
                    };
                    assert_eq!(result, expected);
                    samples.push(
                        serde_json::json!({"round":round,"flavor":flavor,"elapsed_ns":elapsed_ns}),
                    );
                }
            }
            rows.push(serde_json::json!({"width":width,"height":height,"points":count,"samples":samples,"report":expected}));
        }
    }
    println!(
        "{}",
        serde_json::json!({"schema":"spatialrust.lk-pyramid-reuse-benchmark.v1","synthetic_only":true,"rows":rows})
    );
}
