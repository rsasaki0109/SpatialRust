//! Fixed synthetic public-API workload for mask-support comparisons.
use spatialrust_web::{CameraTrackReport, CameraTracker};
use std::hint::black_box;
use std::time::Instant;

fn texture(width: usize, height: usize, frame: usize) -> Vec<u8> {
    (0..width * height)
        .map(|i| {
            let x = (i % width) as i32 - (frame % 3) as i32;
            let y = (i / width) as i32 - (frame % 2) as i32;
            let h = (x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663);
            ((h ^ (h >> 13)) & 255) as u8
        })
        .collect()
}
fn mask(width: usize, height: usize, kind: &str) -> Vec<u8> {
    if kind == "disabled" {
        return Vec::new();
    }
    (0..width * height)
        .map(|i| {
            u8::from(match kind {
                "zero" => false,
                "full" => true,
                "half" => i % width >= width / 2,
                "sparse" => i == (height / 2) * width + width / 2,
                "checker" => (i % width + i / width) % 2 == 0,
                _ => unreachable!(),
            })
        })
        .collect()
}
fn sequence(
    width: usize,
    height: usize,
    images: &[Vec<u8>],
    mask: &[u8],
) -> Vec<CameraTrackReport> {
    let mut tracker = CameraTracker::try_new(width, height).unwrap();
    images
        .iter()
        .enumerate()
        .map(|(i, image)| {
            tracker.track_gray_copy(black_box(image), black_box(mask), i as f64 * 125.).unwrap()
        })
        .collect()
}
fn main() {
    let mut rows = Vec::new();
    for (width, height) in [(192, 144), (256, 192), (320, 240)] {
        let images: Vec<_> = (0..7).map(|frame| texture(width, height, frame)).collect();
        for kind in ["disabled", "zero", "full", "half", "sparse", "checker"] {
            let input = mask(width, height, kind);
            let expected = sequence(width, height, &images, &input);
            let mut elapsed_ns = Vec::new();
            for _ in 0..6 {
                let start = Instant::now();
                let reports = sequence(width, height, &images, &input);
                elapsed_ns.push(start.elapsed().as_nanos() as u64);
                assert_eq!(reports, expected);
                black_box(&reports);
            }
            rows.push(serde_json::json!({"width": width, "height": height, "mask": kind, "frames": 7, "samples_ns": elapsed_ns, "reports": expected}));
        }
    }
    println!(
        "{}",
        serde_json::json!({"schema": "spatialrust.camera-mask-support-benchmark.v1", "synthetic_only": true, "rows": rows})
    );
}
