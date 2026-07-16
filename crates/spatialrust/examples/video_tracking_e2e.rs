//! Loads a deterministic frame sequence, estimates optical flow, and tracks objects.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use spatialrust::image_io::{decode_path, DecodeOptions, DecodedPixels};
use spatialrust::vision::{
    dense_flow_block_match, BoundingBox2, DenseFlowOptions, Detection, MultiObjectTracker,
    MultiObjectTrackerOptions, ObjectTrack,
};
use spatialrust::Image;

fn main() {
    let frame_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/video-tracking-demo/frames"));
    generate_frames(&frame_dir);
    let frames = load_frames(&frame_dir);
    assert!(frames.len() >= 2, "need at least two generated PGM frames");

    let mut tracker = MultiObjectTracker::try_new(MultiObjectTrackerOptions {
        iou_threshold: 0.2,
        max_missed: 1,
        min_confirmed_hits: 2,
    })
    .expect("tracker options");
    let first_detections = detect_objects(&frames[0]);
    tracker.update(&first_detections).expect("first detections");

    let mut flow_pairs = 0usize;
    for (index, pair) in frames.windows(2).enumerate() {
        let flow = dense_flow_block_match(
            pair[0].view(),
            pair[1].view(),
            DenseFlowOptions { block_radius: 1, search_radius: 3, minimum_improvement: 1 },
        )
        .expect("dense optical flow");
        let previous_detections = detect_objects(&pair[0]);
        let detections = detect_objects(&pair[1]);
        let tracks = tracker.update(&detections).expect("tracking update");
        let object_flows = detection_center_flows(&flow, &previous_detections);
        assert_eq!(object_flows, vec![(1, 2.0, 1.0), (2, -2.0, -1.0)]);
        println!(
            "frame={:02} detections={} tracks={} flow_class1=(2,1) flow_class2=(-2,-1)",
            index + 1,
            detections.len(),
            tracks.len()
        );
        assert_stable_tracks(tracks);
        flow_pairs += 1;
    }
    assert_eq!(flow_pairs, frames.len() - 1);
    assert_eq!(tracker.tracks().len(), 2);
    println!(
        "video_tracking_e2e=ok frames={} flow_pairs={} stable_track_ids=1,2",
        frames.len(),
        flow_pairs
    );
}

fn generate_frames(directory: &Path) {
    std::fs::create_dir_all(directory)
        .unwrap_or_else(|error| panic!("create {}: {error}", directory.display()));
    for frame_index in 0..12usize {
        let width = 96usize;
        let height = 72usize;
        let mut pixels = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                pixels[y * width + x] = 20 + ((x * 3 + y * 5) % 20) as u8;
            }
        }
        paint_object(&mut pixels, width, 8 + frame_index * 2, 9 + frame_index, 18, 14, 150);
        paint_object(&mut pixels, width, 70 - frame_index * 2, 46 - frame_index, 16, 12, 220);
        let path = directory.join(format!("frame_{frame_index:02}.pgm"));
        let mut encoded = format!("P5\n{width} {height}\n255\n").into_bytes();
        encoded.extend_from_slice(&pixels);
        std::fs::write(&path, encoded)
            .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    }
}

fn paint_object(
    pixels: &mut [u8],
    width: usize,
    x0: usize,
    y0: usize,
    object_width: usize,
    object_height: usize,
    base: u8,
) {
    for local_y in 0..object_height {
        for local_x in 0..object_width {
            let x = x0 + local_x;
            let y = y0 + local_y;
            pixels[y * width + x] =
                base + ((local_x * 7 + local_y * 11 + local_x * local_y * 3) % 25) as u8;
        }
    }
}

fn load_frames(directory: &Path) -> Vec<Image<u8, 1>> {
    let mut paths = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        .map(|entry| entry.expect("frame directory entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "pgm"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let decoded = decode_path(&path, DecodeOptions::default())
                .unwrap_or_else(|error| panic!("decode {}: {error}", path.display()));
            match decoded.into_pixels() {
                DecodedPixels::Gray8(image) => image,
                _ => panic!("{} is not Gray8", path.display()),
            }
        })
        .collect()
}

fn detect_objects(image: &Image<u8, 1>) -> Vec<Detection> {
    let width = image.width();
    let height = image.height();
    let pixels = image.as_slice();
    let mut visited = vec![false; pixels.len()];
    let mut detections = Vec::new();
    for start in 0..pixels.len() {
        if visited[start] || pixels[start] < 100 {
            continue;
        }
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut maximum = 0u8;
        let mut area = 0usize;
        while let Some(index) = queue.pop_front() {
            let x = index % width;
            let y = index / width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            maximum = maximum.max(pixels[index]);
            area += 1;
            for neighbor in neighbors(x, y, width, height) {
                if !visited[neighbor] && pixels[neighbor] >= 100 {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        if area >= 20 {
            detections.push(Detection {
                bbox: BoundingBox2::try_new(
                    min_x as f32,
                    min_y as f32,
                    (max_x + 1) as f32,
                    (max_y + 1) as f32,
                )
                .expect("component box"),
                score: 1.0,
                class_id: if maximum >= 210 { 2 } else { 1 },
            });
        }
    }
    detections.sort_by_key(|detection| detection.class_id);
    detections
}

fn neighbors(x: usize, y: usize, width: usize, height: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(4);
    if x > 0 {
        result.push(y * width + x - 1);
    }
    if x + 1 < width {
        result.push(y * width + x + 1);
    }
    if y > 0 {
        result.push((y - 1) * width + x);
    }
    if y + 1 < height {
        result.push((y + 1) * width + x);
    }
    result
}

fn detection_center_flows(
    flow: &spatialrust::vision::FlowField,
    detections: &[Detection],
) -> Vec<(i64, f32, f32)> {
    detections
        .iter()
        .map(|detection| {
            let x = ((detection.bbox.x_min + detection.bbox.x_max) * 0.5) as usize;
            let y = ((detection.bbox.y_min + detection.bbox.y_max) * 0.5) as usize;
            let vector = flow.image().get(x, y).expect("box center inside flow");
            (detection.class_id, vector[0], vector[1])
        })
        .collect()
}

fn assert_stable_tracks(tracks: &[ObjectTrack]) {
    assert_eq!(tracks.len(), 2);
    assert_eq!(tracks[0].id, 1);
    assert_eq!(tracks[1].id, 2);
}
