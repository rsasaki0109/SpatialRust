//! Dense motion, background modeling, tracking, and video-source contracts.

#[cfg(feature = "video-adapters")]
use std::collections::VecDeque;

#[cfg(any(feature = "video-adapters", test))]
use spatialrust_image::Image;
use spatialrust_image::ImageView;

use crate::{BinaryMask, BoundingBox2, Detection, FlowField, VisionError, VisionResult};

/// Integer block-matching controls for dense grayscale optical flow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseFlowOptions {
    /// Half-size of the square comparison block.
    pub block_radius: usize,
    /// Maximum integer displacement searched in each axis.
    pub search_radius: usize,
    /// Minimum improvement over zero motion required for a valid vector.
    pub minimum_improvement: u64,
}

impl Default for DenseFlowOptions {
    fn default() -> Self {
        Self { block_radius: 2, search_radius: 4, minimum_improvement: 1 }
    }
}

/// Computes a dense integer flow field with deterministic local block matching.
///
/// Pixels whose full block/search window leaves the image are marked with NaN.
pub fn dense_flow_block_match(
    previous: ImageView<'_, u8, 1>,
    next: ImageView<'_, u8, 1>,
    options: DenseFlowOptions,
) -> VisionResult<FlowField> {
    if (previous.width(), previous.height()) != (next.width(), next.height()) {
        return Err(VisionError::ShapeMismatch(
            "dense-flow frames must share dimensions".to_owned(),
        ));
    }
    if options.block_radius == 0 || options.search_radius == 0 {
        return Err(VisionError::InvalidParameter(
            "dense-flow block/search radii must be positive".to_owned(),
        ));
    }
    let margin = options.block_radius.saturating_add(options.search_radius);
    if previous.width() <= margin * 2 || previous.height() <= margin * 2 {
        return Err(VisionError::InvalidDimensions(
            "dense-flow frames are too small for the configured search".to_owned(),
        ));
    }
    let mut flow = vec![f32::NAN; previous.width() * previous.height() * 2];
    for y in margin..previous.height() - margin {
        for x in margin..previous.width() - margin {
            let zero_cost = block_cost(previous, next, x, y, 0, 0, options.block_radius);
            let mut best_cost = zero_cost;
            let mut best = (0_isize, 0_isize);
            let search = options.search_radius as isize;
            for dy in -search..=search {
                for dx in -search..=search {
                    let cost = block_cost(previous, next, x, y, dx, dy, options.block_radius);
                    if cost < best_cost
                        || (cost == best_cost
                            && (dy.abs() + dx.abs(), dy, dx)
                                < (best.1.abs() + best.0.abs(), best.1, best.0))
                    {
                        best_cost = cost;
                        best = (dx, dy);
                    }
                }
            }
            if zero_cost.saturating_sub(best_cost) >= options.minimum_improvement {
                let offset = (y * previous.width() + x) * 2;
                flow[offset] = best.0 as f32;
                flow[offset + 1] = best.1 as f32;
            }
        }
    }
    FlowField::try_new(previous.width(), previous.height(), flow)
}

fn block_cost(
    previous: ImageView<'_, u8, 1>,
    next: ImageView<'_, u8, 1>,
    x: usize,
    y: usize,
    dx: isize,
    dy: isize,
    radius: usize,
) -> u64 {
    let mut cost = 0_u64;
    let radius = radius as isize;
    for by in -radius..=radius {
        for bx in -radius..=radius {
            let px = (x as isize + bx) as usize;
            let py = (y as isize + by) as usize;
            let nx = (x as isize + bx + dx) as usize;
            let ny = (y as isize + by + dy) as usize;
            let left = previous.get(px, py).expect("validated flow window")[0];
            let right = next.get(nx, ny).expect("validated flow window")[0];
            let difference = i32::from(left) - i32::from(right);
            cost = cost.saturating_add((difference * difference) as u64);
        }
    }
    cost
}

/// Adaptive Gaussian background-model controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackgroundModelOptions {
    /// Exponential update factor in `(0, 1]`.
    pub learning_rate: f32,
    /// Foreground threshold measured in standard deviations.
    pub sigma_threshold: f32,
    /// Initial per-pixel variance.
    pub initial_variance: f32,
}

impl Default for BackgroundModelOptions {
    fn default() -> Self {
        Self { learning_rate: 0.02, sigma_threshold: 3.0, initial_variance: 225.0 }
    }
}

/// Stateful single-Gaussian grayscale background model.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveBackgroundModel {
    width: usize,
    height: usize,
    mean: Vec<f32>,
    variance: Vec<f32>,
    options: BackgroundModelOptions,
    frames_seen: u64,
}

impl AdaptiveBackgroundModel {
    /// Initializes the model from its first frame.
    pub fn try_new(
        frame: ImageView<'_, u8, 1>,
        options: BackgroundModelOptions,
    ) -> VisionResult<Self> {
        validate_background_options(options)?;
        let mean = packed_gray(frame).into_iter().map(f32::from).collect();
        Ok(Self {
            width: frame.width(),
            height: frame.height(),
            mean,
            variance: vec![options.initial_variance; frame.width() * frame.height()],
            options,
            frames_seen: 1,
        })
    }

    /// Segments foreground and updates background statistics in one explicit step.
    pub fn apply(&mut self, frame: ImageView<'_, u8, 1>) -> VisionResult<BackgroundUpdate> {
        if (frame.width(), frame.height()) != (self.width, self.height) {
            return Err(VisionError::ShapeMismatch(
                "background-model frame dimensions changed".to_owned(),
            ));
        }
        let pixels = packed_gray(frame);
        let mut mask = Vec::with_capacity(pixels.len());
        for (index, &pixel) in pixels.iter().enumerate() {
            let value = f32::from(pixel);
            let delta = value - self.mean[index];
            let threshold = self.options.sigma_threshold * self.variance[index].max(1.0).sqrt();
            let foreground = delta.abs() > threshold;
            mask.push(u8::from(foreground));
            let alpha = if foreground {
                self.options.learning_rate * 0.1
            } else {
                self.options.learning_rate
            };
            self.mean[index] += alpha * delta;
            self.variance[index] =
                ((1.0 - alpha) * self.variance[index] + alpha * delta * delta).max(1.0);
        }
        self.frames_seen = self.frames_seen.saturating_add(1);
        let mask = BinaryMask::try_new(self.width, self.height, mask)?;
        let foreground_ratio = mask.area() as f32 / (self.width * self.height).max(1) as f32;
        Ok(BackgroundUpdate { mask, foreground_ratio, frame_index: self.frames_seen - 1 })
    }

    /// Returns the number of frames incorporated into the model.
    #[must_use]
    pub const fn frames_seen(&self) -> u64 {
        self.frames_seen
    }
}

/// Output of one background-model update.
#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundUpdate {
    /// Binary foreground segmentation.
    pub mask: BinaryMask,
    /// Foreground area divided by image area.
    pub foreground_ratio: f32,
    /// Zero-based sequence index of the processed frame.
    pub frame_index: u64,
}

fn validate_background_options(options: BackgroundModelOptions) -> VisionResult<()> {
    if !options.learning_rate.is_finite()
        || !(0.0..=1.0).contains(&options.learning_rate)
        || options.learning_rate == 0.0
        || !options.sigma_threshold.is_finite()
        || options.sigma_threshold <= 0.0
        || !options.initial_variance.is_finite()
        || options.initial_variance <= 0.0
    {
        return Err(VisionError::InvalidParameter(
            "invalid background-model learning/threshold/variance options".to_owned(),
        ));
    }
    Ok(())
}

fn packed_gray(frame: ImageView<'_, u8, 1>) -> Vec<u8> {
    let mut data = Vec::with_capacity(frame.width() * frame.height());
    for y in 0..frame.height() {
        data.extend_from_slice(frame.row(y).expect("frame row in bounds"));
    }
    data
}

/// Track lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackState {
    /// Track has not accumulated the configured hit count.
    Tentative,
    /// Track has accumulated enough consecutive associations.
    Confirmed,
}

/// One deterministic detection track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectTrack {
    /// Monotonic track identifier.
    pub id: u64,
    /// Most recent bounding box.
    pub bbox: BoundingBox2,
    /// Model-defined class identifier.
    pub class_id: i64,
    /// Most recent detection score.
    pub score: f32,
    /// Frames since track creation.
    pub age: u32,
    /// Total successful associations.
    pub hits: u32,
    /// Consecutive frames without an association.
    pub missed: u32,
    /// Tentative/confirmed state.
    pub state: TrackState,
}

/// IoU tracker controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiObjectTrackerOptions {
    /// Minimum IoU for same-class association.
    pub iou_threshold: f32,
    /// Consecutive misses retained before removal.
    pub max_missed: u32,
    /// Hits required to confirm a track.
    pub min_confirmed_hits: u32,
}

impl Default for MultiObjectTrackerOptions {
    fn default() -> Self {
        Self { iou_threshold: 0.3, max_missed: 3, min_confirmed_hits: 2 }
    }
}

/// Deterministic same-class greedy IoU multi-object tracker.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiObjectTracker {
    options: MultiObjectTrackerOptions,
    next_id: u64,
    tracks: Vec<ObjectTrack>,
}

impl MultiObjectTracker {
    /// Creates an empty tracker.
    pub fn try_new(options: MultiObjectTrackerOptions) -> VisionResult<Self> {
        if !options.iou_threshold.is_finite()
            || !(0.0..=1.0).contains(&options.iou_threshold)
            || options.min_confirmed_hits == 0
        {
            return Err(VisionError::InvalidParameter("invalid tracker options".to_owned()));
        }
        Ok(Self { options, next_id: 1, tracks: Vec::new() })
    }

    /// Associates one detection frame and returns current tracks ordered by id.
    pub fn update(&mut self, detections: &[Detection]) -> VisionResult<&[ObjectTrack]> {
        if detections
            .iter()
            .any(|detection| !detection.bbox.is_valid() || !detection.score.is_finite())
        {
            return Err(VisionError::InvalidParameter(
                "tracker detections must contain valid boxes and finite scores".to_owned(),
            ));
        }
        for track in &mut self.tracks {
            track.age = track.age.saturating_add(1);
            track.missed = track.missed.saturating_add(1);
        }
        let mut used = vec![false; detections.len()];
        for track in &mut self.tracks {
            let best = detections
                .iter()
                .enumerate()
                .filter(|(index, detection)| !used[*index] && detection.class_id == track.class_id)
                .map(|(index, detection)| (index, track.bbox.iou(detection.bbox)))
                .filter(|(_, iou)| *iou >= self.options.iou_threshold)
                .max_by(|left, right| {
                    left.1.total_cmp(&right.1).then_with(|| right.0.cmp(&left.0))
                });
            if let Some((index, _)) = best {
                used[index] = true;
                let detection = detections[index];
                track.bbox = detection.bbox;
                track.score = detection.score;
                track.hits = track.hits.saturating_add(1);
                track.missed = 0;
                if track.hits >= self.options.min_confirmed_hits {
                    track.state = TrackState::Confirmed;
                }
            }
        }
        for (index, &detection) in detections.iter().enumerate() {
            if !used[index] {
                let hits = 1;
                self.tracks.push(ObjectTrack {
                    id: self.next_id,
                    bbox: detection.bbox,
                    class_id: detection.class_id,
                    score: detection.score,
                    age: 1,
                    hits,
                    missed: 0,
                    state: if hits >= self.options.min_confirmed_hits {
                        TrackState::Confirmed
                    } else {
                        TrackState::Tentative
                    },
                });
                self.next_id = self.next_id.saturating_add(1);
            }
        }
        self.tracks.retain(|track| track.missed <= self.options.max_missed);
        self.tracks.sort_by_key(|track| track.id);
        Ok(&self.tracks)
    }

    /// Borrows current tracks ordered by id.
    #[must_use]
    pub fn tracks(&self) -> &[ObjectTrack] {
        &self.tracks
    }
}

/// Timestamped owned video frame.
#[cfg(feature = "video-adapters")]
#[derive(Clone, Debug, PartialEq)]
pub struct VideoFrame<T, const CHANNELS: usize> {
    /// Monotonic source sequence.
    pub sequence: u64,
    /// Source timestamp in nanoseconds.
    pub timestamp_ns: i64,
    /// Owned typed image; adapters do not hide host/device copies.
    pub image: Image<T, CHANNELS>,
}

/// Pull-based video source implemented by optional codec/camera adapters.
#[cfg(feature = "video-adapters")]
pub trait VideoFrameSource<T, const CHANNELS: usize> {
    /// Returns the next frame, or `None` at end of stream.
    fn next_frame(&mut self) -> VisionResult<Option<VideoFrame<T, CHANNELS>>>;
}

/// Deterministic in-memory source used for adapter conformance and replay.
#[cfg(feature = "video-adapters")]
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryVideoSource<T, const CHANNELS: usize> {
    frames: VecDeque<VideoFrame<T, CHANNELS>>,
}

#[cfg(feature = "video-adapters")]
impl<T, const CHANNELS: usize> MemoryVideoSource<T, CHANNELS> {
    /// Creates a source and validates strictly increasing sequence/timestamps.
    pub fn try_new(frames: Vec<VideoFrame<T, CHANNELS>>) -> VisionResult<Self> {
        if frames.windows(2).any(|pair| {
            pair[0].sequence >= pair[1].sequence
                || pair[0].timestamp_ns >= pair[1].timestamp_ns
                || (pair[0].image.width(), pair[0].image.height())
                    != (pair[1].image.width(), pair[1].image.height())
        }) {
            return Err(VisionError::InvalidParameter(
                "video frames need increasing sequence/time and stable dimensions".to_owned(),
            ));
        }
        Ok(Self { frames: frames.into() })
    }
}

#[cfg(feature = "video-adapters")]
impl<T, const CHANNELS: usize> VideoFrameSource<T, CHANNELS> for MemoryVideoSource<T, CHANNELS> {
    fn next_frame(&mut self) -> VisionResult<Option<VideoFrame<T, CHANNELS>>> {
        Ok(self.frames.pop_front())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translated_texture(
        width: usize,
        height: usize,
        dx: usize,
        dy: usize,
    ) -> (Image<u8, 1>, Image<u8, 1>) {
        let mut previous = vec![0_u8; width * height];
        for y in 0..height {
            for x in 0..width {
                previous[y * width + x] = ((x * 17 + y * 29 + x * y * 3) % 251) as u8;
            }
        }
        let mut next = vec![0_u8; width * height];
        for y in 0..height - dy {
            for x in 0..width - dx {
                next[(y + dy) * width + x + dx] = previous[y * width + x];
            }
        }
        (
            Image::try_new(width, height, previous).unwrap(),
            Image::try_new(width, height, next).unwrap(),
        )
    }

    #[test]
    fn dense_flow_recovers_integer_translation() {
        let (previous, next) = translated_texture(40, 32, 2, 1);
        let flow =
            dense_flow_block_match(previous.view(), next.view(), DenseFlowOptions::default())
                .unwrap();
        let center = flow.image().get(20, 16).unwrap();
        assert_eq!(center, &[2.0, 1.0]);
        assert!(flow.image().get(0, 0).unwrap().iter().all(|value| value.is_nan()));
    }

    #[test]
    fn background_model_detects_new_foreground() {
        let base = Image::<u8, 1>::try_new(8, 8, vec![20; 64]).unwrap();
        let mut model =
            AdaptiveBackgroundModel::try_new(base.view(), BackgroundModelOptions::default())
                .unwrap();
        let mut changed = vec![20_u8; 64];
        for y in 2..6 {
            for x in 3..5 {
                changed[y * 8 + x] = 200;
            }
        }
        let frame = Image::<u8, 1>::try_new(8, 8, changed).unwrap();
        let update = model.apply(frame.view()).unwrap();
        assert_eq!(update.mask.area(), 8);
        assert_eq!(update.frame_index, 1);
    }

    #[test]
    fn tracker_preserves_id_and_expires_misses() {
        let mut tracker = MultiObjectTracker::try_new(MultiObjectTrackerOptions {
            max_missed: 1,
            ..MultiObjectTrackerOptions::default()
        })
        .unwrap();
        let detection = |x: f32| Detection {
            bbox: BoundingBox2::try_new(x, 0.0, x + 10.0, 10.0).unwrap(),
            score: 0.9,
            class_id: 1,
        };
        assert_eq!(tracker.update(&[detection(0.0)]).unwrap()[0].id, 1);
        let tracks = tracker.update(&[detection(1.0)]).unwrap();
        assert_eq!(tracks[0].id, 1);
        assert_eq!(tracks[0].state, TrackState::Confirmed);
        assert_eq!(tracker.update(&[]).unwrap().len(), 1);
        assert!(tracker.update(&[]).unwrap().is_empty());
    }

    #[cfg(feature = "video-adapters")]
    #[test]
    fn memory_source_preserves_sequence_and_end_of_stream() {
        let frames = (0..3)
            .map(|sequence| VideoFrame {
                sequence,
                timestamp_ns: sequence as i64 * 10,
                image: Image::<u8, 1>::try_new(2, 2, vec![sequence as u8; 4]).unwrap(),
            })
            .collect();
        let mut source = MemoryVideoSource::try_new(frames).unwrap();
        for sequence in 0..3 {
            assert_eq!(source.next_frame().unwrap().unwrap().sequence, sequence);
        }
        assert!(source.next_frame().unwrap().is_none());
    }
}
