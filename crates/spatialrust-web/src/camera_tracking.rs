//! Bounded image-space background flow; not metric camera pose or SLAM.

use serde::Serialize;
use spatialrust_image::Image;
use spatialrust_math::Vec2;
use spatialrust_vision::{
    detect_fast, select_keypoints_grid, track_points_forward_backward, FastOptions,
    GridSelectionOptions, LucasKanadeOptions, RobustTrackOptions,
};

use crate::{WebError, WebResult};

const MAX_AXIS: usize = 320;
const MAX_POINTS: usize = 80;
const PATCH_MARGIN: usize = 14;

/// Whether a frame pair provides usable image-space motion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraTrackingStatus {
    /// First frame supplies features but no displacement.
    Warming,
    /// At least eight background tracks survived consistency checks.
    Tracking,
    /// Insufficient texture, visible background, or consistent tracks.
    Lost,
    /// A gap over 500ms starts a new frame pair.
    Reseeded,
}

/// One forward/backward-consistent background track, in processing pixels.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CameraTrack {
    /// Previous-frame pixel coordinate.
    pub from: [f64; 2],
    /// Current-frame pixel coordinate.
    pub to: [f64; 2],
}

/// Bounded visualization report; contains no image bytes or metric pose.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CameraTrackReport {
    /// Report schema version.
    pub schema: u8,
    /// Processing width.
    pub width: usize,
    /// Processing height.
    pub height: usize,
    /// Caller-supplied monotonic timestamp in milliseconds.
    pub timestamp_ms: f64,
    /// Current tracking state.
    pub status: CameraTrackingStatus,
    /// Component-wise median background displacement, absent when unqualified.
    pub background_flow_px: Option<[f64; 2]>,
    /// At most 80 accepted tracks.
    pub tracks: Vec<CameraTrack>,
    /// Features selected for the next frame pair.
    pub seeded_features: usize,
    /// Fraction of current pixels explicitly marked as person.
    pub person_fraction: f64,
}

/// Small owned frame-pair tracker for an opt-in camera playground.
///
/// Reuses FAST, grid selection and forward/backward LK. It retains one previous
/// gray frame and mask; caller data is explicitly copied by `track_gray_copy`.
/// This estimates 2D background flow, not camera translation, distance or 6DoF.
pub struct CameraTracker {
    width: usize,
    height: usize,
    previous: Option<Image<u8, 1>>,
    previous_mask: Vec<u8>,
    points: Vec<Vec2<f64>>,
    timestamp: Option<f64>,
}

impl CameraTracker {
    /// Creates a tracker bounded to 64..=320 pixels on each axis.
    pub fn try_new(width: usize, height: usize) -> WebResult<Self> {
        if !(64..=MAX_AXIS).contains(&width) || !(64..=MAX_AXIS).contains(&height) {
            return Err(invalid("camera frame axes must be in 64..=320"));
        }
        Ok(Self {
            width,
            height,
            previous: None,
            previous_mask: Vec::new(),
            points: Vec::new(),
            timestamp: None,
        })
    }

    /// Returns the exact accepted gray/mask shape.
    pub fn dimensions(&self) -> [usize; 2] {
        [self.width, self.height]
    }

    /// Drops retained pixels, mask, features and clock state.
    pub fn reset(&mut self) {
        self.previous = None;
        self.previous_mask.clear();
        self.points.clear();
        self.timestamp = None;
    }

    /// Explicitly copies a packed gray frame and optional binary person mask.
    ///
    /// An empty mask disables exclusion; otherwise `1` marks person and `0`
    /// background at the exact frame resolution. Both endpoints and their
    /// 14-pixel neighborhoods must be background. Invalid shape, labels or
    /// non-monotonic/non-finite timestamps reject before retained state changes.
    pub fn track_gray_copy(
        &mut self,
        gray: &[u8],
        mask: &[u8],
        timestamp_ms: f64,
    ) -> WebResult<CameraTrackReport> {
        let pixels = self.width * self.height;
        if gray.len() != pixels || (!mask.is_empty() && mask.len() != pixels) {
            return Err(invalid("gray/mask lengths must match processing dimensions"));
        }
        if mask.iter().any(|&value| value > 1) {
            return Err(invalid("person mask must contain only 0 or 1"));
        }
        if !timestamp_ms.is_finite()
            || timestamp_ms < 0.
            || self.timestamp.is_some_and(|old| timestamp_ms <= old)
        {
            return Err(invalid(
                "frame timestamps must be finite, non-negative and strictly increasing",
            ));
        }
        let current = Image::try_new(self.width, self.height, gray.to_vec()).map_err(invalid)?;
        let gap = self.timestamp.is_some_and(|old| timestamp_ms - old > 500.);
        let mut tracks = Vec::with_capacity(MAX_POINTS);
        if let Some(previous) = self.previous.as_ref().filter(|_| !gap && !self.points.is_empty()) {
            let options = RobustTrackOptions {
                lucas_kanade: LucasKanadeOptions {
                    window_radius: 3,
                    pyramid_levels: 3,
                    max_iterations: 12,
                    epsilon: 0.01,
                    min_eigenvalue: 1e-4,
                },
                max_forward_backward_error: 1.,
            };
            let result = track_points_forward_backward(
                previous.view(),
                current.view(),
                &self.points,
                options,
            )
            .map_err(invalid)?;
            for (i, from) in self.points.iter().enumerate() {
                let to = result.next_points()[i];
                if result.status()[i]
                    && self.background(&self.previous_mask, from.x, from.y)
                    && self.background(mask, to.x, to.y)
                {
                    tracks.push(CameraTrack { from: [from.x, from.y], to: [to.x, to.y] });
                }
            }
        }
        let candidates =
            detect_fast(current.view(), FastOptions { threshold: 12, nonmax_suppression: true })
                .map_err(invalid)?;
        let candidates: Vec<_> = candidates
            .into_iter()
            .filter(|p| self.background(mask, f64::from(p.x()), f64::from(p.y())))
            .collect();
        let candidates = select_keypoints_grid(
            &candidates,
            self.width,
            self.height,
            GridSelectionOptions { cell_width: 24, cell_height: 24, max_per_cell: 2 },
        )
        .map_err(invalid)?;
        let next_points: Vec<_> = candidates
            .into_iter()
            .take(MAX_POINTS)
            .map(|p| Vec2 { x: f64::from(p.x()), y: f64::from(p.y()) })
            .collect();
        let status = if self.previous.is_none() {
            CameraTrackingStatus::Warming
        } else if gap {
            CameraTrackingStatus::Reseeded
        } else if tracks.len() >= 8 {
            CameraTrackingStatus::Tracking
        } else {
            CameraTrackingStatus::Lost
        };
        let background_flow_px = if status == CameraTrackingStatus::Tracking {
            Some(std::array::from_fn(|axis| {
                let mut values: Vec<_> = tracks.iter().map(|p| p.to[axis] - p.from[axis]).collect();
                values.sort_by(f64::total_cmp);
                let middle = values.len() / 2;
                if values.len() % 2 == 0 {
                    (values[middle - 1] + values[middle]) * 0.5
                } else {
                    values[middle]
                }
            }))
        } else {
            None
        };
        let report = CameraTrackReport {
            schema: 1,
            width: self.width,
            height: self.height,
            timestamp_ms,
            status,
            background_flow_px,
            tracks,
            seeded_features: next_points.len(),
            person_fraction: mask.iter().filter(|&&v| v == 1).count() as f64 / pixels as f64,
        };
        self.previous = Some(current);
        self.previous_mask = mask.to_vec();
        self.points = next_points;
        self.timestamp = Some(timestamp_ms);
        Ok(report)
    }

    fn background(&self, mask: &[u8], x: f64, y: f64) -> bool {
        let margin = PATCH_MARGIN as f64;
        if !x.is_finite()
            || !y.is_finite()
            || x < margin
            || y < margin
            || x >= (self.width - PATCH_MARGIN) as f64
            || y >= (self.height - PATCH_MARGIN) as f64
        {
            return false;
        }
        if mask.is_empty() {
            return true;
        }
        let (x, y) = (x as usize, y as usize);
        (y - PATCH_MARGIN..=y + PATCH_MARGIN).all(|row| {
            mask[row * self.width + x - PATCH_MARGIN..=row * self.width + x + PATCH_MARGIN]
                .iter()
                .all(|&v| v == 0)
        })
    }
}

fn invalid(error: impl std::fmt::Display) -> WebError {
    WebError::InvalidState(error.to_string())
}

#[cfg(test)]
#[path = "camera_tracking_tests.rs"]
mod tests;
