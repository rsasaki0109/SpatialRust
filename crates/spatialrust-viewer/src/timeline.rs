use spatialrust_viz::PointCloudView;

use crate::{ViewerError, ViewerResult};

/// RGB, depth, and cloud timestamps associated with one synchronized frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameTimestamps {
    /// RGB timestamp in nanoseconds.
    pub rgb_nanos: u64,
    /// Depth timestamp in nanoseconds.
    pub depth_nanos: u64,
    /// Optional point-cloud timestamp in nanoseconds.
    pub cloud_nanos: Option<u64>,
}

impl FrameTimestamps {
    fn range(self) -> (u64, u64) {
        let mut min = self.rgb_nanos.min(self.depth_nanos);
        let mut max = self.rgb_nanos.max(self.depth_nanos);
        if let Some(cloud) = self.cloud_nanos {
            min = min.min(cloud);
            max = max.max(cloud);
        }
        (min, max)
    }

    /// Deterministic display timestamp at the midpoint of the sensor range.
    #[must_use]
    pub fn display_nanos(self) -> u64 {
        let (min, max) = self.range();
        min + (max - min) / 2
    }
}

/// Borrowed synchronized RGB-D/cloud frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbdFrameView<'a> {
    /// Stable coordinate-frame identifier.
    pub frame_id: &'a str,
    /// Sensor timestamps.
    pub timestamps: FrameTimestamps,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Interleaved RGB8 pixels.
    pub rgb: &'a [u8],
    /// Metric row-major depth.
    pub depth: &'a [f32],
    /// Optional synchronized point-cloud view.
    pub cloud: Option<PointCloudView<'a>>,
}

impl<'a> RgbdFrameView<'a> {
    /// Validates dimensions, finite depth values, timestamps, and cloud alignment.
    pub fn validate(self, max_skew_nanos: u64) -> ViewerResult<()> {
        if self.frame_id.trim().is_empty() || self.width == 0 || self.height == 0 {
            return Err(ViewerError::InvalidState(
                "RGB-D frame ID and dimensions must be non-empty".into(),
            ));
        }
        let pixels = self
            .width
            .checked_mul(self.height)
            .ok_or_else(|| ViewerError::InvalidState("RGB-D pixel count overflow".into()))?;
        let rgb_len = pixels
            .checked_mul(3)
            .ok_or_else(|| ViewerError::InvalidState("RGB byte count overflow".into()))?;
        if self.rgb.len() != rgb_len || self.depth.len() != pixels {
            return Err(ViewerError::InvalidState(
                "RGB/depth lengths must exactly match frame dimensions".into(),
            ));
        }
        if self.depth.iter().any(|depth| depth.is_infinite() || *depth < 0.0) {
            return Err(ViewerError::InvalidState(
                "depth values must be non-negative and not infinite".into(),
            ));
        }
        let (min, max) = self.timestamps.range();
        if max - min > max_skew_nanos {
            return Err(ViewerError::InvalidState(format!(
                "RGB-D/cloud timestamp skew {} exceeds {} ns",
                max - min,
                max_skew_nanos
            )));
        }
        if self.cloud.is_some() != self.timestamps.cloud_nanos.is_some() {
            return Err(ViewerError::InvalidState(
                "cloud payload and cloud timestamp must either both be present or absent".into(),
            ));
        }
        Ok(())
    }
}

/// Pixel-level RGB-D projection inspection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbdPixelSample {
    /// Pixel x coordinate.
    pub x: usize,
    /// Pixel y coordinate.
    pub y: usize,
    /// RGB value.
    pub rgb: [u8; 3],
    /// Metric depth.
    pub depth: f32,
    /// Camera-space XYZ when depth is finite and positive.
    pub camera_point: Option<spatialrust_math::Vec3<f64>>,
}

/// Ordered, borrowed synchronized sensor timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbdTimeline<'a> {
    frames: Vec<RgbdFrameView<'a>>,
    max_skew_nanos: u64,
}

impl<'a> RgbdTimeline<'a> {
    /// Creates a validated timeline in non-decreasing display timestamp order.
    pub fn try_new(
        frames: impl IntoIterator<Item = RgbdFrameView<'a>>,
        max_skew_nanos: u64,
    ) -> ViewerResult<Self> {
        let frames: Vec<_> = frames.into_iter().collect();
        let mut previous = None;
        for frame in &frames {
            frame.validate(max_skew_nanos)?;
            let timestamp = frame.timestamps.display_nanos();
            if previous.is_some_and(|value| timestamp < value) {
                return Err(ViewerError::InvalidState(
                    "RGB-D timeline timestamps must be non-decreasing".into(),
                ));
            }
            previous = Some(timestamp);
        }
        Ok(Self { frames, max_skew_nanos })
    }

    /// Ordered frames.
    #[must_use]
    pub fn frames(&self) -> &[RgbdFrameView<'a>] {
        &self.frames
    }

    /// Configured maximum sensor skew.
    #[must_use]
    pub const fn max_skew_nanos(&self) -> u64 {
        self.max_skew_nanos
    }

    /// Selects the nearest frame, preferring the earlier frame on ties.
    #[must_use]
    pub fn nearest(&self, timestamp_nanos: u64) -> Option<RgbdFrameView<'a>> {
        self.frames.iter().copied().min_by_key(|frame| {
            let stamp = frame.timestamps.display_nanos();
            (stamp.abs_diff(timestamp_nanos), stamp)
        })
    }

    /// Inspects one pixel and unprojects valid depth with a pinhole camera.
    #[cfg(feature = "camera")]
    pub fn inspect_pixel(
        &self,
        frame_index: usize,
        x: usize,
        y: usize,
        camera: &spatialrust_camera::PinholeCamera,
    ) -> ViewerResult<RgbdPixelSample> {
        let frame = self
            .frames
            .get(frame_index)
            .ok_or_else(|| ViewerError::InvalidState("RGB-D frame index out of bounds".into()))?;
        if camera.intrinsics.width != frame.width || camera.intrinsics.height != frame.height {
            return Err(ViewerError::InvalidState(
                "camera dimensions must match RGB-D frame".into(),
            ));
        }
        if x >= frame.width || y >= frame.height {
            return Err(ViewerError::InvalidState("RGB-D pixel out of bounds".into()));
        }
        let index = y * frame.width + x;
        let rgb_offset = index * 3;
        let depth = frame.depth[index];
        let camera_point = if depth.is_finite() && depth > 0.0 {
            Some(
                camera
                    .unproject(spatialrust_math::Vec2 { x: x as f64, y: y as f64 }, depth as f64)
                    .map_err(|error| ViewerError::InvalidState(error.to_string()))?,
            )
        } else {
            None
        };
        Ok(RgbdPixelSample {
            x,
            y,
            rgb: [frame.rgb[rgb_offset], frame.rgb[rgb_offset + 1], frame.rgb[rgb_offset + 2]],
            depth,
            camera_point,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameTimestamps, RgbdFrameView, RgbdTimeline};

    fn frame<'a>(
        rgb: &'a [u8],
        depth: &'a [f32],
        rgb_nanos: u64,
        depth_nanos: u64,
    ) -> RgbdFrameView<'a> {
        RgbdFrameView {
            frame_id: "camera",
            timestamps: FrameTimestamps { rgb_nanos, depth_nanos, cloud_nanos: None },
            width: 2,
            height: 1,
            rgb,
            depth,
            cloud: None,
        }
    }

    #[test]
    fn validates_alignment_and_selects_nearest_with_earlier_tie() {
        let rgb = [1, 2, 3, 4, 5, 6];
        let depth = [1.0, f32::NAN];
        let timeline =
            RgbdTimeline::try_new([frame(&rgb, &depth, 98, 102), frame(&rgb, &depth, 198, 202)], 5)
                .unwrap();
        assert_eq!(timeline.nearest(150).unwrap().timestamps.display_nanos(), 100);
        assert_eq!(timeline.max_skew_nanos(), 5);
    }

    #[test]
    fn rejects_skew_lengths_order_and_unpaired_cloud_timestamp() {
        let rgb = [0; 6];
        let depth = [1.0, 2.0];
        assert!(RgbdTimeline::try_new([frame(&rgb, &depth, 0, 10)], 5).is_err());
        assert!(RgbdTimeline::try_new(
            [frame(&rgb, &depth, 200, 200), frame(&rgb, &depth, 100, 100)],
            0
        )
        .is_err());
        let mut invalid = frame(&rgb, &depth, 0, 0);
        invalid.timestamps.cloud_nanos = Some(0);
        assert!(invalid.validate(0).is_err());
        assert!(frame(&rgb[..3], &depth, 0, 0).validate(0).is_err());
    }

    #[cfg(feature = "camera")]
    #[test]
    fn inspects_rgb_depth_and_projection_deterministically() {
        let rgb = [10, 20, 30, 40, 50, 60];
        let depth = [2.0, f32::NAN];
        let timeline = RgbdTimeline::try_new([frame(&rgb, &depth, 0, 0)], 0).unwrap();
        let camera = spatialrust_camera::PinholeCamera::new(
            spatialrust_camera::CameraIntrinsics::try_new(2.0, 2.0, 0.0, 0.0, 2, 1).unwrap(),
        );
        let sample = timeline.inspect_pixel(0, 0, 0, &camera).unwrap();
        assert_eq!(sample.rgb, [10, 20, 30]);
        assert_eq!(sample.camera_point.unwrap(), spatialrust_math::Vec3::new(0.0, 0.0, 2.0));
        assert!(timeline.inspect_pixel(0, 1, 0, &camera).unwrap().camera_point.is_none());
        assert!(timeline.inspect_pixel(0, 2, 0, &camera).is_err());
    }
}
