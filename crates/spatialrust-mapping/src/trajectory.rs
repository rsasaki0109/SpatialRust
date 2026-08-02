//! Stamped pose trajectories.

use spatialrust_math::{Isometry3, Pose3};
use spatialrust_sync::StampedTime;

use crate::{MappingError, MappingResult};

/// One timed pose sample.
#[derive(Clone, Debug, PartialEq)]
pub struct StampedPose {
    /// Observation / estimate time.
    pub stamp: StampedTime,
    /// Pose in the trajectory frame.
    pub pose: Pose3<f32>,
}

impl StampedPose {
    /// Creates a stamped pose.
    #[must_use]
    pub fn new(stamp: StampedTime, pose: Pose3<f32>) -> Self {
        Self { stamp, pose }
    }
}

/// Ordered trajectory of stamped poses.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trajectory {
    samples: Vec<StampedPose>,
}

impl Trajectory {
    /// Creates an empty trajectory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a sample; timestamps must be non-decreasing.
    pub fn push(&mut self, sample: StampedPose) -> MappingResult<()> {
        if let Some(last) = self.samples.last() {
            if sample.stamp.as_nanos() < last.stamp.as_nanos() {
                return Err(MappingError::InvalidConfiguration(
                    "trajectory timestamps must be non-decreasing".into(),
                ));
            }
        }
        self.samples.push(sample);
        Ok(())
    }

    /// Returns all samples.
    #[must_use]
    pub fn samples(&self) -> &[StampedPose] {
        &self.samples
    }

    /// Returns the latest sample.
    #[must_use]
    pub fn last(&self) -> Option<&StampedPose> {
        self.samples.last()
    }

    /// Linearly interpolates translation between bracketing samples (rotation holds the earlier).
    pub fn interpolate(&self, nanos: u64) -> MappingResult<Pose3<f32>> {
        if self.samples.is_empty() {
            return Err(MappingError::Missing("trajectory samples".into()));
        }
        if nanos <= self.samples[0].stamp.as_nanos() {
            return Ok(self.samples[0].pose);
        }
        if let Some(last) = self.samples.last() {
            if nanos >= last.stamp.as_nanos() {
                return Ok(last.pose);
            }
        }
        for window in self.samples.windows(2) {
            let a = &window[0];
            let b = &window[1];
            if nanos >= a.stamp.as_nanos() && nanos <= b.stamp.as_nanos() {
                let span = b.stamp.as_nanos() - a.stamp.as_nanos();
                if span == 0 {
                    return Ok(a.pose);
                }
                let t = (nanos - a.stamp.as_nanos()) as f32 / span as f32;
                let ta = a.pose.isometry.translation();
                let tb = b.pose.isometry.translation();
                let translation = spatialrust_math::Vec3::new(
                    ta.x + (tb.x - ta.x) * t,
                    ta.y + (tb.y - ta.y) * t,
                    ta.z + (tb.z - ta.z) * t,
                );
                return Ok(Pose3::new(Isometry3::new(a.pose.isometry.rotation(), translation)));
            }
        }
        Err(MappingError::Graph("interpolation failed".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::{StampedPose, Trajectory};
    use spatialrust_core::Timestamp;
    use spatialrust_math::{Isometry3, Pose3, Quat, Vec3};
    use spatialrust_sync::{ClockDomain, StampedTime};

    #[test]
    fn interpolates_translation() {
        let mut traj = Trajectory::new();
        traj.push(StampedPose::new(
            StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(0)),
            Pose3::new(Isometry3::new(Quat::new(0.0, 0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 0.0))),
        ))
        .unwrap();
        traj.push(StampedPose::new(
            StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(10)),
            Pose3::new(Isometry3::new(Quat::new(0.0, 0.0, 0.0, 1.0), Vec3::new(10.0, 0.0, 0.0))),
        ))
        .unwrap();
        let mid = traj.interpolate(5).unwrap();
        assert!((mid.isometry.translation().x - 5.0).abs() < 1e-5);
    }
}
