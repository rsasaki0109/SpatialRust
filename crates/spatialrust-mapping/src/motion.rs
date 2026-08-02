//! Relative motion estimators used by localization pipelines.

use spatialrust_math::{Isometry3, Pose3, Quat, Vec3};
use spatialrust_sync::StampedTime;

use crate::{MappingResult, StampedPose};

/// Relative motion between two times.
#[derive(Clone, Debug, PartialEq)]
pub struct DeltaMotion {
    /// Start stamp.
    pub from: StampedTime,
    /// End stamp.
    pub to: StampedTime,
    /// Transform that maps `from` coordinates into `to` coordinates.
    pub to_t_from: Isometry3<f32>,
}

/// Estimates relative motion for odometry / keyframe tracking.
pub trait RelativeMotionEstimator {
    /// Estimates motion aligning `previous` toward `current` coordinates.
    fn estimate(&self, previous: &StampedPose, current: &StampedPose)
        -> MappingResult<DeltaMotion>;
}

/// Synthetic odometry that trusts successive pose stamps and emits their delta.
#[derive(Clone, Copy, Debug, Default)]
pub struct SyntheticOdometry;

impl RelativeMotionEstimator for SyntheticOdometry {
    fn estimate(
        &self,
        previous: &StampedPose,
        current: &StampedPose,
    ) -> MappingResult<DeltaMotion> {
        let to_t_from = current.pose.isometry.compose(previous.pose.isometry.inverse());
        Ok(DeltaMotion { from: previous.stamp.clone(), to: current.stamp.clone(), to_t_from })
    }
}

impl SyntheticOdometry {
    /// Integrates a translation-only delta onto a pose.
    #[must_use]
    pub fn integrate_translation(pose: Pose3<f32>, delta: Vec3<f32>) -> Pose3<f32> {
        let translation = pose.isometry.translation() + delta;
        Pose3::new(Isometry3::new(pose.isometry.rotation(), translation))
    }

    /// Builds a translation-only delta as an isometry.
    #[must_use]
    pub fn translation_delta(delta: Vec3<f32>) -> Isometry3<f32> {
        Isometry3::new(Quat::new(0.0, 0.0, 0.0, 1.0), delta)
    }
}

#[cfg(test)]
mod tests {
    use super::{RelativeMotionEstimator, SyntheticOdometry};
    use crate::StampedPose;
    use spatialrust_core::Timestamp;
    use spatialrust_math::{Isometry3, Pose3, Quat, Vec3};
    use spatialrust_sync::{ClockDomain, StampedTime};

    #[test]
    fn synthetic_odometry_recovers_translation_delta() {
        let previous = StampedPose::new(
            StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(0)),
            Pose3::new(Isometry3::new(Quat::new(0.0, 0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 0.0))),
        );
        let current = StampedPose::new(
            StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(1)),
            Pose3::new(Isometry3::new(Quat::new(0.0, 0.0, 0.0, 1.0), Vec3::new(1.0, 2.0, 0.0))),
        );
        let delta = SyntheticOdometry.estimate(&previous, &current).unwrap();
        assert!((delta.to_t_from.translation().x - 1.0).abs() < 1e-5);
        assert!((delta.to_t_from.translation().y - 2.0).abs() < 1e-5);
    }
}
