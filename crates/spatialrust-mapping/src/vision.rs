//! Explicit conversion from vision odometry estimates into mapping motion.

use spatialrust_math::{Isometry3, Mat3, Quat, Vec3};
use spatialrust_sync::StampedTime;
use spatialrust_vision::{MonocularOdometryEstimate, RgbdOdometryEstimate};

use crate::{DeltaMotion, MappingError, MappingResult};

/// Converts scale-ambiguous monocular motion after the caller supplies scale.
pub fn delta_from_monocular_odometry(
    from: StampedTime,
    to: StampedTime,
    estimate: &MonocularOdometryEstimate,
    translation_scale: f32,
) -> MappingResult<DeltaMotion> {
    if !translation_scale.is_finite() || translation_scale <= 0.0 {
        return Err(MappingError::InvalidConfiguration(
            "monocular translation scale must be finite and positive".into(),
        ));
    }
    let pose = estimate.pose;
    Ok(delta(from, to, pose.rotation(), pose.translation(), translation_scale))
}

/// Converts a metric RGB-D source-to-target pose into mapping motion.
pub fn delta_from_rgbd_odometry(
    from: StampedTime,
    to: StampedTime,
    estimate: &RgbdOdometryEstimate,
) -> DeltaMotion {
    let pose = estimate.pose;
    delta(from, to, pose.rotation(), pose.translation(), 1.0)
}

fn delta(
    from: StampedTime,
    to: StampedTime,
    rotation: Mat3<f64>,
    translation: Vec3<f64>,
    scale: f32,
) -> DeltaMotion {
    let matrix = Mat3::from_rows(
        [rotation.m[0][0] as f32, rotation.m[0][1] as f32, rotation.m[0][2] as f32],
        [rotation.m[1][0] as f32, rotation.m[1][1] as f32, rotation.m[1][2] as f32],
        [rotation.m[2][0] as f32, rotation.m[2][1] as f32, rotation.m[2][2] as f32],
    );
    let value = matrix_to_quaternion(matrix);
    let translation = Vec3::new(
        translation.x as f32 * scale,
        translation.y as f32 * scale,
        translation.z as f32 * scale,
    );
    DeltaMotion { from, to, to_t_from: Isometry3::new(value, translation) }
}

fn matrix_to_quaternion(matrix: Mat3<f32>) -> Quat<f32> {
    let trace = matrix.m[0][0] + matrix.m[1][1] + matrix.m[2][2];
    let quaternion = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        Quat::new(
            (matrix.m[2][1] - matrix.m[1][2]) / s,
            (matrix.m[0][2] - matrix.m[2][0]) / s,
            (matrix.m[1][0] - matrix.m[0][1]) / s,
            0.25 * s,
        )
    } else if matrix.m[0][0] > matrix.m[1][1] && matrix.m[0][0] > matrix.m[2][2] {
        let s = (1.0 + matrix.m[0][0] - matrix.m[1][1] - matrix.m[2][2]).sqrt() * 2.0;
        Quat::new(
            0.25 * s,
            (matrix.m[0][1] + matrix.m[1][0]) / s,
            (matrix.m[0][2] + matrix.m[2][0]) / s,
            (matrix.m[2][1] - matrix.m[1][2]) / s,
        )
    } else if matrix.m[1][1] > matrix.m[2][2] {
        let s = (1.0 + matrix.m[1][1] - matrix.m[0][0] - matrix.m[2][2]).sqrt() * 2.0;
        Quat::new(
            (matrix.m[0][1] + matrix.m[1][0]) / s,
            0.25 * s,
            (matrix.m[1][2] + matrix.m[2][1]) / s,
            (matrix.m[0][2] - matrix.m[2][0]) / s,
        )
    } else {
        let s = (1.0 + matrix.m[2][2] - matrix.m[0][0] - matrix.m[1][1]).sqrt() * 2.0;
        Quat::new(
            (matrix.m[0][2] + matrix.m[2][0]) / s,
            (matrix.m[1][2] + matrix.m[2][1]) / s,
            0.25 * s,
            (matrix.m[1][0] - matrix.m[0][1]) / s,
        )
    };
    quaternion.normalize()
}

#[cfg(test)]
mod tests {
    use super::delta_from_monocular_odometry;
    use spatialrust_core::Timestamp;
    use spatialrust_math::{Mat3, Vec3};
    use spatialrust_sync::{ClockDomain, StampedTime};
    use spatialrust_vision::{MonocularOdometryEstimate, RelativePose};

    #[test]
    fn monocular_bridge_applies_caller_scale() {
        let estimate = MonocularOdometryEstimate {
            pose: RelativePose::try_new(Mat3::<f64>::identity(), Vec3::new(1.0, 0.0, 0.0)).unwrap(),
            inliers: vec![true; 8],
            positive_depth_count: 8,
        };
        let stamp = |nanos| {
            StampedTime::exact("camera", ClockDomain::HostSteady, Timestamp::from_nanos(nanos))
        };
        let motion = delta_from_monocular_odometry(stamp(1), stamp(2), &estimate, 0.25).unwrap();
        assert!((motion.to_t_from.translation().x - 0.25).abs() < 1e-6);
        assert_eq!(motion.to_t_from.rotation(), spatialrust_math::Quat::<f32>::identity());
    }
}
