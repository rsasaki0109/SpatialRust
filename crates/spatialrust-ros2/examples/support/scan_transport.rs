//! Private scan-to-volume frame contract shared by the mapping examples.

use spatialrust_math::Isometry3;

/// `A_T_B` maps B points into A. The volume gauge is the first body frame.
pub fn sensor_to_volume(
    _body_from_front: Isometry3<f32>,
    body_from_sensor: Isometry3<f32>,
    current_front_from_first_front: Isometry3<f32>,
) -> Result<Isometry3<f32>, &'static str> {
    // Preserve the original consumer algebra for the prospective regression run.
    Ok(body_from_sensor.compose(current_front_from_first_front))
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_math::{Quat, TransformPoint, Vec3};

    const S: f32 = std::f32::consts::FRAC_1_SQRT_2;

    fn rigid(q: [f32; 4], t: [f32; 3]) -> Isometry3<f32> {
        Isometry3::new(Quat::new(q[0], q[1], q[2], q[3]), Vec3::new(t[0], t[1], t[2]))
    }

    fn translation(t: [f32; 3]) -> Isometry3<f32> {
        rigid([0.0, 0.0, 0.0, 1.0], t)
    }

    fn point(p: [f32; 3]) -> Vec3<f32> {
        Vec3::new(p[0], p[1], p[2])
    }

    fn close(actual: Vec3<f32>, expected: [f32; 3]) {
        assert!((actual - point(expected)).norm() < 2e-5, "{actual:?} != {expected:?}");
    }

    #[test]
    fn translating_front_recovers_static_world_point() {
        let identity = Isometry3::identity();
        let pose = sensor_to_volume(identity, identity, translation([-1.0, 0.0, 0.0])).unwrap();
        // Body moves +1 in world x: a static world x=4 is observed at sensor x=3.
        close(pose.transform_point(point([3.0, 0.0, 0.0])), [4.0, 0.0, 0.0]);
    }

    #[test]
    fn rotating_rear_transports_sensor_origin_and_point() {
        let rear = translation([0.0, 2.0, 0.0]);
        let odometry = rigid([0.0, 0.0, -S, S], [0.0; 3]);
        let pose = sensor_to_volume(Isometry3::identity(), rear, odometry).unwrap();
        // Rz(+90) body motion takes the rear origin (0,2,0) to (-2,0,0).
        close(pose.transform_point(point([0.0; 3])), [-2.0, 0.0, 0.0]);
        close(pose.transform_point(point([0.0, -6.0, 0.0])), [4.0, 0.0, 0.0]);
    }

    // Physical fixture, independently expressed as signed-axis observations:
    // body motion Rz(+90), t=(3,-1,2); body_T_front Rx(+90), t=(1,2,3);
    // body_T_rear Ry(+90), t=(-2,1,0.5). No production compose/inverse in oracle.
    fn fixture() -> (Isometry3<f32>, Isometry3<f32>, Isometry3<f32>) {
        let front = rigid([S, 0.0, 0.0, S], [1.0, 2.0, 3.0]);
        let rear = rigid([0.0, S, 0.0, S], [-2.0, 1.0, 0.5]);
        // Rx(-90) Rz(-90) Rx(+90) = Ry(-90), with hand-derived (2,-2,0).
        let odometry = rigid([0.0, -S, 0.0, S], [2.0, -2.0, 0.0]);
        (front, rear, odometry)
    }

    fn observe_front(world: [f32; 3]) -> Vec3<f32> {
        let body = [world[1] + 1.0, 3.0 - world[0], world[2] - 2.0];
        point([body[0] - 1.0, body[2] - 3.0, 2.0 - body[1]])
    }

    fn observe_rear(world: [f32; 3]) -> Vec3<f32> {
        let body = [world[1] + 1.0, 3.0 - world[0], world[2] - 2.0];
        point([0.5 - body[2], body[1] - 1.0, body[0] + 2.0])
    }

    const WORLD: [[f32; 3]; 4] = [[4.0, 0.0, 0.0], [-1.0, 2.0, 5.0], [0.0; 3], [6.0, -3.0, 2.0]];

    #[test]
    fn noncommuting_front_extrinsic_recovers_static_scene() {
        let (front, _, odometry) = fixture();
        let pose = sensor_to_volume(front, front, odometry).unwrap();
        for world in WORLD {
            close(pose.transform_point(observe_front(world)), world);
        }
        close(pose.transform_point(point([0.0; 3])), [1.0, 0.0, 5.0]);
    }

    #[test]
    fn noncommuting_rear_extrinsic_recovers_same_static_scene() {
        let (front, rear, odometry) = fixture();
        let pose = sensor_to_volume(front, rear, odometry).unwrap();
        for world in WORLD {
            close(pose.transform_point(observe_rear(world)), world);
        }
        close(pose.transform_point(point([0.0; 3])), [2.0, -3.0, 2.5]);
    }

    #[test]
    fn inverse_only_rear_is_insufficient_with_lever_arm() {
        let (front, rear, odometry) = fixture();
        let observed = observe_rear(WORLD[0]);
        let wrong = rear.compose(odometry.inverse()).transform_point(observed);
        assert!((wrong - point(WORLD[0])).norm() > 1.0);
        close(sensor_to_volume(front, rear, odometry).unwrap().transform_point(observed), WORLD[0]);
    }

    #[test]
    fn first_pose_preserves_both_static_extrinsics() {
        let (front, rear, _) = fixture();
        let identity = Isometry3::identity();
        for extrinsic in [front, rear] {
            let pose = sensor_to_volume(front, extrinsic, identity).unwrap();
            for p in [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                let expected = extrinsic.transform_point(point(p));
                close(pose.transform_point(point(p)), [expected.x, expected.y, expected.z]);
            }
        }
    }

    #[test]
    fn identity_smoke_gauge_and_inputs_are_preserved() {
        let identity = Isometry3::identity();
        let pose = sensor_to_volume(identity, identity, identity).unwrap();
        close(pose.transform_point(point([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0]);
        let inputs = fixture();
        let before = inputs;
        sensor_to_volume(inputs.0, inputs.1, inputs.2).unwrap();
        assert_eq!(inputs, before);
    }

    #[test]
    fn rejects_nonfinite_front_extrinsic() {
        let identity = Isometry3::identity();
        assert!(sensor_to_volume(translation([f32::NAN, 0.0, 0.0]), identity, identity).is_err());
    }

    #[test]
    fn rejects_nonfinite_other_extrinsic() {
        let identity = Isometry3::identity();
        assert!(sensor_to_volume(
            identity,
            rigid([0.0, f32::INFINITY, 0.0, 1.0], [0.0; 3]),
            identity
        )
        .is_err());
    }

    #[test]
    fn rejects_nonfinite_odometry() {
        let identity = Isometry3::identity();
        assert!(sensor_to_volume(identity, identity, translation([0.0, f32::NEG_INFINITY, 0.0]))
            .is_err());
    }

    #[test]
    fn rejects_finite_input_composition_overflow() {
        let identity = Isometry3::identity();
        let front = translation([f32::MAX, 0.0, 0.0]);
        let odometry = translation([-f32::MAX, 0.0, 0.0]);
        assert!(sensor_to_volume(front, identity, odometry).is_err());
    }

    #[test]
    fn rejects_finite_input_inverse_overflow() {
        let identity = Isometry3::identity();
        let angle = std::f32::consts::FRAC_PI_4 / 2.0;
        let front = rigid([0.0, 0.0, angle.sin(), angle.cos()], [f32::MAX, f32::MAX, 0.0]);
        assert!(sensor_to_volume(front, identity, identity).is_err());
    }
}
