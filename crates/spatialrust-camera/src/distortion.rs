use spatialrust_math::Vec2;

/// Brown–Conrady radial and tangential lens-distortion coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BrownConrady {
    /// First radial coefficient.
    pub k1: f64,
    /// Second radial coefficient.
    pub k2: f64,
    /// First tangential coefficient.
    pub p1: f64,
    /// Second tangential coefficient.
    pub p2: f64,
    /// Third radial coefficient.
    pub k3: f64,
}

/// Kannala–Brandt four-coefficient fisheye model.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KannalaBrandt4 {
    /// Cubic angle coefficient.
    pub k1: f64,
    /// Fifth-order angle coefficient.
    pub k2: f64,
    /// Seventh-order angle coefficient.
    pub k3: f64,
    /// Ninth-order angle coefficient.
    pub k4: f64,
}

impl KannalaBrandt4 {
    /// Distorts normalized pinhole coordinates using the equidistant angle polynomial.
    #[must_use]
    pub fn distort(self, point: Vec2<f64>) -> Vec2<f64> {
        let radius = point.x.hypot(point.y);
        if radius <= f64::EPSILON {
            return point;
        }
        let theta = radius.atan();
        let theta2 = theta * theta;
        let distorted_theta = theta
            * (1.0
                + theta2 * (self.k1 + theta2 * (self.k2 + theta2 * (self.k3 + theta2 * self.k4))));
        let scale = distorted_theta / radius;
        Vec2 { x: point.x * scale, y: point.y * scale }
    }

    /// Iteratively removes fisheye distortion from normalized coordinates.
    #[must_use]
    pub fn undistort(self, point: Vec2<f64>) -> Vec2<f64> {
        let distorted_radius = point.x.hypot(point.y);
        if distorted_radius <= f64::EPSILON {
            return point;
        }
        let mut theta = distorted_radius.min(std::f64::consts::FRAC_PI_2 - 1e-6);
        for _ in 0..16 {
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta4 * theta4;
            let value = theta
                * (1.0 + self.k1 * theta2 + self.k2 * theta4 + self.k3 * theta6 + self.k4 * theta8)
                - distorted_radius;
            let derivative = 1.0
                + 3.0 * self.k1 * theta2
                + 5.0 * self.k2 * theta4
                + 7.0 * self.k3 * theta6
                + 9.0 * self.k4 * theta8;
            if derivative.abs() < 1e-12 {
                break;
            }
            let step = value / derivative;
            theta -= step;
            if step.abs() < 1e-13 {
                break;
            }
        }
        let radius = theta.tan();
        let scale = radius / distorted_radius;
        Vec2 { x: point.x * scale, y: point.y * scale }
    }
}

impl BrownConrady {
    /// Returns whether all coefficients are zero.
    #[must_use]
    pub fn is_identity(self) -> bool {
        self == Self::default()
    }

    /// Distorts normalized pinhole coordinates.
    #[must_use]
    pub fn distort(self, point: Vec2<f64>) -> Vec2<f64> {
        let x = point.x;
        let y = point.y;
        let r2 = x.mul_add(x, y * y);
        let radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3));
        Vec2 {
            x: x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x),
            y: y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y,
        }
    }

    /// Iteratively removes distortion from normalized coordinates.
    ///
    /// Newton iterations use a numerical 2x2 Jacobian and stop once normalized
    /// reprojection error reaches machine precision.
    #[must_use]
    pub fn undistort(self, distorted: Vec2<f64>) -> Vec2<f64> {
        if self.is_identity() {
            return distorted;
        }
        let mut estimate = distorted;
        const STEP: f64 = 1e-7;
        for _ in 0..12 {
            let observed = self.distort(estimate);
            let error = Vec2 { x: observed.x - distorted.x, y: observed.y - distorted.y };
            if error.x.abs().max(error.y.abs()) < 1e-14 {
                break;
            }
            let dx = self.distort(Vec2 { x: estimate.x + STEP, y: estimate.y });
            let dy = self.distort(Vec2 { x: estimate.x, y: estimate.y + STEP });
            let j00 = (dx.x - observed.x) / STEP;
            let j10 = (dx.y - observed.y) / STEP;
            let j01 = (dy.x - observed.x) / STEP;
            let j11 = (dy.y - observed.y) / STEP;
            let determinant = j00.mul_add(j11, -(j01 * j10));
            if determinant.abs() < f64::EPSILON {
                break;
            }
            let delta_x = (j11 * error.x - j01 * error.y) / determinant;
            let delta_y = (-j10 * error.x + j00 * error.y) / determinant;
            estimate.x -= delta_x;
            estimate.y -= delta_y;
        }
        estimate
    }
}

#[cfg(test)]
mod tests {
    use super::{BrownConrady, KannalaBrandt4};
    use spatialrust_math::Vec2;

    #[test]
    fn distortion_roundtrip() {
        let model = BrownConrady { k1: -0.2, k2: 0.03, p1: 0.001, p2: -0.002, k3: 0.0 };
        for point in [Vec2 { x: -0.4, y: 0.3 }, Vec2 { x: 0.2, y: -0.1 }] {
            let recovered = model.undistort(model.distort(point));
            assert!((recovered.x - point.x).abs() < 1e-9);
            assert!((recovered.y - point.y).abs() < 1e-9);
        }
    }

    #[test]
    fn fisheye_roundtrip() {
        let model = KannalaBrandt4 { k1: 0.02, k2: -0.003, k3: 0.0004, k4: -0.00002 };
        for point in [Vec2 { x: -0.8, y: 0.5 }, Vec2 { x: 0.2, y: -0.1 }] {
            let recovered = model.undistort(model.distort(point));
            assert!((recovered.x - point.x).abs() < 1e-10);
            assert!((recovered.y - point.y).abs() < 1e-10);
        }
    }
}
