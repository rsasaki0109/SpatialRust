use crate::{Mat3, Vec3};

/// Accumulates mean and covariance for 3D points.
#[derive(Clone, Debug, PartialEq)]
pub struct CovarianceAccumulator3 {
    count: u64,
    sum: [f64; 3],
    sum_sq: [f64; 6],
}

impl Default for CovarianceAccumulator3 {
    fn default() -> Self {
        Self::new()
    }
}

impl CovarianceAccumulator3 {
    /// Creates an empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self { count: 0, sum: [0.0; 3], sum_sq: [0.0; 6] }
    }

    /// Adds one point sample.
    pub fn push(&mut self, point: Vec3<f32>) {
        self.count += 1;
        self.sum[0] += f64::from(point.x);
        self.sum[1] += f64::from(point.y);
        self.sum[2] += f64::from(point.z);
        self.sum_sq[0] += f64::from(point.x * point.x);
        self.sum_sq[1] += f64::from(point.y * point.y);
        self.sum_sq[2] += f64::from(point.z * point.z);
        self.sum_sq[3] += f64::from(point.x * point.y);
        self.sum_sq[4] += f64::from(point.x * point.z);
        self.sum_sq[5] += f64::from(point.y * point.z);
    }

    /// Returns the number of accumulated samples.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Computes the sample mean.
    #[must_use]
    pub fn mean(&self) -> Option<Vec3<f64>> {
        if self.count == 0 {
            return None;
        }
        let n = self.count as f64;
        Some(Vec3::new(self.sum[0] / n, self.sum[1] / n, self.sum[2] / n))
    }

    /// Computes the unbiased sample covariance matrix.
    #[must_use]
    pub fn covariance(&self) -> Option<Mat3<f64>> {
        if self.count < 2 {
            return None;
        }
        let n = self.count as f64;
        let mean = self.mean()?;
        let inv = 1.0 / (n - 1.0);

        let c00 = inv * (self.sum_sq[0] - n * mean.x * mean.x);
        let c11 = inv * (self.sum_sq[1] - n * mean.y * mean.y);
        let c22 = inv * (self.sum_sq[2] - n * mean.z * mean.z);
        let c01 = inv * (self.sum_sq[3] - n * mean.x * mean.y);
        let c02 = inv * (self.sum_sq[4] - n * mean.x * mean.z);
        let c12 = inv * (self.sum_sq[5] - n * mean.y * mean.z);

        Some(Mat3::from_rows([c00, c01, c02], [c01, c11, c12], [c02, c12, c22]))
    }
}

#[cfg(test)]
mod tests {
    use super::CovarianceAccumulator3;
    use crate::{tolerance::approx_eq_f64, Vec3};

    #[test]
    fn covariance_of_axis_points() {
        let mut acc = CovarianceAccumulator3::new();
        acc.push(Vec3::new(0.0, 0.0, 0.0));
        acc.push(Vec3::new(1.0, 0.0, 0.0));
        acc.push(Vec3::new(2.0, 0.0, 0.0));
        let cov = acc.covariance().unwrap();
        assert!(approx_eq_f64(cov.m[0][0], 1.0, 1e-6));
        assert!(approx_eq_f64(cov.m[1][1], 0.0, 1e-6));
        assert!(approx_eq_f64(cov.m[2][2], 0.0, 1e-6));
    }
}
