/// Robust loss kernel used by registration and estimation algorithms.
pub trait RobustKernel {
    /// Computes the robust weight for a residual magnitude.
    fn weight(&self, residual: f64) -> f64;

    /// Computes the robust rho value for a residual magnitude.
    fn rho(&self, residual: f64) -> f64;
}

/// Huber robust kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HuberKernel {
    /// Threshold parameter.
    pub delta: f64,
}

impl HuberKernel {
    /// Creates a Huber kernel with the given threshold.
    #[must_use]
    pub const fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl RobustKernel for HuberKernel {
    fn weight(&self, residual: f64) -> f64 {
        let r = residual.abs();
        if r <= self.delta {
            1.0
        } else {
            self.delta / r
        }
    }

    fn rho(&self, residual: f64) -> f64 {
        let r = residual.abs();
        if r <= self.delta {
            0.5 * r * r
        } else {
            self.delta * (r - 0.5 * self.delta)
        }
    }
}

/// Cauchy robust kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CauchyKernel {
    /// Scale parameter.
    pub c: f64,
}

impl CauchyKernel {
    /// Creates a Cauchy kernel with the given scale.
    #[must_use]
    pub const fn new(c: f64) -> Self {
        Self { c }
    }
}

impl RobustKernel for CauchyKernel {
    fn weight(&self, residual: f64) -> f64 {
        let r2 = residual * residual;
        let c2 = self.c * self.c;
        1.0 / (1.0 + r2 / c2)
    }

    fn rho(&self, residual: f64) -> f64 {
        let r2 = residual * residual;
        let c2 = self.c * self.c;
        0.5 * c2 * (1.0 + r2 / c2).ln()
    }
}

/// Tukey biweight robust kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TukeyKernel {
    /// Threshold parameter.
    pub c: f64,
}

impl TukeyKernel {
    /// Creates a Tukey kernel with the given threshold.
    #[must_use]
    pub const fn new(c: f64) -> Self {
        Self { c }
    }
}

impl RobustKernel for TukeyKernel {
    fn weight(&self, residual: f64) -> f64 {
        let r = residual.abs();
        if r >= self.c {
            return 0.0;
        }
        let t = 1.0 - (r / self.c).powi(2);
        t * t
    }

    fn rho(&self, residual: f64) -> f64 {
        let r = residual.abs();
        if r >= self.c {
            return self.c * self.c / 6.0;
        }
        let t = 1.0 - (r / self.c).powi(2);
        (self.c * self.c / 6.0) * (1.0 - t.powi(3))
    }
}

#[cfg(test)]
mod tests {
    use super::{CauchyKernel, HuberKernel, RobustKernel, TukeyKernel};

    #[test]
    fn huber_weight_matches_reference() {
        let kernel = HuberKernel::new(1.0);
        assert_eq!(kernel.weight(0.5), 1.0);
        assert!((kernel.weight(2.0) - 0.5).abs() < 1e-12);
        assert!((kernel.rho(2.0) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn cauchy_and_tukey_are_bounded() {
        let cauchy = CauchyKernel::new(1.0);
        assert!(cauchy.weight(10.0) < 0.1);

        let tukey = TukeyKernel::new(1.0);
        assert_eq!(tukey.weight(2.0), 0.0);
    }
}
