/// Default tolerance for `f32` comparisons.
#[must_use]
pub const fn f32_eps() -> f32 {
    1e-6
}

/// Default tolerance for `f64` comparisons.
#[must_use]
pub const fn f64_eps() -> f64 {
    1e-12
}

/// Returns whether two values are approximately equal.
#[must_use]
pub fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() <= epsilon
}

/// Returns whether two values are approximately equal.
#[must_use]
pub fn approx_eq_f64(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() <= epsilon
}

/// Returns whether a value is near zero.
#[must_use]
pub fn near_zero(value: f32, epsilon: f32) -> bool {
    value.abs() <= epsilon
}

/// Returns whether a value is near zero.
#[must_use]
pub fn near_zero_f64(value: f64, epsilon: f64) -> bool {
    value.abs() <= epsilon
}

#[cfg(test)]
mod tests {
    use super::{approx_eq, f32_eps, near_zero};

    #[test]
    fn approx_eq_works() {
        assert!(approx_eq(1.0, 1.0 + 1e-7, f32_eps()));
        assert!(!approx_eq(1.0, 1.1, f32_eps()));
        assert!(near_zero(1e-7, f32_eps()));
    }
}
