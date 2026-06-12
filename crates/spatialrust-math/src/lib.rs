//! Lightweight math primitives for spatial computing.
//!
//! Native small types with optional interop conversions planned for later releases.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod covariance;
mod eigen;
mod linalg;
mod mat;
mod pose;
mod quat;
mod robust;
mod scalar;
mod tolerance;
mod transform;
mod vec;

pub use covariance::CovarianceAccumulator3;
pub use eigen::{smallest_eigenvector, symmetric_eigen3, SymmetricEigen3};
pub use linalg::{solve_linear_system, LeastSquaresResult};
pub use mat::{Mat3, Mat4};
pub use pose::{Cov3, Pose3};
pub use quat::Quat;
pub use robust::{CauchyKernel, HuberKernel, RobustKernel, TukeyKernel};
pub use scalar::{Real, Scalar};
pub use tolerance::{approx_eq, approx_eq_f64, f32_eps, f64_eps, near_zero, near_zero_f64};
pub use transform::{Isometry3, Transform3, TransformPoint};
pub use vec::{Vec2, Vec3, Vec4};
