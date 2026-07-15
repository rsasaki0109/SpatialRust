//! Camera models, projection geometry, and RGB-D conversion.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod distortion;
mod model;
/// Dense RGB-D fill may use audited AVX2 kernels on x86_64.
#[allow(unsafe_code)]
mod rgbd;

pub use distortion::BrownConrady;
pub use model::{CameraError, CameraIntrinsics, PinholeCamera};
pub use rgbd::{
    depth_to_point_cloud, depth_to_xyz_dense, depth_to_xyz_dense_into, rgbd_to_point_cloud,
    DepthConversionOptions, RgbdError,
};
