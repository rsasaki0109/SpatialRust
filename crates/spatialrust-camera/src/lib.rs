//! Camera models, projection geometry, and RGB-D conversion.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod distortion;
mod model;
mod rgbd;

pub use distortion::BrownConrady;
pub use model::{CameraError, CameraIntrinsics, PinholeCamera};
pub use rgbd::{depth_to_point_cloud, rgbd_to_point_cloud, DepthConversionOptions, RgbdError};
