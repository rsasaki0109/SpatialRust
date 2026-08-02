//! Camera models, projection geometry, and RGB-D conversion.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod calibration;
mod distortion;
mod model;
/// Dense RGB-D fill may use audited AVX2 kernels on x86_64.
#[allow(unsafe_code)]
mod rgbd;

pub use calibration::{
    bundle_adjust_points, calibrate_fisheye, calibrate_hand_eye_translation, calibrate_pinhole,
    calibrate_stereo_translation, BundleObservation, BundleProblem, BundleView, CalibrationError,
    CalibrationOptions, CalibrationReport, FisheyeObservation, HandEyeMotionPair,
    PinholeObservation, RigidTransform3, StereoCalibration, StereoPointPair,
};
pub use distortion::{BrownConrady, KannalaBrandt4};
pub use model::{CameraError, CameraIntrinsics, PinholeCamera};
pub use rgbd::{
    depth_to_point_cloud, depth_to_xyz_dense, depth_to_xyz_dense_into, rgbd_to_point_cloud,
    DepthConversionOptions, RgbdError,
};
