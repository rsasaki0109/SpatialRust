//! Point cloud registration for SpatialRust.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod kabsch;
mod registration;
mod transform;

#[cfg(feature = "register-icp")]
mod icp;

#[cfg(feature = "register-icp-point-to-plane")]
mod point_to_plane;

#[cfg(feature = "register-gicp")]
mod gicp;

#[cfg(feature = "register-ndt")]
mod ndt;

#[cfg(feature = "register-fpfh")]
mod fpfh;

pub use kabsch::estimate_rigid_transform;
pub use registration::{PointCloudRegistration, RegistrationResult};
pub use transform::transform_point_cloud;

#[cfg(feature = "register-icp")]
pub use icp::{IcpConfig, IcpRegistration};

#[cfg(feature = "register-icp-point-to-plane")]
pub use point_to_plane::{PointToPlaneIcp, PointToPlaneIcpConfig};

#[cfg(feature = "register-gicp")]
pub use gicp::{GicpConfig, GicpRegistration};

#[cfg(feature = "register-ndt")]
pub use ndt::{NdtConfig, NdtRegistration};

#[cfg(feature = "register-fpfh")]
pub use fpfh::{
    fpfh_descriptors, FpfhDescriptor, FpfhRansacConfig, FpfhRansacRegistration, FPFH_DESCRIPTOR_LEN,
};
