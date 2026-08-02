//! Trajectories, pose graphs, and localization primitives.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod motion;
mod pose_graph;
mod trajectory;
#[cfg(feature = "vision-odometry")]
mod vision;

pub use error::{MappingError, MappingResult};
pub use motion::{DeltaMotion, RelativeMotionEstimator, SyntheticOdometry};
pub use pose_graph::{PoseGraph, PoseGraphEdge, PoseNodeId};
pub use trajectory::{StampedPose, Trajectory};
#[cfg(feature = "vision-odometry")]
pub use vision::{delta_from_monocular_odometry, delta_from_rgbd_odometry};
