//! Trajectories, pose graphs, and localization primitives.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod motion;
mod pose_graph;
mod trajectory;

pub use error::{MappingError, MappingResult};
pub use motion::{DeltaMotion, RelativeMotionEstimator, SyntheticOdometry};
pub use pose_graph::{PoseGraph, PoseGraphEdge, PoseNodeId};
pub use trajectory::{StampedPose, Trajectory};
