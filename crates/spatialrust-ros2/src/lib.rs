//! ROS 2 message and rosbag2 adapters for SpatialRust.
//!
//! Native ROS 2 executors and `rclrs` remain separate integration concerns.
//! Enable `rosbag2-sqlite` for the read-only SQLite bag source.

#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "rosbag2-sqlite")]
mod sqlite;

#[cfg(feature = "rosbag2-sqlite")]
pub use sqlite::{
    list_topics, Rosbag2Error, Rosbag2PointCloudSource, Rosbag2Result, Rosbag2Topic,
    ROSBAG2_POINT_XYZI_SCHEMA_ID, ROSBAG2_POINT_XYZ_SCHEMA_ID,
};
