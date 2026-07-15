//! ROS 2 adaptation contracts (no rclrs dependency).

/// Hint describing a ROS 2 message mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ros2MessageHint {
    /// Fully-qualified ROS type name, e.g. `sensor_msgs/msg/PointCloud2`.
    pub type_name: String,
    /// SpatialRust topic / schema id.
    pub spatial_topic: String,
}

/// Adapter interface for ROS 2 type negotiation.
pub trait Ros2Adapter {
    /// Returns supported type mappings.
    fn supported_types(&self) -> &[Ros2MessageHint];

    /// Negotiates a preferred mapping for one ROS type.
    fn negotiate(&self, ros_type: &str) -> Option<&Ros2MessageHint>;
}

/// In-memory catalog adapter used by default builds with `ros2` enabled.
#[derive(Clone, Debug, Default)]
pub struct CatalogRos2Adapter {
    hints: Vec<Ros2MessageHint>,
}

impl CatalogRos2Adapter {
    /// Creates an adapter from a catalog.
    #[must_use]
    pub fn new(hints: Vec<Ros2MessageHint>) -> Self {
        Self { hints }
    }
}

impl Ros2Adapter for CatalogRos2Adapter {
    fn supported_types(&self) -> &[Ros2MessageHint] {
        &self.hints
    }

    fn negotiate(&self, ros_type: &str) -> Option<&Ros2MessageHint> {
        self.hints.iter().find(|hint| hint.type_name == ros_type)
    }
}
