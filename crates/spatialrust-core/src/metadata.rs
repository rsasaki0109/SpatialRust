use spatialrust_math::Vec3;

/// Coordinate frame identifier.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameId(pub String);

impl FrameId {
    /// Creates a new frame identifier.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

/// Timestamp in nanoseconds since an arbitrary epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Creates a timestamp from nanoseconds.
    #[must_use]
    pub const fn from_nanos(value: u64) -> Self {
        Self(value)
    }

    /// Returns the timestamp in nanoseconds.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }
}

/// Spatial metadata attached to point clouds and maps.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialMetadata {
    /// Coordinate frame identifier.
    pub frame_id: FrameId,
    /// Capture or observation timestamp.
    pub timestamp: Timestamp,
    /// Sensor origin in the frame.
    pub sensor_origin: Option<Vec3<f32>>,
    /// Length unit, defaulting to meters.
    pub unit: String,
}

impl SpatialMetadata {
    /// Creates metadata with the given frame and timestamp.
    #[must_use]
    pub fn new(frame_id: impl Into<FrameId>, timestamp: Timestamp) -> Self {
        Self { frame_id: frame_id.into(), timestamp, unit: "meter".to_owned(), ..Self::default() }
    }
}

impl From<String> for FrameId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for FrameId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}
