use spatialrust_core::PointSchema;

/// Options controlling point cloud reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadOptions {
    /// Optional schema subset to load.
    pub selected_fields: Option<Vec<String>>,
    /// Maximum number of points to read per chunk.
    pub chunk_size: Option<usize>,
}

/// Options controlling point cloud writes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WriteOptions {
    /// Optional schema mapping override.
    pub schema: Option<PointSchema>,
    /// Enable compression when supported by the format.
    pub compress: bool,
}
