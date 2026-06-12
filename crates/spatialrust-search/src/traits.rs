/// One neighbor search result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// Index of the neighbor point in the source cloud.
    pub index: usize,
    /// Squared Euclidean distance to the query point.
    pub distance_squared: f32,
}

/// Common spatial index operations.
pub trait SpatialIndex {
    /// Returns the number of indexed points.
    fn len(&self) -> usize;

    /// Returns whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Exact nearest neighbor queries.
pub trait NearestNeighborIndex: SpatialIndex {
    /// Finds the single nearest neighbor.
    fn nearest_one(&self, x: f32, y: f32, z: f32) -> Option<Neighbor>;

    /// Finds up to `k` nearest neighbors sorted by ascending distance.
    fn nearest_k(&self, x: f32, y: f32, z: f32, k: usize) -> Vec<Neighbor>;
}

/// Radius search queries.
pub trait RadiusSearchIndex: SpatialIndex {
    /// Finds all neighbors within `radius` (not squared).
    fn radius_search(&self, x: f32, y: f32, z: f32, radius: f32) -> Vec<Neighbor>;
}
