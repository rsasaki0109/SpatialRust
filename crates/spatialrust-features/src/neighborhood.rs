use spatialrust_core::{HasPositions3, SpatialError, SpatialResult};
use spatialrust_search::{KdTree, NearestNeighborIndex, RadiusSearchIndex};

/// Neighborhood query abstraction for feature estimation.
pub trait NeighborhoodProvider {
    /// Returns up to `k` neighbor indices excluding the query point itself.
    fn query_k(&self, index: usize, k: usize) -> SpatialResult<Vec<usize>>;

    /// Returns all neighbor indices within `radius`, excluding the query point.
    fn query_radius(&self, index: usize, radius: f32) -> SpatialResult<Vec<usize>>;
}

/// KD-tree backed neighborhood provider.
#[derive(Clone, Debug)]
pub struct KdTreeNeighborhood {
    tree: KdTree,
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
}

impl KdTreeNeighborhood {
    /// Builds a neighborhood provider from a point cloud.
    pub fn from_point_cloud(cloud: &spatialrust_core::PointCloud) -> SpatialResult<Self> {
        let (x, y, z) = cloud.positions3()?;
        Ok(Self {
            tree: KdTree::from_slices(x, y, z),
            x: x.to_vec(),
            y: y.to_vec(),
            z: z.to_vec(),
        })
    }
}

impl NeighborhoodProvider for KdTreeNeighborhood {
    fn query_k(&self, index: usize, k: usize) -> SpatialResult<Vec<usize>> {
        if index >= self.x.len() {
            return Err(SpatialError::InvalidArgument(format!(
                "neighbor query index out of bounds: {index}"
            )));
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let neighbors = self.tree.nearest_k(
            self.x[index],
            self.y[index],
            self.z[index],
            k.saturating_add(1),
        );
        Ok(neighbors
            .into_iter()
            .map(|neighbor| neighbor.index)
            .filter(|&candidate| candidate != index)
            .take(k)
            .collect())
    }

    fn query_radius(&self, index: usize, radius: f32) -> SpatialResult<Vec<usize>> {
        if index >= self.x.len() {
            return Err(SpatialError::InvalidArgument(format!(
                "neighbor query index out of bounds: {index}"
            )));
        }
        if radius < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "radius must be non-negative".to_owned(),
            ));
        }

        let neighbors = self.tree.radius_search(self.x[index], self.y[index], self.z[index], radius);
        Ok(neighbors
            .into_iter()
            .map(|neighbor| neighbor.index)
            .filter(|&candidate| candidate != index)
            .collect())
    }
}
