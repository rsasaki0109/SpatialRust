use crate::{NearestNeighborIndex, Neighbor, RadiusSearchIndex, SpatialIndex};

/// Reference index using brute-force search for correctness tests.
#[derive(Clone, Debug)]
pub struct BruteForceIndex {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
}

impl BruteForceIndex {
    /// Builds a brute-force index from coordinate slices.
    #[must_use]
    pub fn from_slices(x: &[f32], y: &[f32], z: &[f32]) -> Self {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), z.len());
        Self { x: x.to_vec(), y: y.to_vec(), z: z.to_vec() }
    }
}

impl SpatialIndex for BruteForceIndex {
    fn len(&self) -> usize {
        self.x.len()
    }
}

impl NearestNeighborIndex for BruteForceIndex {
    fn nearest_one(&self, x: f32, y: f32, z: f32) -> Option<Neighbor> {
        brute_force_knn(&self.x, &self.y, &self.z, x, y, z, 1).into_iter().next()
    }

    fn nearest_k(&self, x: f32, y: f32, z: f32, k: usize) -> Vec<Neighbor> {
        brute_force_knn(&self.x, &self.y, &self.z, x, y, z, k)
    }
}

impl RadiusSearchIndex for BruteForceIndex {
    fn radius_search(&self, x: f32, y: f32, z: f32, radius: f32) -> Vec<Neighbor> {
        brute_force_radius(&self.x, &self.y, &self.z, x, y, z, radius)
    }
}

/// Finds up to `k` nearest neighbors by brute force.
#[must_use]
pub fn brute_force_knn(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    qx: f32,
    qy: f32,
    qz: f32,
    k: usize,
) -> Vec<Neighbor> {
    if k == 0 || x.is_empty() {
        return Vec::new();
    }

    let mut neighbors: Vec<Neighbor> = x
        .iter()
        .enumerate()
        .map(|(index, &px)| Neighbor {
            index,
            distance_squared: squared_distance(px, y[index], z[index], qx, qy, qz),
        })
        .collect();
    neighbors.sort_by(|a, b| {
        a.distance_squared.partial_cmp(&b.distance_squared).unwrap_or(std::cmp::Ordering::Equal)
    });
    neighbors.truncate(k);
    neighbors
}

/// Finds all neighbors within `radius` by brute force.
#[must_use]
pub fn brute_force_radius(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    qx: f32,
    qy: f32,
    qz: f32,
    radius: f32,
) -> Vec<Neighbor> {
    let radius_sq = radius * radius;
    let mut neighbors = Vec::new();
    for (index, &px) in x.iter().enumerate() {
        let dist_sq = squared_distance(px, y[index], z[index], qx, qy, qz);
        if dist_sq <= radius_sq {
            neighbors.push(Neighbor { index, distance_squared: dist_sq });
        }
    }
    neighbors.sort_by(|a, b| {
        a.distance_squared.partial_cmp(&b.distance_squared).unwrap_or(std::cmp::Ordering::Equal)
    });
    neighbors
}

fn squared_distance(px: f32, py: f32, pz: f32, qx: f32, qy: f32, qz: f32) -> f32 {
    let dx = px - qx;
    let dy = py - qy;
    let dz = pz - qz;
    dx * dx + dy * dy + dz * dz
}
