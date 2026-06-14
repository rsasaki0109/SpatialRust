//! Consistent normal orientation via minimum-spanning-tree propagation.
//!
//! Normal *estimation* recovers each normal only up to sign, so a surface comes
//! out with normals pointing randomly inward/outward. This propagates a single
//! consistent orientation across a k-nearest-neighbor graph: starting from a
//! seed oriented upward, it walks a minimum spanning tree (edges weighted so
//! that near-parallel neighbors are visited first, à la Hoppe) and flips each
//! normal to agree with the one it was reached from.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use spatialrust_core::{
    FieldSemantic, HasNormals3, HasPositions3, PointBuffer, PointBufferSet, PointCloud,
    SpatialError, SpatialResult,
};
use spatialrust_math::Vec3;
use spatialrust_search::{KdTree, NearestNeighborIndex};

/// Configuration for [`orient_normals_consistent`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalOrientationConfig {
    /// Number of nearest neighbors used to build the propagation graph.
    pub k_neighbors: usize,
}

impl Default for NormalOrientationConfig {
    fn default() -> Self {
        Self { k_neighbors: 15 }
    }
}

impl NormalOrientationConfig {
    /// Creates a config with the given neighbor count.
    #[must_use]
    pub const fn new(k_neighbors: usize) -> Self {
        Self { k_neighbors }
    }
}

/// An MST candidate edge, ordered so the binary heap pops the smallest weight.
struct Edge {
    weight: f32,
    parent: u32,
    node: u32,
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}
impl Eq for Edge {}
impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so the max-heap behaves as a min-heap on weight.
        other.weight.total_cmp(&self.weight)
    }
}

/// Re-orients a cloud's normals so neighboring normals agree in sign.
///
/// The cloud must already carry normals. The seed of each connected component is
/// oriented to point along `+Z` (upward); apply a viewpoint convention
/// afterwards (e.g. `orient_normal_towards_viewpoint`) if one is needed instead.
pub fn orient_normals_consistent(
    input: &PointCloud,
    config: NormalOrientationConfig,
) -> SpatialResult<PointCloud> {
    if config.k_neighbors == 0 {
        return Err(SpatialError::InvalidArgument(
            "k_neighbors must be greater than zero".to_owned(),
        ));
    }
    let len = input.len();
    if len == 0 {
        return Ok(input.clone());
    }

    let (x, y, z) = input.positions3()?;
    let (nx, ny, nz) = input.normals3()?;
    let mut normals: Vec<Vec3<f32>> = (0..len).map(|i| Vec3::new(nx[i], ny[i], nz[i])).collect();

    let tree = KdTree::from_slices(x, y, z);
    let neighbors_of = |i: usize| tree.nearest_k(x[i], y[i], z[i], config.k_neighbors + 1);

    let mut visited = vec![false; len];
    let mut heap: BinaryHeap<Edge> = BinaryHeap::new();

    // Process seeds in descending height so each component starts from a point
    // whose "up" orientation is meaningful.
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| z[b].total_cmp(&z[a]));

    for &seed in &order {
        if visited[seed] {
            continue;
        }
        // Orient the seed upward.
        if normals[seed].z < 0.0 {
            normals[seed] = flip(normals[seed]);
        }
        visited[seed] = true;
        push_edges(&mut heap, seed, &neighbors_of(seed), &visited, &normals);

        while let Some(edge) = heap.pop() {
            let node = edge.node as usize;
            if visited[node] {
                continue;
            }
            visited[node] = true;
            // Flip the new normal to agree with the one we reached it from.
            if normals[node].dot(normals[edge.parent as usize]) < 0.0 {
                normals[node] = flip(normals[node]);
            }
            push_edges(&mut heap, node, &neighbors_of(node), &visited, &normals);
        }
    }

    build_output(input, &normals)
}

fn push_edges(
    heap: &mut BinaryHeap<Edge>,
    parent: usize,
    neighbors: &[spatialrust_search::Neighbor],
    visited: &[bool],
    normals: &[Vec3<f32>],
) {
    for neighbor in neighbors {
        let node = neighbor.index;
        if node == parent || visited[node] {
            continue;
        }
        // Near-parallel normals get low weight, so they propagate first.
        let weight = 1.0 - normals[parent].dot(normals[node]).abs();
        heap.push(Edge { weight, parent: parent as u32, node: node as u32 });
    }
}

fn flip(v: Vec3<f32>) -> Vec3<f32> {
    Vec3::new(-v.x, -v.y, -v.z)
}

/// Rebuilds the cloud, replacing only the normal columns.
fn build_output(input: &PointCloud, normals: &[Vec3<f32>]) -> SpatialResult<PointCloud> {
    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let buffer = match field.semantic {
            FieldSemantic::NormalX => PointBuffer::from_f32(normals.iter().map(|n| n.x).collect()),
            FieldSemantic::NormalY => PointBuffer::from_f32(normals.iter().map(|n| n.y).collect()),
            FieldSemantic::NormalZ => PointBuffer::from_f32(normals.iter().map(|n| n.z).collect()),
            _ => clone_buffer(input.field(&field.name)?),
        };
        buffers.insert(field.name.clone(), buffer);
    }
    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
}

fn clone_buffer(buffer: &PointBuffer) -> PointBuffer {
    match buffer {
        PointBuffer::F32(v) => PointBuffer::from_f32(v.clone()),
        PointBuffer::F64(v) => PointBuffer::F64(v.clone()),
        PointBuffer::U8(v) => PointBuffer::U8(v.clone()),
        PointBuffer::U16(v) => PointBuffer::U16(v.clone()),
        PointBuffer::U32(v) => PointBuffer::U32(v.clone()),
        PointBuffer::I32(v) => PointBuffer::I32(v.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::{orient_normals_consistent, NormalOrientationConfig};
    use spatialrust_core::{
        DType, FieldSemantic, HasNormals3, PointCloudBuilder, PointField, PointSchema,
    };

    fn schema() -> PointSchema {
        PointSchema::new()
            .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
            .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
            .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32))
            .with_field(PointField::scalar("normal_x", FieldSemantic::NormalX, DType::F32))
            .with_field(PointField::scalar("normal_y", FieldSemantic::NormalY, DType::F32))
            .with_field(PointField::scalar("normal_z", FieldSemantic::NormalZ, DType::F32))
    }

    #[test]
    fn flips_inconsistent_normals_on_a_plane() {
        // A flat grid whose true normal is +Z, but every other point's normal is
        // flipped to -Z. After orientation they should all agree.
        let mut builder = PointCloudBuilder::new(schema());
        let mut flipped = 0;
        for i in 0..8 {
            for j in 0..8 {
                let nz = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                if nz < 0.0 {
                    flipped += 1;
                }
                builder.push_point([i as f32, j as f32, 0.0, 0.0, 0.0, nz]).unwrap();
            }
        }
        assert!(flipped > 0);
        let cloud = builder.build().unwrap();

        let oriented = orient_normals_consistent(&cloud, NormalOrientationConfig::new(8)).unwrap();
        let (_, _, onz) = oriented.normals3().unwrap();
        // All normals should now point the same way (+Z, since the seed is up).
        assert!(onz.iter().all(|&v| v > 0.5), "normals not consistently +Z");
    }

    #[test]
    fn rejects_zero_neighbors() {
        let mut builder = PointCloudBuilder::new(schema());
        builder.push_point([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let cloud = builder.build().unwrap();
        assert!(orient_normals_consistent(&cloud, NormalOrientationConfig::new(0)).is_err());
    }
}
