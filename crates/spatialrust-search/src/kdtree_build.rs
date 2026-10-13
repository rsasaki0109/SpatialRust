//! Disjoint subtree construction with the unchanged partition/tie convention.
use super::*;

/// Minimum points before the explicit KD-tree builder uses scoped CPU workers.
pub const PARALLEL_KDTREE_BUILD_MIN_POINTS: usize = 32_768;
#[cfg(not(target_arch = "wasm32"))]
const MIN_WORKER_POINTS: usize = PARALLEL_KDTREE_BUILD_MIN_POINTS / 2;

impl KdTree {
    /// Builds with bounded scoped CPU workers under the `parallel` feature.
    /// Point permutation, split bits, preorder node layout and query ties match
    /// `from_slices` exactly. Native workers per call are bounded by available
    /// parallelism and one per 16,384 points; inputs below 32,768 points or one
    /// useful worker keep the serial constructor. WASM also stays serial.
    /// Axes are copied once; one exact-sized node array is split into disjoint
    /// subtree ranges without locks, unsafe, per-worker trees or a merge copy.
    /// This is a per-call bound, not a process-wide thread/stack limit.
    /// Panics on unequal axis lengths, or counts exceeding u32 original indices.
    #[must_use]
    pub fn from_slices_parallel(x: &[f32], y: &[f32], z: &[f32]) -> Self {
        if x.len() < PARALLEL_KDTREE_BUILD_MIN_POINTS {
            return Self::from_slices(x, y, z);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
            from_slices_with_workers(x, y, z, workers)
        }
        #[cfg(target_arch = "wasm32")]
        {
            assert!(x.len() <= u32::MAX as usize, "KD-tree original indices must fit u32");
            Self::from_slices(x, y, z)
        }
    }

    /// Builds an explicit parallel tree from a cloud, retaining position order.
    pub fn from_point_cloud_parallel(cloud: &PointCloud) -> SpatialResult<Self> {
        let (x, y, z) = cloud.positions3()?;
        Ok(Self::from_slices_parallel(x, y, z))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn worker_count(points: usize, available: usize) -> usize {
    if points < PARALLEL_KDTREE_BUILD_MIN_POINTS {
        1
    } else {
        available.max(1).min(points / MIN_WORKER_POINTS)
    }
}

// All levels above a power-of-two frontier split. At that frontier, bucket
// sizes are floor(N/frontier) or ceil(N/frontier). With the power-of-two leaf
// size, either every bucket splits once, or only N-LEAF_SIZE*frontier do.
// This gives exact preorder subtree sizes in O(1), including odd boundaries.
#[cfg(not(target_arch = "wasm32"))]
fn node_count(points: usize) -> usize {
    if points == 0 {
        return 0;
    }
    if points <= LEAF_SIZE {
        return 1;
    }
    debug_assert!(LEAF_SIZE.is_power_of_two());
    let quotient = points / LEAF_SIZE;
    let frontier = 1usize << (usize::BITS - 1 - quotient.leading_zeros());
    if points / frontier == LEAF_SIZE {
        2 * frontier - 1 + 2 * (points - LEAF_SIZE * frontier)
    } else {
        4 * frontier - 1
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn from_slices_with_workers(x: &[f32], y: &[f32], z: &[f32], available: usize) -> KdTree {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());
    assert!(x.len() <= u32::MAX as usize, "KD-tree original indices must fit u32");
    let workers = worker_count(x.len(), available);
    if workers == 1 {
        return KdTree::from_slices(x, y, z);
    }
    let mut points_order: Vec<u32> = (0..x.len() as u32).collect();
    let empty =
        KdNode { split: 0.0, left: INVALID_NODE, right: INVALID_NODE, start: 0, end: 0, axis: 0 };
    let mut nodes = vec![empty; node_count(x.len())];
    build_into([x, y, z], &mut points_order, &mut nodes, 0, 0, workers);
    KdTree { x: x.to_vec(), y: y.to_vec(), z: z.to_vec(), points_order, nodes, root: 0 }
}

#[cfg(not(target_arch = "wasm32"))]
fn build_into(
    xyz: [&[f32]; 3],
    order: &mut [u32],
    nodes: &mut [KdNode],
    point_offset: usize,
    node_offset: usize,
    workers: usize,
) {
    let count = order.len();
    let (node, children) = nodes.split_first_mut().expect("nonempty subtree node range");
    node.start = point_offset as u32;
    node.end = (point_offset + count) as u32;
    if count <= LEAF_SIZE {
        node.axis = AXIS_LEAF;
        return;
    }
    let [x, y, z] = xyz;
    let (axis, minimum) = select_axis(x, y, z, order, 0, count);
    let mid = count / 2;
    select_nth_by_axis(x, y, z, order, 0, count, (axis, minimum), mid);
    node.axis = axis;
    node.split = coordinate(x, y, z, order[mid], axis);
    let left_count = node_count(mid);
    let left_offset = node_offset + 1;
    let right_offset = left_offset + left_count;
    node.left = left_offset as u32;
    node.right = right_offset as u32;
    let (left_order, right_order) = order.split_at_mut(mid);
    let (left_nodes, right_nodes) = children.split_at_mut(left_count);
    if workers == 1 || count < PARALLEL_KDTREE_BUILD_MIN_POINTS {
        build_into(xyz, left_order, left_nodes, point_offset, left_offset, 1);
        build_into(xyz, right_order, right_nodes, point_offset + mid, right_offset, 1);
    } else {
        let left_workers = workers / 2;
        let right_workers = workers - left_workers;
        std::thread::scope(|scope| {
            let left = scope.spawn(move || {
                build_into(xyz, left_order, left_nodes, point_offset, left_offset, left_workers)
            });
            build_into(
                xyz,
                right_order,
                right_nodes,
                point_offset + mid,
                right_offset,
                right_workers,
            );
            left.join().expect("bounded KD-tree build worker panicked");
        });
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "kdtree_build_tests.rs"]
mod tests;
