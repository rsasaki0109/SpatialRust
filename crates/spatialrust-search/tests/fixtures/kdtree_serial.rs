use std::cmp::Ordering;

use spatialrust_core::{HasPositions3, PointCloud, SpatialResult};

use crate::{
    chunked::{ChunkQueryRange, ChunkedNearestNeighborIndex, ChunkedRadiusSearchIndex},
    NearestNeighborIndex, Neighbor, RadiusSearchIndex, SpatialIndex,
};

const LEAF_SIZE: usize = 16;
const SEARCH_STACK_SIZE: usize = 64;
const AXIS_LEAF: u8 = u8::MAX;
const INVALID_NODE: u32 = u32::MAX;

/// Cache-friendly KD-tree for 3D point clouds.
#[derive(Clone, Debug)]
pub struct KdTree {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    points_order: Vec<u32>,
    nodes: Vec<KdNode>,
    root: u32,
}

#[derive(Clone, Copy, Debug)]
struct KdNode {
    split: f32,
    left: u32,
    right: u32,
    start: u32,
    end: u32,
    axis: u8,
}

impl KdTree {
    /// Finds the nearest point inside an inclusive squared-distance bound.
    ///
    /// Uses the same traversal and first-encountered tie policy as `nearest_one`,
    /// without allocating a result vector. Negative/NaN bounds and nonfinite
    /// query coordinates return `None`; positive infinity is an unbounded query.
    #[must_use]
    pub fn nearest_one_within(
        &self,
        x: f32,
        y: f32,
        z: f32,
        max_distance_squared: f32,
    ) -> Option<Neighbor> {
        if max_distance_squared.is_nan()
            || max_distance_squared < 0.0
            || !x.is_finite()
            || !y.is_finite()
            || !z.is_finite()
        {
            return None;
        }
        let mut best = None;
        self.nearest_one_within_recursive(self.root, [x, y, z], max_distance_squared, &mut best);
        best
    }

    fn nearest_one_within_recursive(
        &self,
        node: u32,
        query: [f32; 3],
        bound: f32,
        best: &mut Option<Neighbor>,
    ) {
        if node == INVALID_NODE {
            return;
        }
        let data = self.nodes[node as usize];
        if data.axis == AXIS_LEAF {
            for order in data.start..data.end {
                let (index, x, y, z) = self.ordered_point(order);
                let distance = squared_distance(x, y, z, query[0], query[1], query[2]);
                if distance <= bound
                    && best.as_ref().map_or(true, |n| distance < n.distance_squared)
                {
                    *best = Some(Neighbor { index: index as usize, distance_squared: distance });
                }
            }
            return;
        }
        let diff = query[data.axis as usize] - data.split;
        let (near, far) =
            if diff <= 0.0 { (data.left, data.right) } else { (data.right, data.left) };
        self.nearest_one_within_recursive(near, query, bound, best);
        // Before finding a point, equality must be visited to include the gate
        // boundary. Once found, retain the original strict pruning/tie policy.
        let visit_far =
            best.as_ref().map_or(diff * diff <= bound, |n| diff * diff < n.distance_squared);
        if visit_far {
            self.nearest_one_within_recursive(far, query, bound, best);
        }
    }

    /// Builds a KD-tree from coordinate slices.
    #[must_use]
    pub fn from_slices(x: &[f32], y: &[f32], z: &[f32]) -> Self {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), z.len());

        let len = x.len();
        let mut points_order: Vec<u32> = (0..len as u32).collect();
        let mut nodes = Vec::with_capacity(if len == 0 { 0 } else { len.div_ceil(LEAF_SIZE) * 2 });

        let root = if len == 0 {
            INVALID_NODE
        } else {
            build_node(x, y, z, &mut points_order, 0, len, &mut nodes)
        };

        Self { x: x.to_vec(), y: y.to_vec(), z: z.to_vec(), points_order, nodes, root }
    }

    /// Builds a KD-tree from any point cloud with XYZ positions.
    pub fn from_point_cloud(cloud: &PointCloud) -> SpatialResult<Self> {
        let (x, y, z) = cloud.positions3()?;
        Ok(Self::from_slices(x, y, z))
    }

    fn point(&self, point_index: u32) -> (f32, f32, f32) {
        let idx = point_index as usize;
        (self.x[idx], self.y[idx], self.z[idx])
    }

    fn ordered_point(&self, order_index: u32) -> (u32, f32, f32, f32) {
        let point_index = self.points_order[order_index as usize];
        let (x, y, z) = self.point(point_index);
        (point_index, x, y, z)
    }

    fn nearest_k_recursive(
        &self,
        node: u32,
        qx: f32,
        qy: f32,
        qz: f32,
        k: usize,
        best: &mut KnnAccumulator,
    ) {
        if node == INVALID_NODE {
            return;
        }

        let node_data = self.nodes[node as usize];
        if node_data.axis == AXIS_LEAF {
            let start = node_data.start as usize;
            let end = node_data.end as usize;
            for order_index in start..end {
                let (index, px, py, pz) = self.ordered_point(order_index as u32);
                best.insert(
                    k,
                    Neighbor {
                        index: index as usize,
                        distance_squared: squared_distance(px, py, pz, qx, qy, qz),
                    },
                );
            }
            return;
        }

        let diff = match node_data.axis {
            0 => qx - node_data.split,
            1 => qy - node_data.split,
            _ => qz - node_data.split,
        };

        let (near, far) = if diff <= 0.0 {
            (node_data.left, node_data.right)
        } else {
            (node_data.right, node_data.left)
        };

        self.nearest_k_recursive(near, qx, qy, qz, k, best);

        let worst = best.prune_distance(k);
        if diff * diff < worst || best.len() < k {
            self.nearest_k_recursive(far, qx, qy, qz, k, best);
        }
    }

    fn radius_recursive(
        &self,
        node: u32,
        qx: f32,
        qy: f32,
        qz: f32,
        radius_sq: f32,
        out: &mut Vec<Neighbor>,
    ) {
        if node == INVALID_NODE {
            return;
        }

        let node_data = self.nodes[node as usize];
        if node_data.axis == AXIS_LEAF {
            let start = node_data.start as usize;
            let end = node_data.end as usize;
            for order_index in start..end {
                let (index, px, py, pz) = self.ordered_point(order_index as u32);
                let distance_squared = squared_distance(px, py, pz, qx, qy, qz);
                if distance_squared <= radius_sq {
                    out.push(Neighbor { index: index as usize, distance_squared });
                }
            }
            return;
        }

        let diff = match node_data.axis {
            0 => qx - node_data.split,
            1 => qy - node_data.split,
            _ => qz - node_data.split,
        };

        let (near, far) = if diff <= 0.0 {
            (node_data.left, node_data.right)
        } else {
            (node_data.right, node_data.left)
        };

        self.radius_recursive(near, qx, qy, qz, radius_sq, out);
        if diff * diff <= radius_sq {
            self.radius_recursive(far, qx, qy, qz, radius_sq, out);
        }
    }

    /// Returns whether at least `target` points lie within `radius` of the
    /// query, stopping as soon as the threshold is reached. Unlike
    /// [`radius_search`](RadiusSearchIndex::radius_search) this allocates nothing
    /// and early-exits, which is much faster for density tests (outlier removal).
    #[must_use]
    pub fn radius_reaches(&self, x: f32, y: f32, z: f32, radius: f32, target: usize) -> bool {
        if target == 0 {
            return true;
        }
        if self.is_empty() || radius < 0.0 {
            return false;
        }
        self.radius_count_iterative(x, y, z, radius * radius, target)
    }

    fn radius_count_iterative(
        &self,
        qx: f32,
        qy: f32,
        qz: f32,
        radius_sq: f32,
        target: usize,
    ) -> bool {
        let mut count = 0usize;
        let mut stack = [INVALID_NODE; SEARCH_STACK_SIZE];
        let mut stack_len = 0usize;
        let mut node = self.root;

        loop {
            while node != INVALID_NODE {
                let node_data = self.nodes[node as usize];
                if node_data.axis == AXIS_LEAF {
                    let start = node_data.start as usize;
                    let end = node_data.end as usize;
                    for order_index in start..end {
                        let (_, px, py, pz) = self.ordered_point(order_index as u32);
                        if squared_distance(px, py, pz, qx, qy, qz) <= radius_sq {
                            count += 1;
                            if count >= target {
                                return true;
                            }
                        }
                    }
                    break;
                }

                let diff = match node_data.axis {
                    0 => qx - node_data.split,
                    1 => qy - node_data.split,
                    _ => qz - node_data.split,
                };
                let (near, far) = if diff <= 0.0 {
                    (node_data.left, node_data.right)
                } else {
                    (node_data.right, node_data.left)
                };

                if diff * diff <= radius_sq {
                    if stack_len < stack.len() {
                        stack[stack_len] = far;
                        stack_len += 1;
                    } else if self
                        .radius_count_recursive(far, qx, qy, qz, radius_sq, target, &mut count)
                    {
                        return true;
                    }
                }
                node = near;
            }

            if stack_len == 0 {
                return false;
            }
            stack_len -= 1;
            node = stack[stack_len];
        }
    }

    /// Accumulates points within `radius_sq` into `count`; returns `true` as soon
    /// as `count` reaches `target` so the search can short-circuit.
    fn radius_count_recursive(
        &self,
        node: u32,
        qx: f32,
        qy: f32,
        qz: f32,
        radius_sq: f32,
        target: usize,
        count: &mut usize,
    ) -> bool {
        if node == INVALID_NODE {
            return false;
        }

        let node_data = self.nodes[node as usize];
        if node_data.axis == AXIS_LEAF {
            let start = node_data.start as usize;
            let end = node_data.end as usize;
            for order_index in start..end {
                let (_, px, py, pz) = self.ordered_point(order_index as u32);
                if squared_distance(px, py, pz, qx, qy, qz) <= radius_sq {
                    *count += 1;
                    if *count >= target {
                        return true;
                    }
                }
            }
            return false;
        }

        let diff = match node_data.axis {
            0 => qx - node_data.split,
            1 => qy - node_data.split,
            _ => qz - node_data.split,
        };
        let (near, far) = if diff <= 0.0 {
            (node_data.left, node_data.right)
        } else {
            (node_data.right, node_data.left)
        };

        if self.radius_count_recursive(near, qx, qy, qz, radius_sq, target, count) {
            return true;
        }
        if diff * diff <= radius_sq {
            return self.radius_count_recursive(far, qx, qy, qz, radius_sq, target, count);
        }
        false
    }
}

impl SpatialIndex for KdTree {
    fn len(&self) -> usize {
        self.x.len()
    }
}

impl NearestNeighborIndex for KdTree {
    fn nearest_one(&self, x: f32, y: f32, z: f32) -> Option<Neighbor> {
        self.nearest_k(x, y, z, 1).into_iter().next()
    }

    fn nearest_k(&self, x: f32, y: f32, z: f32, k: usize) -> Vec<Neighbor> {
        let mut best = Vec::with_capacity(k.min(self.len()));
        self.nearest_k_into(x, y, z, k, &mut best);
        best
    }
}

impl KdTree {
    /// Finds up to `k` nearest neighbors sorted by ascending distance, reusing
    /// the caller-provided output buffer.
    pub fn nearest_k_into(&self, x: f32, y: f32, z: f32, k: usize, out: &mut Vec<Neighbor>) {
        self.nearest_k_unsorted_into(x, y, z, k, out);
        out.sort_by(|a, b| {
            a.distance_squared.partial_cmp(&b.distance_squared).unwrap_or(Ordering::Equal)
        });
    }

    /// Finds up to `k` nearest neighbors without sorting the result, reusing the
    /// caller-provided output buffer. This is faster for callers that only need
    /// the neighbor set, such as covariance and mean-distance calculations.
    pub fn nearest_k_unsorted_into(
        &self,
        x: f32,
        y: f32,
        z: f32,
        k: usize,
        out: &mut Vec<Neighbor>,
    ) {
        out.clear();
        if self.is_empty() || k == 0 {
            return;
        }

        out.reserve(k.min(self.len()));
        let mut best = KnnAccumulator::new(out);
        self.nearest_k_recursive(self.root, x, y, z, k, &mut best);
    }
}

impl RadiusSearchIndex for KdTree {
    fn radius_search(&self, x: f32, y: f32, z: f32, radius: f32) -> Vec<Neighbor> {
        if self.is_empty() || radius < 0.0 {
            return Vec::new();
        }

        let radius_sq = radius * radius;
        let mut out = Vec::new();
        self.radius_recursive(self.root, x, y, z, radius_sq, &mut out);
        // Intentionally unsorted: callers count or iterate neighbors, and
        // sorting every query dominates radius search on dense clouds.
        out
    }
}

impl ChunkedRadiusSearchIndex for KdTree {
    fn radius_search_chunk_into(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        chunk: ChunkQueryRange,
        radius: f32,
        out: &mut Vec<(usize, Neighbor)>,
    ) {
        if self.is_empty() || radius < 0.0 || chunk.is_empty() {
            return;
        }

        let radius_sq = radius * radius;
        let mut scratch = Vec::new();
        for index in chunk {
            scratch.clear();
            self.radius_recursive(self.root, x[index], y[index], z[index], radius_sq, &mut scratch);
            for neighbor in scratch.drain(..) {
                out.push((index, neighbor));
            }
        }
    }
}

impl ChunkedNearestNeighborIndex for KdTree {
    fn nearest_k_chunk_into(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        chunk: ChunkQueryRange,
        k: usize,
        out: &mut Vec<(usize, Neighbor)>,
    ) {
        if chunk.is_empty() || k == 0 {
            return;
        }

        let mut scratch = Vec::new();
        for index in chunk {
            scratch.clear();
            self.nearest_k_unsorted_into(x[index], y[index], z[index], k, &mut scratch);
            for neighbor in scratch.drain(..) {
                out.push((index, neighbor));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_node(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    points_order: &mut [u32],
    start: usize,
    end: usize,
    nodes: &mut Vec<KdNode>,
) -> u32 {
    let node_index = nodes.len() as u32;
    nodes.push(KdNode {
        split: 0.0,
        left: INVALID_NODE,
        right: INVALID_NODE,
        start: start as u32,
        end: end as u32,
        axis: 0,
    });

    let count = end - start;
    if count <= LEAF_SIZE {
        nodes[node_index as usize].axis = AXIS_LEAF;
        return node_index;
    }

    let axis = select_axis(x, y, z, points_order, start, end);
    let mid = start + count / 2;
    select_nth_by_axis(x, y, z, points_order, start, end, axis, mid);

    let split_point = points_order[mid];
    let split_value = coordinate(x, y, z, split_point, axis);
    nodes[node_index as usize].axis = axis;
    nodes[node_index as usize].split = split_value;

    let left = build_node(x, y, z, points_order, start, mid, nodes);
    let right = build_node(x, y, z, points_order, mid, end, nodes);

    nodes[node_index as usize].left = left;
    nodes[node_index as usize].right = right;
    node_index
}

fn select_axis(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    points_order: &[u32],
    start: usize,
    end: usize,
) -> u8 {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &point_index in &points_order[start..end] {
        min[0] = min[0].min(x[point_index as usize]);
        min[1] = min[1].min(y[point_index as usize]);
        min[2] = min[2].min(z[point_index as usize]);
        max[0] = max[0].max(x[point_index as usize]);
        max[1] = max[1].max(y[point_index as usize]);
        max[2] = max[2].max(z[point_index as usize]);
    }

    let mut best_axis = 0_u8;
    let mut best_extent = max[0] - min[0];
    for axis in 1_u8..3 {
        let extent = max[axis as usize] - min[axis as usize];
        if extent > best_extent {
            best_extent = extent;
            best_axis = axis;
        }
    }
    best_axis
}

fn select_nth_by_axis(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    points_order: &mut [u32],
    start: usize,
    end: usize,
    axis: u8,
    nth: usize,
) {
    let mut left = start;
    let mut right = end;
    while left < right {
        let pivot = partition_by_axis(x, y, z, points_order, left, right, axis);
        match nth.cmp(&pivot) {
            Ordering::Less => right = pivot,
            Ordering::Greater => left = pivot + 1,
            Ordering::Equal => break,
        }
    }
}

fn partition_by_axis(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    points_order: &mut [u32],
    start: usize,
    end: usize,
    axis: u8,
) -> usize {
    let pivot_index = (start + end) / 2;
    points_order.swap(start, pivot_index);
    let pivot_point = points_order[start];
    let pivot_value = coordinate(x, y, z, pivot_point, axis);

    let mut store = start + 1;
    for i in (start + 1)..end {
        if coordinate(x, y, z, points_order[i], axis) < pivot_value {
            points_order.swap(i, store);
            store += 1;
        }
    }
    points_order.swap(start, store - 1);
    store - 1
}

fn coordinate(x: &[f32], y: &[f32], z: &[f32], point_index: u32, axis: u8) -> f32 {
    match axis {
        0 => x[point_index as usize],
        1 => y[point_index as usize],
        _ => z[point_index as usize],
    }
}

fn squared_distance(px: f32, py: f32, pz: f32, qx: f32, qy: f32, qz: f32) -> f32 {
    let dx = px - qx;
    let dy = py - qy;
    let dz = pz - qz;
    dx * dx + dy * dy + dz * dz
}

#[derive(Debug)]
struct KnnAccumulator<'a> {
    neighbors: &'a mut Vec<Neighbor>,
    worst_index: usize,
    worst_distance_squared: f32,
}

impl<'a> KnnAccumulator<'a> {
    fn new(neighbors: &'a mut Vec<Neighbor>) -> Self {
        Self { neighbors, worst_index: 0, worst_distance_squared: 0.0 }
    }

    fn len(&self) -> usize {
        self.neighbors.len()
    }

    fn prune_distance(&self, k: usize) -> f32 {
        if self.neighbors.len() < k {
            f32::INFINITY
        } else {
            self.worst_distance_squared
        }
    }

    fn insert(&mut self, k: usize, candidate: Neighbor) {
        if k == 0 {
            return;
        }

        if self.neighbors.len() < k {
            let distance_squared = candidate.distance_squared;
            self.neighbors.push(candidate);
            if self.neighbors.len() == 1 || distance_squared > self.worst_distance_squared {
                self.worst_index = self.neighbors.len() - 1;
                self.worst_distance_squared = distance_squared;
            }
            return;
        }

        if candidate.distance_squared >= self.worst_distance_squared {
            return;
        }

        self.neighbors[self.worst_index] = candidate;
        self.refresh_worst();
    }

    fn refresh_worst(&mut self) {
        let mut worst_index = 0usize;
        let mut worst_distance_squared = self.neighbors[0].distance_squared;
        for (index, neighbor) in self.neighbors.iter().enumerate().skip(1) {
            if neighbor.distance_squared > worst_distance_squared {
                worst_index = index;
                worst_distance_squared = neighbor.distance_squared;
            }
        }
        self.worst_index = worst_index;
        self.worst_distance_squared = worst_distance_squared;
    }
}

// Native test-only observers; the parent implementation above is unchanged.
pub(super) fn snapshot(tree: &KdTree) -> (u32, Vec<u32>, Vec<[u32; 6]>) {
    (
        tree.root,
        tree.points_order.clone(),
        tree.nodes
            .iter()
            .map(|n| [n.split.to_bits(), n.left, n.right, n.start, n.end, u32::from(n.axis)])
            .collect(),
    )
}

pub(super) fn select_for_test(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    order: &mut [u32],
    range: std::ops::Range<usize>,
    axis: u8,
    nth: usize,
) {
    select_nth_by_axis(x, y, z, order, range.start, range.end, axis, nth);
}
