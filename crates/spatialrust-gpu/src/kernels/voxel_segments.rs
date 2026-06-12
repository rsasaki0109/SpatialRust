/// Sorted voxel cell segments derived from per-point grid keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoxelSegments {
    /// Unique voxel keys in sorted order.
    pub keys: Vec<(i64, i64, i64)>,
    /// Point indices sorted by voxel key.
    pub point_indices: Vec<u32>,
    /// Start offset into `point_indices` for each cell.
    pub cell_starts: Vec<u32>,
    /// Number of points in each cell.
    pub cell_counts: Vec<u32>,
}

impl VoxelSegments {
    /// Returns the number of occupied voxel cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns whether no voxel cells were found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Builds sorted voxel segments from per-point `(ix, iy, iz)` keys.
#[must_use]
pub fn build_voxel_segments(keys: &[(i64, i64, i64)]) -> VoxelSegments {
    if keys.is_empty() {
        return VoxelSegments {
            keys: Vec::new(),
            point_indices: Vec::new(),
            cell_starts: Vec::new(),
            cell_counts: Vec::new(),
        };
    }

    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&index| keys[index]);

    let sorted_keys: Vec<(i64, i64, i64)> = order.iter().map(|&index| keys[index]).collect();
    let sorted_indices: Vec<u32> = order.iter().map(|&index| index as u32).collect();
    compact_voxel_segments_from_sorted(&sorted_keys, &sorted_indices)
}

/// Compacts already-sorted voxel keys and point indices into segment metadata.
#[must_use]
pub fn compact_voxel_segments_from_sorted(
    sorted_keys: &[(i64, i64, i64)],
    sorted_indices: &[u32],
) -> VoxelSegments {
    assert_eq!(sorted_keys.len(), sorted_indices.len());

    if sorted_keys.is_empty() {
        return VoxelSegments {
            keys: Vec::new(),
            point_indices: Vec::new(),
            cell_starts: Vec::new(),
            cell_counts: Vec::new(),
        };
    }

    let mut segments = VoxelSegments {
        keys: Vec::new(),
        point_indices: Vec::with_capacity(sorted_indices.len()),
        cell_starts: Vec::new(),
        cell_counts: Vec::new(),
    };

    let mut cursor = 0usize;
    while cursor < sorted_keys.len() {
        let key = sorted_keys[cursor];
        let cell_start = segments.point_indices.len() as u32;
        let mut count = 0u32;

        while cursor < sorted_keys.len() && sorted_keys[cursor] == key {
            segments.point_indices.push(sorted_indices[cursor]);
            count += 1;
            cursor += 1;
        }

        segments.keys.push(key);
        segments.cell_starts.push(cell_start);
        segments.cell_counts.push(count);
    }

    for cell_index in 0..segments.len() {
        let start = segments.cell_starts[cell_index] as usize;
        let end = start + segments.cell_counts[cell_index] as usize;
        segments.point_indices[start..end].sort_unstable();
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::build_voxel_segments;

    #[test]
    fn groups_points_by_voxel_key() {
        let keys = vec![(0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0)];
        let segments = build_voxel_segments(&keys);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments.cell_counts, vec![2, 2]);
        assert_eq!(segments.point_indices, vec![0, 2, 1, 3]);
    }
}
