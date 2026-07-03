use spatialrust_core::{SpatialError, SpatialResult};

/// Upper bound on dense grid cells; callers should fall back when exceeded.
pub const MAX_UNIFORM_GRID_CELLS: u64 = 64_000_000;

/// Returns grid origin (min corner) and cell counts for cell size `radius`.
pub fn grid_bounds(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    cell_size: f32,
) -> SpatialResult<([f32; 3], [u32; 3])> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for index in 0..x.len() {
        for (axis, value) in [x[index], y[index], z[index]].into_iter().enumerate() {
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    let inv_cell = 1.0 / cell_size;
    let mut dims = [0u32; 3];
    for axis in 0..3 {
        let span = ((max[axis] - min[axis]) * inv_cell).floor() as i64 + 1;
        dims[axis] = span.max(1) as u32;
    }
    let cells = dims[0] as u64 * dims[1] as u64 * dims[2] as u64;
    if cells > MAX_UNIFORM_GRID_CELLS {
        return Err(SpatialError::InvalidArgument(format!(
            "grid would need {cells} cells (cap {MAX_UNIFORM_GRID_CELLS}); use a larger radius or the CPU path"
        )));
    }
    Ok((min, dims))
}

/// Returns whether a uniform grid with the given cell size fits within the cell cap.
pub fn uniform_grid_fits(x: &[f32], y: &[f32], z: &[f32], cell_size: f32) -> bool {
    grid_bounds(x, y, z, cell_size).is_ok()
}

/// Counting-sort points into grid cells, returning sorted indices and CSR offsets.
pub fn build_grid(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: [f32; 3],
    dims: [u32; 3],
    cell_size: f32,
) -> (Vec<u32>, Vec<u32>) {
    let inv_cell = 1.0 / cell_size;
    let n = x.len();
    let num_cells = dims[0] as usize * dims[1] as usize * dims[2] as usize;

    let cell_of = |index: usize| -> usize {
        let cx = (((x[index] - origin[0]) * inv_cell).floor() as i64).clamp(0, dims[0] as i64 - 1)
            as usize;
        let cy = (((y[index] - origin[1]) * inv_cell).floor() as i64).clamp(0, dims[1] as i64 - 1)
            as usize;
        let cz = (((z[index] - origin[2]) * inv_cell).floor() as i64).clamp(0, dims[2] as i64 - 1)
            as usize;
        (cz * dims[1] as usize + cy) * dims[0] as usize + cx
    };

    let mut counts = vec![0u32; num_cells + 1];
    for index in 0..n {
        counts[cell_of(index)] += 1;
    }
    let mut acc = 0u32;
    for slot in counts.iter_mut() {
        let c = *slot;
        *slot = acc;
        acc += c;
    }
    let cell_start = counts;

    let mut cursor = cell_start.clone();
    let mut sorted = vec![0u32; n];
    for index in 0..n {
        let cell = cell_of(index);
        let slot = cursor[cell];
        sorted[slot as usize] = index as u32;
        cursor[cell] = slot + 1;
    }
    (sorted, cell_start)
}

/// Connected-component roots via uniform-grid union-find (minimum index per component).
pub fn euclidean_cluster_roots(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    cluster_tolerance: f32,
) -> SpatialResult<Vec<u32>> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(SpatialError::InvalidArgument("xyz arrays must have equal length".to_owned()));
    }
    let point_count = x.len();
    if point_count == 0 {
        return Ok(Vec::new());
    }
    if cluster_tolerance <= 0.0 || cluster_tolerance.is_nan() {
        return Err(SpatialError::InvalidArgument("cluster_tolerance must be positive".to_owned()));
    }

    let (origin, dims) = grid_bounds(x, y, z, cluster_tolerance)?;
    let (sorted, cell_start) = build_grid(x, y, z, origin, dims, cluster_tolerance);
    let radius_sq = cluster_tolerance * cluster_tolerance;

    #[cfg(feature = "parallel")]
    if point_count >= PARALLEL_CLUSTER_MIN_POINTS {
        return Ok(cluster_roots_parallel(
            point_count,
            x,
            y,
            z,
            origin,
            dims,
            cluster_tolerance,
            radius_sq,
            &sorted,
            &cell_start,
        ));
    }

    Ok(cluster_roots_sequential(
        point_count,
        x,
        y,
        z,
        origin,
        dims,
        cluster_tolerance,
        radius_sq,
        &sorted,
        &cell_start,
    ))
}

/// Minimum point count before the `parallel` feature uses threaded union-find.
#[cfg(feature = "parallel")]
const PARALLEL_CLUSTER_MIN_POINTS: usize = 4_096;

fn cluster_roots_sequential(
    point_count: usize,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: [f32; 3],
    dims: [u32; 3],
    cluster_tolerance: f32,
    radius_sq: f32,
    sorted: &[u32],
    cell_start: &[u32],
) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..point_count as u32).collect();

    for index in 0..point_count {
        for neighbor in grid_radius_neighbors(
            index,
            x,
            y,
            z,
            origin,
            dims,
            cluster_tolerance,
            radius_sq,
            sorted,
            cell_start,
        ) {
            union_min_root(&mut parent, index as u32, neighbor as u32);
        }
    }

    compress_roots(&mut parent, point_count)
}

#[cfg(feature = "parallel")]
fn cluster_roots_parallel(
    point_count: usize,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: [f32; 3],
    dims: [u32; 3],
    cluster_tolerance: f32,
    radius_sq: f32,
    sorted: &[u32],
    cell_start: &[u32],
) -> Vec<u32> {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    let parent: Arc<[AtomicU32]> = (0..point_count as u32)
        .map(AtomicU32::new)
        .collect::<Vec<_>>()
        .into();

    let thread_count = std::thread::available_parallelism()
        .map_or(1, |count| count.get())
        .min(point_count)
        .max(1);
    let chunk = point_count.div_ceil(thread_count);

    std::thread::scope(|scope| {
        for thread in 0..thread_count {
            let start = thread * chunk;
            if start >= point_count {
                break;
            }
            let end = (start + chunk).min(point_count);
            let parent = Arc::clone(&parent);
            scope.spawn(move || {
                for index in start..end {
                    for neighbor in grid_radius_neighbors(
                        index,
                        x,
                        y,
                        z,
                        origin,
                        dims,
                        cluster_tolerance,
                        radius_sq,
                        sorted,
                        cell_start,
                    ) {
                        atomic_union_min_root(&parent, index as u32, neighbor as u32);
                    }
                }
            });
        }
    });

    let mut parent_vec: Vec<u32> = parent
        .iter()
        .map(|slot| slot.load(Ordering::Relaxed))
        .collect();
    compress_roots(&mut parent_vec, point_count)
}

#[cfg(feature = "parallel")]
fn atomic_union_min_root(parent: &[std::sync::atomic::AtomicU32], a: u32, b: u32) {
    use std::sync::atomic::Ordering;

    let mut ra = atomic_find_root(parent, a);
    let mut rb = atomic_find_root(parent, b);
    while ra != rb {
        let (min_root, max_root) = if ra < rb { (ra, rb) } else { (rb, ra) };
        match parent[max_root as usize].compare_exchange(
            max_root,
            min_root,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(current) => {
                if current == min_root {
                    return;
                }
                ra = atomic_find_root(parent, a);
                rb = atomic_find_root(parent, b);
            }
        }
    }
}

#[cfg(feature = "parallel")]
fn atomic_find_root(parent: &[std::sync::atomic::AtomicU32], mut index: u32) -> u32 {
    use std::sync::atomic::Ordering;

    loop {
        let parent_index = parent[index as usize].load(Ordering::Relaxed);
        if parent_index == index {
            return index;
        }
        index = parent_index;
    }
}

fn compress_roots(parent: &mut [u32], point_count: usize) -> Vec<u32> {
    let mut roots = vec![0u32; point_count];
    for index in 0..point_count {
        roots[index] = find_root(parent, index as u32);
    }
    roots
}

fn find_root(parent: &mut [u32], mut index: u32) -> u32 {
    let mut root = index;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    while parent[index as usize] != root {
        let next = parent[index as usize];
        parent[index as usize] = root;
        index = next;
    }
    root
}

fn union_min_root(parent: &mut [u32], a: u32, b: u32) {
    let ra = find_root_readonly(parent, a);
    let rb = find_root_readonly(parent, b);
    if ra == rb {
        return;
    }
    let (min_root, max_root) = if ra < rb { (ra, rb) } else { (rb, ra) };
    parent[max_root as usize] = min_root;
}

fn find_root_readonly(parent: &[u32], mut index: u32) -> u32 {
    while parent[index as usize] != index {
        index = parent[index as usize];
    }
    index
}

fn grid_radius_neighbors<'a>(
    index: usize,
    x: &'a [f32],
    y: &'a [f32],
    z: &'a [f32],
    origin: [f32; 3],
    dims: [u32; 3],
    tolerance: f32,
    radius_sq: f32,
    sorted: &'a [u32],
    cell_start: &'a [u32],
) -> GridRadiusNeighbors<'a> {
    GridRadiusNeighbors {
        index,
        x,
        y,
        z,
        origin,
        dims,
        inv_cell: 1.0 / tolerance,
        radius_sq,
        sorted,
        cell_start,
        cell: 0,
        slot: 0,
        end: 0,
        dz: -1,
        dy: -1,
        dx: -1,
        cx: cell_coord(x[index], origin[0], 1.0 / tolerance, dims[0]),
        cy: cell_coord(y[index], origin[1], 1.0 / tolerance, dims[1]),
        cz: cell_coord(z[index], origin[2], 1.0 / tolerance, dims[2]),
        started: false,
    }
}

struct GridRadiusNeighbors<'a> {
    index: usize,
    x: &'a [f32],
    y: &'a [f32],
    z: &'a [f32],
    origin: [f32; 3],
    dims: [u32; 3],
    inv_cell: f32,
    radius_sq: f32,
    sorted: &'a [u32],
    cell_start: &'a [u32],
    cell: usize,
    slot: u32,
    end: u32,
    dz: i32,
    dy: i32,
    dx: i32,
    cx: i32,
    cy: i32,
    cz: i32,
    started: bool,
}

impl Iterator for GridRadiusNeighbors<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.started {
                self.started = true;
                if !self.advance_cell() {
                    return None;
                }
            } else if self.slot >= self.end {
                if !self.advance_cell() {
                    return None;
                }
            } else {
                let neighbor = self.sorted[self.slot as usize] as usize;
                self.slot += 1;
                if neighbor == self.index {
                    continue;
                }
                let dx = self.x[neighbor] - self.x[self.index];
                let dy = self.y[neighbor] - self.y[self.index];
                let dz = self.z[neighbor] - self.z[self.index];
                if dx * dx + dy * dy + dz * dz <= self.radius_sq {
                    return Some(neighbor);
                }
            }
        }
    }
}

impl GridRadiusNeighbors<'_> {
    fn advance_cell(&mut self) -> bool {
        let dimx = self.dims[0] as i32;
        let dimy = self.dims[1] as i32;
        let dimz = self.dims[2] as i32;

        loop {
            if self.dz > 1 {
                return false;
            }
            let nx = self.cx + self.dx;
            let ny = self.cy + self.dy;
            let nz = self.cz + self.dz;
            if nx >= 0 && ny >= 0 && nz >= 0 && nx < dimx && ny < dimy && nz < dimz {
                self.cell = cell_index(nx, ny, nz, self.dims[0], self.dims[1]) as usize;
                self.slot = self.cell_start[self.cell];
                self.end = self.cell_start[self.cell + 1];
                self.bump_offset();
                return true;
            }
            self.bump_offset();
        }
    }

    fn bump_offset(&mut self) {
        self.dx += 1;
        if self.dx > 1 {
            self.dx = -1;
            self.dy += 1;
            if self.dy > 1 {
                self.dy = -1;
                self.dz += 1;
            }
        }
    }
}

fn cell_coord(value: f32, origin: f32, inv_cell: f32, dim: u32) -> i32 {
    let cell = ((value - origin) * inv_cell).floor() as i32;
    cell.clamp(0, dim as i32 - 1)
}

fn cell_index(cx: i32, cy: i32, cz: i32, dimx: u32, dimy: u32) -> u32 {
    (cz as u32 * dimy + cy as u32) * dimx + cx as u32
}

#[cfg(test)]
mod tests {
    use super::euclidean_cluster_roots;

    #[test]
    fn long_chain_is_one_component() {
        let len = 300;
        let spacing = 1.0;
        let mut x = Vec::with_capacity(len);
        for index in 0..len {
            x.push(index as f32 * spacing);
        }
        let y = vec![0.0_f32; len];
        let z = vec![0.0_f32; len];
        let roots = euclidean_cluster_roots(&x, &y, &z, 1.5).unwrap();
        assert!(roots.iter().all(|&root| root == 0));
    }
}
