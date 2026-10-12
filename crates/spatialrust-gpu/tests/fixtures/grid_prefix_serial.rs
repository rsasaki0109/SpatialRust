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
        acc += *slot;
        *slot = acc;
    }

    let mut sorted = vec![0u32; n];
    // Descending input indices preserve ascending order within each cell.
    // Decrementing each cumulative end exactly its count leaves its start.
    for index in (0..n).rev() {
        let cell = cell_of(index);
        counts[cell] -= 1;
        sorted[counts[cell] as usize] = index as u32;
    }
    (sorted, counts)
}
