pub(super) fn cpu_roots(xyz: &[Vec<f32>; 3], tolerance: f32) -> Vec<u32> {
    let [x, y, z] = xyz;
    let tree = KdTree::from_slices(x, y, z);
    let len = x.len();
    let mut processed = vec![false; len];
    let mut roots = vec![0u32; len];

    for seed in 0..len {
        if processed[seed] {
            continue;
        }

        let mut queue = VecDeque::from([seed]);
        let mut cluster_indices = Vec::new();
        processed[seed] = true;

        while let Some(index) = queue.pop_front() {
            cluster_indices.push(index);
            let neighbors =
                tree.radius_search(x[index], y[index], z[index], tolerance);
            for neighbor in neighbors {
                let candidate = neighbor.index;
                if processed[candidate] {
                    continue;
                }
                processed[candidate] = true;
                queue.push_back(candidate);
            }
        }

        let root = *cluster_indices.iter().min().unwrap_or(&seed) as u32;
        for index in cluster_indices {
            roots[index] = root;
        }
    }

    roots
}
