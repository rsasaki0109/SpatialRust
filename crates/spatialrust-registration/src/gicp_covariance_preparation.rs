//! GPU-entry-point CPU k-NN preparation: bounded independent rows, same math.
use super::super::{covariance_at, M3};
use spatialrust_search::KdTree;

#[cfg(not(target_arch = "wasm32"))]
const POINTS_PER_WORKER: usize = 16_384;

fn packed(m: M3) -> [f32; 6] {
    [m[0][0] as f32, m[1][1] as f32, m[2][2] as f32, m[0][1] as f32, m[0][2] as f32, m[1][2] as f32]
}

pub(super) fn prepare(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tree: &KdTree,
    k: usize,
    epsilon: f64,
) -> Vec<[f32; 6]> {
    #[cfg(not(target_arch = "wasm32"))]
    let workers = spatialrust_search::parallel_worker_count_with_chunk(x.len(), POINTS_PER_WORKER);
    #[cfg(target_arch = "wasm32")]
    let workers = 1;
    prepare_with_workers(x, y, z, tree, k, epsilon, workers)
}

fn worker_count(len: usize, available: usize) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        available.max(1).min((len / POINTS_PER_WORKER).max(1))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (len, available);
        1
    }
}

fn prepare_with_workers(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tree: &KdTree,
    k: usize,
    epsilon: f64,
    available: usize,
) -> Vec<[f32; 6]> {
    let workers = worker_count(x.len(), available);
    // Pack directly from each local f64 matrix, avoiding the previous 72*N
    // intermediate Vec<M3>. Mutable row slices are disjoint and preserve order.
    let mut out = vec![[0.0; 6]; x.len()];
    let fill = |start: usize, rows: &mut [[f32; 6]]| {
        for (offset, row) in rows.iter_mut().enumerate() {
            *row = packed(covariance_at(x, y, z, tree, start + offset, k, epsilon));
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    if workers > 1 {
        std::thread::scope(|scope| {
            let chunk = x.len().div_ceil(workers);
            for (group, rows) in out.chunks_mut(chunk).enumerate() {
                let fill = &fill;
                scope.spawn(move || fill(group * chunk, rows));
            }
        });
        return out;
    }
    let _ = workers;
    fill(0, &mut out);
    out
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "gicp_covariance_preparation_tests.rs"]
mod tests;
