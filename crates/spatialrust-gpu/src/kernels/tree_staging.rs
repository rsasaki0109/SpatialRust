//! Finite original coordinates packed in the CPU tree's traversal order.
#[cfg(not(target_arch = "wasm32"))]
use super::Zeroable;
use super::{invalid, Point};
use spatialrust_core::SpatialResult;

#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_MIN_POINTS: usize = 524_288;
#[cfg(not(target_arch = "wasm32"))]
const POINTS_PER_WORKER: usize = 131_072;
#[cfg(not(target_arch = "wasm32"))]
const MAX_WORKERS: usize = 4;

pub(super) fn prepare(axes: [&[f32]; 3], order: &[u32]) -> SpatialResult<Vec<Point>> {
    #[cfg(not(target_arch = "wasm32"))]
    let available = if axes[0].len() < PARALLEL_MIN_POINTS {
        1
    } else {
        spatialrust_search::parallel_worker_count_with_chunk(axes[0].len(), POINTS_PER_WORKER)
    };
    #[cfg(target_arch = "wasm32")]
    let available = 1;
    prepare_with_workers(axes, order, available)
}

fn worker_count(len: usize, available: usize) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if len < PARALLEL_MIN_POINTS {
            1
        } else {
            available.clamp(1, MAX_WORKERS).min((len / POINTS_PER_WORKER).max(1))
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (len, available);
        1
    }
}

fn prepare_with_workers(
    axes: [&[f32]; 3],
    order: &[u32],
    available: usize,
) -> SpatialResult<Vec<Point>> {
    let [x, y, z] = axes;
    if x.len() != y.len() || x.len() != z.len() || x.len() != order.len() {
        return Err(invalid());
    }
    let workers = worker_count(x.len(), available);
    #[cfg(not(target_arch = "wasm32"))]
    if workers > 1 {
        let mut points = vec![Point::zeroed(); x.len()];
        let chunk = x.len().div_ceil(workers);
        std::thread::scope(|scope| -> SpatialResult<()> {
            let mut handles = Vec::with_capacity(workers);
            for (group, rows) in points.chunks_mut(chunk).enumerate() {
                let start = group * chunk;
                let end = start + rows.len();
                let indices = &order[start..end];
                handles.push(scope.spawn(move || -> SpatialResult<()> {
                    // Validate all original coordinates, even though this band
                    // gathers a different subset of points in traversal order.
                    if axes.iter().any(|axis| axis[start..end].iter().any(|v| !v.is_finite())) {
                        return Err(invalid());
                    }
                    for (row, &index) in rows.iter_mut().zip(indices) {
                        let i = index as usize;
                        *row = Point { xyz: [x[i], y[i], z[i]], index };
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                handle.join().expect("tree staging worker panicked")?;
            }
            Ok(())
        })?;
        return Ok(points);
    }
    let _ = workers;
    if x.iter().chain(y).chain(z).any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    // The borrowed KdTree layout guarantees valid original indices. Keep the
    // parent collector so one-worker calls retain one exact-capacity allocation.
    Ok(order
        .iter()
        .map(|&index| {
            let i = index as usize;
            Point { xyz: [x[i], y[i], z[i]], index }
        })
        .collect())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "tree_staging_tests.rs"]
mod tests;
