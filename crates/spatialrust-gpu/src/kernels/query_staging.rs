//! Original-order finite SoA queries for the explicit GPU upload boundary.
use super::invalid;
use spatialrust_core::SpatialResult;

#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_MIN_POINTS: usize = 524_288;
#[cfg(not(target_arch = "wasm32"))]
const POINTS_PER_WORKER: usize = 262_144;
#[cfg(not(target_arch = "wasm32"))]
const MAX_WORKERS: usize = 4;

pub(super) fn prepare(axes: [&[f32]; 3]) -> SpatialResult<Vec<[f32; 4]>> {
    #[cfg(not(target_arch = "wasm32"))]
    let available = if axes[0].len() < PARALLEL_MIN_POINTS {
        1
    } else {
        spatialrust_search::parallel_worker_count_with_chunk(axes[0].len(), POINTS_PER_WORKER)
    };
    #[cfg(target_arch = "wasm32")]
    let available = 1;
    prepare_with_workers(axes, available)
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

fn prepare_with_workers(axes: [&[f32]; 3], available: usize) -> SpatialResult<Vec<[f32; 4]>> {
    let [x, y, z] = axes;
    if x.len() != y.len() || x.len() != z.len() {
        return Err(invalid());
    }
    let workers = worker_count(x.len(), available);
    #[cfg(not(target_arch = "wasm32"))]
    if workers > 1 {
        // Safe, disjoint destination bands; one output allocation and only
        // worker-sized dispatch metadata, with no point-sized join buffer.
        let mut rows = vec![[0.0; 4]; x.len()];
        let chunk = x.len().div_ceil(workers);
        std::thread::scope(|scope| -> SpatialResult<()> {
            let mut handles = Vec::with_capacity(workers);
            for (group, rows) in rows.chunks_mut(chunk).enumerate() {
                let start = group * chunk;
                let end = start + rows.len();
                let axes = [&x[start..end], &y[start..end], &z[start..end]];
                handles.push(scope.spawn(move || -> SpatialResult<()> {
                    if axes.iter().any(|axis| axis.iter().any(|v| !v.is_finite())) {
                        return Err(invalid());
                    }
                    for (row, ((&x, &y), &z)) in
                        rows.iter_mut().zip(axes[0].iter().zip(axes[1]).zip(axes[2]))
                    {
                        *row = [x, y, z, 0.0];
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                handle.join().expect("query staging worker panicked")?;
            }
            Ok(())
        })?;
        return Ok(rows);
    }
    let _ = workers;
    if axes.iter().any(|axis| axis.iter().any(|v| !v.is_finite())) {
        return Err(invalid());
    }
    Ok(x.iter().zip(y).zip(z).map(|((&x, &y), &z)| [x, y, z, 0.0]).collect())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "query_staging_tests.rs"]
mod tests;
