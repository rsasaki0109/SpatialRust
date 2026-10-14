//! Bounded original-order AoS query preparation before explicit GPU upload.
use super::invalid;
use spatialrust_core::SpatialResult;

const PARALLEL_MIN_POINTS: usize = 524_288;
const POINTS_PER_WORKER: usize = 262_144;
const MAX_WORKERS: usize = 4;

pub(super) fn prepare(points: &[[f32; 3]]) -> SpatialResult<Vec<[f32; 4]>> {
    // Keep modest uploads independent of worker startup and CPU-count queries.
    if points.len() < PARALLEL_MIN_POINTS {
        return serial(points);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    #[cfg(target_arch = "wasm32")]
    let available = 1;
    prepare_with_workers(points, available)
}

fn workers(len: usize, available: usize) -> usize {
    if cfg!(target_arch = "wasm32") || len < PARALLEL_MIN_POINTS {
        1
    } else {
        available.clamp(1, MAX_WORKERS).min((len / POINTS_PER_WORKER).max(1))
    }
}

fn prepare_with_workers(points: &[[f32; 3]], available: usize) -> SpatialResult<Vec<[f32; 4]>> {
    let count = workers(points.len(), available);
    #[cfg(not(target_arch = "wasm32"))]
    if count > 1 {
        // One existing-sized payload, disjoint writes and worker-sized handles.
        // Large invalid inputs may allocate this payload before rejection.
        let mut output = vec![[0.0; 4]; points.len()];
        let chunk = points.len().div_ceil(count);
        std::thread::scope(|scope| -> SpatialResult<()> {
            let mut handles = Vec::with_capacity(count);
            for (points, rows) in points.chunks(chunk).zip(output.chunks_mut(chunk)) {
                handles.push(scope.spawn(move || -> SpatialResult<()> {
                    for (point, row) in points.iter().zip(rows) {
                        if point.iter().any(|v| !v.is_finite()) {
                            return Err(invalid());
                        }
                        *row = [point[0], point[1], point[2], 0.0];
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                handle.join().expect("AoS query staging worker panicked")?;
            }
            Ok(())
        })?;
        return Ok(output);
    }
    let _ = count;
    serial(points)
}

// Original finite scan and exact-capacity collector for small/one-worker/WASM.
fn serial(points: &[[f32; 3]]) -> SpatialResult<Vec<[f32; 4]>> {
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    Ok(points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "aos_query_staging_tests.rs"]
mod tests;
