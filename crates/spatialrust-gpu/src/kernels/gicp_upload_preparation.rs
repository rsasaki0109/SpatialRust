//! Exact packed-f32 validation and original-order explicit-upload staging.
use super::{invalid, inverse_trace};
use spatialrust_core::SpatialResult;

#[cfg(not(target_arch = "wasm32"))]
const ROWS_PER_WORKER: usize = 16_384;

pub(super) fn prepare(
    source: &[[f32; 6]],
    target: &[[f32; 6]],
) -> SpatialResult<(Vec<[f32; 8]>, f32)> {
    #[cfg(not(target_arch = "wasm32"))]
    let workers = {
        let len = source.len().checked_add(target.len()).ok_or_else(invalid)?;
        spatialrust_search::parallel_worker_count_with_chunk(len, ROWS_PER_WORKER)
    };
    #[cfg(target_arch = "wasm32")]
    let workers = 1;
    prepare_with_workers(source, target, workers)
}

fn worker_count(len: usize, available: usize) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        available.max(1).min((len / ROWS_PER_WORKER).max(1))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (len, available);
        1
    }
}

fn packed(c: &[f32; 6]) -> [f32; 8] {
    [c[0], c[1], c[2], c[3], c[4], c[5], 0.0, 0.0]
}

fn prepare_with_workers(
    source: &[[f32; 6]],
    target: &[[f32; 6]],
    available: usize,
) -> SpatialResult<(Vec<[f32; 8]>, f32)> {
    let len = source.len().checked_add(target.len()).ok_or_else(invalid)?;
    let workers = worker_count(len, available);
    #[cfg(not(target_arch = "wasm32"))]
    if workers > 1 {
        let mut rows = vec![[0.0; 8]; len];
        let chunk = len.div_ceil(workers);
        let bound = std::thread::scope(|scope| -> SpatialResult<f64> {
            let mut handles = Vec::with_capacity(workers);
            for (group, rows) in rows.chunks_mut(chunk).enumerate() {
                handles.push(scope.spawn(move || -> SpatialResult<f64> {
                    let mut bound = 0.0_f64;
                    for (offset, row) in rows.iter_mut().enumerate() {
                        let index = group * chunk + offset;
                        let (c, is_target) = if index < source.len() {
                            (&source[index], false)
                        } else {
                            (&target[index - source.len()], true)
                        };
                        let trace = inverse_trace(c)?;
                        if is_target {
                            bound = bound.max(trace);
                        }
                        *row = packed(c);
                    }
                    Ok(bound)
                }));
            }
            // Every accepted trace is positive finite. Maximum needs no sum,
            // rounding or reordering of each row's original f64 arithmetic.
            let mut bound = 0.0_f64;
            for handle in handles {
                bound = bound.max(handle.join().expect("GICP upload worker panicked")?);
            }
            Ok(bound)
        })?;
        return finish(rows, bound);
    }
    let _ = workers;
    let mut rows = Vec::with_capacity(len);
    let mut bound = 0.0_f64;
    for c in source {
        inverse_trace(c)?;
        rows.push(packed(c));
    }
    for c in target {
        bound = bound.max(inverse_trace(c)?);
        rows.push(packed(c));
    }
    finish(rows, bound)
}

fn finish(out: Vec<[f32; 8]>, bound: f64) -> SpatialResult<(Vec<[f32; 8]>, f32)> {
    let scale = bound as f32;
    if !scale.is_finite() || scale < f32::MIN_POSITIVE {
        return Err(invalid());
    }
    Ok((out, scale))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "gicp_upload_preparation_tests.rs"]
mod tests;
