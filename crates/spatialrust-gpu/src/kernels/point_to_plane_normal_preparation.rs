//! Exact common-scale normal preparation before either explicit upload route.
use super::invalid;
use spatialrust_core::SpatialResult;

const PARALLEL_MIN_ROWS: usize = 262_144;
const ROWS_PER_WORKER: usize = 65_536;
const MAX_WORKERS: usize = 4;

pub(super) fn prepare(normals: &[[f32; 3]]) -> SpatialResult<(Vec<[f32; 4]>, f32)> {
    if normals.len() < PARALLEL_MIN_ROWS {
        return serial(normals);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    #[cfg(target_arch = "wasm32")]
    let available = 1;
    prepare_with_workers(normals, available)
}

fn workers(len: usize, available: usize) -> usize {
    if cfg!(target_arch = "wasm32") || len < PARALLEL_MIN_ROWS {
        1
    } else {
        available.clamp(1, MAX_WORKERS).min((len / ROWS_PER_WORKER).max(1))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn bound(normals: &[[f32; 3]]) -> SpatialResult<f32> {
    let mut scale = 0.0_f32;
    for &value in normals.iter().flatten() {
        if !value.is_finite() {
            return Err(invalid());
        }
        scale = scale.max(value.abs());
    }
    Ok(scale)
}

#[cfg(not(target_arch = "wasm32"))]
fn scale(mut scale: f32) -> SpatialResult<f32> {
    if scale == 0.0 {
        scale = 1.0;
    }
    if scale < f32::MIN_POSITIVE {
        return Err(invalid());
    }
    Ok(scale)
}

#[cfg(not(target_arch = "wasm32"))]
fn packed(normal: [f32; 3], scale: f32) -> SpatialResult<[f32; 4]> {
    // Keep three original f32 divisions, including signed zero. Multiplying
    // by a reciprocal or normalizing each row changes accepted bits/weights.
    let row = normal.map(|value| value / scale);
    if (0..3).any(|i| normal[i] != 0.0 && row[i].abs() < f32::MIN_POSITIVE) {
        return Err(invalid());
    }
    Ok([row[0], row[1], row[2], 0.0])
}

fn prepare_with_workers(
    normals: &[[f32; 3]],
    available: usize,
) -> SpatialResult<(Vec<[f32; 4]>, f32)> {
    let count = workers(normals.len(), available);
    #[cfg(not(target_arch = "wasm32"))]
    if count > 1 {
        let chunk = normals.len().div_ceil(count);
        let maximum = std::thread::scope(|scope| -> SpatialResult<f32> {
            let handles: Vec<_> =
                normals.chunks(chunk).map(|rows| scope.spawn(move || bound(rows))).collect();
            let mut maximum = 0.0_f32;
            for handle in handles {
                maximum = maximum.max(handle.join().expect("normal bounds worker panicked")?);
            }
            Ok(maximum)
        })?;
        // All finite/nonnegative maxima merge exactly. Reject the global scale
        // before allocating the same 16*N-byte packed payload as the serial API.
        let scale = scale(maximum)?;
        let mut output = vec![[0.0; 4]; normals.len()];
        std::thread::scope(|scope| -> SpatialResult<()> {
            let mut handles = Vec::with_capacity(count);
            for (rows, output) in normals.chunks(chunk).zip(output.chunks_mut(chunk)) {
                handles.push(scope.spawn(move || -> SpatialResult<()> {
                    for (&normal, row) in rows.iter().zip(output) {
                        *row = packed(normal, scale)?;
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                handle.join().expect("normal packing worker panicked")?;
            }
            Ok(())
        })?;
        return Ok((output, scale));
    }
    let _ = count;
    serial(normals)
}

// Original small/one-worker/WASM preparation body, including allocation and
// loop shape. Keep this path independent of the parallel packing helpers.
fn serial(target_normals: &[[f32; 3]]) -> SpatialResult<(Vec<[f32; 4]>, f32)> {
    let mut normal_scale = 0.0_f32;
    for &v in target_normals.iter().flatten() {
        if !v.is_finite() {
            return Err(invalid());
        }
        normal_scale = normal_scale.max(v.abs());
    }
    if normal_scale == 0.0 {
        normal_scale = 1.0;
    }
    if normal_scale < f32::MIN_POSITIVE {
        return Err(invalid());
    }
    let mut packed = Vec::with_capacity(target_normals.len());
    for n in target_normals {
        let row = n.map(|v| v / normal_scale);
        if (0..3).any(|i| n[i] != 0.0 && row[i].abs() < f32::MIN_POSITIVE) {
            return Err(invalid());
        }
        packed.push([row[0], row[1], row[2], 0.0]);
    }
    Ok((packed, normal_scale))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "point_to_plane_normal_preparation_tests.rs"]
mod tests;
