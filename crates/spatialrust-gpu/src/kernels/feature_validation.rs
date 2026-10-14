//! CPU validation before feature device allocation; no payload packing.

const PARALLEL_MIN_POINTS: usize = 524_288;
const POINTS_PER_WORKER: usize = 131_072;
const MAX_WORKERS: usize = 4;

pub(super) fn valid(axes: [&[f32]; 3], normals: Option<&[[f32; 3]]>) -> bool {
    // Small uploads avoid worker startup and the CPU-count query.
    #[cfg(not(target_arch = "wasm32"))]
    let available = if axes[0].len() >= PARALLEL_MIN_POINTS {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        1
    };
    #[cfg(target_arch = "wasm32")]
    let available = 1;
    prepare(axes, normals, available).is_some()
}

fn workers(len: usize, available: usize) -> usize {
    if cfg!(target_arch = "wasm32") || len < PARALLEL_MIN_POINTS {
        1
    } else {
        available.clamp(1, MAX_WORKERS).min((len / POINTS_PER_WORKER).max(1))
    }
}

fn unit_normal(n: &[f32; 3]) -> bool {
    // Preserve the original f32 operation order and inclusive tolerance.
    let norm = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    n.iter().all(|v| v.is_finite()) && norm.is_finite() && (norm - 1.0).abs() <= 1e-3
}

#[cfg(not(target_arch = "wasm32"))]
fn extent(lo: [f32; 3], hi: [f32; 3]) -> Option<f64> {
    // Min/max merging changes no nonzero extrema; signed zero differences
    // square to +0. Keep the original axis order and f64 difference/square/sum.
    let squared: f64 = (0..3).map(|a| (f64::from(hi[a]) - f64::from(lo[a])).powi(2)).sum();
    (squared as f32).is_finite().then_some(squared)
}

fn prepare(axes: [&[f32]; 3], normals: Option<&[[f32; 3]]>, available: usize) -> Option<f64> {
    let len = axes[0].len();
    if axes.iter().any(|axis| axis.len() != len)
        || normals.is_some_and(|normals| normals.len() != len)
    {
        return None;
    }
    let count = workers(len, available);
    #[cfg(not(target_arch = "wasm32"))]
    if count > 1 {
        let bounds = std::thread::scope(|scope| {
            let chunk_size = len.div_ceil(count);
            let mut handles = Vec::with_capacity(count);
            for start in (0..len).step_by(chunk_size) {
                let end = (start + chunk_size).min(len);
                handles.push(scope.spawn(move || {
                    let mut lo = [f32::INFINITY; 3];
                    let mut hi = [f32::NEG_INFINITY; 3];
                    for (a, values) in axes.into_iter().enumerate() {
                        for &value in &values[start..end] {
                            if !value.is_finite() {
                                return None;
                            }
                            lo[a] = lo[a].min(value);
                            hi[a] = hi[a].max(value);
                        }
                    }
                    if normals.is_some_and(|n| !n[start..end].iter().all(unit_normal)) {
                        return None;
                    }
                    Some((lo, hi))
                }));
            }
            let mut lo = [f32::INFINITY; 3];
            let mut hi = [f32::NEG_INFINITY; 3];
            for handle in handles {
                let (part_lo, part_hi) =
                    handle.join().expect("feature validation worker panicked")?;
                for a in 0..3 {
                    lo[a] = lo[a].min(part_lo[a]);
                    hi[a] = hi[a].max(part_hi[a]);
                }
            }
            Some((lo, hi))
        })?;
        return extent(bounds.0, bounds.1);
    }
    let _ = count;
    // Retain the former serial scans for small/one-worker/WASM inputs.
    if axes.into_iter().flatten().any(|v| !v.is_finite())
        || normals.is_some_and(|n| !n.iter().all(unit_normal))
    {
        return None;
    }
    if len == 0 {
        return Some(0.0);
    }
    let squared: f64 = axes
        .iter()
        .map(|axis| {
            let lo = axis.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = axis.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (f64::from(hi) - f64::from(lo)).powi(2)
        })
        .sum();
    (squared as f32).is_finite().then_some(squared)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "feature_validation_tests.rs"]
mod tests;
