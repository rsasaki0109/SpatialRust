//! Bounded finite-XYZ/bounds preparation for the shared registration upload.
use super::invalid;
use spatialrust_core::SpatialResult;

#[cfg(not(target_arch = "wasm32"))]
const POINTS_PER_WORKER: usize = 65_536;

#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_MIN_POINTS: usize = 262_144;

pub(super) fn prepare(points: &[[f32; 3]]) -> SpatialResult<([f32; 4], f64)> {
    // Avoid both worker startup and the CPU-count query on modest uploads.
    #[cfg(not(target_arch = "wasm32"))]
    if points.len() < PARALLEL_MIN_POINTS {
        return serial(points);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let workers =
        spatialrust_search::parallel_worker_count_with_chunk(points.len(), POINTS_PER_WORKER);
    #[cfg(target_arch = "wasm32")]
    let workers = 1;
    prepare_with_workers(points, workers)
}

pub(super) fn prepare_soa(axes: [&[f32]; 3]) -> SpatialResult<([f32; 4], f64)> {
    #[cfg(not(target_arch = "wasm32"))]
    if axes[0].len() < PARALLEL_MIN_POINTS {
        return prepare_soa_with_workers(axes, 1);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let workers =
        spatialrust_search::parallel_worker_count_with_chunk(axes[0].len(), POINTS_PER_WORKER);
    #[cfg(target_arch = "wasm32")]
    let workers = 1;
    prepare_soa_with_workers(axes, workers)
}

fn prepare_soa_with_workers(axes: [&[f32]; 3], available: usize) -> SpatialResult<([f32; 4], f64)> {
    let len = axes[0].len();
    if axes.iter().any(|axis| axis.len() != len) {
        return Err(invalid());
    }
    if len == 0 {
        return Ok(([0.0; 4], 0.0));
    }
    let workers = worker_count(len, available);
    #[cfg(not(target_arch = "wasm32"))]
    if workers > 1 {
        let bounds = std::thread::scope(|scope| -> SpatialResult<Bounds> {
            let chunk_size = len.div_ceil(workers);
            let mut handles = Vec::with_capacity(workers);
            for start in (0..len).step_by(chunk_size) {
                let end = (start + chunk_size).min(len);
                handles.push(
                    scope.spawn(move || Bounds::scan_soa(axes.map(|axis| &axis[start..end]))),
                );
            }
            let mut bounds = Bounds::empty();
            for handle in handles {
                bounds.merge(handle.join().expect("registration geometry worker panicked")?);
            }
            Ok(bounds)
        })?;
        return finish_soa(bounds.lo, bounds.hi);
    }
    let _ = workers;
    if axes.iter().any(|axis| axis.iter().any(|v| !v.is_finite())) {
        return Err(invalid());
    }
    let lo = axes.map(|axis| axis.iter().copied().fold(f32::INFINITY, f32::min));
    let hi = axes.map(|axis| axis.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    finish_soa(lo, hi)
}

fn finish_soa(lo: [f32; 3], hi: [f32; 3]) -> SpatialResult<([f32; 4], f64)> {
    let mut anchor = [0.0; 4];
    let mut extent_squared = 0.0;
    for (axis, anchor) in anchor[..3].iter_mut().enumerate() {
        *anchor = midpoint(lo[axis], hi[axis]);
        extent_squared += (f64::from(hi[axis]) - f64::from(lo[axis])).powi(2);
    }
    if !(extent_squared as f32).is_finite() {
        return Err(invalid());
    }
    Ok((anchor, extent_squared))
}

fn worker_count(len: usize, available: usize) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if len < PARALLEL_MIN_POINTS {
            1
        } else {
            available.max(1).min((len / POINTS_PER_WORKER).max(1))
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (len, available);
        1
    }
}

fn prepare_with_workers(points: &[[f32; 3]], available: usize) -> SpatialResult<([f32; 4], f64)> {
    let workers = worker_count(points.len(), available);
    #[cfg(not(target_arch = "wasm32"))]
    if workers > 1 {
        let bounds = std::thread::scope(|scope| -> SpatialResult<Bounds> {
            let mut handles = Vec::with_capacity(workers);
            for chunk in points.chunks(points.len().div_ceil(workers)) {
                handles.push(scope.spawn(move || Bounds::scan(chunk)));
            }
            let mut bounds = Bounds::empty();
            for handle in handles {
                bounds.merge(handle.join().expect("registration geometry worker panicked")?);
            }
            Ok(bounds)
        })?;
        let mut anchor = [0.0; 4];
        let mut extent_squared = 0.0;
        for (axis, anchor) in anchor[..3].iter_mut().enumerate() {
            let lo = bounds.lo[axis];
            let hi = bounds.hi[axis];
            *anchor = midpoint(lo, hi);
            // Keep original axis order and f64 midpoint/difference/square math.
            extent_squared += (f64::from(hi) - f64::from(lo)).powi(2);
        }
        if !(extent_squared as f32).is_finite() {
            return Err(invalid());
        }
        return Ok((anchor, extent_squared));
    }
    let _ = workers;
    serial(points)
}

// Equal signed zeros in f32 min/max are not deterministically ordered.
// Canonicalize the derived frame only; uploaded original point bits survive.
fn midpoint(lo: f32, hi: f32) -> f32 {
    let value = ((f64::from(lo) + f64::from(hi)) * 0.5) as f32;
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

// Original serial scans; only private zero anchors gain a defined sign.
fn serial(points: &[[f32; 3]]) -> SpatialResult<([f32; 4], f64)> {
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    if points.is_empty() {
        return Ok(([0.0; 4], 0.0));
    }
    let mut anchor = [0.0; 4];
    let mut extent_squared = 0.0;
    for axis in 0..3 {
        let lo = points.iter().map(|p| p[axis]).fold(f32::INFINITY, f32::min);
        let hi = points.iter().map(|p| p[axis]).fold(f32::NEG_INFINITY, f32::max);
        anchor[axis] = midpoint(lo, hi);
        extent_squared += (f64::from(hi) - f64::from(lo)).powi(2);
    }
    if !(extent_squared as f32).is_finite() {
        return Err(invalid());
    }
    Ok((anchor, extent_squared))
}

#[cfg(not(target_arch = "wasm32"))]
struct Bounds {
    lo: [f32; 3],
    hi: [f32; 3],
}

#[cfg(not(target_arch = "wasm32"))]
impl Bounds {
    fn empty() -> Self {
        Self { lo: [f32::INFINITY; 3], hi: [f32::NEG_INFINITY; 3] }
    }
    fn scan(points: &[[f32; 3]]) -> SpatialResult<Self> {
        let mut bounds = Self::empty();
        for p in points {
            if p.iter().any(|v| !v.is_finite()) {
                return Err(invalid());
            }
            for (axis, &v) in p.iter().enumerate() {
                bounds.lo[axis] = bounds.lo[axis].min(v);
                bounds.hi[axis] = bounds.hi[axis].max(v);
            }
        }
        Ok(bounds)
    }
    fn scan_soa(axes: [&[f32]; 3]) -> SpatialResult<Self> {
        let mut bounds = Self::empty();
        for (axis, values) in axes.into_iter().enumerate() {
            for &v in values {
                if !v.is_finite() {
                    return Err(invalid());
                }
                bounds.lo[axis] = bounds.lo[axis].min(v);
                bounds.hi[axis] = bounds.hi[axis].max(v);
            }
        }
        Ok(bounds)
    }
    fn merge(&mut self, other: Self) {
        for axis in 0..3 {
            self.lo[axis] = self.lo[axis].min(other.lo[axis]);
            self.hi[axis] = self.hi[axis].max(other.hi[axis]);
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "icp_geometry_preparation_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "icp_soa_upload_tests.rs"]
mod soa_tests;
