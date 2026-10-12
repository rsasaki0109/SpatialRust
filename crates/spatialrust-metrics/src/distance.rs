//! Chamfer and Hausdorff distances between two point clouds.

use spatialrust_core::{
    HasPositions3, PointCloud, SpatialError, SpatialResult, DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE,
};
use spatialrust_search::{parallel_worker_count, KdTree, NearestNeighborIndex};

/// A bundle of directed and symmetric cloud-to-cloud distance statistics.
///
/// "Directed" means nearest-neighbor distances from one cloud to the other; the
/// metrics are not symmetric on their own, so both directions are reported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudDistances {
    /// Mean squared NN distance from `a` to `b`.
    pub mean_sq_a_to_b: f64,
    /// Mean squared NN distance from `b` to `a`.
    pub mean_sq_b_to_a: f64,
    /// Largest NN distance from `a` to `b` (directed Hausdorff).
    pub max_a_to_b: f64,
    /// Largest NN distance from `b` to `a` (directed Hausdorff).
    pub max_b_to_a: f64,
}

impl CloudDistances {
    /// Symmetric Chamfer distance: the sum of the two mean squared NN distances.
    #[must_use]
    pub fn chamfer(&self) -> f64 {
        self.mean_sq_a_to_b + self.mean_sq_b_to_a
    }

    /// Symmetric Hausdorff distance: the larger of the two directed maxima.
    #[must_use]
    pub fn hausdorff(&self) -> f64 {
        self.max_a_to_b.max(self.max_b_to_a)
    }
}

/// Computes directed NN distance statistics between `a` and `b` in both
/// directions with a single KD-tree per side.
///
/// Large query sides share their target tree across bounded scoped CPU workers.
/// A reusable window holds at most one default point chunk per worker. Distance
/// accumulation retains input order; small/one-worker sides remain serial.
pub fn cloud_distances(a: &PointCloud, b: &PointCloud) -> SpatialResult<CloudDistances> {
    cloud_distances_with_workers(
        a,
        b,
        parallel_worker_count(a.len()),
        parallel_worker_count(b.len()),
    )
}

fn cloud_distances_with_workers(
    a: &PointCloud,
    b: &PointCloud,
    workers_a: usize,
    workers_b: usize,
) -> SpatialResult<CloudDistances> {
    if a.is_empty() || b.is_empty() {
        return Err(SpatialError::InvalidArgument("both clouds must be non-empty".to_owned()));
    }
    let (ax, ay, az) = a.positions3()?;
    let (bx, by, bz) = b.positions3()?;

    let tree_b = KdTree::from_slices(bx, by, bz);
    let tree_a = KdTree::from_slices(ax, ay, az);

    let (mean_sq_a_to_b, max_a_to_b) = directed(ax, ay, az, &tree_b, workers_a);
    let (mean_sq_b_to_a, max_b_to_a) = directed(bx, by, bz, &tree_a, workers_b);

    Ok(CloudDistances { mean_sq_a_to_b, mean_sq_b_to_a, max_a_to_b, max_b_to_a })
}

/// Mean squared and max NN distance from each query point to `target`.
fn directed(x: &[f32], y: &[f32], z: &[f32], target: &KdTree, workers: usize) -> (f64, f64) {
    let workers = workers.max(1).min(x.len().max(1));
    if workers == 1 {
        return directed_serial(x, y, z, target);
    }
    // Preserve Some(NaN) versus None, including the original sum/max behavior.
    // Reuse a worker-bounded window instead of staging every query's result.
    let window_size = x.len().min(workers.saturating_mul(DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE));
    let mut nearest = vec![None; window_size];
    let mut sum_sq = 0.0_f64;
    let mut max = 0.0_f64;
    for start in (0..x.len()).step_by(window_size) {
        let count = window_size.min(x.len() - start);
        let active_workers = workers.min(count);
        let fill = |offset: usize, band: &mut [Option<f32>]| {
            // nearest_one uses this same one-neighbor traversal. Reuse its
            // output buffer; nearest_k_into resets the query accumulator.
            let mut neighbor = Vec::new();
            for (i, distance) in band.iter_mut().enumerate() {
                let i = start + offset + i;
                target.nearest_k_into(x[i], y[i], z[i], 1, &mut neighbor);
                *distance = neighbor.first().map(|n| n.distance_squared);
            }
        };
        if active_workers == 1 {
            fill(0, &mut nearest[..count]);
        } else {
            let chunk_size = count.div_ceil(active_workers);
            std::thread::scope(|scope| {
                for (band, chunk) in nearest[..count].chunks_mut(chunk_size).enumerate() {
                    let fill = &fill;
                    scope.spawn(move || fill(band * chunk_size, chunk));
                }
            });
        }
        // Never combine worker-local sums: f64 addition is order-sensitive.
        for distance in nearest[..count].iter().flatten() {
            let d_sq = f64::from(*distance);
            sum_sq += d_sq;
            let d = d_sq.sqrt();
            if d > max {
                max = d;
            }
        }
    }
    (sum_sq / x.len() as f64, max)
}

fn directed_serial(x: &[f32], y: &[f32], z: &[f32], target: &KdTree) -> (f64, f64) {
    let mut sum_sq = 0.0_f64;
    let mut max = 0.0_f64;
    for i in 0..x.len() {
        if let Some(neighbor) = target.nearest_one(x[i], y[i], z[i]) {
            let d_sq = f64::from(neighbor.distance_squared);
            sum_sq += d_sq;
            let d = d_sq.sqrt();
            if d > max {
                max = d;
            }
        }
    }
    (sum_sq / x.len() as f64, max)
}

/// Symmetric Chamfer distance between two clouds (sum of mean squared NN
/// distances in both directions). Lower is better; zero for identical clouds.
pub fn chamfer_distance(a: &PointCloud, b: &PointCloud) -> SpatialResult<f64> {
    Ok(cloud_distances(a, b)?.chamfer())
}

/// Symmetric Hausdorff distance between two clouds (the largest NN distance in
/// either direction). Captures the worst-case discrepancy.
pub fn hausdorff_distance(a: &PointCloud, b: &PointCloud) -> SpatialResult<f64> {
    Ok(cloud_distances(a, b)?.hausdorff())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "../tests/fixtures/distance_serial.rs"]
#[allow(dead_code)]
mod previous;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "distance_parallel_tests.rs"]
mod parallel_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "distance_parallel_benchmark.rs"]
mod parallel_benchmark;

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    fn cloud(points: &[[f32; 3]]) -> PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for p in points {
            builder.push_point(*p).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn identical_clouds_have_zero_distance() {
        let c = cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert!(chamfer_distance(&c, &c).unwrap() < 1e-9);
        assert!(hausdorff_distance(&c, &c).unwrap() < 1e-9);
    }

    #[test]
    fn translation_shows_up_in_distances() {
        let a = cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let b = cloud(&[[0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [2.0, 0.5, 0.0]]);
        // Every point's nearest neighbor is exactly 0.5 away.
        let chamfer = chamfer_distance(&a, &b).unwrap();
        assert!((chamfer - (0.25 + 0.25)).abs() < 1e-6);
        let hausdorff = hausdorff_distance(&a, &b).unwrap();
        assert!((hausdorff - 0.5).abs() < 1e-6);
    }

    #[test]
    fn hausdorff_catches_a_single_outlier() {
        let a = cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let b = cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]);
        // The lone far point in b is 9.0 from its nearest neighbor in a.
        assert!((hausdorff_distance(&a, &b).unwrap() - 9.0).abs() < 1e-6);
    }
}
