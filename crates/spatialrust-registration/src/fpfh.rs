//! Fast Point Feature Histograms (FPFH) and FPFH-based global registration.
//!
//! Global registration recovers a coarse alignment *without* an initial guess,
//! which the local refiners (ICP/GICP/NDT) need. It describes each point by an
//! FPFH descriptor, matches descriptors between the clouds, and runs a RANSAC
//! loop that samples correspondences, estimates a rigid transform, and keeps the
//! pose with the most inliers.

use spatialrust_core::{HasNormals3, HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_math::{Isometry3, TransformPoint, Vec3};
use spatialrust_search::{KdTree, RadiusSearchIndex};

use crate::kabsch::estimate_rigid_transform;
use crate::registration::{PointCloudRegistration, RegistrationResult};

/// Number of bins per angular feature; the full descriptor is `3 * BINS`.
const BINS: usize = 11;
/// FPFH descriptor dimensionality (`alpha`, `phi`, `theta` histograms).
const FPFH_DIM: usize = 3 * BINS;

/// A single FPFH descriptor.
type Descriptor = [f32; FPFH_DIM];

/// Configuration for [`FpfhRansacRegistration`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FpfhRansacConfig {
    /// Radius used to gather neighbors when building FPFH descriptors. Should be
    /// noticeably larger than the normal-estimation radius (≈5× point spacing).
    pub feature_radius: f32,
    /// Distance below which a transformed source point counts as an inlier.
    pub max_correspondence_distance: f32,
    /// Number of RANSAC sampling iterations.
    pub ransac_iterations: usize,
    /// Correspondences per RANSAC sample (minimum 3 for a rigid transform).
    pub sample_size: usize,
    /// Pairwise edge lengths in a sample must agree within this relative
    /// tolerance, which cheaply rejects geometrically inconsistent samples.
    pub edge_length_tolerance: f32,
    /// Seed for the deterministic RNG, so runs are reproducible.
    pub seed: u64,
}

impl Default for FpfhRansacConfig {
    fn default() -> Self {
        Self {
            feature_radius: 0.25,
            max_correspondence_distance: 0.075,
            ransac_iterations: 4000,
            sample_size: 3,
            edge_length_tolerance: 0.9,
            seed: 0x5eed,
        }
    }
}

impl FpfhRansacConfig {
    /// Creates a config from the feature radius and inlier distance.
    #[must_use]
    pub fn with_radius(feature_radius: f32, max_correspondence_distance: f32) -> Self {
        Self { feature_radius, max_correspondence_distance, ..Self::default() }
    }
}

/// FPFH + RANSAC global registration.
///
/// Both `source` and `target` must carry normals (e.g. from normal estimation);
/// FPFH is built from the angular relationships between a point's normal and its
/// neighbors' normals, so normals are mandatory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FpfhRansacRegistration {
    config: FpfhRansacConfig,
}

impl FpfhRansacRegistration {
    /// Creates a registration from config.
    #[must_use]
    pub const fn new(config: FpfhRansacConfig) -> Self {
        Self { config }
    }

    /// Returns the registration config.
    #[must_use]
    pub const fn config(&self) -> FpfhRansacConfig {
        self.config
    }
}

impl PointCloudRegistration for FpfhRansacRegistration {
    fn name(&self) -> &'static str {
        "FpfhRansacRegistration"
    }

    fn align(&self, source: &PointCloud, target: &PointCloud) -> SpatialResult<RegistrationResult> {
        if self.config.feature_radius <= 0.0 || self.config.feature_radius.is_nan() {
            return Err(SpatialError::InvalidArgument(
                "feature_radius must be positive".to_owned(),
            ));
        }
        if self.config.sample_size < 3 {
            return Err(SpatialError::InvalidArgument("sample_size must be at least 3".to_owned()));
        }
        if source.is_empty() || target.is_empty() {
            return Err(SpatialError::InvalidArgument(
                "source and target must be non-empty".to_owned(),
            ));
        }

        let src = PointSet::from_cloud(source)?;
        let tgt = PointSet::from_cloud(target)?;

        let src_features = compute_fpfh(&src, self.config.feature_radius);
        let tgt_features = compute_fpfh(&tgt, self.config.feature_radius);

        // For each source point, its best feature-space match in the target.
        let matches: Vec<usize> =
            src_features.iter().map(|feature| nearest_feature(feature, &tgt_features)).collect();

        self.ransac(&src, &tgt, &matches)
    }
}

impl FpfhRansacRegistration {
    fn ransac(
        &self,
        src: &PointSet,
        tgt: &PointSet,
        matches: &[usize],
    ) -> SpatialResult<RegistrationResult> {
        let n = src.points.len();
        let max_sq = self.config.max_correspondence_distance.powi(2);
        let mut rng = Lcg::new(self.config.seed);

        let mut best_inliers = 0_usize;
        let mut best_error = f64::INFINITY;
        let mut best_transform = Isometry3::<f32>::identity();

        for _ in 0..self.config.ransac_iterations {
            // Draw `sample_size` distinct source indices.
            let mut sample = Vec::with_capacity(self.config.sample_size);
            let mut attempts = 0;
            while sample.len() < self.config.sample_size && attempts < self.config.sample_size * 8 {
                let idx = (rng.next_u32() as usize) % n;
                if !sample.contains(&idx) {
                    sample.push(idx);
                }
                attempts += 1;
            }
            if sample.len() < self.config.sample_size {
                continue;
            }

            let sample_src: Vec<Vec3<f32>> = sample.iter().map(|&i| src.points[i]).collect();
            let sample_tgt: Vec<Vec3<f32>> =
                sample.iter().map(|&i| tgt.points[matches[i]]).collect();

            if !edge_lengths_consistent(&sample_src, &sample_tgt, self.config.edge_length_tolerance)
            {
                continue;
            }

            let Some(transform) = estimate_rigid_transform(&sample_src, &sample_tgt) else {
                continue;
            };

            // Score the candidate over all feature matches.
            let mut inliers = 0_usize;
            let mut error = 0.0_f64;
            for (i, &match_idx) in matches.iter().enumerate() {
                let moved = transform.transform_point(src.points[i]);
                let dist_sq = (moved - tgt.points[match_idx]).length_squared();
                if dist_sq <= max_sq {
                    inliers += 1;
                    error += f64::from(dist_sq);
                }
            }

            let mean_error = if inliers > 0 { error / inliers as f64 } else { f64::INFINITY };
            // Prefer more inliers; break ties by lower mean inlier error.
            if inliers > best_inliers || (inliers == best_inliers && mean_error < best_error) {
                best_inliers = inliers;
                best_error = mean_error;
                best_transform = transform;
            }
        }

        Ok(RegistrationResult {
            transform: best_transform,
            fitness: if best_inliers > 0 { best_error } else { f64::INFINITY },
            iterations: self.config.ransac_iterations,
            converged: best_inliers >= self.config.sample_size,
        })
    }
}

/// Positions + normals lifted out of a cloud into contiguous `Vec3` storage.
struct PointSet {
    points: Vec<Vec3<f32>>,
    normals: Vec<Vec3<f32>>,
    tree: KdTree,
}

impl PointSet {
    fn from_cloud(cloud: &PointCloud) -> SpatialResult<Self> {
        let (x, y, z) = cloud.positions3()?;
        let (nx, ny, nz) = cloud.normals3()?;
        let points: Vec<Vec3<f32>> = (0..x.len()).map(|i| Vec3::new(x[i], y[i], z[i])).collect();
        let normals: Vec<Vec3<f32>> =
            (0..nx.len()).map(|i| Vec3::new(nx[i], ny[i], nz[i])).collect();
        let tree = KdTree::from_slices(x, y, z);
        Ok(Self { points, normals, tree })
    }
}

/// Computes the three Darboux-frame angular features between an anchor point
/// `(p, n_p)` and a neighbor `(q, n_q)`: returns `(alpha, phi, theta)`.
fn darboux(p: Vec3<f32>, np: Vec3<f32>, q: Vec3<f32>, nq: Vec3<f32>) -> Option<(f32, f32, f32)> {
    let diff = q - p;
    let dist = diff.length();
    if dist < 1e-9 {
        return None;
    }
    let u = np;
    let pq = diff.normalize();
    let v = pq.cross(u);
    let v_len = v.length();
    if v_len < 1e-9 {
        return None;
    }
    let v = v.normalize();
    let w = u.cross(v);

    let alpha = v.dot(nq); // [-1, 1]
    let phi = u.dot(pq); // [-1, 1]
    let theta = w.dot(nq).atan2(u.dot(nq)); // [-pi, pi]
    Some((alpha, phi, theta))
}

/// Bins a value from `[lo, hi]` into `BINS` buckets.
fn bin_index(value: f32, lo: f32, hi: f32) -> usize {
    let t = ((value - lo) / (hi - lo)).clamp(0.0, 0.999_999);
    (t * BINS as f32) as usize
}

/// Builds the per-point Simplified Point Feature Histograms (SPFH).
fn compute_spfh(set: &PointSet, radius: f32) -> Vec<Descriptor> {
    let mut spfh = vec![[0.0_f32; FPFH_DIM]; set.points.len()];
    for (i, spfh_i) in spfh.iter_mut().enumerate() {
        let p = set.points[i];
        let np = set.normals[i];
        let neighbors = set.tree.radius_search(p.x, p.y, p.z, radius);
        let mut count = 0_u32;
        for neighbor in neighbors {
            let j = neighbor.index;
            if j == i {
                continue;
            }
            let Some((alpha, phi, theta)) = darboux(p, np, set.points[j], set.normals[j]) else {
                continue;
            };
            spfh_i[bin_index(alpha, -1.0, 1.0)] += 1.0;
            spfh_i[BINS + bin_index(phi, -1.0, 1.0)] += 1.0;
            spfh_i[2 * BINS + bin_index(theta, -std::f32::consts::PI, std::f32::consts::PI)] += 1.0;
            count += 1;
        }
        if count > 0 {
            // Normalize each sub-histogram to percentages.
            for sub in 0..3 {
                let slice = &mut spfh_i[sub * BINS..(sub + 1) * BINS];
                let sum: f32 = slice.iter().sum();
                if sum > 0.0 {
                    for bin in slice {
                        *bin = *bin / sum * 100.0;
                    }
                }
            }
        }
    }
    spfh
}

/// Builds FPFH descriptors by distance-weighting each point's neighbors' SPFH.
fn compute_fpfh(set: &PointSet, radius: f32) -> Vec<Descriptor> {
    let spfh = compute_spfh(set, radius);
    let mut fpfh = vec![[0.0_f32; FPFH_DIM]; set.points.len()];
    for (i, fpfh_i) in fpfh.iter_mut().enumerate() {
        let p = set.points[i];
        let neighbors = set.tree.radius_search(p.x, p.y, p.z, radius);
        let mut sum_weight = 0.0_f32;
        let mut acc = [0.0_f32; FPFH_DIM];
        for neighbor in neighbors {
            let j = neighbor.index;
            if j == i {
                continue;
            }
            let dist = neighbor.distance_squared.sqrt();
            if dist < 1e-9 {
                continue;
            }
            let weight = 1.0 / dist;
            sum_weight += weight;
            for bin in 0..FPFH_DIM {
                acc[bin] += weight * spfh[j][bin];
            }
        }
        for bin in 0..FPFH_DIM {
            fpfh_i[bin] = spfh[i][bin] + if sum_weight > 0.0 { acc[bin] / sum_weight } else { 0.0 };
        }
    }
    fpfh
}

/// Brute-force nearest descriptor (squared L2) in feature space.
fn nearest_feature(feature: &Descriptor, candidates: &[Descriptor]) -> usize {
    let mut best = 0_usize;
    let mut best_dist = f32::INFINITY;
    for (idx, candidate) in candidates.iter().enumerate() {
        let mut dist = 0.0_f32;
        for bin in 0..FPFH_DIM {
            let d = feature[bin] - candidate[bin];
            dist += d * d;
            if dist >= best_dist {
                break;
            }
        }
        if dist < best_dist {
            best_dist = dist;
            best = idx;
        }
    }
    best
}

/// Checks that pairwise edge lengths in the source sample match the target
/// sample within `tolerance` (a relative ratio), rejecting distorted samples.
fn edge_lengths_consistent(src: &[Vec3<f32>], tgt: &[Vec3<f32>], tolerance: f32) -> bool {
    for a in 0..src.len() {
        for b in (a + 1)..src.len() {
            let ls = (src[a] - src[b]).length();
            let lt = (tgt[a] - tgt[b]).length();
            let lo = ls.min(lt);
            let hi = ls.max(lt);
            if hi < 1e-9 {
                continue;
            }
            if lo / hi < tolerance {
                return false;
            }
        }
    }
    true
}

/// Small deterministic xorshift-style PRNG (avoids a `rand` dependency).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        // Avoid a zero state, which xorshift cannot escape.
        Self { state: seed ^ 0x9e37_79b9_7f4a_7c15 }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::{FpfhRansacConfig, FpfhRansacRegistration};
    use crate::registration::PointCloudRegistration;
    use spatialrust_core::{DType, FieldSemantic, PointCloudBuilder, PointField, PointSchema};
    use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};

    fn schema_with_normals() -> PointSchema {
        PointSchema::new()
            .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
            .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
            .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32))
            .with_field(PointField::scalar("normal_x", FieldSemantic::NormalX, DType::F32))
            .with_field(PointField::scalar("normal_y", FieldSemantic::NormalY, DType::F32))
            .with_field(PointField::scalar("normal_z", FieldSemantic::NormalZ, DType::F32))
    }

    /// Two angled planes meeting at an edge, with analytic normals. The corner
    /// gives FPFH something distinctive to latch onto.
    fn corner_cloud() -> (Vec<Vec3<f32>>, Vec<Vec3<f32>>) {
        let mut points = Vec::new();
        let mut normals = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                let (a, b) = (i as f32 * 0.05, j as f32 * 0.05);
                // floor (normal +z)
                points.push(Vec3::new(a, b, 0.0));
                normals.push(Vec3::new(0.0, 0.0, 1.0));
                // wall (normal +y)
                points.push(Vec3::new(a, 0.0, b + 0.05));
                normals.push(Vec3::new(0.0, 1.0, 0.0));
            }
        }
        (points, normals)
    }

    fn build_cloud(points: &[Vec3<f32>], normals: &[Vec3<f32>]) -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(schema_with_normals());
        for (p, n) in points.iter().zip(normals) {
            builder.push_point([p.x, p.y, p.z, n.x, n.y, n.z]).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn recovers_coarse_alignment_without_initial_guess() {
        let (points, normals) = corner_cloud();
        let target = build_cloud(&points, &normals);

        // Apply a sizeable yaw + translation that ICP alone could not recover.
        let misalign = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.6),
            Vec3::new(0.3, -0.2, 0.1),
        );
        let moved_points: Vec<Vec3<f32>> =
            points.iter().map(|p| misalign.transform_point(*p)).collect();
        let moved_normals: Vec<Vec3<f32>> =
            normals.iter().map(|n| misalign.transform_point(*n) - misalign.translation()).collect();
        let source = build_cloud(&moved_points, &moved_normals);

        let config = FpfhRansacConfig {
            feature_radius: 0.2,
            max_correspondence_distance: 0.05,
            ransac_iterations: 6000,
            seed: 7,
            ..FpfhRansacConfig::default()
        };
        let result = FpfhRansacRegistration::new(config).align(&source, &target).unwrap();
        assert!(result.converged);

        // Global registration yields a *coarse* pose meant to seed ICP/GICP, so
        // the residual only needs to land within their convergence basin (a few
        // point spacings, here 0.05), not at sub-voxel accuracy.
        let probe = moved_points[123];
        let restored = result.transform.transform_point(probe);
        let expected = points[123];
        let err = (restored - expected).length();
        assert!(err < 0.15, "coarse alignment residual too large: {err}");
    }

    #[test]
    fn rejects_bad_params() {
        let (points, normals) = corner_cloud();
        let cloud = build_cloud(&points, &normals);
        let config = FpfhRansacConfig { feature_radius: 0.0, ..FpfhRansacConfig::default() };
        assert!(FpfhRansacRegistration::new(config).align(&cloud, &cloud).is_err());
    }
}
