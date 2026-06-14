//! RANSAC fitting of sphere and cylinder primitives.
//!
//! These complement the plane segmenter for detecting man-made shapes — pipes,
//! tanks, poles, balls. Sphere fitting needs only positions; cylinder fitting
//! also needs per-point normals (the axis is recovered from two surface
//! normals), so the input cloud must carry normals.

use spatialrust_core::{HasNormals3, HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_math::{solve_linear_system, LeastSquaresResult, Vec3};

use crate::cloud::extract_mask;
use crate::segmenter::PointCloudSegmenter;

/// Shared RANSAC controls for primitive fitting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RansacPrimitiveConfig {
    /// Maximum surface distance for inlier classification.
    pub distance_threshold: f32,
    /// Maximum number of RANSAC iterations.
    pub max_iterations: usize,
    /// Minimum number of inliers required to accept a model.
    pub min_inliers: usize,
    /// Smallest acceptable radius (rejects degenerate near-flat fits).
    pub min_radius: f32,
    /// Largest acceptable radius.
    pub max_radius: f32,
    /// Seed for deterministic sampling.
    pub seed: u64,
}

impl Default for RansacPrimitiveConfig {
    fn default() -> Self {
        Self {
            distance_threshold: 0.02,
            max_iterations: 1_000,
            min_inliers: 10,
            min_radius: 0.0,
            max_radius: f32::INFINITY,
            seed: 42,
        }
    }
}

/// Sphere model: all surface points are `radius` from `center`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereModel {
    /// Sphere center.
    pub center: Vec3<f32>,
    /// Sphere radius.
    pub radius: f32,
}

impl SphereModel {
    /// Absolute distance from `point` to the sphere surface.
    #[must_use]
    pub fn distance(&self, point: Vec3<f32>) -> f32 {
        ((point - self.center).length() - self.radius).abs()
    }
}

/// Cylinder model: all surface points are `radius` from the axis line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderModel {
    /// A point lying on the cylinder axis.
    pub axis_point: Vec3<f32>,
    /// Unit-length axis direction.
    pub axis_direction: Vec3<f32>,
    /// Cylinder radius.
    pub radius: f32,
}

impl CylinderModel {
    /// Perpendicular distance from `point` to the axis line.
    #[must_use]
    pub fn axis_distance(&self, point: Vec3<f32>) -> f32 {
        let v = point - self.axis_point;
        let along = scale(self.axis_direction, v.dot(self.axis_direction));
        (v - along).length()
    }

    /// Absolute distance from `point` to the cylinder surface.
    #[must_use]
    pub fn distance(&self, point: Vec3<f32>) -> f32 {
        (self.axis_distance(point) - self.radius).abs()
    }
}

/// Result of fitting a primitive, partitioning the cloud into inliers/outliers.
#[derive(Clone, Debug, PartialEq)]
pub struct PrimitiveSegmentation<M> {
    /// Fitted model.
    pub model: M,
    /// Points classified as inliers.
    pub inliers: PointCloud,
    /// Points classified as outliers.
    pub outliers: PointCloud,
    /// Number of inlier points.
    pub inlier_count: usize,
}

/// RANSAC sphere segmenter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RansacSphereSegmenter {
    config: RansacPrimitiveConfig,
}

impl RansacSphereSegmenter {
    /// Creates a segmenter from config.
    #[must_use]
    pub const fn new(config: RansacPrimitiveConfig) -> Self {
        Self { config }
    }

    /// Returns the segmenter config.
    #[must_use]
    pub const fn config(&self) -> RansacPrimitiveConfig {
        self.config
    }

    /// Fits the dominant sphere and partitions the cloud.
    pub fn segment(&self, input: &PointCloud) -> SpatialResult<PrimitiveSegmentation<SphereModel>> {
        let (x, y, z) = input.positions3()?;
        let len = input.len();
        if len < 4 {
            return Err(SpatialError::InvalidArgument(
                "sphere fitting requires at least four points".to_owned(),
            ));
        }

        let mut rng = Rng::new(self.config.seed);
        let mut best_inliers: Vec<usize> = Vec::new();
        let mut best_model = None;

        for _ in 0..self.config.max_iterations {
            let Some(sample) = sample_distinct::<4>(&mut rng, len) else {
                continue;
            };
            let Some(model) = sphere_from_points(x, y, z, sample) else {
                continue;
            };
            if model.radius < self.config.min_radius || model.radius > self.config.max_radius {
                continue;
            }
            let inliers = collect_inliers(len, self.config.distance_threshold, |i| {
                model.distance(Vec3::new(x[i], y[i], z[i]))
            });
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_model = Some(model);
            }
        }

        finalize(input, best_model, &best_inliers, self.config.min_inliers)
    }
}

impl PointCloudSegmenter for RansacSphereSegmenter {
    fn name(&self) -> &'static str {
        "RansacSphereSegmenter"
    }
}

/// RANSAC cylinder segmenter. The input cloud must carry normals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RansacCylinderSegmenter {
    config: RansacPrimitiveConfig,
}

impl RansacCylinderSegmenter {
    /// Creates a segmenter from config.
    #[must_use]
    pub const fn new(config: RansacPrimitiveConfig) -> Self {
        Self { config }
    }

    /// Returns the segmenter config.
    #[must_use]
    pub const fn config(&self) -> RansacPrimitiveConfig {
        self.config
    }

    /// Fits the dominant cylinder and partitions the cloud.
    pub fn segment(
        &self,
        input: &PointCloud,
    ) -> SpatialResult<PrimitiveSegmentation<CylinderModel>> {
        let (x, y, z) = input.positions3()?;
        let (nx, ny, nz) = input.normals3()?;
        let len = input.len();
        if len < 2 {
            return Err(SpatialError::InvalidArgument(
                "cylinder fitting requires at least two points".to_owned(),
            ));
        }

        let mut rng = Rng::new(self.config.seed);
        let mut best_inliers: Vec<usize> = Vec::new();
        let mut best_model = None;

        for _ in 0..self.config.max_iterations {
            let Some(sample) = sample_distinct::<2>(&mut rng, len) else {
                continue;
            };
            let Some(model) = cylinder_from_points(x, y, z, nx, ny, nz, sample) else {
                continue;
            };
            if model.radius < self.config.min_radius || model.radius > self.config.max_radius {
                continue;
            }
            let inliers = collect_inliers(len, self.config.distance_threshold, |i| {
                model.distance(Vec3::new(x[i], y[i], z[i]))
            });
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_model = Some(model);
            }
        }

        finalize(input, best_model, &best_inliers, self.config.min_inliers)
    }
}

impl PointCloudSegmenter for RansacCylinderSegmenter {
    fn name(&self) -> &'static str {
        "RansacCylinderSegmenter"
    }
}

/// Builds the inlier/outlier partition once the best model is known.
fn finalize<M>(
    input: &PointCloud,
    best_model: Option<M>,
    best_inliers: &[usize],
    min_inliers: usize,
) -> SpatialResult<PrimitiveSegmentation<M>> {
    if best_inliers.len() < min_inliers || best_model.is_none() {
        return Err(SpatialError::InvalidArgument(format!(
            "RANSAC found only {} inliers, minimum is {min_inliers}",
            best_inliers.len()
        )));
    }
    let model = best_model.expect("checked above");

    let mut inlier_mask = vec![false; input.len()];
    for &index in best_inliers {
        inlier_mask[index] = true;
    }
    let outlier_mask: Vec<bool> = inlier_mask.iter().map(|&keep| !keep).collect();

    Ok(PrimitiveSegmentation {
        model,
        inliers: extract_mask(input, &inlier_mask)?,
        outliers: extract_mask(input, &outlier_mask)?,
        inlier_count: best_inliers.len(),
    })
}

fn collect_inliers(len: usize, threshold: f32, distance: impl Fn(usize) -> f32) -> Vec<usize> {
    (0..len).filter(|&i| distance(i) <= threshold).collect()
}

/// Solves for the sphere through four points (subtract one equation from the
/// others to linearize, then solve the 3×3 system for the center).
fn sphere_from_points(x: &[f32], y: &[f32], z: &[f32], idx: [usize; 4]) -> Option<SphereModel> {
    let p: Vec<[f64; 3]> =
        idx.iter().map(|&i| [f64::from(x[i]), f64::from(y[i]), f64::from(z[i])]).collect();
    let sq = |q: [f64; 3]| q[0] * q[0] + q[1] * q[1] + q[2] * q[2];

    let mut a = Vec::with_capacity(3);
    let mut b = Vec::with_capacity(3);
    for row in 1..4 {
        a.push(vec![
            2.0 * (p[row][0] - p[0][0]),
            2.0 * (p[row][1] - p[0][1]),
            2.0 * (p[row][2] - p[0][2]),
        ]);
        b.push(sq(p[row]) - sq(p[0]));
    }

    let center = match solve_linear_system(a, b) {
        LeastSquaresResult::Solved(c) => c,
        LeastSquaresResult::Singular => return None,
    };
    let center = Vec3::new(center[0] as f32, center[1] as f32, center[2] as f32);
    let radius = (center - Vec3::new(x[idx[0]], y[idx[0]], z[idx[0]])).length();
    if !radius.is_finite() {
        return None;
    }
    Some(SphereModel { center, radius })
}

/// Recovers a cylinder from two points with surface normals: the axis is the
/// cross product of the normals, and projecting into the plane perpendicular to
/// the axis turns the problem into fitting a circle through two points whose
/// (projected) normals point at the center.
#[allow(clippy::too_many_arguments)]
fn cylinder_from_points(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    nx: &[f32],
    ny: &[f32],
    nz: &[f32],
    idx: [usize; 2],
) -> Option<CylinderModel> {
    let (i0, i1) = (idx[0], idx[1]);
    let p0 = Vec3::new(x[i0], y[i0], z[i0]);
    let p1 = Vec3::new(x[i1], y[i1], z[i1]);
    let n0 = Vec3::new(nx[i0], ny[i0], nz[i0]);
    let n1 = Vec3::new(nx[i1], ny[i1], nz[i1]);

    let axis = n0.cross(n1);
    if axis.length_squared() < 1e-10 {
        return None; // normals parallel: axis undefined.
    }
    let axis = axis.normalize();

    // Orthonormal basis (u, w) spanning the plane perpendicular to the axis.
    let helper =
        if axis.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let u = axis.cross(helper).normalize();
    let w = axis.cross(u);

    let proj = |v: Vec3<f32>| (v.dot(u), v.dot(w));
    let (p0u, p0w) = proj(p0);
    let (p1u, p1w) = proj(p1);
    let (mut n0u, mut n0w) = proj(n0);
    let (mut n1u, mut n1w) = proj(n1);
    let l0 = (n0u * n0u + n0w * n0w).sqrt();
    let l1 = (n1u * n1u + n1w * n1w).sqrt();
    if l0 < 1e-6 || l1 < 1e-6 {
        return None; // a normal nearly parallel to the axis projects to ~0.
    }
    n0u /= l0;
    n0w /= l0;
    n1u /= l1;
    n1w /= l1;

    // Intersect the two in-plane normal lines: P0 + t0 N0 = P1 + t1 N1.
    // [N0.u, -N1.u; N0.w, -N1.w] [t0; t1] = [P1.u - P0.u; P1.w - P0.w].
    let det = n0u * (-n1w) - (-n1u) * n0w;
    if det.abs() < 1e-9 {
        return None;
    }
    let rhs = (p1u - p0u, p1w - p0w);
    let t0 = (rhs.0 * (-n1w) - (-n1u) * rhs.1) / det;

    let center_u = p0u + t0 * n0u;
    let center_w = p0w + t0 * n0w;
    let radius = ((center_u - p0u).powi(2) + (center_w - p0w).powi(2)).sqrt();
    if !radius.is_finite() {
        return None;
    }

    let axis_point = scale(u, center_u) + scale(w, center_w);
    Some(CylinderModel { axis_point, axis_direction: axis, radius })
}

/// Scales a vector by a scalar (`Vec3` has no scalar-multiply operator).
fn scale(v: Vec3<f32>, s: f32) -> Vec3<f32> {
    Vec3::new(v.x * s, v.y * s, v.z * s)
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        self.state = self.state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        // Map the high, well-mixed bits of the LCG into `0..upper` (its low bits
        // have a short period, which biases naive `% upper` sampling).
        (((self.state >> 32) * upper as u64) >> 32) as usize
    }
}

/// Draws `N` distinct indices in `0..len`, or `None` if it cannot.
fn sample_distinct<const N: usize>(rng: &mut Rng, len: usize) -> Option<[usize; N]> {
    if len < N {
        return None;
    }
    let mut out = [0usize; N];
    let mut filled = 0;
    let mut attempts = 0;
    while filled < N && attempts < N * 16 {
        let candidate = rng.next_usize(len);
        if !out[..filled].contains(&candidate) {
            out[filled] = candidate;
            filled += 1;
        }
        attempts += 1;
    }
    (filled == N).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_core::{DType, FieldSemantic, PointCloudBuilder, PointField, PointSchema};
    use std::f32::consts::PI;

    fn xyz_cloud(points: &[Vec3<f32>]) -> PointCloud {
        let mut builder = PointCloudBuilder::new(
            PointSchema::new()
                .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
                .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
                .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32)),
        );
        for p in points {
            builder.push_point([p.x, p.y, p.z]).unwrap();
        }
        builder.build().unwrap()
    }

    fn xyz_normal_cloud(points: &[(Vec3<f32>, Vec3<f32>)]) -> PointCloud {
        let mut builder = PointCloudBuilder::new(
            PointSchema::new()
                .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
                .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
                .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32))
                .with_field(PointField::scalar("normal_x", FieldSemantic::NormalX, DType::F32))
                .with_field(PointField::scalar("normal_y", FieldSemantic::NormalY, DType::F32))
                .with_field(PointField::scalar("normal_z", FieldSemantic::NormalZ, DType::F32)),
        );
        for (p, n) in points {
            builder.push_point([p.x, p.y, p.z, n.x, n.y, n.z]).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn fits_sphere_with_outliers() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let radius = 0.5_f32;
        let mut pts = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                let theta = PI * i as f32 / 11.0;
                let phi = 2.0 * PI * j as f32 / 12.0;
                pts.push(
                    center
                        + Vec3::new(
                            radius * theta.sin() * phi.cos(),
                            radius * theta.sin() * phi.sin(),
                            radius * theta.cos(),
                        ),
                );
            }
        }
        // Scatter some outliers far from the surface.
        pts.push(center + Vec3::new(3.0, 0.0, 0.0));
        pts.push(center + Vec3::new(0.0, 3.0, 0.0));

        let cloud = xyz_cloud(&pts);
        let seg = RansacSphereSegmenter::new(RansacPrimitiveConfig {
            distance_threshold: 0.02,
            max_iterations: 800,
            min_inliers: 50,
            seed: 3,
            ..RansacPrimitiveConfig::default()
        });
        let result = seg.segment(&cloud).unwrap();
        assert!((result.model.radius - radius).abs() < 0.02);
        assert!((result.model.center - center).length() < 0.02);
        assert_eq!(result.outliers.len(), 2);
    }

    #[test]
    fn fits_sphere_among_many_distractors() {
        // A sphere plus a cloud of scattered random distractors of comparable
        // size. Random points fit no sphere, so the only high-inlier model is the
        // true sphere, and finding it requires reliably sampling 4 sphere points
        // among the ~50% noise -- a stress test for the RANSAC sampler.
        let center = Vec3::new(0.0, 0.0, 0.0);
        let radius = 0.4_f32;
        let mut pts = Vec::new();
        for i in 0..16 {
            for j in 0..16 {
                let theta = PI * i as f32 / 15.0;
                let phi = 2.0 * PI * j as f32 / 16.0;
                pts.push(Vec3::new(
                    radius * theta.sin() * phi.cos(),
                    radius * theta.sin() * phi.sin(),
                    radius * theta.cos(),
                ));
            }
        }
        // ~256 scattered distractors in a box well away from the sphere surface.
        let mut s = 1234_u64;
        let mut rand = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as f32 / (1u64 << 24) as f32 // in [0, 1)
        };
        for _ in 0..256 {
            pts.push(Vec3::new(rand() * 4.0 - 2.0, rand() * 4.0 - 2.0, rand() * 4.0 + 2.0));
        }

        let cloud = xyz_cloud(&pts);
        let seg = RansacSphereSegmenter::new(RansacPrimitiveConfig {
            distance_threshold: 0.02,
            max_iterations: 2000,
            min_inliers: 100,
            seed: 11,
            ..RansacPrimitiveConfig::default()
        });
        let result = seg.segment(&cloud).unwrap();
        assert!((result.model.radius - radius).abs() < 0.03, "radius {}", result.model.radius);
        assert!((result.model.center - center).length() < 0.03);
    }

    #[test]
    fn fits_cylinder_axis_and_radius() {
        let radius = 0.4_f32;
        // Cylinder along +z through the origin.
        let mut samples = Vec::new();
        for i in 0..20 {
            for j in 0..24 {
                let h = i as f32 * 0.1;
                let phi = 2.0 * PI * j as f32 / 24.0;
                let dir = Vec3::new(phi.cos(), phi.sin(), 0.0);
                samples.push((Vec3::new(radius * phi.cos(), radius * phi.sin(), h), dir));
            }
        }
        let cloud = xyz_normal_cloud(&samples);
        let seg = RansacCylinderSegmenter::new(RansacPrimitiveConfig {
            distance_threshold: 0.02,
            max_iterations: 800,
            min_inliers: 100,
            seed: 5,
            ..RansacPrimitiveConfig::default()
        });
        let result = seg.segment(&cloud).unwrap();
        assert!((result.model.radius - radius).abs() < 0.03, "radius {}", result.model.radius);
        // Axis should be (anti)parallel to +z.
        assert!(result.model.axis_direction.z.abs() > 0.98);
    }

    #[test]
    fn sphere_rejects_too_few_points() {
        let cloud = xyz_cloud(&[Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)]);
        assert!(RansacSphereSegmenter::new(RansacPrimitiveConfig::default())
            .segment(&cloud)
            .is_err());
    }
}
