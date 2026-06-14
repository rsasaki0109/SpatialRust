//! Moving Least Squares (MLS) surface smoothing.
//!
//! For each point a local reference plane is fit to its neighborhood, a bivariate
//! polynomial height field is fit over that plane (weighted least squares), and
//! the point is projected onto the polynomial surface. This removes scanner noise
//! while preserving curvature far better than a plain average, and yields cleaner
//! normals for downstream estimation.

use spatialrust_core::{
    FieldSemantic, HasPositions3, PointBuffer, PointBufferSet, PointCloud, SpatialError,
    SpatialResult,
};
use spatialrust_math::{solve_linear_system, symmetric_eigen3, LeastSquaresResult, Mat3, Vec3};
use spatialrust_search::{KdTree, RadiusSearchIndex};

use crate::filter::PointCloudFilter;

/// Configuration for [`MlsSmoothing`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlsConfig {
    /// Neighborhood radius used to fit the local surface.
    pub search_radius: f32,
    /// Polynomial order of the fitted height field (1 = plane, 2 = quadratic).
    pub polynomial_order: u8,
    /// Minimum neighbors required to smooth a point (else it is left in place).
    pub min_neighbors: usize,
}

impl Default for MlsConfig {
    fn default() -> Self {
        Self { search_radius: 0.1, polynomial_order: 2, min_neighbors: 6 }
    }
}

impl MlsConfig {
    /// Creates a config with the given search radius (quadratic fit).
    #[must_use]
    pub const fn with_radius(search_radius: f32) -> Self {
        Self { search_radius, polynomial_order: 2, min_neighbors: 6 }
    }
}

/// Moving Least Squares smoothing filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlsSmoothing {
    config: MlsConfig,
}

impl MlsSmoothing {
    /// Creates a smoother from config.
    #[must_use]
    pub const fn new(config: MlsConfig) -> Self {
        Self { config }
    }

    /// Returns the smoother config.
    #[must_use]
    pub const fn config(&self) -> MlsConfig {
        self.config
    }

    /// Returns the smoothed XYZ positions, one per input point.
    pub fn smoothed_positions(&self, input: &PointCloud) -> SpatialResult<Vec<Vec3<f32>>> {
        if self.config.search_radius <= 0.0 || self.config.search_radius.is_nan() {
            return Err(SpatialError::InvalidArgument("search_radius must be positive".to_owned()));
        }
        if self.config.polynomial_order > 2 {
            return Err(SpatialError::InvalidArgument(
                "polynomial_order must be 1 or 2".to_owned(),
            ));
        }

        let (x, y, z) = input.positions3()?;
        let len = input.len();
        let tree = KdTree::from_slices(x, y, z);
        // Gaussian weight scale: ~1/2 of the radius averages out noise while
        // still down-weighting the far neighborhood to preserve curvature.
        let h_sq = (self.config.search_radius / 2.0).powi(2).max(1e-12);

        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let p = Vec3::new(x[i], y[i], z[i]);
            let neighbors = tree.radius_search(p.x, p.y, p.z, self.config.search_radius);
            if neighbors.len() < self.config.min_neighbors {
                out.push(p);
                continue;
            }
            // Leave-one-out: exclude the query point itself so the surface is
            // determined by its neighbors, which smooths the point's own noise.
            let pts: Vec<Vec3<f32>> = neighbors
                .iter()
                .filter(|n| n.index != i)
                .map(|n| Vec3::new(x[n.index], y[n.index], z[n.index]))
                .collect();
            out.push(project_point(p, &pts, self.config.polynomial_order, h_sq).unwrap_or(p));
        }
        Ok(out)
    }
}

impl PointCloudFilter for MlsSmoothing {
    fn name(&self) -> &'static str {
        "MlsSmoothing"
    }

    fn filter(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        let smoothed = self.smoothed_positions(input)?;
        build_output(input, &smoothed)
    }
}

/// Fits a local frame + height polynomial and returns the projection of `p`.
fn project_point(p: Vec3<f32>, neighbors: &[Vec3<f32>], order: u8, h_sq: f32) -> Option<Vec3<f32>> {
    // Weighted centroid and covariance about the query point.
    let mut sum_w = 0.0_f64;
    let mut centroid = [0.0_f64; 3];
    let mut weights = Vec::with_capacity(neighbors.len());
    for q in neighbors {
        let d_sq = (*q - p).length_squared();
        let w = f64::from((-d_sq / h_sq).exp());
        weights.push(w);
        sum_w += w;
        centroid[0] += w * f64::from(q.x);
        centroid[1] += w * f64::from(q.y);
        centroid[2] += w * f64::from(q.z);
    }
    if sum_w < 1e-12 {
        return None;
    }
    let c = Vec3::new(
        (centroid[0] / sum_w) as f32,
        (centroid[1] / sum_w) as f32,
        (centroid[2] / sum_w) as f32,
    );

    let mut cov = [[0.0_f64; 3]; 3];
    for (q, &w) in neighbors.iter().zip(&weights) {
        let d = *q - c;
        let d = [f64::from(d.x), f64::from(d.y), f64::from(d.z)];
        for r in 0..3 {
            for col in 0..3 {
                cov[r][col] += w * d[r] * d[col];
            }
        }
    }
    let covariance = Mat3::<f64>::from_rows(cov[0], cov[1], cov[2]);
    let eigen = symmetric_eigen3(covariance);
    // Smallest eigenvector (column 0) is the plane normal.
    let normal = Vec3::new(
        eigen.eigenvectors.m[0][0] as f32,
        eigen.eigenvectors.m[1][0] as f32,
        eigen.eigenvectors.m[2][0] as f32,
    )
    .normalize();

    // In-plane orthonormal basis (u, w).
    let helper =
        if normal.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let u = normal.cross(helper).normalize();
    let v = normal.cross(u);

    // Query point's in-plane coordinates relative to the centroid.
    let rel = p - c;
    let qu = rel.dot(u);
    let qv = rel.dot(v);

    // Fit height(s, t) = sum coeff_k * basis_k(s, t) by weighted least squares.
    let basis = |s: f32, t: f32| match order {
        0 => vec![1.0_f64],
        1 => vec![1.0, f64::from(s), f64::from(t)],
        _ => vec![
            1.0,
            f64::from(s),
            f64::from(t),
            f64::from(s * s),
            f64::from(s * t),
            f64::from(t * t),
        ],
    };
    let terms = basis(0.0, 0.0).len();
    if neighbors.len() < terms {
        // Not enough support for this order: fall back to the plane.
        return Some(c + scale(u, qu) + scale(v, qv));
    }

    let mut ata = vec![vec![0.0_f64; terms]; terms];
    let mut atb = vec![0.0_f64; terms];
    for (q, &w) in neighbors.iter().zip(&weights) {
        let d = *q - c;
        let s = d.dot(u);
        let t = d.dot(v);
        let height = f64::from(d.dot(normal));
        let row = basis(s, t);
        for a in 0..terms {
            atb[a] += w * row[a] * height;
            for b in 0..terms {
                ata[a][b] += w * row[a] * row[b];
            }
        }
    }

    let coeffs = match solve_linear_system(ata, atb) {
        LeastSquaresResult::Solved(c) => c,
        LeastSquaresResult::Singular => return Some(c + scale(u, qu) + scale(v, qv)),
    };
    let query_basis = basis(qu, qv);
    let height: f64 = coeffs.iter().zip(&query_basis).map(|(c, b)| c * b).sum();

    Some(c + scale(u, qu) + scale(v, qv) + scale(normal, height as f32))
}

fn scale(v: Vec3<f32>, s: f32) -> Vec3<f32> {
    Vec3::new(v.x * s, v.y * s, v.z * s)
}

/// Rebuilds the cloud with smoothed positions, preserving every other field.
fn build_output(input: &PointCloud, positions: &[Vec3<f32>]) -> SpatialResult<PointCloud> {
    let schema = input.schema().clone();
    let name_for = |sem| schema.find_semantic(sem).map(|f| f.name.clone());
    let (xn, yn, zn) = (
        name_for(FieldSemantic::PositionX),
        name_for(FieldSemantic::PositionY),
        name_for(FieldSemantic::PositionZ),
    );

    let mut buffers = PointBufferSet::new();
    for field in schema.fields() {
        let name = &field.name;
        let buffer = if Some(name) == xn.as_ref() {
            PointBuffer::from_f32(positions.iter().map(|p| p.x).collect())
        } else if Some(name) == yn.as_ref() {
            PointBuffer::from_f32(positions.iter().map(|p| p.y).collect())
        } else if Some(name) == zn.as_ref() {
            PointBuffer::from_f32(positions.iter().map(|p| p.z).collect())
        } else {
            clone_buffer(input.field(name)?)
        };
        buffers.insert(name.clone(), buffer);
    }
    PointCloud::try_from_parts(schema, buffers, input.metadata().clone())
}

fn clone_buffer(buffer: &PointBuffer) -> PointBuffer {
    match buffer {
        PointBuffer::F32(v) => PointBuffer::from_f32(v.clone()),
        PointBuffer::F64(v) => PointBuffer::F64(v.clone()),
        PointBuffer::U8(v) => PointBuffer::U8(v.clone()),
        PointBuffer::U16(v) => PointBuffer::U16(v.clone()),
        PointBuffer::U32(v) => PointBuffer::U32(v.clone()),
        PointBuffer::I32(v) => PointBuffer::I32(v.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    /// A noisy plane: MLS should pull points back toward z = 0.
    #[test]
    fn flattens_noisy_plane() {
        // Deterministic pseudo-noise so the test is stable.
        let mut seed = 12345_u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // Top 32 bits mapped to a centered [-0.02, 0.02] perturbation.
            let unit = (seed >> 32) as u32 as f32 / u32::MAX as f32;
            (unit * 2.0 - 1.0) * 0.02
        };

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        let mut z_before = Vec::new();
        for i in 0..25 {
            for j in 0..25 {
                let z = noise();
                z_before.push(z);
                builder.push_point([i as f32 * 0.05, j as f32 * 0.05, z]).unwrap();
            }
        }
        let cloud = builder.build().unwrap();

        // Order-1 (plane) fit is the right tool for a planar surface.
        let smoother = MlsSmoothing::new(MlsConfig {
            search_radius: 0.2,
            polynomial_order: 1,
            min_neighbors: 6,
        });
        let out = smoother.filter(&cloud).unwrap();
        assert_eq!(out.len(), cloud.len());

        let (_, _, z) = out.positions3().unwrap();
        let interior = |k: usize| {
            let (i, j) = (k / 25, k % 25);
            (4..21).contains(&i) && (4..21).contains(&j)
        };
        // Compare the RMS deviation from the true plane (z = 0) before/after.
        let rms = |get: &dyn Fn(usize) -> f32| {
            let vals: Vec<f32> = (0..cloud.len()).filter(|&k| interior(k)).map(get).collect();
            (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt()
        };
        let before = rms(&|k| z_before[k]);
        let after = rms(&|k| z[k]);
        assert!(after < before * 0.6, "MLS did not flatten: rms {after} vs {before}");
    }

    #[test]
    fn rejects_bad_params() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();
        assert!(MlsSmoothing::new(MlsConfig::with_radius(0.0)).smoothed_positions(&cloud).is_err());
        assert!(MlsSmoothing::new(MlsConfig {
            search_radius: 0.1,
            polynomial_order: 3,
            min_neighbors: 6
        })
        .smoothed_positions(&cloud)
        .is_err());
    }
}
