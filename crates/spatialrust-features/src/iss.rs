//! Intrinsic Shape Signatures (ISS) keypoint detection.
//!
//! ISS picks a sparse set of geometrically salient points — corners and other
//! spots with variation in all three directions — by thresholding the ratios of
//! the eigenvalues of each point's neighborhood covariance, then keeping local
//! maxima of the smallest eigenvalue. The keypoints are a natural front-end for
//! descriptor-based global registration: far fewer points to describe and match.

use spatialrust_core::{
    HasPositions3, PointBuffer, PointBufferSet, PointCloud, SpatialError, SpatialResult,
};
use spatialrust_math::{symmetric_eigen3, Mat3};
use spatialrust_search::{KdTree, RadiusSearchIndex};

/// Configuration for [`IssKeypointDetector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IssKeypointConfig {
    /// Radius of the neighborhood used to build each covariance matrix.
    pub salient_radius: f32,
    /// Radius for non-maximum suppression of the saliency (smallest eigenvalue).
    pub non_max_radius: f32,
    /// Upper bound on `lambda2 / lambda1` (rejects flat / planar points).
    pub gamma_21: f32,
    /// Upper bound on `lambda3 / lambda2` (rejects edge / linear points).
    pub gamma_32: f32,
    /// Minimum neighbors within `salient_radius` for a point to be considered.
    pub min_neighbors: usize,
}

impl Default for IssKeypointConfig {
    fn default() -> Self {
        Self {
            salient_radius: 0.2,
            non_max_radius: 0.15,
            gamma_21: 0.975,
            gamma_32: 0.975,
            min_neighbors: 5,
        }
    }
}

impl IssKeypointConfig {
    /// Creates a config from the salient and non-max-suppression radii.
    #[must_use]
    pub fn with_radii(salient_radius: f32, non_max_radius: f32) -> Self {
        Self { salient_radius, non_max_radius, ..Self::default() }
    }
}

/// Result of ISS keypoint detection.
#[derive(Clone, Debug, PartialEq)]
pub struct IssKeypointResult {
    /// Indices of the keypoints in the input cloud.
    pub indices: Vec<usize>,
    /// Sub-cloud containing only the keypoints (input schema preserved).
    pub keypoints: PointCloud,
}

/// Intrinsic Shape Signatures keypoint detector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IssKeypointDetector {
    config: IssKeypointConfig,
}

impl IssKeypointDetector {
    /// Creates a detector from config.
    #[must_use]
    pub const fn new(config: IssKeypointConfig) -> Self {
        Self { config }
    }

    /// Returns the detector config.
    #[must_use]
    pub const fn config(&self) -> IssKeypointConfig {
        self.config
    }

    /// Detects keypoints and returns their indices and a keypoint sub-cloud.
    pub fn detect(&self, input: &PointCloud) -> SpatialResult<IssKeypointResult> {
        if self.config.salient_radius <= 0.0 || self.config.non_max_radius <= 0.0 {
            return Err(SpatialError::InvalidArgument("ISS radii must be positive".to_owned()));
        }

        let (x, y, z) = input.positions3()?;
        let len = input.len();
        let tree = KdTree::from_slices(x, y, z);

        // Saliency = smallest eigenvalue (lambda3) where the eigenvalue-ratio
        // tests pass; NaN marks a non-salient point.
        let mut saliency = vec![f32::NAN; len];
        for i in 0..len {
            let neighbors = tree.radius_search(x[i], y[i], z[i], self.config.salient_radius);
            if neighbors.len() < self.config.min_neighbors {
                continue;
            }
            let Some(eigenvalues) = neighborhood_eigenvalues(x, y, z, &neighbors) else {
                continue;
            };
            // Ascending order: l3 <= l2 <= l1.
            let (l3, l2, l1) = (eigenvalues[0], eigenvalues[1], eigenvalues[2]);
            if l1 <= 0.0 || l2 <= 0.0 {
                continue;
            }
            if (l2 / l1) < f64::from(self.config.gamma_21)
                && (l3 / l2) < f64::from(self.config.gamma_32)
            {
                saliency[i] = l3 as f32;
            }
        }

        // Non-maximum suppression: keep points whose saliency is the strict
        // maximum within `non_max_radius`.
        let mut indices = Vec::new();
        for i in 0..len {
            let s = saliency[i];
            if s.is_nan() {
                continue;
            }
            let neighbors = tree.radius_search(x[i], y[i], z[i], self.config.non_max_radius);
            let is_local_max = neighbors.iter().all(|n| {
                let other = saliency[n.index];
                n.index == i || other.is_nan() || s >= other
            });
            if is_local_max {
                indices.push(i);
            }
        }

        let keypoints = gather_indices(input, &indices)?;
        Ok(IssKeypointResult { indices, keypoints })
    }
}

/// Eigenvalues (ascending) of the neighborhood covariance about its centroid.
fn neighborhood_eigenvalues(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    neighbors: &[spatialrust_search::Neighbor],
) -> Option<[f64; 3]> {
    let count = neighbors.len() as f64;
    if count < 3.0 {
        return None;
    }
    let (mut mx, mut my, mut mz) = (0.0_f64, 0.0_f64, 0.0_f64);
    for n in neighbors {
        mx += f64::from(x[n.index]);
        my += f64::from(y[n.index]);
        mz += f64::from(z[n.index]);
    }
    mx /= count;
    my /= count;
    mz /= count;

    let (mut c00, mut c11, mut c22) = (0.0_f64, 0.0, 0.0);
    let (mut c01, mut c02, mut c12) = (0.0_f64, 0.0, 0.0);
    for n in neighbors {
        let dx = f64::from(x[n.index]) - mx;
        let dy = f64::from(y[n.index]) - my;
        let dz = f64::from(z[n.index]) - mz;
        c00 += dx * dx;
        c11 += dy * dy;
        c22 += dz * dz;
        c01 += dx * dy;
        c02 += dx * dz;
        c12 += dy * dz;
    }
    let inv = 1.0 / count;
    let covariance = Mat3::<f64>::from_rows(
        [c00 * inv, c01 * inv, c02 * inv],
        [c01 * inv, c11 * inv, c12 * inv],
        [c02 * inv, c12 * inv, c22 * inv],
    );
    Some(symmetric_eigen3(covariance).eigenvalues)
}

/// Gathers the selected indices into a new cloud, preserving schema.
fn gather_indices(input: &PointCloud, indices: &[usize]) -> SpatialResult<PointCloud> {
    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        buffers.insert(field.name.clone(), gather_buffer(source, indices));
    }
    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
}

fn gather_buffer(source: &PointBuffer, indices: &[usize]) -> PointBuffer {
    match source {
        PointBuffer::F32(v) => PointBuffer::from_f32(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::F64(v) => PointBuffer::F64(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U8(v) => PointBuffer::U8(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U16(v) => PointBuffer::U16(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U32(v) => PointBuffer::U32(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::I32(v) => PointBuffer::I32(indices.iter().map(|&i| v[i]).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::{IssKeypointConfig, IssKeypointDetector};
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    /// A flat floor plus a sharp spike — the spike tip should be a keypoint and
    /// the flat region should not.
    fn floor_with_spike() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..30 {
            for j in 0..30 {
                builder.push_point([i as f32 * 0.05, j as f32 * 0.05, 0.0]).unwrap();
            }
        }
        // A small dense cluster rising off the plane near its center.
        for k in 0..40 {
            let h = k as f32 * 0.02;
            builder.push_point([0.75, 0.75, h]).unwrap();
            builder.push_point([0.77, 0.75, h]).unwrap();
            builder.push_point([0.75, 0.77, h]).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn finds_keypoints_on_salient_structure() {
        let cloud = floor_with_spike();
        let detector = IssKeypointDetector::new(IssKeypointConfig {
            salient_radius: 0.12,
            non_max_radius: 0.08,
            gamma_21: 0.9,
            gamma_32: 0.9,
            min_neighbors: 5,
        });
        let result = detector.detect(&cloud).unwrap();
        // Some keypoints found, far fewer than the input size.
        assert!(!result.indices.is_empty());
        assert!(result.indices.len() < cloud.len() / 4);
        assert_eq!(result.keypoints.len(), result.indices.len());
    }

    #[test]
    fn flat_plane_interior_is_not_salient() {
        let n = 20;
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..n {
            for j in 0..n {
                builder.push_point([i as f32 * 0.05, j as f32 * 0.05, 0.0]).unwrap();
            }
        }
        let cloud = builder.build().unwrap();
        let detector = IssKeypointDetector::new(IssKeypointConfig::with_radii(0.12, 0.08));
        let result = detector.detect(&cloud).unwrap();
        // ISS legitimately flags the plane's boundary (a real edge), but the
        // isotropic interior (lambda1 ~ lambda2) must be rejected by gamma_21.
        let center = (n / 2) * n + (n / 2);
        assert!(!result.indices.contains(&center), "plane interior should not be salient");
    }

    #[test]
    fn rejects_bad_radii() {
        let cloud = floor_with_spike();
        let detector = IssKeypointDetector::new(IssKeypointConfig::with_radii(0.0, 0.1));
        assert!(detector.detect(&cloud).is_err());
    }
}
