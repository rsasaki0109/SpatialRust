use spatialrust_core::{
    DType, FieldSemantic, HasPositions3, PointBuffer, PointBufferSet, PointCloud, PointField,
    PointSchema, SpatialResult,
};
use spatialrust_math::{Mat3, Vec3, symmetric_eigen3};

use crate::estimator::FeatureEstimator;
use crate::neighborhood::{KdTreeNeighborhood, NeighborhoodProvider};

/// Configuration for covariance-based normal estimation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalEstimationConfig {
    /// Number of nearest neighbors to use when `search_radius` is `None`.
    pub k_neighbors: usize,
    /// Optional radius search instead of fixed `k`.
    pub search_radius: Option<f32>,
    /// Minimum number of neighbors required to estimate a valid normal.
    pub min_neighbors: usize,
    /// Optional viewpoint used to orient normals consistently.
    pub viewpoint: Option<Vec3<f32>>,
}

impl Default for NormalEstimationConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 20,
            search_radius: None,
            min_neighbors: 3,
            viewpoint: None,
        }
    }
}

impl NormalEstimationConfig {
    /// Creates a k-NN normal estimation config.
    #[must_use]
    pub const fn k_neighbors(k_neighbors: usize) -> Self {
        Self {
            k_neighbors,
            search_radius: None,
            min_neighbors: 3,
            viewpoint: None,
        }
    }
}

/// Result metadata for normal estimation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NormalEstimationResult {
    /// Number of points with valid normals.
    pub valid_count: usize,
    /// Number of points with invalid normals.
    pub invalid_count: usize,
}

/// Covariance-based normal estimator.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalEstimator {
    config: NormalEstimationConfig,
}

impl NormalEstimator {
    /// Creates a normal estimator from config.
    #[must_use]
    pub const fn new(config: NormalEstimationConfig) -> Self {
        Self { config }
    }

    /// Returns the estimator config.
    #[must_use]
    pub const fn config(&self) -> NormalEstimationConfig {
        self.config
    }

    /// Estimates normals and curvature, returning output cloud and diagnostics.
    pub fn estimate_with_diagnostics(
        &self,
        input: &PointCloud,
    ) -> SpatialResult<(PointCloud, NormalEstimationResult)> {
        if input.is_empty() {
            return Ok((input.clone(), NormalEstimationResult::default()));
        }

        let (x, y, z) = input.positions3()?;
        let neighborhood = KdTreeNeighborhood::from_point_cloud(input)?;

        let mut nx = vec![f32::NAN; input.len()];
        let mut ny = vec![f32::NAN; input.len()];
        let mut nz = vec![f32::NAN; input.len()];
        let mut curvature = vec![0.0_f32; input.len()];
        let mut valid_count = 0usize;
        let mut invalid_count = 0usize;

        for index in 0..input.len() {
            let neighbors = self.query_neighbors(&neighborhood, index)?;
            if neighbors.len() < self.config.min_neighbors {
                invalid_count += 1;
                continue;
            }

            let Some((normal, curv)) = estimate_normal_from_neighbors(x, y, z, index, &neighbors)
            else {
                invalid_count += 1;
                continue;
            };

            let oriented = if let Some(viewpoint) = self.config.viewpoint {
                orient_normal_towards_viewpoint(normal, point_xyz(x, y, z, index), viewpoint)
            } else {
                normal
            };

            nx[index] = oriented.x;
            ny[index] = oriented.y;
            nz[index] = oriented.z;
            curvature[index] = curv;
            valid_count += 1;
        }

        let output = build_output_cloud(input, nx, ny, nz, curvature)?;
        Ok((
            output,
            NormalEstimationResult {
                valid_count,
                invalid_count,
            },
        ))
    }

    fn query_neighbors(
        &self,
        neighborhood: &KdTreeNeighborhood,
        index: usize,
    ) -> SpatialResult<Vec<usize>> {
        if let Some(radius) = self.config.search_radius {
            neighborhood.query_radius(index, radius)
        } else {
            neighborhood.query_k(index, self.config.k_neighbors)
        }
    }
}

impl FeatureEstimator for NormalEstimator {
    fn name(&self) -> &'static str {
        "NormalEstimator"
    }

    fn estimate(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        self.estimate_with_diagnostics(input).map(|(cloud, _)| cloud)
    }
}

/// Orients a normal to point towards the viewpoint when possible.
#[must_use]
pub fn orient_normal_towards_viewpoint(
    mut normal: Vec3<f32>,
    point: Vec3<f32>,
    viewpoint: Vec3<f32>,
) -> Vec3<f32> {
    let view_direction = Vec3::new(viewpoint.x - point.x, viewpoint.y - point.y, viewpoint.z - point.z);
    if normal.dot(view_direction) < 0.0 {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }
    normal.normalize()
}

fn point_xyz(x: &[f32], y: &[f32], z: &[f32], index: usize) -> Vec3<f32> {
    Vec3::new(x[index], y[index], z[index])
}

fn estimate_normal_from_neighbors(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    _center_index: usize,
    neighbors: &[usize],
) -> Option<(Vec3<f32>, f32)> {
    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    let mut mean_z = 0.0_f64;
    for &index in neighbors {
        mean_x += f64::from(x[index]);
        mean_y += f64::from(y[index]);
        mean_z += f64::from(z[index]);
    }
    let count = neighbors.len() as f64;
    mean_x /= count;
    mean_y /= count;
    mean_z /= count;

    let mut c00 = 0.0_f64;
    let mut c11 = 0.0_f64;
    let mut c22 = 0.0_f64;
    let mut c01 = 0.0_f64;
    let mut c02 = 0.0_f64;
    let mut c12 = 0.0_f64;
    for &index in neighbors {
        let dx = f64::from(x[index]) - mean_x;
        let dy = f64::from(y[index]) - mean_y;
        let dz = f64::from(z[index]) - mean_z;
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

    let eigen = symmetric_eigen3(covariance);

    let normal = Vec3::new(
        eigen.eigenvectors.m[0][0] as f32,
        eigen.eigenvectors.m[1][0] as f32,
        eigen.eigenvectors.m[2][0] as f32,
    )
    .normalize();

    let sum = eigen.eigenvalues[0] + eigen.eigenvalues[1] + eigen.eigenvalues[2];
    let curvature = if sum > 0.0 {
        (eigen.eigenvalues[0] / sum) as f32
    } else {
        0.0
    };

    Some((normal, curvature))
}

fn build_output_cloud(
    input: &PointCloud,
    nx: Vec<f32>,
    ny: Vec<f32>,
    nz: Vec<f32>,
    curvature: Vec<f32>,
) -> SpatialResult<PointCloud> {
    let mut schema = input.schema().clone();
    ensure_field(
        &mut schema,
        "normal_x",
        FieldSemantic::NormalX,
        DType::F32,
    );
    ensure_field(
        &mut schema,
        "normal_y",
        FieldSemantic::NormalY,
        DType::F32,
    );
    ensure_field(
        &mut schema,
        "normal_z",
        FieldSemantic::NormalZ,
        DType::F32,
    );
    ensure_field(
        &mut schema,
        "curvature",
        FieldSemantic::Curvature,
        DType::F32,
    );

    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        buffers.insert(field.name.clone(), clone_buffer(source)?);
    }
    buffers.insert("normal_x".to_owned(), PointBuffer::from_f32(nx));
    buffers.insert("normal_y".to_owned(), PointBuffer::from_f32(ny));
    buffers.insert("normal_z".to_owned(), PointBuffer::from_f32(nz));
    buffers.insert("curvature".to_owned(), PointBuffer::from_f32(curvature));

    PointCloud::try_from_parts(schema, buffers, input.metadata().clone())
}

fn ensure_field(schema: &mut PointSchema, name: &str, semantic: FieldSemantic, dtype: DType) {
    if schema.find_semantic(semantic).is_none() {
        *schema = schema.clone().with_field(PointField::scalar(name, semantic, dtype));
    }
}

fn clone_buffer(buffer: &PointBuffer) -> SpatialResult<PointBuffer> {
    Ok(match buffer {
        PointBuffer::F32(values) => PointBuffer::from_f32(values.clone()),
        PointBuffer::F64(values) => PointBuffer::F64(values.clone()),
        PointBuffer::U8(values) => PointBuffer::U8(values.clone()),
        PointBuffer::U16(values) => PointBuffer::U16(values.clone()),
        PointBuffer::U32(values) => PointBuffer::U32(values.clone()),
        PointBuffer::I32(values) => PointBuffer::I32(values.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::{NormalEstimationConfig, NormalEstimator, orient_normal_towards_viewpoint};
    use crate::FeatureEstimator;
    use spatialrust_core::{HasNormals3, PointCloudBuilder, StandardSchemas};
    use spatialrust_math::Vec3;

    fn plane_cloud() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for x in 0..5 {
            for y in 0..5 {
                builder.push_point([x as f32, y as f32, 0.0]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn estimates_plane_normals_upwards() {
        let input = plane_cloud();
        let estimator = NormalEstimator::new(NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
            ..NormalEstimationConfig::default()
        });
        let (output, stats) = estimator.estimate_with_diagnostics(&input).unwrap();
        assert_eq!(stats.valid_count, input.len());
        assert_eq!(stats.invalid_count, 0);

        let (_, _, nz) = output.normals3().unwrap();
        for value in nz {
            assert!((*value - 1.0).abs() < 0.1, "expected upward normal, got {value}");
        }
    }

    #[test]
    fn orient_normal_towards_viewpoint_works() {
        let normal = Vec3::new(0.0, 0.0, -1.0);
        let point = Vec3::new(0.0, 0.0, 0.0);
        let viewpoint = Vec3::new(0.0, 0.0, 1.0);
        let oriented = orient_normal_towards_viewpoint(normal, point, viewpoint);
        assert!(oriented.z > 0.0);
    }

    #[test]
    fn adds_curvature_field() {
        let input = plane_cloud();
        let estimator = NormalEstimator::new(NormalEstimationConfig::k_neighbors(10));
        let output = estimator.estimate(&input).unwrap();
        assert!(output.field("curvature").is_ok());
    }
}
