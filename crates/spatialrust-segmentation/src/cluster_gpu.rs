use spatialrust_core::{HasPositions3, PointCloud, SpatialResult, TransferStats};
use spatialrust_gpu::{euclidean_cluster_roots_gpu_with_receipt, WgpuRuntime};

use crate::cluster::{finalize_euclidean_clusters, EuclideanClusterConfig, EuclideanClusterResult};
use crate::segmenter::PointCloudSegmenter;

/// GPU-grid-backed Euclidean cluster extractor.
///
/// Sparse grid construction and minimum-index component labeling run on wgpu.
/// Cluster-size filtering and output-cloud assembly remain on the host, matching
/// [`crate::EuclideanClusterExtractor`] semantics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuEuclideanClusterExtractor {
    config: EuclideanClusterConfig,
}

impl GpuEuclideanClusterExtractor {
    /// Creates a GPU extractor from config.
    #[must_use]
    pub const fn new(config: EuclideanClusterConfig) -> Self {
        Self { config }
    }

    /// Returns the extractor config.
    #[must_use]
    pub const fn config(&self) -> EuclideanClusterConfig {
        self.config
    }

    /// Clusters the input cloud using the grid union-find backend.
    pub fn extract(&self, input: &PointCloud) -> SpatialResult<EuclideanClusterResult> {
        self.extract_with_receipt(input).map(|(result, _)| result)
    }

    /// Clusters the input cloud and returns GPU grid transfer accounting.
    pub fn extract_with_receipt(
        &self,
        input: &PointCloud,
    ) -> SpatialResult<(EuclideanClusterResult, TransferStats)> {
        if input.is_empty() {
            return Ok((
                EuclideanClusterResult {
                    cloud: input.clone(),
                    cluster_count: 0,
                    cluster_sizes: Vec::new(),
                },
                TransferStats::default(),
            ));
        }

        let (x, y, z) = input.positions3()?;
        let runtime = WgpuRuntime::shared()?;
        let (roots, transfers) = euclidean_cluster_roots_gpu_with_receipt(
            &runtime,
            x,
            y,
            z,
            self.config.cluster_tolerance,
        )?;
        Ok((finalize_euclidean_clusters(input, &roots, self.config)?, transfers))
    }
}

impl PointCloudSegmenter for GpuEuclideanClusterExtractor {
    fn name(&self) -> &'static str {
        "GpuEuclideanClusterExtractor"
    }
}

#[cfg(test)]
mod tests {
    use super::GpuEuclideanClusterExtractor;
    use crate::cluster::EuclideanClusterConfig;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    fn three_clusters() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for center in [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)] {
            for dx in 0..3 {
                for dy in 0..3 {
                    builder
                        .push_point([center.0 + dx as f32, center.1 + dy as f32, center.2])
                        .unwrap();
                }
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn gpu_finds_three_separated_clusters() {
        let input = three_clusters();
        let extractor = GpuEuclideanClusterExtractor::new(EuclideanClusterConfig {
            cluster_tolerance: 1.5,
            min_cluster_size: 3,
            max_cluster_size: usize::MAX,
            gpu_min_points: None,
        });
        let result = extractor.extract(&input).unwrap();
        assert_eq!(result.cluster_count, 3);
        assert!(result.cluster_sizes.iter().all(|&size| size == 9));
    }

    fn long_chain(len: usize, spacing: f32) -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for index in 0..len {
            builder.push_point([index as f32 * spacing, 0.0, 0.0]).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn gpu_matches_cpu_on_long_chain() {
        use crate::cluster::EuclideanClusterExtractor;

        let input = long_chain(300, 1.0);
        let config = EuclideanClusterConfig {
            cluster_tolerance: 1.5,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
            gpu_min_points: None,
        };
        let cpu = EuclideanClusterExtractor::new(config).extract(&input).unwrap();
        let gpu = GpuEuclideanClusterExtractor::new(config).extract(&input).unwrap();
        assert_eq!(cpu.cluster_count, 1);
        assert_eq!(gpu.cluster_count, cpu.cluster_count);
        assert_eq!(gpu.cluster_sizes, cpu.cluster_sizes);
    }

    #[test]
    fn gpu_matches_cpu_on_three_clusters() {
        use crate::cluster::EuclideanClusterExtractor;

        let input = three_clusters();
        let config = EuclideanClusterConfig {
            cluster_tolerance: 1.5,
            min_cluster_size: 3,
            max_cluster_size: usize::MAX,
            gpu_min_points: None,
        };
        let cpu = EuclideanClusterExtractor::new(config).extract(&input).unwrap();
        let gpu = GpuEuclideanClusterExtractor::new(config).extract(&input).unwrap();
        assert_eq!(gpu.cluster_count, cpu.cluster_count);
        assert_eq!(gpu.cluster_sizes, cpu.cluster_sizes);
    }

    #[test]
    fn gpu_policy_preserves_all_columns_metadata_size_filtering_and_receipts() {
        use crate::cluster::EuclideanClusterExtractor;
        use spatialrust_core::{
            DeviceKind, ExecutionPolicy, FieldSemantic, PointBuffer, PointCloud, PointField,
            SpatialMetadata, Timestamp,
        };
        let (mut schema, mut buffers, _) = three_clusters().into_parts();
        for (name, buffer) in [
            ("a_f32", PointBuffer::F32((0..27).map(|i| i as f32 * 0.5).collect())),
            ("a_f64", PointBuffer::F64((0..27).map(|i| i as f64 * 0.25).collect())),
            ("a_u8", PointBuffer::U8((0..27).map(|i| i as u8).collect())),
            ("a_u16", PointBuffer::U16((0..27).map(|i| i as u16 * 37).collect())),
            ("a_u32", PointBuffer::U32((0..27).map(|i| i * 104729).collect())),
            ("a_i32", PointBuffer::I32((0..27).map(|i| i - 13).collect())),
        ] {
            schema =
                schema.with_field(PointField::scalar(name, FieldSemantic::Unknown, buffer.dtype()));
            buffers.insert(name.to_owned(), buffer);
        }
        let metadata = SpatialMetadata::new("synthetic-clusters", Timestamp::from_nanos(42));
        let input = PointCloud::try_from_parts(schema, buffers, metadata).unwrap();
        for (min_cluster_size, max_cluster_size) in [(1, 9), (10, usize::MAX), (1, 8)] {
            let extractor = EuclideanClusterExtractor::new(EuclideanClusterConfig {
                cluster_tolerance: 1.5,
                min_cluster_size,
                max_cluster_size,
                gpu_min_points: None,
            });
            let expected = extractor.extract(&input).unwrap();
            let actual = extractor
                .extract_with_policy_and_receipt(&input, ExecutionPolicy::Gpu(DeviceKind::Wgpu))
                .unwrap();
            assert_eq!(actual.output().cluster_sizes, expected.cluster_sizes);
            assert_eq!(actual.output().cluster_count, expected.cluster_count);
            assert_eq!(actual.output().cloud.schema(), expected.cloud.schema());
            assert_eq!(actual.output().cloud.metadata(), expected.cloud.metadata());
            for field in expected.cloud.schema().fields() {
                assert_eq!(
                    actual.output().cloud.field(&field.name).unwrap(),
                    expected.cloud.field(&field.name).unwrap()
                );
            }
            assert_eq!(actual.receipt().host_to_device_bytes(), 27 * 12);
            assert_eq!(actual.receipt().device_to_host_bytes(), 27 * 4 + 4);
            assert_eq!(
                actual.receipt().stages(),
                &["euclidean-clustering", "gpu-sparse-grid", "gpu-component-labeling"]
            );
        }
    }

    #[test]
    fn infinite_and_empty_gpu_policy_receipts_name_only_executed_device_stages() {
        use crate::cluster::EuclideanClusterExtractor;
        use spatialrust_core::{DeviceKind, ExecutionPolicy, PointBuffer, PointCloud};
        let extractor = EuclideanClusterExtractor::new(EuclideanClusterConfig {
            cluster_tolerance: f32::INFINITY,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
            gpu_min_points: None,
        });
        let actual = extractor
            .extract_with_policy_and_receipt(
                &three_clusters(),
                ExecutionPolicy::Gpu(DeviceKind::Wgpu),
            )
            .unwrap();
        assert_eq!(actual.output().cloud.field("label").unwrap(), &PointBuffer::I32(vec![0; 27]));
        assert_eq!(actual.receipt().host_to_device_bytes(), 0);
        assert_eq!(actual.receipt().device_to_host_bytes(), 27 * 4);
        assert_eq!(actual.receipt().stages(), &["euclidean-clustering", "gpu-component-labeling"]);
        let (schema, mut buffers, metadata) = three_clusters().into_parts();
        for field in schema.fields() {
            buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, 0));
        }
        let empty = PointCloud::try_from_parts(schema, buffers, metadata).unwrap();
        let actual = extractor
            .extract_with_policy_and_receipt(&empty, ExecutionPolicy::Gpu(DeviceKind::Wgpu))
            .unwrap();
        assert_eq!(actual.receipt().host_to_device_bytes(), 0);
        assert_eq!(actual.receipt().device_to_host_bytes(), 0);
        assert_eq!(actual.receipt().stages(), &["euclidean-clustering"]);
    }
}
