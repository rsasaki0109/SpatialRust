use std::path::Path;

use copc_streaming::{ByteSource, CopcStreamingReader, FileSource};
use las::Header;
use spatialrust_core::{PointCloud, PointSchema, SpatialMetadata};

use crate::copc::query::{CopcFileInfo, CopcQuery};
use crate::error::{copc_parse, IoError};
use crate::las::{metadata_from_las_header, point_cloud_from_las_points, schema_for_las_header};
use crate::{PointReader, ReadOptions};

/// Reads point clouds from COPC files.
pub struct CopcReader {
    path: std::path::PathBuf,
    metadata: SpatialMetadata,
    schema: PointSchema,
    file_info: CopcFileInfo,
}

impl CopcReader {
    /// Opens a COPC file and parses its header eagerly.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref().to_path_buf();
        let source = FileSource::open(&path).map_err(|error| copc_parse(error.to_string()))?;
        let (header, file_info) = pollster::block_on(read_header_info(source))?;
        Ok(Self {
            schema: schema_for_las_header(&header),
            metadata: metadata_from_las_header(),
            file_info,
            path,
        })
    }

    /// Returns COPC header metadata parsed at open time.
    #[must_use]
    pub fn file_info(&self) -> &CopcFileInfo {
        &self.file_info
    }

    /// Returns the root octree bounds for this file.
    #[must_use]
    pub fn root_bounds(&self) -> crate::copc::CopcBounds {
        self.file_info.root_bounds
    }

    /// Reads points matching a spatial query.
    pub fn read_query(&mut self, query: &CopcQuery) -> Result<PointCloud, IoError> {
        read_copc_file_with_query(&self.path, query)
    }
}

impl PointReader for CopcReader {
    fn schema(&self) -> spatialrust_core::SpatialResult<PointSchema> {
        Ok(self.schema.clone())
    }

    fn metadata(&self) -> spatialrust_core::SpatialResult<SpatialMetadata> {
        Ok(self.metadata.clone())
    }

    fn read(&mut self, _options: &ReadOptions) -> spatialrust_core::SpatialResult<PointCloud> {
        read_copc_file(&self.path)
            .map_err(|error| spatialrust_core::SpatialError::Io(error.to_string()))
    }
}

pub(crate) async fn read_header_info<S: ByteSource>(
    source: S,
) -> Result<(Header, CopcFileInfo), IoError> {
    let reader = CopcStreamingReader::open(source)
        .await
        .map_err(|error| copc_parse(error.to_string()))?;
    let las_header = reader.header().las_header().clone();
    let copc_info = reader.copc_info();
    let root = copc_info.root_bounds();
    let file_info = CopcFileInfo {
        root_bounds: crate::copc::CopcBounds::new(root.min, root.max),
        spacing: copc_info.spacing,
        point_count: las_header.number_of_points(),
    };
    Ok((las_header, file_info))
}

/// Reads COPC header metadata without loading points.
pub fn read_copc_file_info(path: impl AsRef<Path>) -> Result<CopcFileInfo, IoError> {
    let source = FileSource::open(path.as_ref()).map_err(|error| copc_parse(error.to_string()))?;
    pollster::block_on(async {
        read_header_info(source)
            .await
            .map(|(_, info)| info)
    })
}

/// Reads all points from a COPC file on disk.
pub fn read_copc(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    read_copc_file(path)
}

/// Reads all points from a COPC file on disk.
pub fn read_copc_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    let source = FileSource::open(path.as_ref()).map_err(|error| copc_parse(error.to_string()))?;
    pollster::block_on(read_copc_from_byte_source(source, None))
}

/// Reads points inside a bounding box at full available detail.
pub fn read_copc_file_in_bounds(
    path: impl AsRef<Path>,
    bounds: crate::copc::CopcBounds,
) -> Result<PointCloud, IoError> {
    read_copc_file_with_query(path, &CopcQuery::bounds(bounds))
}

/// Reads points using a spatial bounds and optional LOD limit.
pub fn read_copc_file_with_query(
    path: impl AsRef<Path>,
    query: &CopcQuery,
) -> Result<PointCloud, IoError> {
    query.validate()?;
    let source = FileSource::open(path.as_ref()).map_err(|error| copc_parse(error.to_string()))?;
    pollster::block_on(read_copc_from_byte_source(source, Some(query)))
}

pub(crate) async fn read_copc_from_byte_source<S: ByteSource>(
    source: S,
    query: Option<&CopcQuery>,
) -> Result<PointCloud, IoError> {
    let mut reader = CopcStreamingReader::open(source)
        .await
        .map_err(|error| copc_parse(error.to_string()))?;

    let las_header = reader.header().las_header().clone();
    let schema = schema_for_las_header(&las_header);
    let metadata = metadata_from_las_header();

    let points = match query {
        None => read_all_points(&mut reader).await?,
        Some(query) => read_query_points(&mut reader, query).await?,
    };

    point_cloud_from_las_points(schema, metadata, points)
}

async fn read_all_points<S: ByteSource>(
    reader: &mut CopcStreamingReader<S>,
) -> Result<Vec<las::Point>, IoError> {
    reader
        .load_all_hierarchy()
        .await
        .map_err(|error| copc_parse(error.to_string()))?;

    let mut points = Vec::new();
    for (key, entry) in reader.entries() {
        if entry.point_count == 0 {
            continue;
        }
        let chunk = reader
            .fetch_chunk(key)
            .await
            .map_err(|error| copc_parse(error.to_string()))?;
        let chunk_points = reader
            .read_points(&chunk)
            .map_err(|error| copc_parse(error.to_string()))?;
        points.extend(chunk_points);
    }
    Ok(points)
}

async fn read_query_points<S: ByteSource>(
    reader: &mut CopcStreamingReader<S>,
    query: &CopcQuery,
) -> Result<Vec<las::Point>, IoError> {
    let bounds = query.bounds.to_aabb();
    if let Some(max_level) = query.max_level_for_spacing(reader.copc_info().spacing) {
        reader
            .query_points_to_level(&bounds, max_level)
            .await
            .map_err(|error| copc_parse(error.to_string()))
    } else {
        reader
            .query_points(&bounds)
            .await
            .map_err(|error| copc_parse(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{read_copc_file, read_copc_file_info, read_copc_file_with_query, CopcQuery};
    use crate::copc::{copc_level_for_resolution, CopcBounds};
    use crate::copc::writer::write_copc_file;
    use crate::{write_las_file, LasWriteFormat};
    use spatialrust_core::PointCloudBuilder;

    #[test]
    fn rejects_non_copc_laz() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        let cloud = builder.build().unwrap();

        let path = std::env::temp_dir().join(format!("spatialrust_laz_{}.laz", std::process::id()));
        write_las_file(&path, &cloud, LasWriteFormat::Laz).unwrap();

        let error = read_copc_file(&path).unwrap_err();
        let _ = std::fs::remove_file(path);
        assert!(matches!(error, crate::IoError::CopcParse(_)));
    }

    #[test]
    fn rejects_invalid_query_bounds() {
        let path = std::env::temp_dir().join(format!("spatialrust_copc_query_{}.copc.laz", std::process::id()));
        let query = CopcQuery::bounds(CopcBounds::from_ranges((1.0, 0.0), (0.0, 1.0), (0.0, 1.0)));
        let error = read_copc_file_with_query(&path, &query).unwrap_err();
        assert!(matches!(error, crate::IoError::CopcFormat(_)));
    }

    #[test]
    fn write_copc_rejects_empty_cloud() {
        use spatialrust_core::{PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas};

        let schema = StandardSchemas::point_xyz();
        let mut buffers = PointBufferSet::new();
        for field in schema.fields() {
            buffers.insert(
                field.name.clone(),
                PointBuffer::with_capacity(field.dtype, 0),
            );
        }
        let cloud =
            PointCloud::try_from_parts(schema, buffers, SpatialMetadata::default()).unwrap();
        assert!(cloud.is_empty());

        let path = std::env::temp_dir().join(format!("spatialrust_copc_{}.copc.laz", std::process::id()));
        let error = write_copc_file(&path, &cloud).unwrap_err();
        assert!(matches!(error, crate::IoError::CopcFormat(_)));
    }

    #[test]
    fn resolution_level_helper_is_usable_from_reader_tests() {
        assert_eq!(copc_level_for_resolution(4.0, 1.0), 2);
    }

    #[test]
    fn multi_resolution_copc_resolution_query_reduces_point_count() {
        use copc_writer::CopcWriterParams;

        use crate::copc::writer::write_copc_file_with_params;

        let cloud = dense_grid_cloud(7_000);
        let path = std::env::temp_dir().join(format!(
            "spatialrust_copc_multires_{}.copc.laz",
            std::process::id()
        ));
        write_copc_file_with_params(
            &path,
            &cloud,
            &CopcWriterParams {
                max_points_per_node: 96,
                max_depth: 8,
            },
        )
        .unwrap();

        let info = read_copc_file_info(&path).unwrap();
        let full = read_copc_file(&path).unwrap();
        assert_eq!(full.len(), cloud.len());

        let coarse = read_copc_file_with_query(
            &path,
            &CopcQuery::with_resolution(info.root_bounds, info.spacing * 4.0),
        )
        .unwrap();
        let medium = read_copc_file_with_query(
            &path,
            &CopcQuery::with_resolution(info.root_bounds, info.spacing),
        )
        .unwrap();
        let fine = read_copc_file_with_query(
            &path,
            &CopcQuery::with_resolution(info.root_bounds, info.spacing / 4.0),
        )
        .unwrap();

        assert!(coarse.len() <= medium.len());
        assert!(medium.len() <= fine.len());
        assert!(fine.len() <= full.len());
        assert!(
            coarse.len() < full.len(),
            "coarse resolution should load fewer points than full detail"
        );

        let level0 = read_copc_file_with_query(&path, &CopcQuery::with_level(info.root_bounds, 0))
            .unwrap();
        let level2 = read_copc_file_with_query(&path, &CopcQuery::with_level(info.root_bounds, 2))
            .unwrap();
        assert!(level0.len() <= level2.len());
        assert!(level2.len() <= full.len());

        let _ = std::fs::remove_file(path);
    }

    fn dense_grid_cloud(count: usize) -> spatialrust_core::PointCloud {
        use spatialrust_core::PointCloudBuilder;

        let mut builder = PointCloudBuilder::xyz();
        for index in 0..count {
            let x = (index % 31) as f32 - 15.0;
            let y = ((index / 31) % 29) as f32 - 14.0;
            let z = ((index / (31 * 29)) % 23) as f32 - 11.0;
            builder.push_point([x, y, z]).unwrap();
        }
        builder.build().unwrap()
    }
}
