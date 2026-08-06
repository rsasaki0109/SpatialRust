use std::path::Path;

use copc_streaming::{ByteSource, CopcStreamingReader, FileSource};
#[cfg(feature = "streaming")]
use copc_streaming::{DecompressedChunk, VoxelKey};
use las::Header;
use spatialrust_core::{PointCloud, PointSchema, SpatialMetadata};

use crate::copc::query::{CopcFileInfo, CopcQuery};
use crate::error::{copc_parse, IoError};
use crate::las::{metadata_from_las_header, point_cloud_from_las_points, schema_for_las_header};
use crate::{PointReader, ReadOptions};

#[cfg(feature = "streaming")]
use crate::streaming::{records_io, FormatStreamState};
#[cfg(feature = "streaming")]
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, MemoryReservation, MemoryTracker, RecordsResult,
    SchemaDescriptor, SpatialRecordChunk, StreamOptions,
};

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
    let reader =
        CopcStreamingReader::open(source).await.map_err(|error| copc_parse(error.to_string()))?;
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
    pollster::block_on(async { read_header_info(source).await.map(|(_, info)| info) })
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
    let mut reader =
        CopcStreamingReader::open(source).await.map_err(|error| copc_parse(error.to_string()))?;

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
    reader.load_all_hierarchy().await.map_err(|error| copc_parse(error.to_string()))?;

    let mut points = Vec::new();
    for (key, entry) in reader.entries() {
        if entry.point_count == 0 {
            continue;
        }
        let chunk = reader.fetch_chunk(key).await.map_err(|error| copc_parse(error.to_string()))?;
        let chunk_points =
            reader.read_points(&chunk).map_err(|error| copc_parse(error.to_string()))?;
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
        reader.query_points(&bounds).await.map_err(|error| copc_parse(error.to_string()))
    }
}

/// Bounded, deterministic COPC source over any random-access byte source.
#[cfg(feature = "streaming")]
pub struct CopcChunkSource<S: ByteSource> {
    reader: CopcStreamingReader<S>,
    keys: Vec<VoxelKey>,
    query_bounds: Option<copc_streaming::Aabb>,
    metadata: SpatialMetadata,
    state: FormatStreamState,
    key_index: usize,
    current: Option<DecompressedChunk>,
    current_reservation: Option<MemoryReservation>,
    current_offset: u32,
}

#[cfg(feature = "streaming")]
impl<S: ByteSource> CopcChunkSource<S> {
    /// Opens a byte source, loads matching hierarchy metadata, and orders nodes
    /// by `(level, x, y, z)` for repeatable chunk identities.
    pub fn from_source(
        source: S,
        query: Option<CopcQuery>,
        options: StreamOptions,
        cancellation: CancellationToken,
    ) -> Result<Self, IoError> {
        if let Some(query) = query {
            query.validate()?;
        }
        let mut reader = pollster::block_on(CopcStreamingReader::open(source))
            .map_err(|error| copc_parse(error.to_string()))?;
        let query_bounds = query.map(|query| query.bounds.to_aabb());
        pollster::block_on(async {
            match (query, query_bounds.as_ref()) {
                (Some(query), Some(bounds)) => {
                    if let Some(level) = query.max_level_for_spacing(reader.copc_info().spacing) {
                        reader.load_hierarchy_for_bounds_to_level(bounds, level).await
                    } else {
                        reader.load_hierarchy_for_bounds(bounds).await
                    }
                }
                _ => reader.load_all_hierarchy().await,
            }
        })
        .map_err(|error| copc_parse(error.to_string()))?;

        let root = reader.copc_info().root_bounds();
        let max_level =
            query.and_then(|query| query.max_level_for_spacing(reader.copc_info().spacing));
        let mut keys: Vec<_> = reader
            .entries()
            .filter(|(key, entry)| {
                entry.point_count > 0
                    && max_level.map_or(true, |level| key.level <= level)
                    && query_bounds
                        .as_ref()
                        .map_or(true, |bounds| key.bounds(&root).intersects(bounds))
            })
            .map(|(key, _)| *key)
            .collect();
        keys.sort_by_key(|key| (key.level, key.x, key.y, key.z));

        let schema = schema_for_las_header(reader.header().las_header());
        let metadata = metadata_from_las_header();
        let state = FormatStreamState::new("copc", schema, options, cancellation)?;
        Ok(Self {
            reader,
            keys,
            query_bounds,
            metadata,
            state,
            key_index: 0,
            current: None,
            current_reservation: None,
            current_offset: 0,
        })
    }

    /// Returns the declared LAS point count from the COPC header.
    #[must_use]
    pub fn declared_point_count(&self) -> u64 {
        self.reader.header().las_header().number_of_points()
    }

    fn load_next_node(&mut self) -> RecordsResult<bool> {
        let Some(key) = self.keys.get(self.key_index).copied() else {
            return Ok(false);
        };
        self.state.cancellation.check()?;
        let entry = self.reader.get(&key).ok_or_else(|| {
            spatialrust_records::RecordsError::InvalidChunk(format!(
                "COPC hierarchy entry disappeared for {key:?}"
            ))
        })?;
        let point_bytes = u64::from(entry.point_count)
            .checked_mul(u64::from(
                self.reader.header().las_header().point_format().len()
                    + self.reader.header().las_header().point_format().extra_bytes,
            ))
            .ok_or_else(|| {
                spatialrust_records::RecordsError::InvalidChunk(
                    "COPC node byte size overflow".into(),
                )
            })?;
        let working_bytes =
            point_bytes.checked_add(u64::from(entry.byte_size)).ok_or_else(|| {
                spatialrust_records::RecordsError::InvalidChunk(
                    "COPC node working set overflow".into(),
                )
            })?;
        let reservation = self.state.tracker.try_reserve(working_bytes)?;
        let chunk = pollster::block_on(self.reader.fetch_chunk(&key))
            .map_err(|error| records_io(copc_parse(error.to_string())))?;
        self.current = Some(chunk);
        self.current_reservation = Some(reservation);
        self.current_offset = 0;
        self.key_index += 1;
        Ok(true)
    }
}

#[cfg(feature = "streaming")]
impl CopcChunkSource<FileSource> {
    /// Opens a local COPC file for bounded node and record chunk reads.
    pub fn open(
        path: impl AsRef<Path>,
        query: Option<CopcQuery>,
        options: StreamOptions,
        cancellation: CancellationToken,
    ) -> Result<Self, IoError> {
        let source =
            FileSource::open(path.as_ref()).map_err(|error| copc_parse(error.to_string()))?;
        Self::from_source(source, query, options, cancellation)
    }
}

#[cfg(all(feature = "streaming", feature = "io-copc-http"))]
impl CopcChunkSource<crate::copc::HttpByteSource> {
    /// Opens a remote COPC URL using bounded HTTP range requests.
    pub fn open_url(
        url: &str,
        query: Option<CopcQuery>,
        options: StreamOptions,
        cancellation: CancellationToken,
    ) -> Result<Self, IoError> {
        Self::from_source(crate::copc::HttpByteSource::new(url)?, query, options, cancellation)
    }
}

#[cfg(feature = "streaming")]
impl<S: ByteSource> BoundedSpatialRecordSource for CopcChunkSource<S> {
    fn schema(&self) -> &SchemaDescriptor {
        &self.state.schema
    }

    fn options(&self) -> &StreamOptions {
        &self.state.options
    }

    fn memory_tracker(&self) -> &MemoryTracker {
        &self.state.tracker
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.state.cancellation.clone()
    }

    fn max_chunk_bytes(&self) -> u64 {
        self.state.max_chunk_bytes
    }

    fn next_chunk(&mut self) -> Option<RecordsResult<SpatialRecordChunk>> {
        loop {
            if self.current.is_none() {
                match self.load_next_node() {
                    Ok(true) => {}
                    Ok(false) => return None,
                    Err(error) => return Some(Err(error)),
                }
            }
            let chunk = self.current.as_ref().expect("loaded above");
            let remaining = chunk.point_count - self.current_offset;
            let count = remaining.min(self.state.options.chunk_points() as u32);
            let point_vec_bytes = match u64::try_from(std::mem::size_of::<las::Point>())
                .ok()
                .and_then(|size| size.checked_mul(u64::from(count)))
            {
                Some(bytes) => bytes,
                None => {
                    return Some(Err(spatialrust_records::RecordsError::InvalidChunk(
                        "COPC decoded point buffer size overflow".into(),
                    )));
                }
            };
            let reservation =
                match self.state.reserve_points_with_scratch(count as usize, point_vec_bytes) {
                    Ok(reservation) => reservation,
                    Err(error) => return Some(Err(error)),
                };
            let end = self.current_offset + count;
            let mut points = match self.reader.read_points_range(chunk, self.current_offset..end) {
                Ok(points) => points,
                Err(error) => return Some(Err(records_io(copc_parse(error.to_string())))),
            };
            if let Some(bounds) = &self.query_bounds {
                points.retain(|point| {
                    point.x >= bounds.min[0]
                        && point.x <= bounds.max[0]
                        && point.y >= bounds.min[1]
                        && point.y <= bounds.max[1]
                        && point.z >= bounds.min[2]
                        && point.z <= bounds.max[2]
                });
            }
            self.current_offset = end;
            if end == chunk.point_count {
                self.current = None;
                self.current_reservation = None;
            }
            if points.is_empty() {
                continue;
            }
            let cloud = match point_cloud_from_las_points(
                self.state.schema.point_schema().clone(),
                self.metadata.clone(),
                points,
            ) {
                Ok(cloud) => cloud,
                Err(error) => return Some(Err(records_io(error))),
            };
            return Some(self.state.lease(cloud, reservation));
        }
    }
}

/// One COPC octree node exposed by [`CopcNodeReader`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CopcNode {
    /// Octree level; 0 is the root.
    pub level: i32,
    /// Node X index at `level`.
    pub x: i32,
    /// Node Y index at `level`.
    pub y: i32,
    /// Node Z index at `level`.
    pub z: i32,
    /// World-space bounds of this node.
    pub bounds: crate::copc::CopcBounds,
    /// Points materialized by this node's chunk.
    pub point_count: u64,
}

/// Bounded per-node COPC reader.
///
/// Opens a COPC file once, loads the full hierarchy (metadata only), and
/// exposes one [`PointCloud`] per node on demand. Nodes are returned in
/// deterministic `(level, x, y, z)` order so the hierarchy can be turned
/// into a tile set without materializing the whole cloud.
pub struct CopcNodeReader {
    reader: CopcStreamingReader<FileSource>,
    keys: Vec<copc_streaming::VoxelKey>,
    root: copc_streaming::Aabb,
    nodes: Vec<CopcNode>,
    schema: PointSchema,
    metadata: SpatialMetadata,
}

impl CopcNodeReader {
    /// Opens a local COPC file and loads its hierarchy eagerly.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let source =
            FileSource::open(path.as_ref()).map_err(|error| copc_parse(error.to_string()))?;
        let mut reader = pollster::block_on(CopcStreamingReader::open(source))
            .map_err(|error| copc_parse(error.to_string()))?;
        pollster::block_on(reader.load_all_hierarchy())
            .map_err(|error| copc_parse(error.to_string()))?;
        let root = reader.copc_info().root_bounds();
        let schema = schema_for_las_header(reader.header().las_header());
        let metadata = metadata_from_las_header();

        let mut keys: Vec<_> = reader
            .entries()
            .filter(|(_, entry)| entry.point_count > 0)
            .map(|(key, _)| *key)
            .collect();
        keys.sort_by_key(|key| (key.level, key.x, key.y, key.z));

        let mut nodes = Vec::with_capacity(keys.len());
        for key in &keys {
            let bounds = key.bounds(&root);
            let point_count =
                reader.get(key).map(|entry| u64::from(entry.point_count)).unwrap_or(0);
            nodes.push(CopcNode {
                level: key.level,
                x: key.x,
                y: key.y,
                z: key.z,
                bounds: crate::copc::CopcBounds::new(bounds.min, bounds.max),
                point_count,
            });
        }
        Ok(Self { reader, keys, root, nodes, schema, metadata })
    }

    /// Node descriptors in deterministic `(level, x, y, z)` order.
    #[must_use]
    pub fn nodes(&self) -> &[CopcNode] {
        &self.nodes
    }

    /// Root octree bounds.
    #[must_use]
    pub fn root_bounds(&self) -> crate::copc::CopcBounds {
        crate::copc::CopcBounds::new(self.root.min, self.root.max)
    }

    /// Reads the points of one node by index into [`Self::nodes`].
    pub fn read_node(&mut self, index: usize) -> Result<PointCloud, IoError> {
        let key = *self
            .keys
            .get(index)
            .ok_or_else(|| copc_parse(format!("COPC node index {index} is out of range")))?;
        let chunk = pollster::block_on(self.reader.fetch_chunk(&key))
            .map_err(|error| copc_parse(error.to_string()))?;
        let points =
            self.reader.read_points(&chunk).map_err(|error| copc_parse(error.to_string()))?;
        point_cloud_from_las_points(self.schema.clone(), self.metadata.clone(), points)
    }
}

#[cfg(test)]
mod tests {
    use super::{read_copc_file, read_copc_file_info, read_copc_file_with_query, CopcQuery};
    use crate::copc::writer::write_copc_file;
    use crate::copc::{copc_level_for_resolution, CopcBounds};
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
        let path = std::env::temp_dir()
            .join(format!("spatialrust_copc_query_{}.copc.laz", std::process::id()));
        let query = CopcQuery::bounds(CopcBounds::from_ranges((1.0, 0.0), (0.0, 1.0), (0.0, 1.0)));
        let error = read_copc_file_with_query(&path, &query).unwrap_err();
        assert!(matches!(error, crate::IoError::CopcFormat(_)));
    }

    #[test]
    fn write_copc_rejects_empty_cloud() {
        use spatialrust_core::{
            PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas,
        };

        let schema = StandardSchemas::point_xyz();
        let mut buffers = PointBufferSet::new();
        for field in schema.fields() {
            buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, 0));
        }
        let cloud =
            PointCloud::try_from_parts(schema, buffers, SpatialMetadata::default()).unwrap();
        assert!(cloud.is_empty());

        let path =
            std::env::temp_dir().join(format!("spatialrust_copc_{}.copc.laz", std::process::id()));
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
        let path = std::env::temp_dir()
            .join(format!("spatialrust_copc_multires_{}.copc.laz", std::process::id()));
        write_copc_file_with_params(
            &path,
            &cloud,
            &CopcWriterParams { max_points_per_node: 96, max_depth: 8 },
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

        let level0 =
            read_copc_file_with_query(&path, &CopcQuery::with_level(info.root_bounds, 0)).unwrap();
        let level2 =
            read_copc_file_with_query(&path, &CopcQuery::with_level(info.root_bounds, 2)).unwrap();
        assert!(level0.len() <= level2.len());
        assert!(level2.len() <= full.len());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn copc_node_reader_enumerates_hierarchy_bounded() {
        use copc_writer::CopcWriterParams;

        use crate::copc::writer::write_copc_file_with_params;

        let cloud = dense_grid_cloud(7_000);
        let path = std::env::temp_dir()
            .join(format!("spatialrust_copc_nodes_{}.copc.laz", std::process::id()));
        write_copc_file_with_params(
            &path,
            &cloud,
            &CopcWriterParams { max_points_per_node: 96, max_depth: 8 },
        )
        .unwrap();

        let mut reader = super::CopcNodeReader::open(&path).unwrap();
        let nodes = reader.nodes().to_vec();
        assert!(nodes.len() > 1, "expected a multi-node hierarchy");
        assert!(nodes.iter().all(|node| node.point_count > 0));
        assert!(
            nodes.windows(2).all(|pair| (pair[0].level, pair[0].x, pair[0].y, pair[0].z)
                <= (pair[1].level, pair[1].x, pair[1].y, pair[1].z)),
            "nodes must be deterministically ordered"
        );

        let mut total = 0u64;
        for index in 0..nodes.len() {
            let cloud = reader.read_node(index).unwrap();
            assert_eq!(cloud.len() as u64, nodes[index].point_count);
            total += nodes[index].point_count;
        }
        assert_eq!(total, cloud.len() as u64, "per-node reads must reconstruct the cloud");
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
