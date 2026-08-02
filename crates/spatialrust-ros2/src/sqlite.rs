//! Read-only rosbag2 SQLite storage for bounded PointCloud2 streams.

use std::path::Path;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use spatialrust_core::{
    PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas, Timestamp,
};
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, ChunkIdentity, MemoryReservation, MemoryTracker,
    RecordsError, RecordsResult, SchemaDescriptor, SchemaVersion, SpatialRecord,
    SpatialRecordChunk, StreamOptions,
};
use spatialrust_runtime::{
    decode_point_cloud2_xyz, PointCloud2Xyz, RuntimeError, POINT_CLOUD2_TYPE,
};
use thiserror::Error;

/// Stable schema family for XYZ columns decoded from ROS 2 PointCloud2.
pub const ROSBAG2_POINT_XYZ_SCHEMA_ID: &str = "ros2.sensor_msgs.msg.PointCloud2.xyz";

const CDR_SCRATCH_OVERHEAD_BYTES: u64 = 64 * 1024;

/// Result type for rosbag2 SQLite operations.
pub type Rosbag2Result<T> = Result<T, Rosbag2Error>;

/// Failures while inspecting or streaming a rosbag2 SQLite bag.
#[derive(Debug, Error)]
pub enum Rosbag2Error {
    /// SQLite storage failure.
    #[error("rosbag2 SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// The file is not a supported rosbag2 SQLite database.
    #[error("invalid rosbag2 SQLite bag: {0}")]
    InvalidBag(String),
    /// The selected topic or serialization is outside this adapter's scope.
    #[error("unsupported rosbag2 message: {0}")]
    Unsupported(String),
    /// PointCloud2 CDR decoding failure.
    #[error("ROS 2 CDR error: {0}")]
    Cdr(#[from] RuntimeError),
    /// Bounded-record contract failure.
    #[error(transparent)]
    Records(#[from] RecordsError),
    /// Core point-cloud construction failure.
    #[error(transparent)]
    Core(#[from] spatialrust_core::SpatialError),
}

/// Metadata for one rosbag2 topic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rosbag2Topic {
    /// SQLite topic id.
    pub id: i64,
    /// ROS topic name.
    pub name: String,
    /// Fully-qualified ROS message type.
    pub type_name: String,
    /// Storage serialization identifier, normally `cdr`.
    pub serialization_format: String,
    /// Number of stored messages for this topic.
    pub message_count: u64,
}

/// Lists topics from a rosbag2 SQLite file without modifying it.
pub fn list_topics(path: impl AsRef<Path>) -> Rosbag2Result<Vec<Rosbag2Topic>> {
    let connection = open_connection(path)?;
    read_topics(&connection)
}

/// Bounded, deterministic PointCloud2 XYZ source backed by rosbag2 SQLite.
///
/// The SQLite connection is opened read-only. One selected topic is traversed
/// in `(timestamp, id)` order. A PointCloud2 message may be split into several
/// leased chunks when it exceeds `StreamOptions::chunk_points()`.
pub struct Rosbag2PointCloudSource {
    connection: Connection,
    topic: Rosbag2Topic,
    schema: SchemaDescriptor,
    options: StreamOptions,
    tracker: MemoryTracker,
    cancellation: CancellationToken,
    max_chunk_bytes: u64,
    max_message_bytes: u64,
    cursor: Option<MessageCursor>,
    pending: Option<PendingPointCloud>,
    next_sequence: u64,
    next_point_offset: u64,
    finished: bool,
}

impl Rosbag2PointCloudSource {
    /// Opens one PointCloud2 topic from a rosbag2 SQLite file.
    pub fn open(
        path: impl AsRef<Path>,
        topic_name: &str,
        options: StreamOptions,
        cancellation: CancellationToken,
    ) -> Rosbag2Result<Self> {
        let connection = open_connection(path)?;
        let topics = read_topics(&connection)?;
        let topic = topics.into_iter().find(|topic| topic.name == topic_name).ok_or_else(|| {
            Rosbag2Error::InvalidBag(format!("topic `{topic_name}` was not found"))
        })?;

        if topic.type_name != POINT_CLOUD2_TYPE {
            return Err(Rosbag2Error::Unsupported(format!(
                "topic `{}` has type `{}`, expected `{POINT_CLOUD2_TYPE}`",
                topic.name, topic.type_name
            )));
        }
        if !topic.serialization_format.eq_ignore_ascii_case("cdr") {
            return Err(Rosbag2Error::Unsupported(format!(
                "topic `{}` uses serialization `{}`, only CDR is supported",
                topic.name, topic.serialization_format
            )));
        }

        let max_message_bytes = max_message_bytes(&connection, topic.id)?;
        let schema = SchemaDescriptor::try_new(
            ROSBAG2_POINT_XYZ_SCHEMA_ID,
            SchemaVersion::new(1, 0),
            StandardSchemas::point_xyz(),
        )?;
        let max_record_bytes = xyz_capacity_bytes(options.chunk_points())?;
        let max_message_working_bytes = message_working_bytes(max_message_bytes)?;
        let max_chunk_bytes = max_record_bytes
            .checked_add(max_message_working_bytes)
            .ok_or_else(|| Rosbag2Error::InvalidBag("maximum source memory overflow".into()))?;
        if max_chunk_bytes > options.memory_budget().limit_bytes() {
            return Err(Rosbag2Error::Records(RecordsError::MemoryBudgetExceeded {
                requested: max_chunk_bytes,
                current: 0,
                limit: options.memory_budget().limit_bytes(),
            }));
        }
        let budget = options.memory_budget();

        Ok(Self {
            connection,
            topic,
            schema,
            options,
            tracker: MemoryTracker::new(budget),
            cancellation,
            max_chunk_bytes,
            max_message_bytes,
            cursor: None,
            pending: None,
            next_sequence: 0,
            next_point_offset: 0,
            finished: false,
        })
    }

    /// Returns the selected topic metadata.
    #[must_use]
    pub fn topic(&self) -> &Rosbag2Topic {
        &self.topic
    }

    /// Returns the largest raw CDR payload in the selected topic.
    #[must_use]
    pub const fn max_message_bytes(&self) -> u64 {
        self.max_message_bytes
    }

    fn next_message(&mut self) -> Rosbag2Result<Option<(MessageCursor, Vec<u8>)>> {
        let row = match self.cursor {
            None => self.connection.query_row(
                "SELECT id, timestamp, data \
                 FROM messages WHERE topic_id = ?1 \
                 ORDER BY timestamp ASC, id ASC LIMIT 1",
                params![self.topic.id],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, Vec<u8>>(2)?)),
            ),
            Some(cursor) => self.connection.query_row(
                "SELECT id, timestamp, data \
                 FROM messages WHERE topic_id = ?1 \
                   AND (timestamp > ?2 OR (timestamp = ?2 AND id > ?3)) \
                 ORDER BY timestamp ASC, id ASC LIMIT 1",
                params![self.topic.id, cursor.timestamp, cursor.id],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, Vec<u8>>(2)?)),
            ),
        }
        .optional()?;

        row.map(|(id, timestamp, data)| {
            if timestamp < 0 {
                return Err(Rosbag2Error::InvalidBag(format!(
                    "message {id} has a negative timestamp {timestamp}"
                )));
            }
            Ok((MessageCursor { id, timestamp }, data))
        })
        .transpose()
    }

    fn load_pending(&mut self) -> Rosbag2Result<bool> {
        loop {
            let Some((cursor, data)) = self.next_message()? else {
                return Ok(false);
            };
            self.cursor = Some(cursor);

            let working_bytes = message_working_bytes(
                u64::try_from(data.len())
                    .map_err(|_| Rosbag2Error::InvalidBag("CDR payload is too large".into()))?,
            )?;
            let mut reservation = self.tracker.try_reserve(working_bytes)?;
            let message = decode_point_cloud2_xyz(&data)?;
            drop(data);
            let decoded_bytes = xyz_capacity_bytes(message.point_count())?;
            reservation.shrink_to(
                decoded_bytes
                    .checked_add(CDR_SCRATCH_OVERHEAD_BYTES)
                    .ok_or_else(|| Rosbag2Error::InvalidBag("decoded memory overflow".into()))?,
            )?;
            if message.point_count() == 0 {
                continue;
            }

            let timestamp = ros_timestamp_ns(&message)?;
            self.pending = Some(PendingPointCloud {
                message,
                offset: 0,
                timestamp,
                _reservation: reservation,
            });
            return Ok(true);
        }
    }

    fn build_chunk(
        &self,
        pending: &PendingPointCloud,
        count: usize,
        reservation: MemoryReservation,
    ) -> Rosbag2Result<SpatialRecordChunk> {
        let start = pending
            .offset
            .checked_mul(3)
            .ok_or_else(|| Rosbag2Error::InvalidBag("point offset overflow".into()))?;
        let end = start
            .checked_add(
                count
                    .checked_mul(3)
                    .ok_or_else(|| Rosbag2Error::InvalidBag("chunk point count overflow".into()))?,
            )
            .ok_or_else(|| Rosbag2Error::InvalidBag("chunk range overflow".into()))?;
        let values =
            pending.message.xyz.get(start..end).ok_or_else(|| {
                Rosbag2Error::InvalidBag("decoded XYZ range is out of bounds".into())
            })?;

        let mut x = Vec::with_capacity(count);
        let mut y = Vec::with_capacity(count);
        let mut z = Vec::with_capacity(count);
        for point in values.chunks_exact(3) {
            x.push(point[0]);
            y.push(point[1]);
            z.push(point[2]);
        }
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(x));
        buffers.insert("y", PointBuffer::from_f32(y));
        buffers.insert("z", PointBuffer::from_f32(z));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::new(
                pending.message.frame_id.as_str(),
                Timestamp::from_nanos(pending.timestamp),
            ),
        )?;
        let record = SpatialRecord::try_new(self.schema.clone(), cloud)?;
        Ok(SpatialRecordChunk::try_from_reserved(
            ChunkIdentity { sequence: self.next_sequence, point_offset: self.next_point_offset },
            record,
            reservation,
        )?)
    }

    fn fail(&mut self, error: Rosbag2Error) -> Option<RecordsResult<SpatialRecordChunk>> {
        self.pending.take();
        self.finished = true;
        Some(Err(match error {
            Rosbag2Error::Records(error) => error,
            other => RecordsError::InvalidChunk(other.to_string()),
        }))
    }
}

impl BoundedSpatialRecordSource for Rosbag2PointCloudSource {
    fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    fn options(&self) -> &StreamOptions {
        &self.options
    }

    fn memory_tracker(&self) -> &MemoryTracker {
        &self.tracker
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    fn max_chunk_bytes(&self) -> u64 {
        self.max_chunk_bytes
    }

    fn next_chunk(&mut self) -> Option<RecordsResult<SpatialRecordChunk>> {
        if self.finished {
            return None;
        }
        if let Err(error) = self.cancellation.check() {
            return self.fail(Rosbag2Error::Records(error));
        }

        if self.pending.is_none() {
            match self.load_pending() {
                Ok(true) => {}
                Ok(false) => {
                    self.finished = true;
                    return None;
                }
                Err(error) => return self.fail(error),
            }
        }

        let mut pending = self.pending.take().expect("pending message was loaded");
        let remaining = pending.message.point_count().saturating_sub(pending.offset);
        let count = remaining.min(self.options.chunk_points());
        if count == 0 {
            drop(pending);
            return self.next_chunk();
        }

        let reservation = match self.tracker.try_reserve(match xyz_capacity_bytes(count) {
            Ok(bytes) => bytes,
            Err(error) => return self.fail(error),
        }) {
            Ok(reservation) => reservation,
            Err(error) => {
                self.pending = Some(pending);
                return self.fail(Rosbag2Error::Records(error));
            }
        };
        let next_offset = pending.offset + count;
        let chunk = match self.build_chunk(&pending, count, reservation) {
            Ok(chunk) => chunk,
            Err(error) => {
                self.pending = Some(pending);
                return self.fail(error);
            }
        };
        pending.offset = next_offset;
        if next_offset < pending.message.point_count() {
            self.pending = Some(pending);
        }
        self.next_sequence = match self.next_sequence.checked_add(1) {
            Some(value) => value,
            None => {
                return self.fail(Rosbag2Error::Records(RecordsError::ReceiptOverflow(
                    "rosbag2 chunk sequence".into(),
                )))
            }
        };
        let count = match u64::try_from(count) {
            Ok(count) => count,
            Err(_) => {
                return self.fail(Rosbag2Error::Records(RecordsError::ReceiptOverflow(
                    "rosbag2 point count".into(),
                )))
            }
        };
        self.next_point_offset = match self.next_point_offset.checked_add(count) {
            Some(value) => value,
            None => {
                return self.fail(Rosbag2Error::Records(RecordsError::ReceiptOverflow(
                    "rosbag2 point offset".into(),
                )))
            }
        };
        Some(Ok(chunk))
    }
}

#[derive(Clone, Copy, Debug)]
struct MessageCursor {
    id: i64,
    timestamp: i64,
}

struct PendingPointCloud {
    message: PointCloud2Xyz,
    offset: usize,
    timestamp: u64,
    _reservation: MemoryReservation,
}

fn open_connection(path: impl AsRef<Path>) -> Rosbag2Result<Connection> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    ensure_schema(&connection)?;
    Ok(connection)
}

fn ensure_schema(connection: &Connection) -> Rosbag2Result<()> {
    for table in ["topics", "messages"] {
        let present: i64 = connection.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            params![table],
            |row| row.get(0),
        )?;
        if present != 1 {
            return Err(Rosbag2Error::InvalidBag(format!(
                "required rosbag2 table `{table}` is missing"
            )));
        }
    }
    Ok(())
}

fn read_topics(connection: &Connection) -> Rosbag2Result<Vec<Rosbag2Topic>> {
    let mut statement = connection.prepare(
        "SELECT t.id, t.name, t.\"type\", t.serialization_format, COUNT(m.id) \
         FROM topics AS t LEFT JOIN messages AS m ON m.topic_id = t.id \
         GROUP BY t.id, t.name, t.\"type\", t.serialization_format \
         ORDER BY t.id",
    )?;
    let rows = statement.query_map([], |row| {
        let message_count: i64 = row.get(4)?;
        Ok(Rosbag2Topic {
            id: row.get(0)?,
            name: row.get(1)?,
            type_name: row.get(2)?,
            serialization_format: row.get(3)?,
            message_count: u64::try_from(message_count).unwrap_or(0),
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Rosbag2Error::from)
}

fn max_message_bytes(connection: &Connection, topic_id: i64) -> Rosbag2Result<u64> {
    let value: Option<i64> = connection.query_row(
        "SELECT MAX(length(data)) FROM messages WHERE topic_id = ?1",
        params![topic_id],
        |row| row.get(0),
    )?;
    value
        .unwrap_or(0)
        .try_into()
        .map_err(|_| Rosbag2Error::InvalidBag("negative message length".into()))
}

fn message_working_bytes(raw_bytes: u64) -> Rosbag2Result<u64> {
    raw_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(CDR_SCRATCH_OVERHEAD_BYTES))
        .ok_or_else(|| Rosbag2Error::InvalidBag("message working-set overflow".into()))
}

fn xyz_capacity_bytes(point_count: usize) -> Rosbag2Result<u64> {
    u64::try_from(point_count)
        .ok()
        .and_then(|count| count.checked_mul(12))
        .ok_or_else(|| Rosbag2Error::InvalidBag("XYZ capacity overflow".into()))
}

fn ros_timestamp_ns(message: &PointCloud2Xyz) -> Rosbag2Result<u64> {
    if message.stamp_sec < 0 || message.stamp_nanosec >= 1_000_000_000 {
        return Err(Rosbag2Error::InvalidBag(format!(
            "invalid PointCloud2 header timestamp {}.{:09}",
            message.stamp_sec, message.stamp_nanosec
        )));
    }
    u64::try_from(message.stamp_sec)
        .ok()
        .and_then(|seconds| seconds.checked_mul(1_000_000_000))
        .and_then(|seconds| seconds.checked_add(u64::from(message.stamp_nanosec)))
        .ok_or_else(|| Rosbag2Error::InvalidBag("PointCloud2 timestamp overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::{list_topics, Rosbag2PointCloudSource, ROSBAG2_POINT_XYZ_SCHEMA_ID};
    use rusqlite::{params, Connection};
    use spatialrust_core::PointBuffer;
    use spatialrust_records::{
        BoundedSpatialRecordSource, CancellationToken, MemoryBudget, StreamOptions,
    };
    use spatialrust_runtime::{encode_point_cloud2_xyz, PointCloud2Xyz, POINT_CLOUD2_TYPE};

    fn bag_file() -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sample.db3");
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, serialization_format TEXT NOT NULL);\
                 CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, timestamp INTEGER NOT NULL, data BLOB NOT NULL);",
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO topics(id,name,type,serialization_format) VALUES(1,?1,?2,'cdr')",
                params!["/lidar/points", POINT_CLOUD2_TYPE],
            )
            .unwrap();
        for (id, stamp, offset) in [(1_i64, 20_i64, 0.0_f32), (2, 10, 10.0)] {
            let message = PointCloud2Xyz::try_new(
                "lidar",
                7,
                id as u32,
                vec![offset, 1.0, 2.0, offset + 1.0, 3.0, 4.0, offset + 2.0, 5.0, 6.0],
            )
            .unwrap();
            connection
                .execute(
                    "INSERT INTO messages(id,topic_id,timestamp,data) VALUES(?1,1,?2,?3)",
                    params![id, stamp, encode_point_cloud2_xyz(&message).unwrap()],
                )
                .unwrap();
        }
        drop(connection);
        (directory, path)
    }

    #[test]
    fn lists_topics_and_counts_messages() {
        let (_directory, path) = bag_file();
        let topics = list_topics(path).unwrap();
        assert_eq!(topics.len(), 1);
        assert_eq!(topics[0].name, "/lidar/points");
        assert_eq!(topics[0].message_count, 2);
    }

    #[test]
    fn source_orders_messages_and_splits_chunks() {
        let (_directory, path) = bag_file();
        let options = StreamOptions::new(2, MemoryBudget::new(1024 * 1024).unwrap()).unwrap();
        let mut source = Rosbag2PointCloudSource::open(
            path,
            "/lidar/points",
            options,
            CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(source.schema().id.as_str(), ROSBAG2_POINT_XYZ_SCHEMA_ID);

        let first = source.next_chunk().unwrap().unwrap();
        assert_eq!(first.identity().sequence, 0);
        assert_eq!(first.identity().point_offset, 0);
        assert_eq!(first.record().metadata().timestamp.as_nanos(), 7_000_000_002);
        assert_eq!(first.record().metadata().frame_id.0, "lidar");
        assert_eq!(first.record().cloud().len(), 2);
        assert_eq!(
            first.record().cloud().field("x").unwrap(),
            &PointBuffer::from_f32(vec![10.0, 11.0])
        );
        drop(first);

        let second = source.next_chunk().unwrap().unwrap();
        assert_eq!(second.identity().sequence, 1);
        assert_eq!(second.identity().point_offset, 2);
        assert_eq!(second.record().cloud().field("x").unwrap(), &PointBuffer::from_f32(vec![12.0]));
        drop(second);

        let third = source.next_chunk().unwrap().unwrap();
        assert_eq!(third.identity().sequence, 2);
        assert_eq!(third.identity().point_offset, 3);
        assert_eq!(third.record().metadata().timestamp.as_nanos(), 7_000_000_001);
        assert_eq!(
            third.record().cloud().field("x").unwrap(),
            &PointBuffer::from_f32(vec![0.0, 1.0])
        );
        drop(third);

        let fourth = source.next_chunk().unwrap().unwrap();
        assert_eq!(fourth.identity().sequence, 3);
        assert_eq!(fourth.identity().point_offset, 5);
        drop(fourth);
        assert!(source.next_chunk().is_none());
        assert_eq!(source.memory_tracker().snapshot().current_bytes, 0);
    }
}
