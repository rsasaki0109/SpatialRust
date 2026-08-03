//! Read-only rosbag2 SQLite storage for bounded PointCloud2 streams.

use std::path::Path;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use spatialrust_core::{
    PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas, Timestamp,
};
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, ChunkIdentity, MemoryReservation, MemoryTracker,
    RecordProvenance, RecordsError, RecordsResult, SchemaDescriptor, SchemaVersion, SpatialRecord,
    SpatialRecordChunk, StreamOptions,
};
use spatialrust_runtime::{
    decode_point_cloud2_xyz, decode_tf_message, point_cloud2_has_intensity, PointCloud2Xyz,
    RuntimeError, TfTransform, POINT_CLOUD2_TYPE, TF_MESSAGE_TYPE,
};
use thiserror::Error;

/// Stable schema family for XYZ columns decoded from ROS 2 PointCloud2.
pub const ROSBAG2_POINT_XYZ_SCHEMA_ID: &str = "ros2.sensor_msgs.msg.PointCloud2.xyz";
/// Stable schema family for XYZ-I columns decoded from ROS 2 PointCloud2.
pub const ROSBAG2_POINT_XYZI_SCHEMA_ID: &str = "ros2.sensor_msgs.msg.PointCloud2.xyzi";

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

/// One bounded rosbag2 TFMessage with its SQLite capture timestamp.
#[derive(Clone, Debug, PartialEq)]
pub struct Rosbag2TfMessage {
    /// SQLite message timestamp in nanoseconds.
    pub bag_timestamp: u64,
    /// Ordered transforms carried by the TFMessage.
    pub transforms: Vec<TfTransform>,
}

/// Lists topics from a rosbag2 SQLite file without modifying it.
pub fn list_topics(path: impl AsRef<Path>) -> Rosbag2Result<Vec<Rosbag2Topic>> {
    let connection = open_connection(path)?;
    read_topics(&connection)
}

/// Reads at most `max_messages` TFMessage payloads from one rosbag2 topic.
///
/// The SQLite source remains read-only and message order is `(timestamp, id)`.
/// This function decodes TF edges but does not compose them, select a root, or
/// claim that the result belongs to another bag or sensor naming scheme.
pub fn list_tf_messages(
    path: impl AsRef<Path>,
    topic_name: &str,
    max_messages: usize,
) -> Rosbag2Result<Vec<Rosbag2TfMessage>> {
    if max_messages == 0 {
        return Err(Rosbag2Error::InvalidBag("TF message limit must be greater than zero".into()));
    }
    let connection = open_connection(path)?;
    let topics = read_topics(&connection)?;
    let topic = topics
        .into_iter()
        .find(|topic| topic.name == topic_name)
        .ok_or_else(|| Rosbag2Error::InvalidBag(format!("topic `{topic_name}` was not found")))?;
    if topic.type_name != TF_MESSAGE_TYPE {
        return Err(Rosbag2Error::Unsupported(format!(
            "topic `{}` has type `{}`, expected `{TF_MESSAGE_TYPE}`",
            topic.name, topic.type_name
        )));
    }
    if !topic.serialization_format.eq_ignore_ascii_case("cdr") {
        return Err(Rosbag2Error::Unsupported(format!(
            "topic `{}` uses serialization `{}`, only CDR is supported",
            topic.name, topic.serialization_format
        )));
    }
    let limit = i64::try_from(max_messages)
        .map_err(|_| Rosbag2Error::InvalidBag("TF message limit exceeds SQLite bounds".into()))?;
    let mut statement = connection.prepare(
        "SELECT timestamp, data FROM messages WHERE topic_id = ?1 \
         ORDER BY timestamp ASC, id ASC LIMIT ?2",
    )?;
    let rows = statement.query_map(params![topic.id, limit], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
    })?;
    let mut messages = Vec::new();
    for row in rows {
        let (timestamp, data) = row?;
        let bag_timestamp = u64::try_from(timestamp).map_err(|_| {
            Rosbag2Error::InvalidBag(format!("TF message has negative timestamp {timestamp}"))
        })?;
        let transforms = decode_tf_message(&data)?;
        messages.push(Rosbag2TfMessage { bag_timestamp, transforms });
    }
    Ok(messages)
}

/// Bounded, deterministic PointCloud2 XYZ/XYZI source backed by rosbag2 SQLite.
///
/// The SQLite connection is opened read-only. One selected topic is traversed
/// in `(timestamp, id)` order. A PointCloud2 message may be split into several
/// leased chunks when it exceeds `StreamOptions::chunk_points()`.
pub struct Rosbag2PointCloudSource {
    connection: Connection,
    source_id: String,
    topic: Rosbag2Topic,
    schema: SchemaDescriptor,
    options: StreamOptions,
    tracker: MemoryTracker,
    cancellation: CancellationToken,
    max_chunk_bytes: u64,
    max_message_bytes: u64,
    has_intensity: bool,
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
        let path = path.as_ref();
        let source_uri = path.display().to_string();
        let source_id = format!("rosbag2-sqlite:{source_uri}");
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

        let has_intensity = first_message_has_intensity(&connection, topic.id)?;
        let (schema_id, point_schema) = if has_intensity {
            (ROSBAG2_POINT_XYZI_SCHEMA_ID, StandardSchemas::point_xyzi())
        } else {
            (ROSBAG2_POINT_XYZ_SCHEMA_ID, StandardSchemas::point_xyz())
        };
        let max_message_bytes = max_message_bytes(&connection, topic.id)?;
        let schema = SchemaDescriptor::try_new(schema_id, SchemaVersion::new(1, 0), point_schema)?;
        let max_record_bytes = point_capacity_bytes(options.chunk_points(), has_intensity)?;
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
            source_id,
            topic,
            schema,
            options,
            tracker: MemoryTracker::new(budget),
            cancellation,
            max_chunk_bytes,
            max_message_bytes,
            has_intensity,
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
            if message.intensity.is_some() != self.has_intensity {
                return Err(Rosbag2Error::InvalidBag(format!(
                    "PointCloud2 fields changed in topic `{}`",
                    self.topic.name
                )));
            }
            let decoded_bytes = point_capacity_bytes(message.point_count(), self.has_intensity)?;
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
        let point_end = pending
            .offset
            .checked_add(count)
            .ok_or_else(|| Rosbag2Error::InvalidBag("chunk point range overflow".into()))?;
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
        if self.has_intensity {
            let intensity = pending
                .message
                .intensity
                .as_ref()
                .and_then(|values| values.get(pending.offset..point_end))
                .ok_or_else(|| {
                    Rosbag2Error::InvalidBag("decoded intensity range is out of bounds".into())
                })?;
            buffers.insert("intensity", PointBuffer::from_f32(intensity.to_vec()));
        }
        let cloud = PointCloud::try_from_parts(
            if self.has_intensity {
                StandardSchemas::point_xyzi()
            } else {
                StandardSchemas::point_xyz()
            },
            buffers,
            SpatialMetadata::new(
                pending.message.frame_id.as_str(),
                Timestamp::from_nanos(pending.timestamp),
            ),
        )?;
        let provenance = RecordProvenance::try_new(self.source_id.clone())
            .map_err(Rosbag2Error::Records)?
            .with_source_uri(
                self.source_id.strip_prefix("rosbag2-sqlite:").unwrap_or(&self.source_id),
            )
            .with_stream_id(self.topic.name.clone())
            .with_sequence(Some(self.next_sequence));
        let record =
            SpatialRecord::try_new_with_provenance(self.schema.clone(), cloud, provenance)?;
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

        let reservation =
            match self.tracker.try_reserve(match point_capacity_bytes(count, self.has_intensity) {
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

fn first_message_has_intensity(connection: &Connection, topic_id: i64) -> Rosbag2Result<bool> {
    let data: Option<Vec<u8>> = connection
        .query_row(
            "SELECT data FROM messages WHERE topic_id = ?1 \
             ORDER BY timestamp ASC, id ASC LIMIT 1",
            params![topic_id],
            |row| row.get(0),
        )
        .optional()?;
    let Some(data) = data else {
        return Ok(false);
    };
    Ok(point_cloud2_has_intensity(&data)?)
}

fn message_working_bytes(raw_bytes: u64) -> Rosbag2Result<u64> {
    raw_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(CDR_SCRATCH_OVERHEAD_BYTES))
        .ok_or_else(|| Rosbag2Error::InvalidBag("message working-set overflow".into()))
}

fn point_capacity_bytes(point_count: usize, has_intensity: bool) -> Rosbag2Result<u64> {
    u64::try_from(point_count)
        .ok()
        .and_then(|count| count.checked_mul(if has_intensity { 16 } else { 12 }))
        .ok_or_else(|| Rosbag2Error::InvalidBag("point-column capacity overflow".into()))
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
    use super::{
        list_tf_messages, list_topics, Rosbag2PointCloudSource, ROSBAG2_POINT_XYZI_SCHEMA_ID,
        ROSBAG2_POINT_XYZ_SCHEMA_ID,
    };
    use rusqlite::{params, Connection};
    use spatialrust_core::PointBuffer;
    use spatialrust_records::{
        BoundedSpatialRecordSource, CancellationToken, MemoryBudget, StreamOptions,
    };
    use spatialrust_runtime::{
        encode_point_cloud2_xyz, PointCloud2Xyz, TfTransform, POINT_CLOUD2_TYPE, TF_MESSAGE_TYPE,
    };

    fn align_from(bytes: &mut Vec<u8>, alignment: usize, origin: usize) {
        let relative = bytes.len().saturating_sub(origin);
        let remainder = relative % alignment;
        if remainder != 0 {
            bytes.resize(bytes.len() + alignment - remainder, 0);
        }
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&u32::try_from(value.len() + 1).unwrap().to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
        bytes.push(0);
    }

    fn encode_tf_message(transform: &TfTransform) -> Vec<u8> {
        let mut bytes = vec![0x00, 0x01, 0x00, 0x00];
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&transform.stamp_sec.to_le_bytes());
        bytes.extend_from_slice(&transform.stamp_nanosec.to_le_bytes());
        write_string(&mut bytes, &transform.frame_id);
        write_string(&mut bytes, &transform.child_frame_id);
        align_from(&mut bytes, 8, 4);
        for value in transform.translation.into_iter().chain(transform.rotation_xyzw) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

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
    fn lists_and_decodes_bounded_tf_messages() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tf.db3");
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
                params!["/tf_static", TF_MESSAGE_TYPE],
            )
            .unwrap();
        let transform = TfTransform {
            stamp_sec: 12,
            stamp_nanosec: 34,
            frame_id: "base_link".into(),
            child_frame_id: "lidar_front".into(),
            translation: [1.0, -2.0, 3.5],
            rotation_xyzw: [0.0, 0.0, 0.707, 0.707],
        };
        connection
            .execute(
                "INSERT INTO messages(id,topic_id,timestamp,data) VALUES(1,1,42,?1)",
                params![encode_tf_message(&transform)],
            )
            .unwrap();
        drop(connection);

        let messages = list_tf_messages(&path, "/tf_static", 1).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].bag_timestamp, 42);
        assert_eq!(messages[0].transforms, vec![transform]);
        assert!(list_tf_messages(&path, "/tf_static", 0).is_err());
    }

    #[test]
    fn source_orders_messages_and_splits_chunks() {
        let (_directory, path) = bag_file();
        let source_uri = path.display().to_string();
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
        assert_eq!(first.record().provenance().source_id, format!("rosbag2-sqlite:{source_uri}"));
        assert_eq!(first.record().provenance().source_uri.as_deref(), Some(source_uri.as_str()));
        assert_eq!(first.record().provenance().stream_id.as_deref(), Some("/lidar/points"));
        assert_eq!(first.record().provenance().sequence, Some(0));
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

    fn bag_file_with_intensity() -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sample-intensity.db3");
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
            let message = PointCloud2Xyz::try_new_with_intensity(
                "lidar",
                7,
                id as u32,
                vec![offset, 1.0, 2.0, offset + 1.0, 3.0, 4.0, offset + 2.0, 5.0, 6.0],
                vec![100.0 + offset, 101.0 + offset, 102.0 + offset],
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
    fn source_preserves_intensity_and_selects_xyzi_schema() {
        let (_directory, path) = bag_file_with_intensity();
        let options = StreamOptions::new(2, MemoryBudget::new(1024 * 1024).unwrap()).unwrap();
        let mut source = Rosbag2PointCloudSource::open(
            path,
            "/lidar/points",
            options,
            CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(source.schema().id.as_str(), ROSBAG2_POINT_XYZI_SCHEMA_ID);

        let first = source.next_chunk().unwrap().unwrap();
        assert_eq!(
            first.record().cloud().field("intensity").unwrap(),
            &PointBuffer::from_f32(vec![110.0, 111.0])
        );
        drop(first);
        let second = source.next_chunk().unwrap().unwrap();
        assert_eq!(
            second.record().cloud().field("intensity").unwrap(),
            &PointBuffer::from_f32(vec![112.0])
        );
        drop(second);
        assert!(source.next_chunk().is_some());
    }
}
