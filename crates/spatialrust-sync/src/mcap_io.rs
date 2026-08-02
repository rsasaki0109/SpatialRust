//! MCAP read/write for SpatialRust stamped XYZ records.
//!
//! Payload encoding: `application/x-spatialrust-xyz-v1`
//!
//! ```text
//! u32 le magic = 0x5352_5859 ("SRXY")
//! u32 le point_count
//! u32 le schema_id_len
//! utf8 schema_id
//! u32 le schema_major
//! u32 le schema_minor
//! u64 le stamp_ns
//! u8  clock_domain (0 HostSteady, 1 HostWall, 2 Sensor, 3 External)
//! f32 le xyz[point_count * 3]
//! ```

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Cursor, Read},
    path::Path,
};

use mcap::{records::MessageHeader, Message, Writer};
use spatialrust_core::{
    PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas, Timestamp,
};
use spatialrust_records::{SchemaVersion, SpatialRecord};

use crate::{
    ClockDomain, ClockId, MemoryEpisode, StampedRecord, StampedTime, SyncError, SyncResult, TopicId,
};

const MAGIC: u32 = 0x5352_5859;
const ENCODING: &str = "application/x-spatialrust-xyz-v1";
const SCHEMA_NAME: &str = "spatialrust/xyz/v1";

/// Writes a [`MemoryEpisode`] of XYZ records into an MCAP file.
pub fn write_memory_episode_mcap(
    path: impl AsRef<Path>,
    episode: &MemoryEpisode,
) -> SyncResult<()> {
    let file = File::create(path.as_ref()).map_err(io_err)?;
    let mut writer = Writer::new(BufWriter::new(file)).map_err(mcap_err)?;
    let schema_id = writer.add_schema(SCHEMA_NAME, ENCODING, &[]).map_err(mcap_err)?;

    let mut channels = BTreeMap::<String, u16>::new();
    for (sequence, stamped) in episode.records().iter().enumerate() {
        let topic = stamped.topic.as_str().to_owned();
        let channel_id = if let Some(id) = channels.get(&topic) {
            *id
        } else {
            let id = writer
                .add_channel(schema_id, &topic, ENCODING, &BTreeMap::new())
                .map_err(mcap_err)?;
            channels.insert(topic, id);
            id
        };
        let payload = encode_stamped_xyz(stamped)?;
        let stamp = stamped.stamp.as_nanos();
        writer
            .write_to_known_channel(
                &MessageHeader {
                    channel_id,
                    sequence: sequence as u32,
                    log_time: stamp,
                    publish_time: stamp,
                },
                &payload,
            )
            .map_err(mcap_err)?;
    }
    writer.finish().map_err(mcap_err)?;
    Ok(())
}

/// Reads stamped XYZ records from an MCAP file into a [`MemoryEpisode`].
pub fn read_memory_episode_mcap(path: impl AsRef<Path>) -> SyncResult<MemoryEpisode> {
    let bytes = std::fs::read(path.as_ref()).map_err(io_err)?;
    let mut records = Vec::new();
    for message in mcap::MessageStream::new(&bytes).map_err(mcap_err)? {
        let message = message.map_err(mcap_err)?;
        if message.channel.message_encoding != ENCODING {
            return Err(SyncError::InvalidConfiguration(format!(
                "unsupported MCAP encoding `{}`",
                message.channel.message_encoding
            )));
        }
        records.push(decode_stamped_xyz(&message)?);
    }
    Ok(MemoryEpisode::from_records(records))
}

fn encode_stamped_xyz(stamped: &StampedRecord) -> SyncResult<Vec<u8>> {
    let cloud = stamped.record.cloud();
    let xs = cloud.field("x")?.as_f32()?;
    let ys = cloud.field("y")?.as_f32()?;
    let zs = cloud.field("z")?.as_f32()?;
    if xs.len() != ys.len() || ys.len() != zs.len() {
        return Err(SyncError::InvalidConfiguration("xyz lengths disagree".into()));
    }
    let schema = stamped.record.schema();
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&(xs.len() as u32).to_le_bytes());
    let id = schema.id.as_str().as_bytes();
    out.extend_from_slice(&(id.len() as u32).to_le_bytes());
    out.extend_from_slice(id);
    out.extend_from_slice(&schema.version.major.to_le_bytes());
    out.extend_from_slice(&schema.version.minor.to_le_bytes());
    out.extend_from_slice(&stamped.stamp.as_nanos().to_le_bytes());
    out.push(domain_byte(stamped.stamp.domain));
    for i in 0..xs.len() {
        out.extend_from_slice(&xs[i].to_le_bytes());
        out.extend_from_slice(&ys[i].to_le_bytes());
        out.extend_from_slice(&zs[i].to_le_bytes());
    }
    Ok(out)
}

fn decode_stamped_xyz(message: &Message<'_>) -> SyncResult<StampedRecord> {
    let mut cur = Cursor::new(message.data.as_ref());
    let magic = read_u32(&mut cur)?;
    if magic != MAGIC {
        return Err(SyncError::InvalidConfiguration("bad spatialrust MCAP magic".into()));
    }
    let count = read_u32(&mut cur)? as usize;
    let id_len = read_u32(&mut cur)? as usize;
    let mut id_bytes = vec![0_u8; id_len];
    cur.read_exact(&mut id_bytes).map_err(io_err)?;
    let schema_id = String::from_utf8(id_bytes)
        .map_err(|_| SyncError::InvalidConfiguration("schema id is not UTF-8".into()))?;
    let major = read_u32(&mut cur)?;
    let minor = read_u32(&mut cur)?;
    let stamp_ns = read_u64(&mut cur)?;
    let mut domain_byte = [0_u8; 1];
    cur.read_exact(&mut domain_byte).map_err(io_err)?;
    let domain = byte_domain(domain_byte[0])?;

    let mut xs = Vec::with_capacity(count);
    let mut ys = Vec::with_capacity(count);
    let mut zs = Vec::with_capacity(count);
    for _ in 0..count {
        xs.push(read_f32(&mut cur)?);
        ys.push(read_f32(&mut cur)?);
        zs.push(read_f32(&mut cur)?);
    }

    let mut buffers = PointBufferSet::new();
    buffers.insert("x", PointBuffer::from_f32(xs));
    buffers.insert("y", PointBuffer::from_f32(ys));
    buffers.insert("z", PointBuffer::from_f32(zs));
    let cloud = PointCloud::try_from_parts(
        StandardSchemas::point_xyz(),
        buffers,
        SpatialMetadata::default(),
    )?;
    let record = SpatialRecord::try_from_cloud(schema_id, SchemaVersion::new(major, minor), cloud)?;
    let stamp = StampedTime {
        clock: ClockId::new(message.channel.topic.clone()),
        domain,
        timestamp: Timestamp::from_nanos(stamp_ns),
        quality: crate::SyncQuality::exact(),
    };
    Ok(StampedRecord::new(TopicId::new(message.channel.topic.clone()), stamp, record))
}

fn domain_byte(domain: ClockDomain) -> u8 {
    match domain {
        ClockDomain::HostSteady => 0,
        ClockDomain::HostWall => 1,
        ClockDomain::Sensor => 2,
        ClockDomain::External => 3,
    }
}

fn byte_domain(value: u8) -> SyncResult<ClockDomain> {
    match value {
        0 => Ok(ClockDomain::HostSteady),
        1 => Ok(ClockDomain::HostWall),
        2 => Ok(ClockDomain::Sensor),
        3 => Ok(ClockDomain::External),
        other => Err(SyncError::InvalidConfiguration(format!("unknown clock domain byte {other}"))),
    }
}

fn read_u32(cur: &mut Cursor<&[u8]>) -> SyncResult<u32> {
    let mut buf = [0_u8; 4];
    cur.read_exact(&mut buf).map_err(io_err)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(cur: &mut Cursor<&[u8]>) -> SyncResult<u64> {
    let mut buf = [0_u8; 8];
    cur.read_exact(&mut buf).map_err(io_err)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32(cur: &mut Cursor<&[u8]>) -> SyncResult<f32> {
    let mut buf = [0_u8; 4];
    cur.read_exact(&mut buf).map_err(io_err)?;
    Ok(f32::from_le_bytes(buf))
}

fn io_err(error: impl std::fmt::Display) -> SyncError {
    SyncError::Io(error.to_string())
}

fn mcap_err(error: impl std::fmt::Display) -> SyncError {
    SyncError::Mcap(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{read_memory_episode_mcap, write_memory_episode_mcap};
    use crate::{ClockDomain, MemoryEpisode, StampedRecord, StampedTime};
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas, Timestamp,
    };
    use spatialrust_records::{SchemaVersion, SpatialRecord};

    #[test]
    fn roundtrips_xyz_episode() {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![1.0, 2.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0, 0.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![3.0, 4.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let record =
            SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 2), cloud).unwrap();
        let stamped = StampedRecord::new(
            "lidar",
            StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(42)),
            record,
        );
        let episode = MemoryEpisode::from_records(vec![stamped]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("episode.mcap");
        write_memory_episode_mcap(&path, &episode).unwrap();
        let loaded = read_memory_episode_mcap(&path).unwrap();
        assert_eq!(loaded.records().len(), 1);
        assert_eq!(loaded.records()[0].topic.as_str(), "lidar");
        assert_eq!(loaded.records()[0].stamp.as_nanos(), 42);
        assert_eq!(
            loaded.records()[0].record.cloud().field("x").unwrap().as_f32().unwrap(),
            &[1.0, 2.0]
        );
    }
}
