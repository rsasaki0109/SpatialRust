//! Bounded construction of deterministic in-memory episodes.

use spatialrust_records::record_storage_bytes;

use crate::{MemoryEpisode, StampedRecord, SyncError, SyncResult};

/// Hard limits applied while collecting an episode.
///
/// The byte limit accounts for the allocated scalar capacity of every record,
/// which is deliberately conservative for external or untrusted inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpisodeLimits {
    /// Maximum number of stamped records.
    pub max_records: u64,
    /// Maximum total point count across all records.
    pub max_points: u64,
    /// Maximum allocated column-storage bytes across all records.
    pub max_bytes: u64,
}

impl EpisodeLimits {
    /// Creates episode limits; [`MemoryEpisodeBuilder::try_new`] validates them.
    #[must_use]
    pub const fn new(max_records: u64, max_points: u64, max_bytes: u64) -> Self {
        Self { max_records, max_points, max_bytes }
    }

    fn validate(self) -> SyncResult<Self> {
        for (name, value) in [
            ("max_records", self.max_records),
            ("max_points", self.max_points),
            ("max_bytes", self.max_bytes),
        ] {
            if value == 0 {
                return Err(SyncError::InvalidConfiguration(format!(
                    "episode {name} must be greater than zero"
                )));
            }
        }
        Ok(self)
    }
}

impl Default for EpisodeLimits {
    fn default() -> Self {
        Self::new(4_096, 16_777_216, 512 * 1024 * 1024)
    }
}

/// Bounded collector for a [`MemoryEpisode`].
///
/// Records are retained only after all three limits have been checked. Call
/// [`MemoryEpisodeBuilder::finish`] to sort them into the episode's stable
/// timestamp/topic order.
#[derive(Debug)]
pub struct MemoryEpisodeBuilder {
    limits: EpisodeLimits,
    records: Vec<StampedRecord>,
    points: u64,
    bytes: u64,
}

impl MemoryEpisodeBuilder {
    /// Creates an empty builder with validated hard limits.
    pub fn try_new(limits: EpisodeLimits) -> SyncResult<Self> {
        Ok(Self { limits: limits.validate()?, records: Vec::new(), points: 0, bytes: 0 })
    }

    /// Adds one stamped record if all configured limits remain satisfied.
    pub fn push(&mut self, stamped: StampedRecord) -> SyncResult<()> {
        let record_points = u64::try_from(stamped.record.cloud().len())
            .map_err(|_| SyncError::InvalidConfiguration("episode point count overflow".into()))?;
        let record_bytes = record_storage_bytes(&stamped.record)?;
        let current_records = u64::try_from(self.records.len())
            .map_err(|_| SyncError::InvalidConfiguration("episode record count overflow".into()))?;
        let next_points = self.points.checked_add(record_points).ok_or_else(|| {
            SyncError::InvalidConfiguration("episode point count overflow".into())
        })?;
        let next_bytes = self
            .bytes
            .checked_add(record_bytes)
            .ok_or_else(|| SyncError::InvalidConfiguration("episode byte count overflow".into()))?;

        check_limit("records", 1, current_records, self.limits.max_records)?;
        check_limit("points", record_points, self.points, self.limits.max_points)?;
        check_limit("bytes", record_bytes, self.bytes, self.limits.max_bytes)?;

        self.records.push(stamped);
        self.points = next_points;
        self.bytes = next_bytes;
        Ok(())
    }

    /// Returns the number of records currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns whether the builder contains no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns the total point count currently retained.
    #[must_use]
    pub const fn points(&self) -> u64 {
        self.points
    }

    /// Returns the conservative allocated column-storage total.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Returns the hard limits used by this builder.
    #[must_use]
    pub const fn limits(&self) -> EpisodeLimits {
        self.limits
    }

    /// Finishes the bounded collection as a deterministic in-memory episode.
    #[must_use]
    pub fn finish(self) -> MemoryEpisode {
        MemoryEpisode::from_records(self.records)
    }
}

fn check_limit(resource: &'static str, requested: u64, current: u64, limit: u64) -> SyncResult<()> {
    let next = current.checked_add(requested).ok_or(SyncError::EpisodeLimitExceeded {
        resource,
        requested,
        current,
        limit,
    })?;
    if next > limit {
        return Err(SyncError::EpisodeLimitExceeded { resource, requested, current, limit });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{EpisodeLimits, MemoryEpisodeBuilder};
    use crate::{ClockDomain, StampedRecord, StampedTime, TopicId};
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas, Timestamp,
    };
    use spatialrust_records::{SchemaVersion, SpatialRecord};

    fn sample(topic: &str, timestamp: u64, point_count: usize) -> StampedRecord {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![1.0; point_count]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0; point_count]));
        buffers.insert("z", PointBuffer::from_f32(vec![0.0; point_count]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let record =
            SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 0), cloud).unwrap();
        StampedRecord::new(
            TopicId::new(topic),
            StampedTime::exact("test", ClockDomain::External, Timestamp::from_nanos(timestamp)),
            record,
        )
    }

    #[test]
    fn enforces_limits_before_retaining_record() {
        let mut builder = MemoryEpisodeBuilder::try_new(EpisodeLimits::new(1, 4, 100)).unwrap();
        builder.push(sample("lidar", 20, 1)).unwrap();
        let error = builder.push(sample("lidar", 10, 2)).unwrap_err();
        assert!(error.to_string().contains("episode records limit exceeded"));
        assert_eq!(builder.len(), 1);
        assert_eq!(builder.points(), 1);
        assert_eq!(builder.bytes(), 12);
    }

    #[test]
    fn finish_uses_episode_deterministic_order() {
        let mut builder = MemoryEpisodeBuilder::try_new(EpisodeLimits::new(4, 4, 48)).unwrap();
        builder.push(sample("lidar", 20, 1)).unwrap();
        builder.push(sample("camera", 10, 1)).unwrap();
        let episode = builder.finish();
        assert_eq!(episode.records()[0].topic.as_str(), "camera");
        assert_eq!(episode.records()[1].topic.as_str(), "lidar");
    }
}
