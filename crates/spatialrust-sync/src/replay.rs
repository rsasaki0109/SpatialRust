//! In-memory episodes and deterministic multimodal replay.

use std::collections::{BTreeMap, HashMap};

use crate::{StampedRecord, SyncResult};

/// Logical multimodal topic / channel name.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TopicId(pub String);

impl TopicId {
    /// Creates a topic identifier.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrows the topic string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TopicId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for TopicId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// Inclusive time-window options for approximate sync.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SyncWindow {
    /// Maximum absolute time delta accepted when pairing topics.
    pub max_delta_ns: u64,
    /// Maximum sync uncertainty accepted on each stamp.
    pub max_uncertainty_ns: u64,
}

impl Default for SyncWindow {
    fn default() -> Self {
        Self { max_delta_ns: 10_000_000, max_uncertainty_ns: 5_000_000 }
    }
}

/// Ordered episode index keyed by timestamp then topic.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EpisodeIndex {
    /// Stable sorted keys: (timestamp_ns, topic, insertion ordinal).
    entries: BTreeMap<(u64, String, u64), usize>,
}

impl EpisodeIndex {
    /// Builds an index over stamped records.
    pub fn build(records: &[StampedRecord]) -> Self {
        let mut entries = BTreeMap::new();
        for (ordinal, record) in records.iter().enumerate() {
            entries.insert(
                (record.stamp.as_nanos(), record.topic.0.clone(), ordinal as u64),
                ordinal,
            );
        }
        Self { entries }
    }

    /// Returns record indices in deterministic sorted order.
    pub fn ordered_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.entries.values().copied()
    }
}

/// In-memory multimodal episode used as the default MCAP substitute.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MemoryEpisode {
    records: Vec<StampedRecord>,
    index: EpisodeIndex,
}

impl MemoryEpisode {
    /// Creates an episode and builds a deterministic index.
    #[must_use]
    pub fn from_records(mut records: Vec<StampedRecord>) -> Self {
        // Stable pre-sort so insertion-order ties remain deterministic.
        records.sort_by(|a, b| {
            (a.stamp.as_nanos(), a.topic.as_str()).cmp(&(b.stamp.as_nanos(), b.topic.as_str()))
        });
        let index = EpisodeIndex::build(&records);
        Self { records, index }
    }

    /// Returns all records.
    #[must_use]
    pub fn records(&self) -> &[StampedRecord] {
        &self.records
    }

    /// Returns the deterministic index.
    #[must_use]
    pub fn index(&self) -> &EpisodeIndex {
        &self.index
    }
}

/// Replays an episode as a sorted stream of stamped records.
pub struct DeterministicReplayer<'a> {
    episode: &'a MemoryEpisode,
    cursor: usize,
    order: Vec<usize>,
}

impl<'a> DeterministicReplayer<'a> {
    /// Creates a replayer that walks the episode in index order.
    #[must_use]
    pub fn new(episode: &'a MemoryEpisode) -> Self {
        let order = episode.index.ordered_indices().collect();
        Self { episode, cursor: 0, order }
    }

    /// Returns the next record, if any.
    pub fn next_record(&mut self) -> Option<&'a StampedRecord> {
        let index = *self.order.get(self.cursor)?;
        self.cursor += 1;
        self.episode.records.get(index)
    }

    /// Bundles nearest-neighbor matches across required topics around the next
    /// primary-topic message.
    pub fn next_bundle(
        &mut self,
        primary: &TopicId,
        others: &[TopicId],
        window: SyncWindow,
    ) -> SyncResult<Option<HashMap<TopicId, &'a StampedRecord>>> {
        while let Some(candidate) = self.next_record() {
            if &candidate.topic != primary {
                continue;
            }
            if !candidate.stamp.quality.is_tight(window.max_uncertainty_ns) {
                continue;
            }
            let mut bundle = HashMap::new();
            bundle.insert(primary.clone(), candidate);
            let mut ok = true;
            for topic in others {
                match nearest_match(self.episode, topic, candidate.stamp.as_nanos(), window) {
                    Some(matched) => {
                        bundle.insert(topic.clone(), matched);
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                return Ok(Some(bundle));
            }
        }
        Ok(None)
    }
}

fn nearest_match<'a>(
    episode: &'a MemoryEpisode,
    topic: &TopicId,
    center_ns: u64,
    window: SyncWindow,
) -> Option<&'a StampedRecord> {
    episode
        .records
        .iter()
        .filter(|record| {
            &record.topic == topic
                && record.stamp.quality.is_tight(window.max_uncertainty_ns)
                && record.stamp.as_nanos().abs_diff(center_ns) <= window.max_delta_ns
        })
        .min_by_key(|record| record.stamp.as_nanos().abs_diff(center_ns))
}

#[cfg(feature = "mcap")]
mod mcap_placeholder {
    //! Feature gate reserved for file-backed MCAP codecs.
    //!
    //! The default episode contract is [`super::MemoryEpisode`]. Enabling
    //! `mcap` currently documents intent without pulling a file codec until a
    //! checked binding lands in a follow-up slice.

    /// Marker ensuring the `mcap` feature compiles.
    pub const MCAP_FEATURE_ENABLED: bool = true;
}

#[cfg(test)]
mod tests {
    use super::{DeterministicReplayer, MemoryEpisode, SyncWindow, TopicId};
    use crate::{ClockDomain, StampedRecord, StampedTime};
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas, Timestamp,
    };
    use spatialrust_records::{SchemaVersion, SpatialRecord};

    fn sample(topic: &str, nanos: u64, x: f32) -> StampedRecord {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![x]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![0.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let record = SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 0), cloud).unwrap();
        StampedRecord::new(
            topic,
            StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(nanos)),
            record,
        )
    }

    #[test]
    fn replays_in_timestamp_order_and_bundles_topics() {
        let episode = MemoryEpisode::from_records(vec![
            sample("lidar", 30, 3.0),
            sample("camera", 10, 1.0),
            sample("lidar", 12, 2.0),
            sample("camera", 40, 4.0),
        ]);
        let mut replayer = DeterministicReplayer::new(&episode);
        let first = replayer.next_record().unwrap();
        assert_eq!(first.topic.as_str(), "camera");
        assert_eq!(first.stamp.as_nanos(), 10);

        let mut replayer = DeterministicReplayer::new(&episode);
        let bundle = replayer
            .next_bundle(
                &TopicId::new("camera"),
                &[TopicId::new("lidar")],
                SyncWindow { max_delta_ns: 5, max_uncertainty_ns: 0 },
            )
            .unwrap()
            .unwrap();
        assert_eq!(bundle.get(&TopicId::new("camera")).unwrap().stamp.as_nanos(), 10);
        assert_eq!(bundle.get(&TopicId::new("lidar")).unwrap().stamp.as_nanos(), 12);
    }
}
