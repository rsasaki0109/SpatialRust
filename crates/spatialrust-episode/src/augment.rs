//! Deterministic episode augmentation operators.

use spatialrust_sync::MemoryEpisode;

use crate::{Episode, EpisodeResult};

/// Augmentation operators for embodied datasets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AugmentationOp {
    /// Drop every Nth record to thin the episode.
    TemporalSubsample {
        /// Keep stride (must be >= 1).
        stride: usize,
    },
    /// Reverse record order (for stress tests).
    Reverse,
}

/// Applies augmentation ops while preserving schema contracts.
#[derive(Clone, Copy, Debug, Default)]
pub struct EpisodeAugmentor;

impl EpisodeAugmentor {
    /// Applies one operator to an episode and returns a new episode id suffix.
    pub fn apply(&self, episode: &Episode, op: AugmentationOp) -> EpisodeResult<Episode> {
        let records = episode.memory.records();
        let next = match op {
            AugmentationOp::TemporalSubsample { stride } => {
                if stride == 0 {
                    return Err(crate::EpisodeError::InvalidConfiguration(
                        "stride must be positive".into(),
                    ));
                }
                records.iter().step_by(stride).cloned().collect::<Vec<_>>()
            }
            AugmentationOp::Reverse => records.iter().rev().cloned().collect::<Vec<_>>(),
        };
        let memory = MemoryEpisode::from_records(next);
        let mut out = Episode::try_new(format!("{}::aug", episode.id.0), memory)?;
        out.annotations = episode.annotations.clone();
        out.provenance = episode.provenance.clone();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{AugmentationOp, EpisodeAugmentor};
    use crate::Episode;
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas, Timestamp,
    };
    use spatialrust_records::{SchemaVersion, SpatialRecord};
    use spatialrust_sync::{ClockDomain, MemoryEpisode, StampedRecord, StampedTime};

    fn sample(nanos: u64) -> StampedRecord {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![0.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let record =
            SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 0), cloud).unwrap();
        StampedRecord::new(
            "lidar",
            StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(nanos)),
            record,
        )
    }

    #[test]
    fn subsamples_episode() {
        let memory = MemoryEpisode::from_records(vec![sample(1), sample(2), sample(3), sample(4)]);
        let episode = Episode::try_new("ep0", memory).unwrap();
        let out = EpisodeAugmentor
            .apply(&episode, AugmentationOp::TemporalSubsample { stride: 2 })
            .unwrap();
        assert_eq!(out.memory.records().len(), 2);
    }
}
