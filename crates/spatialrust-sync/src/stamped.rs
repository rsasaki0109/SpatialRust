//! Stamped spatial records for multimodal fusion.

use spatialrust_records::SpatialRecord;

use crate::{StampedTime, TopicId};

/// A spatial record tagged with topic and clocked time.
#[derive(Clone, Debug, PartialEq)]
pub struct StampedRecord {
    /// Logical multimodal topic / channel.
    pub topic: TopicId,
    /// Observation time with sync metadata.
    pub stamp: StampedTime,
    /// Versioned spatial payload.
    pub record: SpatialRecord,
}

impl StampedRecord {
    /// Creates a stamped record.
    #[must_use]
    pub fn new(topic: impl Into<TopicId>, stamp: StampedTime, record: SpatialRecord) -> Self {
        Self { topic: topic.into(), stamp, record }
    }
}
