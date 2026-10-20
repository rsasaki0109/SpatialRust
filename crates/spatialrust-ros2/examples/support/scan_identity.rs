//! Private full-bag scan/pose association seam.

use spatialrust_mapping::Trajectory;
use spatialrust_math::Isometry3;
use spatialrust_sync::{MemoryEpisode, StampedRecord, TopicId};
use std::{collections::BTreeMap, error::Error};

pub(super) fn topic_records<'a>(
    episode: &'a MemoryEpisode,
    topic: &TopicId,
) -> Result<Vec<&'a StampedRecord>, Box<dyn Error>> {
    Ok(episode.records().iter().filter(|record| &record.topic == topic).collect())
}

pub(super) fn front_pose_lookup(
    records: &[&StampedRecord],
    trajectory: &Trajectory,
) -> Result<BTreeMap<u64, Isometry3<f32>>, Box<dyn Error>> {
    let mut poses = BTreeMap::new();
    for (record, pose) in records.iter().zip(trajectory.samples()) {
        poses.insert(record.stamp.as_nanos(), pose.pose.isometry);
    }
    Ok(poses)
}
