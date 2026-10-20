//! Fail-closed private full-bag timestamp/pose admission.
//!
//! Unique corrected times are an admission requirement, not message IDs or
//! evidence of physical synchronization. Source grouping is checked separately.

use spatialrust_mapping::Trajectory;
use spatialrust_math::Isometry3;
use spatialrust_sync::{MemoryEpisode, StampedRecord, TopicId};
use std::{collections::BTreeMap, error::Error};

pub(super) fn topic_records<'a>(
    episode: &'a MemoryEpisode,
    topic: &TopicId,
) -> Result<Vec<&'a StampedRecord>, Box<dyn Error>> {
    let records: Vec<_> =
        episode.records().iter().filter(|record| &record.topic == topic).collect();
    for pair in records.windows(2) {
        if pair[1].stamp.as_nanos() <= pair[0].stamp.as_nanos() {
            return Err(format!(
                "topic `{}` requires strictly increasing corrected scan timestamps; repeated {} ns",
                topic.as_str(),
                pair[1].stamp.as_nanos()
            )
            .into());
        }
    }
    Ok(records)
}

pub(super) fn front_pose_lookup(
    records: &[&StampedRecord],
    trajectory: &Trajectory,
) -> Result<BTreeMap<u64, Isometry3<f32>>, Box<dyn Error>> {
    if records.len() != trajectory.samples().len() {
        return Err(format!(
            "front trajectory sample count {} differs from scan count {}",
            trajectory.samples().len(),
            records.len()
        )
        .into());
    }
    let mut poses = BTreeMap::new();
    for (record, pose) in records.iter().zip(trajectory.samples()) {
        // This loop cannot execute for an empty slice.
        let first = records[0];
        if record.topic != first.topic {
            return Err("front pose lookup requires one topic".into());
        }
        if record.record.metadata().frame_id != first.record.metadata().frame_id {
            return Err("front pose lookup requires one source frame".into());
        }
        if record.stamp != pose.stamp {
            return Err("front trajectory stamp differs from its scan clock/time/quality".into());
        }
        if poses.insert(record.stamp.as_nanos(), pose.pose.isometry).is_some() {
            return Err("duplicate corrected front timestamp in pose lookup".into());
        }
    }
    Ok(poses)
}
