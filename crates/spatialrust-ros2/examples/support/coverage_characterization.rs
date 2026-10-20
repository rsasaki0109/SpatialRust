//! Existing replay semantics before the additive coverage implementation.

use spatialrust_core::{
    PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas, Timestamp,
};
use spatialrust_records::{SchemaVersion, SpatialRecord};
use spatialrust_sync::{
    ClockDomain, DeterministicReplayer, MemoryEpisode, StampedRecord, StampedTime, SyncQuality,
    SyncWindow, TopicId,
};

pub(super) fn sample(topic: &str, nanos: u64) -> StampedRecord {
    let mut buffers = PointBufferSet::new();
    buffers.insert("x", PointBuffer::from_f32(vec![0.0, 1.0, 0.0, 1.0]));
    buffers.insert("y", PointBuffer::from_f32(vec![0.0, 0.0, 1.0, 1.0]));
    buffers.insert("z", PointBuffer::from_f32(vec![1.0, 1.0, 1.0, 2.0]));
    let cloud = PointCloud::try_from_parts(
        StandardSchemas::point_xyz(),
        buffers,
        SpatialMetadata::new(
            if topic == "/front" { "front" } else { "rear" },
            Timestamp::from_nanos(nanos),
        ),
    )
    .unwrap();
    StampedRecord::new(
        topic,
        StampedTime {
            clock: "fixture-time".into(),
            domain: ClockDomain::External,
            timestamp: Timestamp::from_nanos(nanos),
            quality: SyncQuality { estimated: true, ..SyncQuality::exact() },
        },
        SpatialRecord::try_from_cloud("synthetic", SchemaVersion::new(1, 0), cloud).unwrap(),
    )
}

pub(super) fn episode(items: &[(&str, u64)]) -> MemoryEpisode {
    MemoryEpisode::from_records(items.iter().map(|(topic, nanos)| sample(topic, *nanos)).collect())
}

fn window(delta: u64, uncertainty: u64) -> SyncWindow {
    SyncWindow { max_delta_ns: delta, max_uncertainty_ns: uncertainty }
}

fn pairs(episode: &MemoryEpisode, window: SyncWindow) -> Vec<(usize, usize)> {
    let front = TopicId::new("/front");
    let rear = TopicId::new("/rear");
    let mut replay = DeterministicReplayer::new(episode);
    let mut result = Vec::new();
    while let Some(bundle) =
        replay.next_bundle(&front, std::slice::from_ref(&rear), window).unwrap()
    {
        let index = |topic: &TopicId| {
            episode
                .records()
                .iter()
                .position(|record| std::ptr::eq(record, *bundle.get(topic).unwrap()))
                .unwrap()
        };
        result.push((index(&front), index(&rear)));
    }
    result
}

#[test]
fn nearest_tie_uses_first_stable_rear() {
    assert_eq!(
        pairs(&episode(&[("/rear", 30), ("/front", 20), ("/rear", 10)]), window(10, 0)),
        vec![(1, 0)]
    );
}

#[test]
fn delta_window_is_inclusive() {
    assert_eq!(pairs(&episode(&[("/front", 20), ("/rear", 25)]), window(5, 0)), vec![(0, 1)]);
}

#[test]
fn outside_delta_window_is_unmatched() {
    assert!(pairs(&episode(&[("/front", 20), ("/rear", 26)]), window(5, 0)).is_empty());
}

#[test]
fn primary_uncertainty_boundary_is_inclusive() {
    let mut front = sample("/front", 20);
    front.stamp.quality.uncertainty_ns = 5;
    assert_eq!(
        pairs(&MemoryEpisode::from_records(vec![front, sample("/rear", 20)]), window(0, 5)),
        vec![(0, 1)]
    );
}

#[test]
fn uncertain_primary_is_skipped() {
    let mut front = sample("/front", 20);
    front.stamp.quality.uncertainty_ns = 6;
    assert!(pairs(&MemoryEpisode::from_records(vec![front, sample("/rear", 20)]), window(0, 5))
        .is_empty());
}

#[test]
fn uncertain_nearest_rear_does_not_hide_eligible_rear() {
    let mut rear = sample("/rear", 20);
    rear.stamp.quality.uncertainty_ns = 6;
    let mut eligible = sample("/rear", 24);
    eligible.stamp.quality.uncertainty_ns = 5;
    assert_eq!(
        pairs(
            &MemoryEpisode::from_records(vec![sample("/front", 20), rear, eligible]),
            window(5, 5)
        ),
        vec![(0, 2)]
    );
}

#[test]
fn rear_record_can_be_reused_by_three_fronts() {
    assert_eq!(
        pairs(
            &episode(&[("/front", 10), ("/front", 20), ("/front", 30), ("/rear", 20)]),
            window(100, 0)
        ),
        vec![(0, 2), (1, 2), (3, 2)]
    );
}

#[test]
fn unmatched_rear_remains_in_retained_episode() {
    let episode = episode(&[("/front", 10), ("/front", 20), ("/rear", 11), ("/rear", 100)]);
    assert_eq!(pairs(&episode, window(2, 0)), vec![(0, 1)]);
    assert_eq!(episode.records().len(), 4);
}

#[test]
fn equal_cross_topic_times_have_distinct_episode_indices() {
    assert_eq!(pairs(&episode(&[("/rear", 20), ("/front", 20)]), window(0, 0)), vec![(0, 1)]);
}

#[test]
fn empty_episode_has_no_bundle() {
    assert!(pairs(&episode(&[]), window(10, 0)).is_empty());
}

#[test]
fn missing_front_has_no_bundle() {
    assert!(pairs(&episode(&[("/rear", 20)]), window(10, 0)).is_empty());
}

#[test]
fn missing_rear_has_no_bundle() {
    assert!(pairs(&episode(&[("/front", 20)]), window(10, 0)).is_empty());
}

#[test]
fn full_bag_admission_rejects_duplicate_topic_times() {
    let episode = episode(&[("/front", 20), ("/front", 20), ("/rear", 20)]);
    assert!(super::scan_identity::topic_records(&episode, &TopicId::new("/front")).is_err());
}
