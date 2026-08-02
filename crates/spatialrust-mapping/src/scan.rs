//! Bounded scan-sequence odometry over synchronized spatial records.

use spatialrust_core::{FrameId, PointCloud};
use spatialrust_math::{Isometry3, Pose3, Quat};
use spatialrust_sync::{MemoryEpisode, StampedRecord, TopicId};

use crate::{
    DeltaMotion, MappingError, MappingResult, PoseGraph, PoseGraphEdge, PoseNodeId, StampedPose,
    Trajectory,
};

/// Contract for an estimator that maps a previous scan into the current scan.
pub trait ScanMatcher {
    /// Estimates `current_T_previous` from two same-frame scans.
    fn match_scans(
        &self,
        previous: &PointCloud,
        current: &PointCloud,
    ) -> MappingResult<Isometry3<f32>>;
}

/// Bounds and validation settings for scan-sequence odometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanOdometryConfig {
    /// Maximum number of scans retained from the selected topic prefix.
    pub max_scans: usize,
    /// Minimum points required in every scan passed to the matcher.
    pub min_points: usize,
}

impl ScanOdometryConfig {
    /// Creates scan odometry limits.
    #[must_use]
    pub const fn new(max_scans: usize, min_points: usize) -> Self {
        Self { max_scans, min_points }
    }

    fn validate(self) -> MappingResult<Self> {
        if self.max_scans == 0 {
            return Err(MappingError::InvalidConfiguration(
                "scan odometry max_scans must be greater than zero".into(),
            ));
        }
        if self.min_points == 0 {
            return Err(MappingError::InvalidConfiguration(
                "scan odometry min_points must be greater than zero".into(),
            ));
        }
        Ok(self)
    }
}

impl Default for ScanOdometryConfig {
    fn default() -> Self {
        Self::new(1_024, 3)
    }
}

/// Deterministic scan odometry runner over one topic in a memory episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanOdometry {
    config: ScanOdometryConfig,
}

impl ScanOdometry {
    /// Creates a scan odometry runner with validated limits.
    pub fn try_new(config: ScanOdometryConfig) -> MappingResult<Self> {
        Ok(Self { config: config.validate()? })
    }

    /// Returns the configured scan limits.
    #[must_use]
    pub const fn config(&self) -> ScanOdometryConfig {
        self.config
    }

    /// Estimates a deterministic trajectory and relative pose graph.
    ///
    /// The episode is already timestamp/topic ordered. Only the selected
    /// topic is borrowed, and at most `max_scans` records are inspected. Every
    /// selected record must use the same frame and clock domain. The first
    /// scan is the graph root; each matcher result becomes the edge
    /// `current_T_previous` and is composed using the pose-graph convention.
    pub fn estimate<M: ScanMatcher>(
        &self,
        episode: &MemoryEpisode,
        topic: &TopicId,
        matcher: &M,
    ) -> MappingResult<ScanOdometryResult> {
        let scans: Vec<&StampedRecord> = episode
            .records()
            .iter()
            .filter(|record| &record.topic == topic)
            .take(self.config.max_scans)
            .collect();
        if scans.is_empty() {
            return Err(MappingError::Missing(format!("scan topic `{}`", topic.as_str())));
        }
        let truncated =
            episode.records().iter().filter(|record| &record.topic == topic).count() > scans.len();
        let frame_id = scans[0].record.metadata().frame_id.clone();
        let clock = scans[0].stamp.clock.clone();
        let domain = scans[0].stamp.domain;
        for (index, scan) in scans.iter().enumerate() {
            if scan.record.metadata().frame_id != frame_id {
                return Err(MappingError::InvalidConfiguration(format!(
                    "scan {index} frame `{}` differs from `{}`",
                    scan.record.metadata().frame_id.0,
                    frame_id.0
                )));
            }
            if scan.stamp.clock != clock || scan.stamp.domain != domain {
                return Err(MappingError::InvalidConfiguration(
                    "scan timestamps must share one clock and domain".into(),
                ));
            }
            if scan.record.cloud().len() < self.config.min_points {
                return Err(MappingError::InvalidConfiguration(format!(
                    "scan {index} has {} points, minimum is {}",
                    scan.record.cloud().len(),
                    self.config.min_points
                )));
            }
            if let Some(previous) = index.checked_sub(1).and_then(|value| scans.get(value)) {
                if scan.stamp.as_nanos() < previous.stamp.as_nanos() {
                    return Err(MappingError::InvalidConfiguration(
                        "scan timestamps must be non-decreasing".into(),
                    ));
                }
            }
        }

        let root = node_id(topic, 0);
        let mut graph = PoseGraph::new();
        graph.upsert_node(
            root.clone(),
            StampedPose::new(
                scans[0].stamp.clone(),
                Pose3::new(Isometry3::new(
                    Quat::<f32>::identity(),
                    spatialrust_math::Vec3::new(0.0, 0.0, 0.0),
                )),
            ),
        );
        let mut motions = Vec::with_capacity(scans.len().saturating_sub(1));
        for index in 1..scans.len() {
            let previous = scans[index - 1];
            let current = scans[index];
            let to_t_from = matcher.match_scans(previous.record.cloud(), current.record.cloud())?;
            let previous_id = node_id(topic, index - 1);
            let current_id = node_id(topic, index);
            let previous_pose = graph
                .nodes()
                .get(&previous_id.0)
                .ok_or_else(|| MappingError::Missing(format!("pose node `{}`", previous_id.0)))?
                .pose
                .isometry;
            let current_pose = to_t_from.compose(previous_pose);
            let current_sample = StampedPose::new(current.stamp.clone(), Pose3::new(current_pose));
            graph.upsert_node(current_id.clone(), current_sample);
            graph.add_edge(PoseGraphEdge {
                from: previous_id,
                to: current_id,
                to_t_from,
                loop_closure: false,
            })?;
            motions.push(DeltaMotion {
                from: previous.stamp.clone(),
                to: current.stamp.clone(),
                to_t_from,
            });
        }
        graph.localize_from_root(&root)?;

        let mut trajectory = Trajectory::new();
        for index in 0..scans.len() {
            let id = node_id(topic, index);
            let pose = graph
                .nodes()
                .get(&id.0)
                .cloned()
                .ok_or_else(|| MappingError::Missing(format!("pose node `{}`", id.0)))?;
            trajectory.push(pose)?;
        }

        Ok(ScanOdometryResult {
            topic: topic.clone(),
            frame_id,
            trajectory,
            pose_graph: graph,
            motions,
            truncated,
        })
    }
}

/// Output of one bounded scan odometry run.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanOdometryResult {
    /// Selected topic.
    pub topic: TopicId,
    /// Common source frame of all selected scans.
    pub frame_id: FrameId,
    /// Deterministic timestamped trajectory.
    pub trajectory: Trajectory,
    /// Relative pose graph containing sequential edges.
    pub pose_graph: PoseGraph,
    /// Sequential motions in timestamp order.
    pub motions: Vec<DeltaMotion>,
    /// Whether the topic contained more scans than the configured prefix.
    pub truncated: bool,
}

fn node_id(topic: &TopicId, index: usize) -> PoseNodeId {
    PoseNodeId::new(format!("{}#{index}", topic.as_str()))
}

#[cfg(feature = "scan-icp")]
/// ICP-backed scan matcher for sequential point-cloud odometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IcpScanMatcher {
    registration: spatialrust_registration::IcpRegistration,
}

#[cfg(feature = "scan-icp")]
impl IcpScanMatcher {
    /// Creates an ICP scan matcher.
    #[must_use]
    pub const fn new(config: spatialrust_registration::IcpConfig) -> Self {
        Self { registration: spatialrust_registration::IcpRegistration::new(config) }
    }

    /// Returns the underlying ICP configuration.
    #[must_use]
    pub const fn config(&self) -> spatialrust_registration::IcpConfig {
        self.registration.config()
    }
}

#[cfg(feature = "scan-icp")]
impl ScanMatcher for IcpScanMatcher {
    fn match_scans(
        &self,
        previous: &PointCloud,
        current: &PointCloud,
    ) -> MappingResult<Isometry3<f32>> {
        use spatialrust_registration::PointCloudRegistration;

        Ok(self.registration.align(previous, current)?.transform)
    }
}

#[cfg(test)]
mod tests {
    use super::{ScanMatcher, ScanOdometry, ScanOdometryConfig};
    use spatialrust_core::{
        FrameId, PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas,
        Timestamp,
    };
    use spatialrust_math::{Isometry3, Quat, Vec3};
    use spatialrust_records::{SchemaVersion, SpatialRecord};
    use spatialrust_sync::{ClockDomain, MemoryEpisode, StampedRecord, StampedTime, TopicId};

    #[derive(Clone, Copy)]
    struct ShiftMatcher;

    impl ScanMatcher for ShiftMatcher {
        fn match_scans(
            &self,
            _previous: &PointCloud,
            _current: &PointCloud,
        ) -> super::MappingResult<Isometry3<f32>> {
            Ok(Isometry3::new(Quat::<f32>::identity(), Vec3::new(1.0, 0.0, 0.0)))
        }
    }

    fn scan(topic: &str, stamp: u64, frame: &str) -> StampedRecord {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![0.0, 1.0, 0.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0, 0.0, 1.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![0.0, 0.0, 0.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::new(frame, Timestamp::from_nanos(stamp)),
        )
        .unwrap();
        let record =
            SpatialRecord::try_from_cloud("scan", SchemaVersion::new(1, 0), cloud).unwrap();
        StampedRecord::new(
            topic,
            StampedTime::exact("ros2", ClockDomain::External, Timestamp::from_nanos(stamp)),
            record,
        )
    }

    #[test]
    fn builds_trajectory_and_pose_graph_from_bounded_topic_prefix() {
        let topic = TopicId::new("/lidar");
        let episode = MemoryEpisode::from_records(vec![
            scan(topic.as_str(), 20, "lidar"),
            scan(topic.as_str(), 10, "lidar"),
            scan(topic.as_str(), 30, "lidar"),
        ]);
        let odometry = ScanOdometry::try_new(ScanOdometryConfig::new(2, 3)).unwrap();
        let result = odometry.estimate(&episode, &topic, &ShiftMatcher).unwrap();
        assert_eq!(result.trajectory.samples().len(), 2);
        assert_eq!(result.motions.len(), 1);
        assert!(result.truncated);
        assert_eq!(result.frame_id, FrameId::new("lidar"));
        assert!((result.trajectory.samples()[1].pose.isometry.translation().x - 1.0).abs() < 1e-5);
        assert_eq!(result.pose_graph.edges().len(), 1);
    }

    #[test]
    fn rejects_mixed_frames() {
        let topic = TopicId::new("/lidar");
        let episode = MemoryEpisode::from_records(vec![
            scan(topic.as_str(), 1, "front"),
            scan(topic.as_str(), 2, "rear"),
        ]);
        let odometry = ScanOdometry::try_new(ScanOdometryConfig::default()).unwrap();
        let error = odometry.estimate(&episode, &topic, &ShiftMatcher).unwrap_err();
        assert!(error.to_string().contains("differs"));
    }

    #[cfg(feature = "scan-icp")]
    #[test]
    fn icp_matcher_estimates_previous_to_current_motion() {
        use super::IcpScanMatcher;
        use spatialrust_core::PointCloudBuilder;
        use spatialrust_registration::{transform_point_cloud, IcpConfig};

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for x in 0..6 {
            for y in 0..6 {
                for z in 0..3 {
                    builder
                        .push_point([x as f32 * 0.05, y as f32 * 0.05, z as f32 * 0.05])
                        .unwrap();
                }
            }
        }
        let previous = builder.build().unwrap();
        let expected = Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.02, -0.01, 0.0));
        let current = transform_point_cloud(&previous, expected).unwrap();
        let matcher = IcpScanMatcher::new(IcpConfig {
            max_correspondence_distance: 0.1,
            max_iterations: 30,
            ..IcpConfig::default()
        });
        let estimated = matcher.match_scans(&previous, &current).unwrap();
        assert!((estimated.translation().x - expected.translation().x).abs() < 5e-3);
        assert!((estimated.translation().y - expected.translation().y).abs() < 5e-3);
    }
}
