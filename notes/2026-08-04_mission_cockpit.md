# 145J-A Interactive Mission Cockpit

Date: 2026-08-04

## Direction

Similar OSS was surveyed before implementation. Foxglove's 3D panel emphasizes
selection, measurement, click tools, and timestamp synchronization; RViz
organizes displays, tools, panels, and pluggable frame transformation; Open3D
demonstrates browser/Jupyter 3D visualization. SpatialRust already owns the
portable viewer, explicit wgpu picking/readback, synchronized timestamps, ROS
PointCloud2 receipts, and edge-partition receipts. The next useful slice was
therefore a source-bound interaction surface that composes those contracts
without copying an OSS implementation or adding a UI dependency to core.

References:

- https://docs.foxglove.dev/docs/visualization/panels/3d
- https://docs.foxglove.dev/docs/extensions
- https://github.com/ros2/rviz
- https://www.open3d.org/html/tutorial/visualization/web_visualizer.html

## Implementation

`spatialrust-viewer::MissionCockpitState` keeps a versioned, fail-closed
contract for expected topic/frame identities, packet frames, source-indexed
XYZ samples, timeline bounds, visual layers, execution nodes/links, artifacts,
and separate publish/partition/mapping admissions. Each frame is capped at
`MISSION_COCKPIT_MAX_SAMPLED_POINTS`; the sample indices remain tied to the
original packet so browser selection can report exact source indices.

`rosbag2_mission_cockpit` consumes:

- `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145h-live-publish-v2/live-publish.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145i-edge-partition-v2/edge-partition.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/143b-calibration-survey/canonical.calibration.readiness.json`

The generated HTML is self-contained: it has bounded 3D orbit/zoom, timeline
playback, point-cloud layer toggles, point selection, shift-click distance
measurement, packet metadata, and the edge-to-host graph. No full sensor dump
or derived sensor asset is committed to the repository.

## External evidence

Positive:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-mission-cockpit-v2/mission-cockpit.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-mission-cockpit-v2/mission-cockpit.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-mission-cockpit-v2/mission-cockpit.manifest.json`

Negative:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-mission-cockpit-validation-probe-v2/mission-cockpit.json`

Positive state: four frames, 115,972 source points, 768 bounded samples,
four completed transfers, and `mapping_admitted:false` because calibration is
not registered. The negative state has zero admitted frames and exits 2.
