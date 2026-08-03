# SpatialRust v1.3 real-data acceptance contract

Status: active baseline, observed 2026-08-03. This document defines the
external-data acceptance boundary for the post-Epic-140 operational program;
it does not claim that the full downstream mapping or semantic pipeline has
already passed.

## Scope and storage boundary

The canonical source is read-only and remains outside the repository. Source
and derived sensor data must not be copied into
`/home/sasaki/workspace/SpatialRust` or committed to Git.

| Role | Canonical path |
| --- | --- |
| Code workspace | `/home/sasaki/workspace/SpatialRust` |
| Input root | `/media/sasaki/aiueo/datasets/migrated` |
| Result root | `/media/sasaki/aiueo/spatialrust-results` |
| Canonical input | `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3` |
| Input metadata | `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/metadata.yaml` |

New runs should use a run-scoped directory of the form
`/media/sasaki/aiueo/spatialrust-results/v1-3/<run-id>/`. Each run must write a
manifest before it is considered accepted. The manifest includes every input,
output, receipt, and temporary-spool survivor with role, byte size, and
SHA-256. Temporary files are confined to the result root and must be removed
or listed explicitly when a run fails.

The initial operational free-space floor is 20 GiB on the result filesystem.
A preflight check must fail before opening the bag when the floor is not met,
and every run receipt records available bytes before and after execution. The
2026-08-03 observation was 155 GiB available on `/media/sasaki/aiueo` and
52 GiB available on the workspace filesystem; the latter is not an output
target.

The `storage-preflight` feature exposes `StoragePreflight::check` and
`StorageRoots::preflight_output`. Both require an existing absolute directory
and reject the run before source admission when the requested floor is not
available. The rosbag2 batch example exposes this as
`--min-output-free-bytes`; `--verify-manifest` re-hashes all local manifest
entries before accepting the generated manifest.

## Canonical input snapshot

The rosbag2 SQLite snapshot is:

| Field | Value |
| --- | --- |
| Storage | `sqlite3` |
| Size | `713670656` bytes |
| SHA-256 | `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c` |
| Duration | `76854162986` ns |
| Starting timestamp | `1600901887524769241` ns since epoch |
| Total messages | `1535` |
| Supported PointCloud2 topics | `2` |
| Explicitly skipped non-PointCloud2 topics | `11` |
| Failed topics | `0` |

The inventory and batch receipt evidence is retained at
`/media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/batch-inventory.json` and
`/media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/batch-provenance/rosbag2.ingest.receipt.json`.

## Topic baseline

| Topic | Messages | Schema | Chunks | Points | Peak tracked bytes |
| --- | ---: | --- | ---: | ---: | ---: |
| `/lidar_front/points_raw` | 768 | `ros2.sensor_msgs.msg.PointCloud2.xyzi` | 1536 | 22,266,624 | 993,608 |
| `/lidar_rear/points_raw` | 767 | `ros2.sensor_msgs.msg.PointCloud2.xyzi` | 1534 | 22,237,631 | 993,608 |

The corresponding batch-provenance LAS outputs are:

| Output | Size | SHA-256 |
| --- | ---: | --- |
| `/media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/batch-provenance/topic-2-lidar_front_points_raw.las` | 445,332,707 bytes | `0c7fff23b82bdd94b5a9af2ad858190269509df3973b5ed95f2b99fa00121700` |
| `/media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/batch-provenance/topic-8-lidar_rear_points_raw.las` | 444,752,847 bytes | `b77be6f9fa959a877f3c0a2738d18d72855259c50dce97949738d4e96c4c7189` |

The equivalent `batch-ingest` outputs have the same two hashes. This is the
current byte-level deterministic output evidence for identical input and
conversion settings.

## Bounded synchronization baseline

The existing bounded preview uses eight records per topic, a 100 ms matching
window, 256 MiB source and episode byte budgets, and a two-million-point
episode limit. Its receipt is
`/media/sasaki/aiueo/spatialrust-results/rosbag2-sync-preview.receipt.json`.

Observed values:

- 16 retained records and 463,888 retained points;
- 7,422,208 retained episode bytes;
- 8 matched bundles;
- maximum matched delta `92,435,944` ns;
- front/rear frames remain `lidar_front`/`lidar_rear`.

The preview treats PointCloud2 header stamps as one external ROS clock domain.
It performs no clock calibration and invents no front/rear extrinsic transform.
This is accepted as a bounded synchronization smoke test only, not as a fused
mapping result.

## 141B preflight and manifest smoke evidence

The read-only inventory smoke confirmed the preflight before opening the bag:

```text
root=/media/sasaki/aiueo/spatialrust-results
available_bytes=163900334080
required_free_bytes=21474836480
```

The bounded front-topic run used
`--output-root /media/sasaki/aiueo/spatialrust-results`,
`--output-dir v1-3/141b-smoke`,
`--min-output-free-bytes 21474836480`, and `--verify-manifest`. It produced
the external receipt and manifest under
`/media/sasaki/aiueo/spatialrust-results/v1-3/141b-smoke/`, converted 768
messages into 1,536 chunks and 22,266,624 points, and re-hashed four local
manifest files totaling 1,159,004,968 bytes. No repository data was created.

## 142A checkpoint and resume smoke evidence

The E2E runner now writes an atomic, run-scoped control-plane checkpoint at
`/media/sasaki/aiueo/spatialrust-results/v1-3/142a-checkpoint-smoke/rosbag2.e2e.checkpoint.json`.
The real-data smoke advanced through ingest, synchronization, odometry, TSDF,
interchange, Viewer, receipt, manifest verification, and `complete`. The run
used the 20 GiB floor, observed 159,474,601,984 bytes before processing and
159,360,425,984 bytes after the pipeline, and left no checkpoint temp file.

Running the same run directory with `--resume` did not reopen the bag or
overwrite outputs. It loaded the complete checkpoint and re-hashed the three
manifested local files, totaling 736,379,864 bytes. Partial checkpoints fail
closed and remain available for audit; only the narrow atomic checkpoint temp
file is eligible for cleanup. The checkpoint is intentionally excluded from
the data manifest because it is advanced after the manifest and is control
plane state rather than a dataset payload.

## 142B partial-ingest resume smoke evidence

The E2E runner now persists the bounded `MemoryEpisode` immediately after
ingest as
`/media/sasaki/aiueo/spatialrust-results/v1-3/142b-ingest-resume-smoke-v2/rosbag2.e2e.episode.bin`
with an atomic ingest summary at
`/media/sasaki/aiueo/spatialrust-results/v1-3/142b-ingest-resume-smoke-v2/rosbag2.e2e.ingest.json`.
The binary checkpoint preserves the PointXYZ/PointXYZI schema family, ROS
clock stamp and quality, frame/timestamp metadata, sensor origin, and
`RecordProvenance`; the reader validates its magic, version, bounds, counts,
and trailing bytes before admitting records.

The real-data run used `--stop-after ingest`, then resumed the same directory
with `--resume` without reopening the DB3. It loaded four XYZI records and
115,972 points, matched two bundles, completed the same ICP/TSDF/semantic/
Viewer/glTF path, and re-hashed five local manifest entries totaling
738,238,733 bytes. The glTF hash remained
`94a2d1405d392bed35182ecd2a69aba80cda3891562799904966fb1350bd1330`.
Episode and ingest artifacts remain as auxiliary manifest survivors for audit
and a future deeper-stage resume; the run-scoped checkpoint remains excluded
because it is advanced after manifest creation.

## 143A bounded E2E smoke evidence

The bounded E2E example
`rosbag2_e2e` ran against the canonical input with two retained chunks per
topic under
`/media/sasaki/aiueo/spatialrust-results/v1-3/143a-e2e-smoke/`. The run
checked the 20 GiB floor before opening the bag and observed 162,144,653,312
bytes available before the pipeline and 162,025,279,488 bytes after the
pipeline stages. It retained four records and 115,972 points within the
256 MiB source/episode budgets, then matched two front/rear bundles with a
maximum delta of 92,435,944 ns.

The front-only, same-frame downstream smoke produced one bounded ICP motion,
integrated 57,986 points into a 200×200×80 TSDF, extracted 1,064,304 vertices
and 354,768 triangles, created two deterministic semantic record entities,
and validated two Viewer layers without device uploads. The embedded glTF
output is 22,705,791 bytes with SHA-256
`94a2d1405d392bed35182ecd2a69aba80cda3891562799904966fb1350bd1330`.
The manifest re-hashed the canonical input, glTF output, and E2E receipt as
three local entries totaling 736,379,566 bytes. The receipt and manifest are:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/143a-e2e-smoke/rosbag2.e2e.receipt.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/143a-e2e-smoke/rosbag2.e2e.manifest.json`

This is accepted as vertical contract smoke evidence only. It does not apply
clock calibration or a front/rear extrinsic, does not fuse the two frames, and
does not claim full-bag mapping or semantic model quality.

## 144A performance baseline evidence

Receipt version 2 adds an explicit `performance` section with a run mode,
stage wall-clock measurements, bounded memory observations, and host/device
transfer counters. The fresh-source smoke at
`/media/sasaki/aiueo/spatialrust-results/v1-3/144-performance-fresh/` retained
the same four records and 115,972 points. Its observed pipeline time was
68,684,129,437 ns (about 68.7 s): ingest 119,269,194 ns, synchronization
210,747 ns, ICP odometry 65,411,209,963 ns, TSDF 2,478,270,941 ns, glTF
interchange 669,718,884 ns, and semantic/Viewer 5,329,307 ns.

The configured source and episode budgets were 268,435,456 bytes each; the
retained episode was 1,855,552 bytes and peak source tracking was 993,608
bytes. Host-to-device, device-to-host, and hidden-device-copy counters were
all zero. The manifest re-hashed five local entries totaling 738,239,481
bytes. The receipt and manifest are:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/144-performance-fresh/rosbag2.e2e.receipt.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/144-performance-fresh/rosbag2.e2e.manifest.json`

## Acceptance gates

| Gate | Requirement | Current status |
| --- | --- | --- |
| Input identity | Exact input size, SHA-256, metadata, and topic inventory | Accepted |
| Topic scope | Two supported PointCloud2 topics, eleven explicit skips, zero failures | Accepted |
| Bounded ingest | 256 MiB configured memory ceiling and receipt peak below the ceiling | Accepted |
| Deterministic ingest | Repeated output trees produce the same per-topic LAS hashes and counts | Accepted |
| Output-root preflight | Absolute external result root and minimum free-space floor checked before bag access | Accepted |
| Run-scoped checkpoint | Atomic stage state, complete-run verification, persisted ingest artifact, and no-overwrite partial resume | Accepted as 142A/142B smoke |
| Performance receipt | Stage wall time, bounded memory observations, and explicit host/device transfer counters | Accepted as 144A smoke |
| Bounded sync | Eight-bundle smoke preview stays within the declared point/byte/time limits | Accepted |
| Bounded vertical E2E | rosbag2 → records → sync → ICP → TSDF → semantic → Viewer/glTF receipt | Accepted as 143A smoke only |
| Clock calibration | Calibrated clock model and uncertainty receipt | Not accepted; no calibration evidence |
| Frame calibration | Explicit front/rear `FrameGraph` path and extrinsic provenance | Not accepted; no extrinsic evidence |
| Mapping/reconstruction | Bounded odometry, pose graph, TSDF, and mesh receipt on this input | Not accepted; 143A prefix smoke only |
| Semantic/Viewer/interchange | Record semantic entities, Viewer layer, and glTF/OpenUSD receipt on this input | Not accepted; deterministic adapter smoke only |
| Data hygiene | No sensor or derived artifacts in the repository; all paths are external | Accepted |

Full v1.3 acceptance requires all rows to be accepted. Until clock and frame
calibration evidence exists, front/rear data may be inspected separately but
must not be reported as a fused world reconstruction.

## Required full-run receipt

Every future E2E run must record:

- input and output manifest entries with role, size, and SHA-256;
- repository commit, enabled feature set, command/configuration, host, and
  available result-disk bytes before/after;
- source topic/message/chunk/point counts, skipped and failed topics, and
  peak tracked bytes;
- clock-domain, calibration artifact identity, frame-graph path, and
  uncertainty/quality values;
- per-stage record/point/byte counts, dropped items with reasons, generated
  output hashes, and explicit host/device transfer bytes;
- deterministic run identity so a rerun can be compared without overwriting
  the prior result directory.

## Next implementation gates

1. Provide or register the clock calibration and front/rear extrinsic artifact.
2. Extend the 143A prefix smoke to a bounded full-bag, frame-aware odometry and
   TSDF run before adding any semantic model runtime.
3. Add semantic, Viewer, and interchange quality receipts only after the
   geometric output has a valid frame and provenance chain.
