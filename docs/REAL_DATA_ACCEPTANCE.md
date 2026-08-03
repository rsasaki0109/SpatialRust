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

## 143B calibration/frame readiness inventory

The feature-gated `rosbag2_calibration_readiness` example now records the
calibration boundary before any frame-aware mapping is admitted. It inventories
the canonical bag's requested front/rear PointCloud2 topics and the relevant
`/clock`, `/tf`, `/tf_static`, and `/odom` topic names. Optional clock and frame
artifacts can be registered only through absolute paths; registration records
the external file size and SHA-256. The gate deliberately does not interpret an
artifact format, apply a clock correction, or invent a `FrameGraph` edge.

The canonical run wrote
`/media/sasaki/aiueo/spatialrust-results/v1-3/143b-calibration-readiness/rosbag2.calibration.readiness.json`.
Both sensor topics were present and supported (768 front messages and 767 rear
messages), while `/clock`, `/tf`, `/tf_static`, and `/odom` were absent. No
clock or frame artifact was registered, so `registration_ready` is `false` and
the command exits non-zero after atomically preserving the blocker receipt.
This is an intentional fail-closed result: the two sensor frames remain
separate and no calibration-aware mapping claim is made.

### Separate-source TF parser evidence

An additional read-only inventory was run against
`/media/sasaki/aiueo/datasets/migrated/autoware_data/all-sensors-bag1/all-sensors-bag1_compressed_0.db3`,
whose SHA-256 is
`74e5915719a7b7b4820b5339207eeade0c656deaa38b8e5b5e8d18787a58ac22`. This is
not the canonical 2020 capture: it is a separate 2022 sensor fixture with
`/tf_static`, `/tf`, and `velodyne_*` frame names. The source-bound receipt
decoded one `/tf_static` message containing 14 transforms, observed all four
requested Velodyne frames, and passed its own identity and truncation checks:

`/media/sasaki/aiueo/spatialrust-results/v1-3/143b-tf-inventory/all-sensors-bag1.tf-static.inventory.json`

The guard was also exercised with the canonical bag and the fixture SHA as the
expected identity. It wrote
`/media/sasaki/aiueo/spatialrust-results/v1-3/143b-tf-inventory/canonical-wrong-source.tf-static.inventory.json`
and exited non-zero because the observed canonical SHA is
`b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`. This
evidence validates the TF parser and source binding only; it does not register
clock or front/rear calibration for the canonical input.

### Canonical calibration artifact survey

On 2026-08-03, the external SSD was searched read-only for artifacts matching
the canonical bag identity. The canonical directory
`/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/`
contains only the bag, its `metadata.yaml`, and SQLite WAL/SHM sidecars. The
companion archive
`/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2-astuff-1-lidar-only.tar.gz`
contains only the same bag directory, metadata, and DB3 payload. The migration
manifest
`/media/sasaki/aiueo/datasets/migrated/autoware_data-migration-20260802.sha256`
contains no clock, TF, extrinsic, or frame-calibration artifact for this input.
An external filename/content search for the canonical bag name, starting
timestamp, and `/lidar_front/points_raw`/`/lidar_rear/points_raw` identifiers
found no additional source candidate outside the existing SpatialRust result
receipts.

The refreshed fail-closed readiness receipt is
`/media/sasaki/aiueo/spatialrust-results/v1-3/143b-calibration-survey/canonical.calibration.readiness.json`.
It re-hashed the canonical input as
`b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`, confirmed
768 front and 767 rear PointCloud2 messages, confirmed that `/clock`, `/tf`,
`/tf_static`, and `/odom` are absent, and exited non-zero with both calibration
artifacts `not_registered`. No candidate was registered or treated as usable
for mapping. The separate `all-sensors-bag1` receipt remains diagnostic only.

## 145A source-bound Spatial Studio dashboard

`rosbag2_studio` creates a portable JSON state and a static HTML dashboard from
the canonical bag plus explicit readiness, TF, and E2E receipts. The viewer
crate owns only renderer-independent state; ROS/SQLite parsing remains in the
feature-gated example. The dashboard joins the point-cloud/derived layer
inventory, timeline, calibration gate, frame inventory, and explicit
performance counters on one screen.

The external evidence is:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145a-spatial-studio-v2/spatial-studio.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145a-spatial-studio-v2/spatial-studio.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145a-spatial-studio-v2/spatial-studio.manifest.json`

The state records four renderable receipt-backed layers, 1,535 timeline
samples, the uncalibrated PointCloud2 header-stamp basis, the 66.7-second
performance receipt, and zero hidden device copies. It remains
`mapping_admitted:false`: the canonical readiness receipt has no registered
clock/frame artifacts, while the supplied TF inventory has a different
expected input SHA and is rejected before any frame edge enters the graph.
This is an inspection/dashboard acceptance slice, not acceptance of fused
world mapping.

## 145B TF / Calibration Observatory evidence

rosbag2_calibration_observatory exposes calibration artifact status, clock
diagnostics, source-bound TF edges, graph topology, and composition state in a
portable JSON/HTML observatory. It does not interpret opaque calibration
files, apply a clock model, compose TF, or accept edges from another input
identity.

The external evidence is:

- /media/sasaki/aiueo/spatialrust-results/v1-3/145b-tf-calibration-observatory/calibration-observatory.json
- /media/sasaki/aiueo/spatialrust-results/v1-3/145b-tf-calibration-observatory/calibration-observatory.html
- /media/sasaki/aiueo/spatialrust-results/v1-3/145b-tf-calibration-observatory/calibration-observatory.manifest.json

The canonical state has an exact input SHA match, but both calibration
artifacts are not_registered, the clock model is not applied, the TF
inventory has a different expected input SHA, zero edges are accepted, and
calibration_admitted:false. The HTML panel renders NO ACCEPTED EDGES and
the source mismatch as blockers. This is observability evidence; it does not
change the blocked mapping acceptance row.

## 145C one-command deterministic replay evidence

`rosbag2_replay_demo` provides a bounded, one-command replay trace for the
canonical bag. It requires an absolute input and output path plus the exact
expected input SHA-256, retains two records per canonical lidar topic, verifies
the deterministic replay order with a second walk, and writes a portable JSON
state, static HTML dashboard, and checksummed manifest. It uses the
PointCloud2 header stamp in the explicit `ros2-external` domain; it does not
apply clock calibration, TF composition, or front/rear fusion.

The external evidence is:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145c-one-command-replay-demo/replay-demo.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145c-one-command-replay-demo/replay-demo.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145c-one-command-replay-demo/replay-demo.manifest.json`

The observed input SHA-256 exactly matches
`b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`. The
bounded episode contains four records and 115,972 points across the front and
rear topics, with two matched bundles. The maximum matched timestamp delta is
92,435,944 ns inside the configured 100,000,000 ns window, and deterministic
order verification passed. The manifest re-hashes three local entries totaling
713,683,963 bytes; the output directory was created on the external SSD after
a free-space preflight.

The resulting state is `replay_ready:true` and
`mapping_admitted:false`. `calibration_applied:false` remains visible, and the
dashboard lists the missing clock calibration, unapplied TF/frame composition,
and source-bound calibration requirement as blockers. This is a replay and
inspection acceptance slice, not acceptance of calibrated or fused-world
mapping.

## 145D source-bound TSDF/glTF Map Diff evidence

`rosbag2_map_diff` compares two existing E2E run directories without reopening
the rosbag2 source. It re-validates both manifests, decodes the embedded
SpatialRust glTF position/index buffers, requires the configured canonical
input SHA and matching map frame, and computes stable-index displacement,
added/removed vertex counts, topology/hash equality, and a bounded 16×16
spatial heatmap. It never transforms either map or fuses uncalibrated frames.

The primary external evidence is:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145d-map-diff-v2/map-diff.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145d-map-diff-v2/map-diff.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145d-map-diff-v2/map-diff.manifest.json`

The base map is the 143A two-record canonical smoke at
`/media/sasaki/aiueo/spatialrust-results/v1-3/143a-e2e-smoke/`; the candidate
map is a fresh three-record canonical run at
`/media/sasaki/aiueo/spatialrust-results/v1-3/145d-map-diff-candidate/`. Both
input manifest entries re-hash to
`b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8` and both
maps use `lidar_front` coordinates. The candidate contains 1,074,930 vertices
and 358,310 triangles versus 1,064,304 vertices and 354,768 triangles in the
base.

With a 1,000 µm change threshold, the diff compared 1,064,304 stable-index
vertices, found 1,063,615 changed vertices and 10,626 candidate-only vertices,
and populated 143 of 256 heatmap cells. Maximum displacement was 121,954,730
µm, mean displacement 25,504,301 µm, and p95 displacement 82,791,440 µm. The
map hashes and decoded topology are different, so the dashboard visibly
renders a non-zero change surface. The manifest re-hashes nine local entries
totaling 759,440,844 bytes.

The state is `compare_ready:true` but `mapping_admitted:false`: the time basis
is the uncalibrated PointCloud2 header stamp, TF/frame composition is not
applied, and calibrated source-bound map evidence is still required. A
same-hash 143A-versus-144 performance comparison under
`/media/sasaki/aiueo/spatialrust-results/v1-3/145d-map-diff-identical-smoke/`
also produced zero changed vertices. The negative source-binding probe at
`/media/sasaki/aiueo/spatialrust-results/v1-3/145d-map-diff-validation-probe/map-diff.json`
withholds all comparison cells and reports `compare_ready:false` when the
expected input identity is wrong.

## 145E AI Semantic Overlay evidence

`rosbag2_semantic_overlay` reads the existing receipt-backed 143A glTF map
without reopening or modifying the canonical DB3. It requires the exact
canonical input SHA-256 and an expected frame, samples at most 4,096 map
vertices, and feeds explicit CPU-owned `[1,4,N]` features
`(normalized x, y, z, horizontal radius)` into the deterministic
`spatialrust-ai` `MockProfile::SemanticClasses`. The mock profile is a
visualization fixture, not a learned ontology or production model.

The primary external evidence is:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145e-ai-semantic-overlay/semantic-overlay.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145e-ai-semantic-overlay/semantic-overlay.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145e-ai-semantic-overlay/semantic-overlay.manifest.json`

The run checked the canonical SHA-256
`b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`, matched
the `lidar_front` frame, and produced `overlay_ready:true` for 4,094 bounded
predictions over 1,064,304 input vertices. All three declared classes were
represented: `ground` 497, `structure` 3,096, and `object` 501. Mean
confidence was `797875` millionths and p95 was `960000` millionths. The model
receipt records 65,504 input host bytes, 32,752 output host bytes, and zero
device upload/readback bytes. The manifest re-hashes six local entries totaling
737,996,457 bytes.

`mapping_admitted:false` is intentional. The state and dashboard expose the
header-stamp time basis, missing clock calibration, unapplied TF/frame
composition, visualization-only deterministic mock, and missing
source-bound production model receipt as blockers. No unapproved conversion or
fusion is accepted as mapping evidence.

The negative source-binding probe is retained at
`/media/sasaki/aiueo/spatialrust-results/v1-3/145e-ai-semantic-overlay-validation-probe/semantic-overlay.json`.
It used a deliberately wrong 64-character SHA, emitted zero predictions,
recorded `runtime:"not run: source/frame admission failed"`, set
`overlay_ready:false`, wrote its six-entry manifest, and exited with status 2.

## 145F glTF/USD Digital Twin evidence

`rosbag2_digital_twin` consumes the existing receipt-backed 143A glTF map
without reopening the canonical DB3. It requires the exact canonical input
SHA-256 and expected frame, copies the accepted glTF payload byte-for-byte,
and writes an ASCII USDA companion/reference layer that points explicitly to
`@digital-twin.gltf@`. The slice intentionally does not add an OpenUSD
runtime, reinterpret arbitrary USD geometry, or apply an implicit calibration
transform. An optional semantic overlay is attached only after its own
source/frame state has passed the same identity checks.

The positive external evidence is:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin/digital-twin.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin/digital-twin.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin/digital-twin.gltf`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin/digital-twin.usda`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin/digital-twin.manifest.json`

The canonical input SHA-256 is
`b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`, and the
observed frame is `lidar_front`. The bundle preserves 1,064,304 vertices and
354,768 triangles. The copied glTF SHA-256 is
`94a2d1405d392bed35182ecd2a69aba80cda3891562799904966fb1350bd1330`, equal
to the 143A source map by byte comparison. The USDA receipt is 631 bytes and
contains the canonical SHA, frame, source time basis, geometry counts, and
the explicit glTF asset reference. The 145E semantic receipt is attached as a
source-bound auxiliary layer. The manifest re-hashes nine local entries
totaling 760,092,196 bytes.

The state is `twin_ready:true` and `mapping_admitted:false`. The dashboard
keeps the header-stamp time basis, absent clock calibration, unapplied TF/frame
composition, and inspection-only USDA companion semantics visible as
blockers/notices. No unapproved conversion, calibration, or fusion is treated
as acceptance evidence.

The negative source-binding probe is retained at
`/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin-validation-probe/digital-twin.json`.
It used a deliberately wrong expected SHA, emitted no glTF or USDA files,
withheld the semantic layer, wrote a seven-entry manifest with
`twin_ready:false`, and exited with status 2.

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

## 144B repeated-run comparison evidence

`rosbag2_e2e_compare` compares at least two absolute receipt paths without
reopening the bag. It rejects mixed receipt versions, inputs, topics, limits,
or resume modes; compares the canonical input, persisted episode, and glTF
hashes; and writes an atomic non-overwriting comparison receipt.

Three sequential `fresh-source-ingest` runs under
`/media/sasaki/aiueo/spatialrust-results/v1-3/144-performance-fresh`,
`/media/sasaki/aiueo/spatialrust-results/v1-3/144-performance-fresh-2`, and
`/media/sasaki/aiueo/spatialrust-results/v1-3/144-performance-fresh-3` all
matched the input/episode/glTF hashes. The observed-pipeline samples were
66,701,571,887, 67,060,768,484, and 68,684,129,437 ns; median was
67,060,768,484 ns, p95/max was 68,684,129,437 ns, and coefficient of variation
was 1.28%. All bounded-smoke stage budgets passed, including the 70 s ICP,
5 s TSDF, 2 s interchange, and 80 s pipeline ceilings. Memory observations
and zero-transfer counters were stable across all three runs.

The comparison receipt is
`/media/sasaki/aiueo/spatialrust-results/v1-3/144-comparison-smoke-v2/rosbag2.e2e.comparison.json`.

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
| Calibration/frame readiness | External inventory receipt for required topics and registered calibration artifacts | Recorded as 143B blocker; not ready |
| Source-bound TF inventory | Exact input SHA, bounded `/tf_static` CDR decode, and required-frame receipt | Accepted as separate-source parser evidence; not canonical calibration |
| Canonical artifact availability | Clock and front/rear frame artifacts matching the canonical input identity | Blocked; read-only SSD survey found no matching artifacts |
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

1. Provide and register clock calibration and front/rear extrinsic artifacts for
   the canonical input; the separate `all-sensors-bag1` TF receipt cannot
   satisfy this gate, and the latest survey receipt records both as missing.
2. Extend the 143A prefix smoke to a bounded full-bag, frame-aware odometry and
   TSDF run before adding any semantic model runtime.
3. Add semantic, Viewer, and interchange quality receipts only after the
   geometric output has a valid frame and provenance chain.
