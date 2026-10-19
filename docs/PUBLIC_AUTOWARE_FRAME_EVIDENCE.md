# Public canonical Autoware frame evidence

## Scope and fixed protocol

This named Epic 141C slice restores the exact public canonical rosbag and
registers the publisher's front/rear LiDAR configuration. It does not measure
physical calibration accuracy or sensor-clock offset/drift. Overall calibrated
mapping must remain blocked while clock evidence is missing.

The protocol and adapter are committed before the native acceptance run. The
source is the official [Autoware localization demo](https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/rosbag-localization-howto.html).
Its archive is [the public LiDAR-only bag](https://autoware-auto.s3.us-east-2.amazonaws.com/rosbag2/rosbag2-astuff-1-lidar-only.tar.gz).
The [publisher benchmark instructions](https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/benchmark-tool-nodes-lidar-localization.html)
describe recorded AS parking-lot data and the exact extracted bag name.
The data license is not established by these download links; this repository
redistributes neither the archive nor derived sensor records.

| Identity | Bytes | SHA-256 |
| --- | ---: | --- |
| Public archive | 335328022 | `317f5d3b249bb81fdb00e259552ebc6c1c21b8016a4c2491f5cc5889aa9746ac` |
| Canonical SQLite bag | 713670656 | `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8` |
| Publisher metadata | 611 | `92d3782d4a093d3de3bdfbdb80e151a528982487e0f3fd39d4d2698351328e62` |

The original snapshot table omitted the final hex digit of the bag checksum.
The corrected 64-character identity agrees with the later source-binding
negative test and the newly acquired bytes; it does not change the fixture.

## Publisher association and direction audit

All six complete upstream files are pinned to Autoware.Auto commit
`15d2e91a2b3c8cd61b20b7e6e79da32919c2cbcd`. The adapter verifies their byte
sizes and SHA-256 values before parsing YAML. Upstream Python/C++ is never
executed. The fixed sources, public URLs and complete identities are in
`scripts/prepare_autoware_frame_evidence.py`; external provenance repeats them.

- [Launch association](https://gitlab.com/autowarefoundation/autoware.auto/AutowareAuto/-/blob/15d2e91a2b3c8cd61b20b7e6e79da32919c2cbcd/src/launch/autoware_demos/launch/localization_rosbag.launch.py): the default names the exact canonical bag and passes the same VLP16 Lexus parameter file to both named sensor nodes, remapping `points_in` to `points_raw`.
- [Published parameters](https://gitlab.com/autowarefoundation/autoware.auto/AutowareAuto/-/blob/15d2e91a2b3c8cd61b20b7e6e79da32919c2cbcd/src/perception/filters/point_cloud_filter_transform_nodes/param/vlp16_lexus_filter_transform.param.yaml): input frames are `lidar_front`/`lidar_rear`, output is `base_link`.
- [Filter node](https://gitlab.com/autowarefoundation/autoware.auto/AutowareAuto/-/blob/15d2e91a2b3c8cd61b20b7e6e79da32919c2cbcd/src/perception/filters/point_cloud_filter_transform_nodes/src/point_cloud_filter_transform_node.cpp) and [node header](https://gitlab.com/autowarefoundation/autoware.auto/AutowareAuto/-/blob/15d2e91a2b3c8cd61b20b7e6e79da32919c2cbcd/src/perception/filters/point_cloud_filter_transform_nodes/include/point_cloud_filter_transform_nodes/point_cloud_filter_transform_node.hpp): the configured numeric `.transform` goes directly to `StaticTransformer`; fallback TF lookup is `(output,input)`, and output points are labeled `base_link`. The helper's unused `TransformStamped` header labels must not invert the numeric values.
- [Transformer construction](https://gitlab.com/autowarefoundation/autoware.auto/AutowareAuto/-/blob/15d2e91a2b3c8cd61b20b7e6e79da32919c2cbcd/src/common/lidar_utils/src/point_cloud_utils.cpp) and [application](https://gitlab.com/autowarefoundation/autoware.auto/AutowareAuto/-/blob/15d2e91a2b3c8cd61b20b7e6e79da32919c2cbcd/src/common/lidar_utils/include/lidar_utils/point_cloud_utils.hpp): the affine operation is `p_base_link = R(q) * p_lidar + t`, without inversion.

| Child sensor | Parent | Translation metres (xyz) | Quaternion (xyzw) |
| --- | --- | --- | --- |
| `lidar_front` | `base_link` | `[1.42849,-0.017811,1.4802]` | `[0.01423,0.0617607,0.0562799,0.9964014]` |
| `lidar_rear` | `base_link` | `[0.1,0.01,1.44803]` | `[-0.0732027,0.0004243,0.9973002,0.0057807]` |

The JSON keeps the raw rounded values. The existing native mapping consumer
normalizes its float32 quaternion. The upstream Eigen implementation accepts
near-unit input and forms a float32 matrix without explicit normalization;
bit-identical transformed coordinates are not asserted. No point filtering,
motion estimation, mapping, or TF convention elsewhere is changed.

The pinned later demo explicitly associates these values with this bag. This
supports publisher-configured provenance; it is not an independent metrology
report, a statement of physical residuals, or a measured uncertainty bound.

## Storage and preparation

All raw data, pinned upstream source files and generated JSON stay outside the
code workspace. The offline adapter requires absolute paths, an existing output
parent, a new run directory, exact source identities, all 768 front and 767 rear
headers, matching CDR frames, and nondecreasing timestamps per sensor. Header
reads are limited to 256 bytes; point payloads are not materialized. SQLite is
opened read-only/immutable. Hashing is streamed twice and source files are
rechecked before output admission.

A 20 GiB output-filesystem floor is checked before opening the bag and again
before creating the run directory. No bulk output or build is needed. The
provenance manifest records free bytes before work, at admission and after the
frame artifact; the later native-run receipt records final free space. Tests
cover low space, a falling free-space observation, and existing outputs.

Install the script-only dependency `PyYAML>=6,<7` (also in the Python test
extra); the runtime wheel and `spatialrust-core` have no new dependency.

Example local paths (public payloads are external):

```powershell
python C:\Users\rsasa\Workspace\SpatialRust\scripts\prepare_autoware_frame_evidence.py `
  --bag E:\SpatialRustData\maturity-80\autoware-astuff-2020\rosbag2_2020_09_23-15_58_07\rosbag2_2020_09_23-15_58_07.db3 `
  --upstream-dir E:\SpatialRustData\maturity-80\autoware-public-configs-v1 `
  --output-dir E:\SpatialRustData\maturity-80\autoware-canonical-frame-v1
```

`frame-evidence.json` uses the existing
`spatialrust.calibration.frame-evidence`, version 1, schema. Its absolute
source path, size and SHA-256 bind both `base_link`-parent edges to this bag.
`manifest.json` holds upstream/source/preparation provenance, full header
coverage and the evidence limitations. No clock artifact is generated.
Recording-minus-header delay and node timeouts do not independently measure
sensor clock offset, drift or uncertainty.

## Native acceptance expectations (fixed before execution)

Reuse the already-built release examples from
`E:\SpatialRustBuild\maturity-80-target\release\examples\` after confirming
their source has no difference from the parent. The run receipt records both
binary hashes and invocation/exit status. Existing gate conditions are retained.

1. `rosbag2_calibration_readiness` with the real bag and frame artifact must
   register the frame file, retain `clock=not_registered`, write its external
   receipt and exit nonzero with overall `registration_ready=false`.
2. `rosbag2_calibration_evidence` consumes that readiness receipt, the expected
   canonical SHA, root `base_link`, front `lidar_front` and rear `lidar_rear`.
   It must establish matching input identity and a source-bound two-edge ready
   frame graph, retain a blocked clock and blocked overall registration, write
   JSON/HTML/manifest externally and exit nonzero.
3. A separate frame JSON changing only its source SHA to all zeros must be
   rejected as `source_mismatch`, with no ready frame graph and blocked overall
   registration. Preserve the negative receipt; do not overwrite the valid run.
4. Rehash the bag, six sources, frame artifact, native executables and all
   listed receipt outputs; record storage and bytes added. Unit tests and
   complete hosted required checks must pass before merging.

## Execution results

Protocol/source commit: `3a9eec871a12bb03c864d85b4507bd1bdfe5d388`.
Run: 2026-10-19, with existing Windows release executables and Python 3.12.10,
PyYAML 6.0.3. Full result provenance and receipt identities are in
[the compact record](receipts/2026-10-19_public_autoware_frames.json).

| Check | Observed outcome |
| --- | --- |
| Canonical source identity | Exact size and complete SHA-256 match |
| Every front/rear header | 768/767 checked; expected frames and nondecreasing stamps |
| Publisher source configuration | All six complete file hashes match the pinned revision |
| Readiness inventory | Frame registered, clock not registered, overall false; exit 1 |
| Native frame graph | Two source-bound accepted edges and ready root paths |
| Native clock/overall | No samples or clock artifact; registration false; exit 2 |
| Wrong frame source SHA | `source_mismatch`, frame graph false, overall false; exit 2 |
| Independent manifest rehash | All ten entries across the two native manifests match |
| Python regressions | 801 passed, three optional ONNX tests skipped; 35 new focused cases included |
| Python compatibility | Both new files parse with Python 3.8 syntax; hosted runtime checks still required before merge |

The audit runner originally expected exit 1 for the native evidence command;
the implementation uses exit 2 for its error result. Original outputs were
preserved and verified, then the runner resumed only the unexecuted wrong-source
probe. Gate conditions and the committed adapter/protocol did not change.

The 20 external result files total **72,203 bytes**, including run receipts and
the preserved partial control receipt. No native rebuild, bag copy or new bulk
download was needed for execution. Free bytes were 507,830,841,344 before and
512,271,282,176 after; shared-drive activity prevents attributing that difference
to this run. Use measured file sizes for its footprint.

Local output paths:

- `E:\SpatialRustData\maturity-80\autoware-canonical-frame-v1\frame-evidence.json`
- `E:\SpatialRustData\maturity-80\autoware-canonical-frame-v1\manifest.json`
- `E:\SpatialRustData\maturity-80\autoware-canonical-native-v1\readiness.json`
- `E:\SpatialRustData\maturity-80\autoware-canonical-native-v1\publisher-frame\calibration-evidence.json`
- `E:\SpatialRustData\maturity-80\autoware-canonical-native-v1\wrong-source\calibration-evidence.json`
- `E:\SpatialRustData\maturity-80\autoware-canonical-native-v1\run.json`

Missing clock evidence remains an expected retained blocker. Publisher-configured
frame paths and correct rejection do not establish physically calibrated full-bag
mapping, accurate motion, or completion of the operational 80% target.
