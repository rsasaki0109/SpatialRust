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

Pending execution of this committed protocol. Missing clock evidence is an
expected retained blocker, not a successful calibrated mapping result.
