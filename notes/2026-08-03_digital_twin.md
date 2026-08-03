# 145F glTF/USD Digital Twin

Date: 2026-08-03

## Scope

This slice makes the existing receipt-backed SpatialRust map presentable as a
portable Digital Twin bundle while keeping the stable viewer crate small. The
new `DigitalTwinState` records source/frame identity, glTF and USDA assets,
geometry identity, optional semantic attachment, and independent twin versus
mapping gates.

`rosbag2_digital_twin` reads the external 143A E2E receipt and manifest,
decodes the SpatialRust glTF for count verification, and copies the accepted
glTF bytes without modification. It writes a small ASCII USDA companion
layer containing source/frame/time/count metadata and an explicit
`@digital-twin.gltf@` asset reference. It does not introduce an OpenUSD
runtime, hide a device transfer, or apply an unregistered transform.

## External evidence

Canonical input:

- `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`
- SHA-256: `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`

Positive bundle:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin/digital-twin.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin/digital-twin.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin/digital-twin.gltf`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin/digital-twin.usda`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin/digital-twin.manifest.json`

Observed positive state:

- `twin_ready=true`, `mapping_admitted=false`
- frame `lidar_front`
- 1,064,304 vertices and 354,768 triangles in source, glTF, and USDA metadata
- glTF byte-identical to the 143A map; SHA-256
  `94a2d1405d392bed35182ecd2a69aba80cda3891562799904966fb1350bd1330`
- 145E semantic overlay attached after source/frame validation
- manifest: 9 local files, 760,092,196 bytes re-hashed

Negative probe:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145f-digital-twin-validation-probe/digital-twin.json`
- wrong expected SHA, no glTF/USDA output, semantic layer withheld
- `twin_ready=false`, CLI exit status 2

## Verification

Focused checks passed:

- `cargo fmt --all -- --check`
- `cargo test -p spatialrust-viewer digital_twin --features serde`
- `cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_digital_twin`
- positive/negative external runs with manifest re-hashing
- byte comparison of the positive glTF against the 143A map

The remaining mapping blocker is intentional: the source uses PointCloud2
header stamps, and no source-bound clock calibration or TF/frame composition
has been registered. The USDA layer is therefore a portable visualization and
asset-reference surface, not calibrated-world mapping evidence.
