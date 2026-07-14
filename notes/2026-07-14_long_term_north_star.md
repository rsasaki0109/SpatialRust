# Long-term north star research note

Date: 2026-07-14

## Decision

Epic 83–90 remains the active foundation program. Epic 91–100 is reserved as a
successor program whose outcome is a Rust-native spatial-intelligence data plane:
synchronized capture, deterministic replay, localization and mapping, classical
and learned scene reconstruction, semantic spatial query, robotics execution,
and standards-based scene exchange.

The reservations deliberately start from contracts and data movement rather
than a list of model architectures. Model families change quickly; ownership,
time, coordinate frames, schema evolution, provenance, and reproducibility are
the durable platform requirements.

## Evidence used

- Apache Arrow's C Data Interface defines a small ABI-stable interface for
  zero-copy sharing across runtimes. Its C Device Data Interface extends the
  exchange model to buffers residing in device memory. This supports an
  optional spatial record/stream boundary without adding Arrow to the core.
  <https://arrow.apache.org/docs/format/CDataInterface.html>
  <https://arrow.apache.org/docs/format/CDeviceDataInterface.html>
- ROS 2 REP-2007 formalizes custom-type adaptation partly to avoid unnecessary
  conversions, and REP-2009 formalizes format negotiation between publishers
  and subscriptions. These match SpatialRust's capability and explicit-copy
  principles.
  <https://ros.org/reps/rep-2007.html>
  <https://www.ros.org/reps/rep-2009.html>
- MCAP is a container for timestamped pub/sub messages with indexes for time and
  topic lookup, making it a suitable optional boundary for deterministic sensor
  episode capture and replay.
  <https://mcap.dev/spec>
- glTF is optimized for efficient runtime delivery of 3D scenes and models,
  while OpenUSD targets scalable composition and interchange of complex scenes.
  They serve different adapter roles and should not be core storage models.
  <https://www.khronos.org/gltf/>
  <https://openusd.org/dev/intro.html>
- The original 3D Gaussian Splatting work demonstrates a point-adjacent,
  anisotropic Gaussian representation with real-time visibility-aware rendering.
  It is important enough to reserve an optional scene representation, but not
  mature or universal enough to replace meshes, surfels, or TSDF volumes.
  <https://arxiv.org/abs/2308.04079>

## Guardrails

- Epic 91–100 does not expand the scope or completion criteria of Epic 83–90.
- External standards and heavy runtimes live in dedicated, feature-gated crates.
- Device and network transfers are explicit graph operations with observable
  byte counts, latency, and synchronization.
- Learned outputs carry model identity, preprocessing contract, confidence or
  uncertainty where available, source timestamps, and coordinate frames.
- Reproducible episode replay and conformance tests precede distributed autonomy.
