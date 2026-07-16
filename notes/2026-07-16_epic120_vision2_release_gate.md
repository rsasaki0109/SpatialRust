# Epic 120 Vision 2 release gate — 2026-07-16

## Scope

Epic 120 closes the Vision 2 performance program with a fail-closed typed gate.
It extends the Vision 1 stability surface without changing stable ownership or
CPU/GPU placement contracts.

`Vision2ReleaseGate` requires eleven named conformance cases, six dated receipt
families, the runnable `vision_2_release_gate` example, the four-item security
baseline, and the `vision-2` migration policy. Missing, skipped, failed,
duplicated, or over-budget evidence is denied.

## Typed budgets

The canonical evidence uses already-recorded 2026-07-16 measurements from the
Intel Core i7-9750H Windows host:

| Dimension | Observed | Ceiling |
| --- | ---: | ---: |
| Native RGB-to-gray allocate, 1080p | 648 us | 1,000 us |
| Native RGB-to-gray reuse, 1080p | 195 us | 400 us |
| Python RGB-to-gray allocate, 1080p | 825 us | 1,200 us |
| Python RGB-to-gray reuse, 1080p | 232 us | 400 us |
| Peak accounted CPU bytes | 6,220,800 | 64 MiB |
| Caller-output steady-state allocations | 0 | 0 |
| Default worker policy | 12 | 12 |
| 4K GPU source upload | 33,177,600 bytes | 33,177,600 bytes |
| Resident GPU readback before request | 0 bytes | 0 bytes |

Latency values come from
`C:\Users\rsasa\Workspace\SpatialRust\notes\2026-07-16_rgb_gray_acceleration.md`.
CPU memory/thread accounting comes from
`C:\Users\rsasa\Workspace\SpatialRust\notes\2026-07-16_vision2_baseline.md`.
GPU transfer accounting comes from
`C:\Users\rsasa\Workspace\SpatialRust\notes\2026-07-16_epic119_gpu_vision_chain.md`.

These ceilings apply only to the named release workloads. They are not
portable performance guarantees or blanket OpenCV claims.

## Cross-platform and documentation delivery

The existing Linux/Windows/macOS Vision conformance matrix now runs the Vision
2 denial tests and receipt example. The Pages workflow generates
`vision2-release.md` from the runnable example, while README, API stability,
the Vision 2 page, and `docs/VISION_2_MIGRATION.md` link the policy and evidence.

## Verification

```powershell
cargo test -p spatialrust-platform
cargo clippy -p spatialrust-platform --all-targets -- -D warnings
cargo run -p spatialrust --no-default-features --features platform --example vision_2_release_gate
cargo doc -p spatialrust-platform --no-deps
```

The focused denial tests cover complete evidence, missing/skip/duplicate
evidence, migration mismatch, generated Markdown, and each resource dimension
over its inclusive ceiling.
