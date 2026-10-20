# Rooted pose-graph propagation

Slice 146X fixes the propagation contract before adding drift correction.
The [prospective protocol](../benchmarks/pose-graph-rooted-localization-plan.json)
retains the existing `to_T_from * from_T_anchor` algebra and all timestamps.
Root-reachable non-loop constraints must form a directed tree; root and
disconnected records remain fixed. Complete isometries propagate without a
translation-only convergence threshold. Validation is atomic on failure.

Independent public-API regression tests will run against the original source
before implementation. Native default/all-feature, scan/integration, Clippy,
WASM and formatting checks follow. Existing real-motion evidence and all
operational 80% requirements remain unchanged. Results are pending.
