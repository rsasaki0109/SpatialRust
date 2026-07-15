# Epic 109: bounded spatial execution graph

Epic 109 composes the existing distribute DAG, watermark, and named-transfer
contracts with runtime operators. Compilation rejects cycles, multiple sources,
and every cross-device edge lacking an explicit transfer. Fusion groups contain
only fusable linear neighbors on the same placement.

The conformance graph fuses `decode -> gray`, retains GPU `infer` as its own
group, executes `10 -> 17`, and records exactly one named `gray-upload` copy of
1024 bytes. A hard-limit-one source queue accepts one value and rejects the
second without unbounded buffering.

On the reference Windows host, the eight-stage fused CPU graph measured
`3.8234–4.0696 µs` per submitted value (Criterion, 10-sample short receipt).
