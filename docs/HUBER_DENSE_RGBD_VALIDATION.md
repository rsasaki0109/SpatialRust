# Fixed Huber dense RGB-D fitting comparison

Slice 146V prospectively selects one explicit `ReferencePrecisionHuber` /
`reference-precision-huber` candidate before implementation and new fitting.
The [complete protocol](../benchmarks/huber-dense-plan.json) specifies numerical
order, units, finite admission, legacy/ownership/memory, exact data and unchanged
full acceptance, verification and storage boundaries.

Use independent scalar Huber IRLS factors with delta 0.02 for normalized gray
and one third of the original depth gate (0.01m here). These are engineering
priors; the room audit does not establish outliers or an error cause. All six
old choices/defaults and original raw final admission remain. One common fixed
candidate is tested across all four already observed development recordings;
all four complete trajectories must be frozen before any score.

No post-score parameter changes or per-scene candidate selection are allowed.
Room/unobserved motion, both raw Linux hardware classes, canonical measured
clock/physical accuracy and full canonical mapping/semantic/release acceptance
remain required for operational 80%. Implementation and results are pending.
