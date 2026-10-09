# Controlled convergence and failure study

567 native ICP runs: clean volume, Gaussian noise (sigma .005 base units), 20%
source replacement outliers, 60% partial source, and a 180-point symmetric ring;
scales .001, 1 and 1000; controlled prior errors 0, 20 and 40 degrees; default,
scale-aware transform and budget-only stopping. Volume conditions use five seeds;
ring uses one fixture. All native runs completed. Each prior also includes a
fixed small translation perturbation. Gate .3 and evaluation distance .05 are
scaled with geometry. There is no voxel stage in this focused ICP-policy study.

Generating truth is used to construct controlled priors and assess final output,
never to select a policy or candidate. Recovery requires rotation error <1 degree
and translation error <.01 base units. A stopped-but-not-recovered result is
counted as false convergence for this study; ring disagreement is intrinsically
ambiguous, not observable physical pose error.

| Scale | Policy | Recovery /63 | Stopped without recovery /63 | Iteration-limit exits /63 |
| --- | --- | --- | --- | --- |
| .001 | defaults | 20 | 43 | 0 |
| .001 | scaled_transform | 45 | 18 | 0 |
| .001 | budget_only | 45 | 0 | 63 |
| 1 | defaults | 45 | 14 | 19 |
| 1 | scaled_transform | 45 | 18 | 0 |
| 1 | budget_only | 45 | 0 | 63 |
| 1000 | defaults | 45 | 2 | 38 |
| 1000 | scaled_transform | 45 | 18 | 0 |
| 1000 | budget_only | 45 | 0 | 63 |

The default absolute fitness threshold 1e-6 squared coordinate units is sensitive
to scale: on .001-scaled data it accepts early residuals that still correspond to
the wrong generating pose. Scale-aware transform stopping recovers 45/63 in each
scale, but scaled_transform and budget_only still have 18/63 nonrecovered runs
at every scale. Disabling
stopping makes the false-convergence count zero by definition; it does not repair
those poses. Default policies at larger scales can avoid false-convergence labels
by exhausting the budget instead, without increased recovery. The ring recovers
its generating pose for only 1/3 priors under every policy, despite high proximity
support. These are controlled-fixture observations, not universal rankings.

The standalone HTML has stacked outcome fractions plus environment/prior tables.
study.json contains every update, effective thresholds, errors, forward/reverse
support, native extension hash, NumPy version and source hashes. Source hashes
were verified after the final run. Output directories are exclusively created.
CI now runs the full study and retains JSON/HTML for 14 days; a smaller study
contract is part of the binding tests. Remote CI status remains unconfirmed.

Validation: all 280 Python tests pass; native registration 23 CPU-feature tests,
strict Clippy and Python stubtest passed during this milestone. The new end-to-end
study tests cover truthful ambiguity interpretation, disabled stopping behavior,
receipt hashes, invalid settings and existing-output protection. No dataset bytes
or generated results enter Git.

Artifacts:
/workspace/SpatialRust-python-delivery/target/icp-convergence-study-final/study.json
/workspace/SpatialRust-python-delivery/target/icp-convergence-study-final/report.html

Subjective maturity assessment: 55%, provisional, documented in
/workspace/SpatialRust-python-delivery/docs/MATURITY.md. This is an engineering
estimate, not measured external-library parity. Multi-dataset real-reference
validation and stronger robust/global registration remain outstanding.
