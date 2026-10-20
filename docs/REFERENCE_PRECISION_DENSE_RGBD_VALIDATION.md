# Explicit reference-precision dense RGB-D model

Slice 146T prospectively fixes an additive `ReferencePrecision` /
`reference-precision` CPU fitting model before implementation and new fits.
The complete arithmetic, pose update, ownership, memory, source, resource and
original acceptance contracts are in
[the fixed protocol](../benchmarks/reference-precision-dense-plan.json).

The six synthetic pairs in the prior [precision audit](DENSE_PRECISION_AUDIT.md)
motivate float32 intermediate image/XYZ arithmetic and a proper left Euler
increment. They do not prove a real trajectory error's cause. Existing fitting
models, default selection and original raw final admission remain unchanged.
The new owned buffers still store f64; this is not a memory-saving claim.

All four reused TUM recordings are already observed development data, including
plant and room. Their old results remain evidence. Four complete new timelines
must be frozen before any score, with no retry, reset, trimming or post-score
model changes. Independent future motion qualification and every operational
80% acceptance requirement remain necessary.

Implementation and results are pending. The plan is committed first.
