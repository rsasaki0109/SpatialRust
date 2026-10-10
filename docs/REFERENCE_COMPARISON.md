# Public reference pairs and refinement failures

For original 3DMatch fragment/reference binding and the publisher's saved-log
recall/precision metric, see [Redwood/3DMatch reference](3DMATCH_REFERENCE.md).
For precommitted noisy synthetic depth-frame validation on a separate dataset,
see [fixed Redwood frame validation](REDWOOD_FRAME_VALIDATION.md).

The comparison tools separate declared reference pose accuracy from distance-gated
proximity. They never pass the reference pose to a registration algorithm or use
it to select a candidate or a trim fraction. Library algorithms differ in feature
neighborhoods, RANSAC sampling, numeric precision and stopping. Recorded elapsed
times are diagnostic, not evidence of equal work or a speed ranking.

## Prepare a pair

GeoTransformer supplies `data/demo/src.npy`, `ref.npy` and `gt.npy`. The demo code
uses the transform to map source to reference. The tested snapshot is
[`e7a135af4c318ff3b8d7f6c963df094d7e4ea540`](https://github.com/qinzheng93/GeoTransformer/tree/e7a135af4c318ff3b8d7f6c963df094d7e4ea540/data/demo).
It is a publicly supplied demo reference; original sensor lineage and independent
physical ground truth are not established here.

```bash
python scripts/prepare_numpy_reference.py \
  --source /path/to/GeoTransformer/data/demo/src.npy \
  --target /path/to/GeoTransformer/data/demo/ref.npy \
  --pose /path/to/GeoTransformer/data/demo/gt.npy --length-unit m \
  --provenance-url https://github.com/qinzheng93/GeoTransformer/tree/e7a135af4c318ff3b8d7f6c963df094d7e4ea540/data/demo \
  --description 'Public demo reference; sensor lineage not independently established' \
  --project-reference-rotation --output-dir target/public-pair
```

NumPy loading disables pickle. Numeric shape, finiteness and f32 representability
are validated. PCD XYZ roundtrips must exactly match converted arrays. Original
array byte hashes, generated PCD byte hashes and maximum f32 conversion errors
are retained. All output directories must be new.

The supplied rotation has maximum orthogonality deviation about `7.09e-5` and
determinant `0.999898`. Strict evaluation rejects it. The explicit projection
flag computes the nearest proper rotation by SVD, rejects reflections or singular
values deviating from one by more than `1e-3`, and records the original rotation,
singular values and correction norm. Translation is unchanged. This is a declared
reference correction, not a fitted pose or silently relaxed validity check.

## Compare without an initial pose

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/compare_public_global.py \
  --source target/public-pair/source.pcd --target target/public-pair/target.pcd \
  --reference target/public-pair/reference.json --seeds 7 8 9 10 11 \
  --output-dir target/public-comparison
```

Both methods use the same input coordinates and SpatialRust voxel points, 0.1 m
coarse leaf, 0.5 m feature radius, 20-neighbor normal estimates, 0.2 m global gate,
10,000 RANSAC cap, then 0.05 m voxel ICP at 0.2 m and full-resolution ICP at
0.05 m, with 50-update caps per stage. Open3D's descriptor neighborhood is capped
at 100; SpatialRust's feature calculation differs. Seeds are recorded but do not
imply equivalent random samples. The common final support evaluator uses native
full-source proximity at 0.05 m for both poses. Native libraries, algorithm helper
files, runner, reference and inputs are fingerprinted; changes during running
are rejected. Failure rows remain in the receipt.

## Replay fixed global initializations

```bash
python scripts/study_public_refinement.py \
  --source target/public-pair/source.pcd --target target/public-pair/target.pcd \
  --reference target/public-pair/reference.json \
  --comparison target/public-comparison/comparison.json \
  --trim-fractions 1 .8 .6 .4 --output-dir target/refinement-study
```

The replay requires matching inputs and native build, and one generated native
candidate per seed. It reuses each saved global pose and changes only trimming
in the two refinement stages. Global sampling is not rerun. Reports retain the
full update histories. HTML plots translation error versus proximity fraction;
all seeds/fractions and exact errors are tabulated. No reference-based trim or
pose selection is performed.

## Observations on the supplied demo

The pair has 15,953 source and 18,977 target points. Five seeds per method produce
ten registrations. Native final translation errors range 0.170–0.224 m and
Open3D errors 0.176–0.226 m relative to the corrected supplied reference. This is
shared failure evidence, not native accuracy superiority.

Twenty fixed-initialization replays give native median translation errors:

| Retained fraction | Median translation error | Median rotation error |
| --- | --- | --- |
| 1.0 | 0.206075 m | 2.61190° |
| 0.8 | 0.093657 m | 0.95571° |
| 0.6 | 0.050511 m | 0.48106° |
| 0.4 | 0.049118 m | 0.61698° |

All five untrimmed replay poses exactly match the original native results.
At seed 8, retaining 0.4 gives 3.77° rotation error versus 0.87° at 0.6.
Aggressive trimming can preserve a wrong local subset when initialization is
poor. Untrimmed proximity fractions around 0.46–0.48 coexist with larger pose
errors than some lower-support trimmed results. Proximity alone can favor biased
alignment on partially overlapping clouds. Trimming remains opt-in; one demo
does not justify a new universal default or establish multiple-dataset parity.
