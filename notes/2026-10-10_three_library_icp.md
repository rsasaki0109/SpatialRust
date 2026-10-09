# PCL, Open3D and SpatialRust controlled ICP study

Artifacts: `/workspace/SpatialRust-python-delivery/target/pcl-open3d-icp-failure-study-final/`.
702 registrations over 234 paired conditions. PCL 1.15.0, Open3D 0.19.0 and
SpatialRust all recover 138 poses, with identical recovery classification in
every paired condition. Cases/seeds/controlled priors/gates match the preceding
Open3D study; normal XYZ updates use f32 in PCL/SpatialRust and f64 in Open3D.
Common SciPy evaluation avoids comparing incompatible library fitness metrics.

PCL comparator explicitly instantiates its header ICP and transformation-SVD
implementations against the existing local PCL common/search libraries; built
with g++ -std=c++17 -O2. It uses a persistent bounded request/response protocol.
Kernel time excludes payload conversion/parsing and process creation. Absolute,
relative, translation and rotation stopping thresholds are zero; PCL's inclusive
zero-delta check still allows earlier stopping. Actual PCL updates range 15–100,
SpatialRust uses 100; Open3D does not expose actual updates. No universal timing
ranking or bitwise-equivalent numerical-work claim follows from these settings.

Optional local verification: 5 tests pass across Open3D/PCL comparison suites,
including all 162 three-library one-seed registrations, known clean alignment,
protocol validation and the existing full-source denominator checks. Native
binary and C++/Python/fixture code hashes are recorded. The CMake target is
provided for conventional PCL installations; the local provisioned-library
build used explicit include/link flags and LD_LIBRARY_PATH instead.

Broader gates (.6/1.2 m) recover clean poor-prior clouds yet consistently bias
ordinary ICP in nearby-outlier and replacement cases. Narrow .3 m gates recover
the outlier cases with good priors but shrink the poor-prior capture basin.
The same tradeoff in three implementations supports investigating correspondence
assumptions rather than attributing it to one library. Controlled priors are
constructed relative to known truth, not independent global initialization.
The ring has equivalent poses and no observable generating-pose correctness.
No real-sensor, OpenCV ICP or global-registration comparison is claimed.
Provisional maturity 64%; long-term target 90%.
