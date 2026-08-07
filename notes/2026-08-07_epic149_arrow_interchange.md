# Epic 149: Arrow canonical interchange in Python

Date: 2026-08-07. Slices 149A/149B complete.

## Why

SpatialRust already owns the audited Arrow C Data / Stream / Device substrate
(91C/91D). Epic 149 promotes Arrow to the canonical cross-language zero-copy
interchange so PyArrow, pandas, and DuckDB consume SpatialRust records without
copying — a differentiator with no OSS precedent in point-cloud libraries.

## What was built

- `crates/spatialrust-py/src/arrow_capsule.rs` — CPython capsule boundary for
  the Arrow C Data Interface:
  - `__arrow_c_array__`: exports a `PointCloud` as `(arrow_schema, arrow_array)`
    capsules (spec order). Destructors call the Arrow release callbacks exactly
    once; `PyTuple_SetItem` steals references so ownership transfers cleanly.
  - `__arrow_c_stream__`: exports a `PyPointCloudStream` as a single
    `arrow_array_stream` capsule with `get_schema`/`get_next`/`get_last_error`/
    `release` callbacks driving the chunk iterator directly (no `Send` bound
    needed, matching the `unsendable` pyclass).
- `PyPointCloud::__arrow_c_array__` and `PyPointCloudStream::__arrow_c_stream__`
  methods, typed `.pyi` stubs, and two wheel-gate tests (PyArrow round trips,
  bounded batch sizes, concatenated column equality).
- `spatialrust-pipeline::StreamingPipelineIter::schema()` additive accessor.
- CI installs `pyarrow` for the Python binding job.

## Verification (PyArrow 25 in a venv)

- `pa.array(cloud)` → struct<float32 x,y,z>; field buffers are CPU-backed and
  sized `N*4`, and field values match the source — zero-copy.
- `pa.RecordBatchReader.from_stream(stream)` over a 1000-point PCD yields
  256/256/256/232 batches; concatenated `x` column matches a direct read.
- `gc.collect()` after consuming shows no double free.
- pytest `-k arrow_c` passes; Rust workspace tests pass; clippy `-D warnings`
  clean for the py crate.

## Contract notes

- The capsule destructor only acts when the capsule name is still the original
  `arrow_array`/`arrow_schema`/`arrow_array_stream`; consumers that move data
  mark the release callback null, so there is exactly one release per export.
- `__arrow_c_stream__` consumes the stream: the underlying iterator is moved
  into the capsule, so the Python object becomes exhausted afterwards.
