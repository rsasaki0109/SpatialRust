# Defer owned RGBA row copies after measurement

Parent: `5b413653a1c85c0149e86338c25cdea525d5c3a4` (merged PR 252).

The proposed safe row-append path remains a native test fixture. Shared-host
measurements did not establish a latency benefit, and slower results occurred.
Production RGBA packing, padded readback and earlier contiguous readback/borrowed
upload paths retain their immediate-parent implementation. This deliverable
records a rejected optimization candidate and its evidence; it adds no production
GPU speedup or transfer change.

## Candidate and source work

The candidate reserves the validated output length once and appends complete
logical RGBA rows with safe slice copies. ROI/stride padding and mapped staging
padding are excluded. Total appended length equals the reserved extent; exact
capacity and owned output remain. Non-RGBA helpers, the earlier packed upload
borrowing check and the contiguous RGBA readback shortcut are preserved.
There is no new unsafe code or public API.

For affected width W, height H and B = 4*W*H:

| Route | Owned outputs original / candidate | Explicit zero initialization original / candidate | Logical copies |
| --- | --- | --- | --- |
| Strided/ROI RGBA upload packing | 1 / 1 | B / 0 bytes | H rows and B bytes in both |
| Dense RGBA upload | 0 / 0 | 0 / 0 bytes | Borrowed queue write in both |
| Padded RGBA readback | 1 / 1 | B / 0 bytes | H rows and B bytes in both |
| Contiguous RGBA readback | 1 / 1 | 0 / 0 bytes | One-prefix copy in both |

At 1921 x 1080, B is 8,298,720 bytes. Source-level zero initialization does not
establish physical memory traffic or peak-memory savings. Each append checks
capacity and updates length; copies and allocations remain. Compiler/allocator
choices affect actual cost. The source work reduction did not justify adoption
without a demonstrated completed-call benefit.

The complete parent and candidate public upload/readback methods are native-test
copies of the original bodies, differing only in selection of packing/unpacking
helpers. Validation/error order, staging extents, queue writes, texture copies,
submits, maps, waits, mapping guards, metadata and success-only receipts retain
their original code. The candidate is reachable only through
`cfg(all(test, not(target_arch = "wasm32")))`.

## Validation of final test-only placement

- **343 native correctness tests:** 139 image, 184 voxel/readback, 18 filtering
  and 2 facade, including ignored GPU tests and excluding benchmarks.
- **328,948,048 exact original/CPU host bytes:** 80 channel/layout cases
  plus empty outputs. Independent pixels cover channels one through four,
  packed/ROI layouts, zero unused lanes, alignment boundaries and large-worker
  thresholds. Guard padding, exact capacities and ownership after source mutation
  are checked against complete frozen parent helpers.
- **100,999,808 original/CPU RGBA bytes:** ten input fixtures, each
  covering all four candidate/production upload/readback combinations. Tests
  retain dense borrowing, ROI selection, BGRA/alpha/range metadata, matching
  receipts, texture recycling and owned data after runtime drop. Prior lifetime,
  panic/remap, concurrent and rejection fixtures remain and pass.
- All-features/all-targets Clippy with warnings denied, CPU-only and minimal/CI
  WASM builds with WASM warnings denied, formatting and whitespace pass.
- **500 unchanged original GPU/image Rust/WGSL sources** and **504 validated
  source/executable hashes**. The existing native test file is append-only.
  Complete frozen helpers and both parent/candidate public methods match the
  immediate parent after namespace normalization. Candidate pack/unpack helpers
  differ only in the two explicit owned-RGBA branches.

## Paired diagnostic measurements

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 output_drop_assertions_recycle_excluded=true`.

Each run has three warmup pairs and thirty measured pairs with alternating
original/candidate order for five shapes at four boundaries. All three complete
runs, including slower results, are retained below. Runs one and two preceded
the decision to keep the candidate test-only. Run three verifies the final native
fixture placement with the same candidate helper bodies; moving test functions
can affect code generation, so cross-build timing comparisons are not controlled.
The ignored benchmark passes separately in every run with exact byte, metadata
and receipt checks.

Checks, output destruction and texture recycling occur outside timing. Pure host
pack always constructs an owned result, including the dense control; public dense
uploads still borrow. Public upload timing includes an explicit empty submit and
`wait_idle` to measure completion. These waits are benchmark instrumentation;
production upload remains asynchronous. Its verification readback is outside
timing. Completed readback includes staging, copy, submit/map/wait, host output
and metadata/receipt construction. Unpadded host/public readbacks are unchanged
controls.

In run one, 4K upload completion was 15.379950 ms original versus 16.049550 ms
candidate. In run two it was 19.562800 ms versus 17.444500 ms. Host-only 4K packing
medians were slower with the candidate in both initial runs (8.390200/8.667650 ms
and 9.108750/10.848250 ms). Paired ratios and unchanged controls also vary. These
observations establish no latency win. Adoption is deferred until a controlled
measurement demonstrates a repeatable benefit without adverse cases.

Quiet requires no other sampled process above 0.25 CPU cores and all GPU/memory
samples at most 5%. Surveys wait two seconds between snapshots; measured CPU
intervals include snapshot overhead and exclude the coordinator process. Pre/post
samples cannot prove isolation for every timed pair. These shared-host medians
are diagnostic observations. Zero-duration samples remain in raw records; no
paired ratio is reported for affected rows. Process identities and the adapter
name are omitted from published evidence.

### Run 1

Observed-quiet: **false**.
Busy-process counts: 0 before / 3 after.
GPU/memory utilization samples: `[[[0, 0]], [[0, 0]]]` before /
`[[[1, 0]], [[2, 0]]]` after.

| Boundary | Pixels | ROI | Original median ms | Candidate median ms | Median paired original/candidate |
| --- | --- | --- | ---: | ---: | ---: |
| host_pack_only | 64 x 3 | false | 0.000100 | 0.000100 | 1.0000 |
| public_upload_and_wait_idle | 64 x 3 | false | 0.099550 | 0.098500 | 0.9908 |
| host_unpack_only | 64 x 3 | false | 0.000250 | 0.000200 | 1.0000 |
| completed_public_readback | 64 x 3 | false | 1.022850 | 1.018850 | 0.9889 |
| host_pack_only | 65 x 3 | true | 0.000200 | 0.000200 | 1.0000 |
| public_upload_and_wait_idle | 65 x 3 | true | 0.111400 | 0.113250 | 1.0034 |
| host_unpack_only | 65 x 3 | true | 0.000200 | 0.000200 | 1.0000 |
| completed_public_readback | 65 x 3 | true | 0.959800 | 0.961650 | 1.0002 |
| host_pack_only | 1920 x 1080 | true | 1.774900 | 1.747900 | 0.9975 |
| public_upload_and_wait_idle | 1920 x 1080 | true | 3.845550 | 3.869950 | 1.0025 |
| host_unpack_only | 1920 x 1080 | true | 1.462500 | 1.457800 | 0.9958 |
| completed_public_readback | 1920 x 1080 | true | 4.154200 | 4.018500 | 1.0227 |
| host_pack_only | 1921 x 1080 | true | 1.955500 | 1.982350 | 1.0001 |
| public_upload_and_wait_idle | 1921 x 1080 | true | 4.106400 | 4.133800 | 0.9816 |
| host_unpack_only | 1921 x 1080 | true | 1.845150 | 1.873150 | 0.9959 |
| completed_public_readback | 1921 x 1080 | true | 6.015250 | 6.017350 | 1.0110 |
| host_pack_only | 3840 x 2160 | true | 8.390200 | 8.667650 | 0.9793 |
| public_upload_and_wait_idle | 3840 x 2160 | true | 15.379950 | 16.049550 | 0.9654 |
| host_unpack_only | 3840 x 2160 | true | 6.894950 | 6.974550 | 0.9952 |
| completed_public_readback | 3840 x 2160 | true | 16.197200 | 16.501950 | 1.0182 |

### Run 2

Observed-quiet: **false**.
Busy-process counts: 1 before / 2 after.
GPU/memory utilization samples: `[[[0, 0]], [[0, 0]]]` before /
`[[[3, 0]], [[18, 1]]]` after.

| Boundary | Pixels | ROI | Original median ms | Candidate median ms | Median paired original/candidate |
| --- | --- | --- | ---: | ---: | ---: |
| host_pack_only | 64 x 3 | false | 0.000100 | 0.000100 | timer resolution |
| public_upload_and_wait_idle | 64 x 3 | false | 0.113700 | 0.118650 | 0.9832 |
| host_unpack_only | 64 x 3 | false | 0.000100 | 0.000100 | timer resolution |
| completed_public_readback | 64 x 3 | false | 1.059350 | 1.039150 | 1.0294 |
| host_pack_only | 65 x 3 | true | 0.000100 | 0.000100 | timer resolution |
| public_upload_and_wait_idle | 65 x 3 | true | 0.106100 | 0.107000 | 0.9949 |
| host_unpack_only | 65 x 3 | true | 0.000100 | 0.000100 | timer resolution |
| completed_public_readback | 65 x 3 | true | 1.071500 | 1.090350 | 1.0102 |
| host_pack_only | 1920 x 1080 | true | 2.074600 | 2.059800 | 1.0044 |
| public_upload_and_wait_idle | 1920 x 1080 | true | 4.137000 | 4.216050 | 0.9944 |
| host_unpack_only | 1920 x 1080 | true | 1.865450 | 1.833450 | 1.0066 |
| completed_public_readback | 1920 x 1080 | true | 4.361850 | 4.373700 | 1.0113 |
| host_pack_only | 1921 x 1080 | true | 2.218800 | 2.207350 | 1.0046 |
| public_upload_and_wait_idle | 1921 x 1080 | true | 4.354500 | 4.547850 | 0.9764 |
| host_unpack_only | 1921 x 1080 | true | 2.231500 | 2.210950 | 0.9999 |
| completed_public_readback | 1921 x 1080 | true | 8.885850 | 8.793850 | 1.0073 |
| host_pack_only | 3840 x 2160 | true | 9.108750 | 10.848250 | 0.9909 |
| public_upload_and_wait_idle | 3840 x 2160 | true | 19.562800 | 17.444500 | 1.0089 |
| host_unpack_only | 3840 x 2160 | true | 6.696900 | 6.986550 | 0.9697 |
| completed_public_readback | 3840 x 2160 | true | 16.611950 | 16.943950 | 0.9903 |

### Run 3

Observed-quiet: **false**.
Busy-process counts: 2 before / 1 after.
GPU/memory utilization samples: `[[[0, 0]], [[0, 0]]]` before /
`[[[0, 0]], [[0, 0]]]` after.

| Boundary | Pixels | ROI | Original median ms | Candidate median ms | Median paired original/candidate |
| --- | --- | --- | ---: | ---: | ---: |
| host_pack_only | 64 x 3 | false | 0.000100 | 0.000100 | timer resolution |
| public_upload_and_wait_idle | 64 x 3 | false | 0.081950 | 0.081250 | 1.0128 |
| host_unpack_only | 64 x 3 | false | 0.000100 | 0.000100 | timer resolution |
| completed_public_readback | 64 x 3 | false | 0.985600 | 0.966700 | 1.0174 |
| host_pack_only | 65 x 3 | true | 0.000200 | 0.000200 | 1.0000 |
| public_upload_and_wait_idle | 65 x 3 | true | 0.096900 | 0.100650 | 1.0168 |
| host_unpack_only | 65 x 3 | true | 0.000200 | 0.000200 | 1.0000 |
| completed_public_readback | 65 x 3 | true | 1.003550 | 1.015800 | 0.9753 |
| host_pack_only | 1920 x 1080 | true | 1.784300 | 1.788150 | 1.0039 |
| public_upload_and_wait_idle | 1920 x 1080 | true | 3.857100 | 3.829900 | 1.0032 |
| host_unpack_only | 1920 x 1080 | true | 1.454150 | 1.437150 | 1.0110 |
| completed_public_readback | 1920 x 1080 | true | 3.935000 | 4.011300 | 0.9985 |
| host_pack_only | 1921 x 1080 | true | 1.877600 | 1.813450 | 1.0225 |
| public_upload_and_wait_idle | 1921 x 1080 | true | 4.112500 | 4.189200 | 1.0084 |
| host_unpack_only | 1921 x 1080 | true | 2.136550 | 2.129550 | 1.0074 |
| completed_public_readback | 1921 x 1080 | true | 8.676050 | 8.694750 | 1.0055 |
| host_pack_only | 3840 x 2160 | true | 8.650350 | 8.272100 | 1.0354 |
| public_upload_and_wait_idle | 3840 x 2160 | true | 23.111900 | 23.518600 | 0.9926 |
| host_unpack_only | 3840 x 2160 | true | 5.486750 | 5.456800 | 1.0029 |
| completed_public_readback | 3840 x 2160 | true | 22.329350 | 21.858500 | 1.0095 |

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Unchanged production helper:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/transfer_packing.rs`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/mapped_readback_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/rgba_row_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/rgba_row_benchmark.rs`.
Ignored validation, raw measurements and provenance:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/rgba-rows-*`.
