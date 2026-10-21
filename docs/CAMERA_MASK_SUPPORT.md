# Camera mask support (146AS)

The optional browser camera tracker now summarizes each validated person mask
once. Mixed masks use a zero-bordered u32 summed-area table, so each feature's
inclusive 45x45 support test reads four entries instead of scanning its pixels.
An empty or all-zero mask uses an empty summary; an all-one mask uses a full
summary. Uniform masks allocate no prefix table.

The finite checks, strict 22-pixel margin, positive-coordinate truncation and
inclusive square are unchanged. The two bounded diagonal sums are compared for
equality, avoiding unsigned subtraction underflow. Shape, binary-label and clock
validation keep their original order and reject before retained state changes.
Label validation also counts person pixels for the unchanged report fraction.

The tracker owns its derived summary independently of caller storage. Reset
releases it. At most one previous and one current prefix are live: each holds
at most 321x321 u32 entries (412,164 bytes), or 824,328 logical bytes together.
These numbers exclude inputs, images, features, allocator overhead and browser
memory. A mixed mask uses more retained support memory than its old byte mask.
Public Rust/WASM signatures and report schema, gray copies, JS mask/image
readbacks, FAST/LK and grid-selection math remain unchanged. The summary is CPU
work inside the existing optional web feature; it adds no device transfer,
dependency or GPU performance claim.

## Correctness evidence

The [source-bound receipt](receipts/2026-10-22_camera_mask_support.json) preserves
the parent executable and WASM, all command logs and their source hashes.
The parent native suite passed 21 tests; the candidate passed 24, including
the independent frozen brute-force support predicate, finite/fractional/strict
border admission, every small-frame pixel with single exclusions, deterministic
mixed masks, caller ownership, maximum prefix mass and invalid-call/reset state.
The initial parent-fixture compile failure (an omitted integer type, E0689)
and its corrected retry are both retained; it was not a product regression.

All 126 complete reports in each native benchmark survey match, including
serialized floating-point values. Separately, installed Chromium loaded the
actual parent and candidate WASM packages together: 42 shape/mask cases produced
252 identical complete JSON reports and 84 matching label/clock rejections.
Changing masks and reset are included, with no camera acquisition or external
requests. The existing six asset, 84 JS and 82 real WASM/model browser controls
also pass. Release Clippy, feature-disabled tests, Rustfmt and WASM compilation
pass. Exact-source CI, merge and matching public Pages checks are verified after
this local receipt.

## Native timing observations

The fixed protocol is
`C:/Users/rsasa/Workspace/SpatialRust/benchmarks/camera-mask-support-plan.json`.
It uses one native Windows worker and release builds, three synthetic image
sizes and six masks. Each row measures the median of six seven-frame sequences
after one warmup sequence. Construction, mask validation/summary, tracking and
report construction are timed; fixture generation and JSON output are outside
the timer. Runs execute parent, candidate, candidate, parent. Ratios below 1
favor the candidate. Both surveys and every adverse row are shown, with all raw
samples in the receipt. This is a local observation without confidence intervals
or a general speed guarantee; it does not measure physical phones, GPU work,
AI inference or end-to-end browser frame time.

| Frame | Mask | Parent 1 ms | New 1 ms | New/parent 1 | Parent 2 ms | New 2 ms | New/parent 2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 192x144 | disabled | 552.720 | 514.129 | 0.930 | 487.813 | 545.661 | 1.119 |
| 192x144 | zero | 880.452 | 855.027 | 0.971 | 890.205 | 675.062 | 0.758 |
| 192x144 | full | 79.708 | 73.761 | 0.925 | 64.863 | 62.110 | 0.958 |
| 192x144 | half | 511.761 | 527.577 | 1.031 | 500.485 | 370.918 | 0.741 |
| 192x144 | sparse | 834.581 | 973.739 | 1.167 | 876.727 | 644.375 | 0.735 |
| 192x144 | checker | 83.346 | 64.436 | 0.773 | 72.063 | 68.369 | 0.949 |
| 256x192 | disabled | 1047.319 | 786.910 | 0.751 | 899.596 | 862.256 | 0.958 |
| 256x192 | zero | 1089.399 | 884.246 | 0.812 | 805.224 | 984.273 | 1.222 |
| 256x192 | full | 152.824 | 113.189 | 0.741 | 115.072 | 119.688 | 1.040 |
| 256x192 | half | 1019.405 | 783.617 | 0.769 | 659.305 | 651.047 | 0.987 |
| 256x192 | sparse | 1088.303 | 802.858 | 0.738 | 714.537 | 682.069 | 0.955 |
| 256x192 | checker | 133.236 | 118.538 | 0.890 | 129.641 | 107.424 | 0.829 |
| 320x240 | disabled | 1007.336 | 927.754 | 0.921 | 862.563 | 929.542 | 1.078 |
| 320x240 | zero | 1100.728 | 1094.732 | 0.995 | 1008.091 | 860.486 | 0.854 |
| 320x240 | full | 205.945 | 186.721 | 0.907 | 193.191 | 194.161 | 1.005 |
| 320x240 | half | 1142.988 | 953.183 | 0.834 | 1084.946 | 1034.584 | 0.954 |
| 320x240 | sparse | 1051.278 | 886.199 | 0.843 | 1009.860 | 919.626 | 0.911 |
| 320x240 | checker | 192.847 | 209.024 | 1.084 | 189.582 | 215.363 | 1.136 |

Of 36 paired medians, 27 are lower and 9 are higher for the candidate. Mask
distribution and unrelated tracking work affect the total; small differences
may be noise.

The public workload is
`crates/spatialrust-web/examples/camera_mask_support.rs`. With
`RAYON_NUM_THREADS=1`, run:

```sh
cargo run -p spatialrust-web --release --no-default-features --features camera-tracking --example camera_mask_support --locked --offline
```

The independent browser comparison is
`scripts/test_camera_mask_support.cjs`; it expects separate frozen parent and
candidate packages as described by its local allowlist.

## Resource and qualification limits

The owned E-drive root has a prospective 1GiB cap, with two Cargo build jobs,
one owned command at a time and a 900-second command deadline. Rust builds are
locked/offline in a fresh isolated target; the existing shared compiler cache
and raw release WASM are preserved. Nine vendor files reuse 41,087,603
hard-linked bytes after SHA checks. No model, browser, toolchain or sensor data
is acquired. Storage snapshots retry transient vanished files only inside the
owned live Cargo target; unrelated or persistent inventory failures still fail.

The receipt preserves 786 prior camera/renderer files, 493 scientific JSON,
42 frozen native artifacts, room/reference outputs and the resolved lockfile.
Public verification has a separate 40MiB streaming cap and retains no downloaded
vendor payload files. Physical phones/iOS Safari, calibrated camera clocks,
metric pose/full-room accuracy and operational 80% qualification remain open.
