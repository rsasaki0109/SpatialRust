# Direct unpacking of mapped GPU image readbacks

This slice removes the intermediate host vector from `GpuImage::readback_u8`
for logical channels 1..=4. The GPU still copies the texture to a padded
COPY_DST/MAP_READ staging buffer. After the existing map callback and wait,
the unchanged `unpack_rgba8` reads the borrowed mapped bytes directly into its
owned logical output vector. The previous whole-staging `to_vec` is removed.

The eliminated payload is
`ceil((4 * width) / 256) * 256 * height` host bytes per successful image
download: 33,177,600 bytes (31.640625 MiB) at 3840x2160, regardless of logical
channels. GPU staging and the final `width * height * channels` host output
remain. This counts one framework payload allocation/copy removed over the
call; process peak memory is not measured.

A private generic mapping helper executes its decoder while the mapping is
live. The mapped view is declared after an unmap guard, so reverse local-drop
order releases the view first on normal return or decoder panic. Scoped
unpacking workers join before the decoder returns or unwinds. No mapped
borrow escapes in the returned image, and there is no unsafe code or new
dependency. The mapping now stays live through unpacking; mapped-memory cache
behavior can affect latency, so allocation removal alone is not a speed claim.

Existing tensor/histogram byte consumers retain the same owned-byte helper
signature and copy through that decoder. Texture upload, GPU copy/map/wait,
transfer plans and limits, packing/unpacking arithmetic, bounded CPU workers,
metadata, receipts and shader sources remain unchanged. Image construction
and receipt updates still occur after the mapping is released.

## Verification design

The complete public readback and its original mapping helper are frozen from
parent merge `95805bcac9fdbc208a55a7c73eff326cd5c62720`; only test names and the
call to that frozen helper change. This baseline retains its own original
whole-buffer copy and map/unmap sequence.

The new native matrix covers all channels, packed and strided ROI uploads,
1x1, 64x3, 65x3, 1023x1025, 1024x1024 and 1024x1025. These include aligned
and padded readback rows, below/exact/above native worker thresholds. It checks
logical bytes against the frozen API and independent CPU rows, every physical
RGBA component including unused zero lanes, metadata and repeated receipts.
Source mutation/drop precedes downloads. Returned images remain live after
later downloads, GPU recycling and pool clearing; mutating an owned result
does not change subsequent downloads. Four concurrent callers repeatedly read
large RGB images so mapped input must outlive all scoped unpacking workers.

A separate mapping test returns a non-Clone owned result, verifies exact
prefix and empty borrows, returns a decoder error, deliberately panics, and
then remaps the same buffer. This exercises successful/error/unwind cleanup
and preserves the existing owned-byte wrapper behavior.

## Measurement design

The ignored paired fixture compares complete old/current public readbacks:
staging allocation, encoder/copy/submit, mapping wait, unpacking, receipt
update and owned image construction. Input uploads are outside the timer;
output assertions and output drop are outside. Two uploaded live images
share the same current runtime and payload, with alternating order, three
warmups and 30 paired samples. Final receipts must agree.

Eight conditions cover RGBA 64x3, 65x3, 1024x1024, padded 1921x1080 and
3840x2160; gray 1024x1024; two-channel 1920x1080; and RGB 1920x1080 from an
ROI. Two isolated surveys are attempted after correctness checks and source/
executable hash capture. Competing observable build/test/benchmark/Python
computation invalidates an attempt; unrelated jobs are never terminated.
The quiet-window deadline is three minutes per scheduling window to bound waiting.
No speed is attributed to an incomplete or overlapping trial.

Analysis uses paired log old/new latency ratios and 10,000 bootstrap resamples,
sorted indices 249/9749 for the 95% interval. Seeds are
2026101511 + 100*survey + 10*condition_index + JSON_field_index. Ratios above
one favor direct decoding; intervals crossing one are inconclusive, and any
zero sample suppresses that row's ratio. Every configured shape/channel/ROI
condition must appear once per survey with exactly 30 nonnegative finite
paired samples. All rows, including adverse and inconclusive results, remain
in the report.

## Completed correctness validation

- All 113 native image regression tests passed, including three new tests.
- Four public facade tests passed (median, bilateral and two image transfers).
- The new 48-fixture matrix compared 352,479,568 physical/logical bytes.
- Concurrent repeated reads kept independent outputs, and deliberate decoder
  panic was caught with successful remapping afterward.
- GPU all-feature/all-target Clippy with warnings denied, CPU-only and WASM
  builds, formatting and diff checks passed.
- Source audit leaves 344 other preexisting GPU/image sources, including all
  97 WGSL files, unchanged. Both complete frozen functions match the parent;
  production changes are confined to mapped decoding, owned-byte wrapping and
  map cleanup. Plans, uploads, receipt bodies and unpacking math are unchanged.
- 350 source/executable SHA-256 values were pinned after validation. The same
  files are verified after the measurement attempt and at delivery.

## Measured completed readback results

Two surveys completed: survey 1 / attempt 1 in 7.66 seconds, and survey 2 /
attempt 2 in 7.46 seconds. Both exact benchmark invocations passed, with zero
stderr and no detected competing work. Survey 2's first three-minute quiet
window timed out without launching a child. The first survey was then
summarized outside any running measurement. A second scheduling window
repeated the same complete eight-condition matrix without changing source,
executable, fixtures, ordering or sample counts. No condition was selected or
dropped based on that interim analysis. No overlapping or incomplete samples
contribute to the table.

Conditions: Windows x86_64; rustc 1.98.1 (48a229cea, 2026-09-01), release
with thin LTO and one codegen unit; Vulkan discrete backend; 12 available CPU
workers; the existing 64 MiB image pool retention budget. RGBA unpacking stays
serial; the existing policy chooses four workers for 1024x1024 gray and eight
for the 1920x1080 two-/three-channel conditions. Uploads precede the timer,
outputs are validated on every iteration, and old/new final receipts agree.
The old intermediate allocation, copy and drop are inside the public call;
returned-output assertions and drop are outside its timer in both routes.

All 350 pinned source/executable hashes match after both accepted surveys.
The 16 rows contain 12 favorable intervals, four inconclusive intervals and
no wholly adverse intervals or zero samples. All six large-image conditions
favor direct decoding in both surveys. Small 64x3/65x3 conditions are
inconclusive; the slightly higher new 64x3 medians remain in the table.

- RGBA 3840x2160: 23.764850 to 16.102100 ms and 22.276550 to 14.740100 ms,
  median reductions of 32.2% and 33.8%. Paired old/new intervals are
  [1.389263, 1.559711] and [1.433069, 1.616462].
- RGBA 1024x1024: median reductions of 29.8% and 30.9%; padded 1921x1080:
  29.4% and 27.1%.
- Gray 1024x1024: 11.0% and 11.5%; two-channel 1920x1080: 19.4% and 17.8%;
  RGB ROI 1920x1080: 19.2% and 15.3%.

These are warm completed-readback observations for the stated backend and
conditions. GPU kernels are unchanged; complete upload/processing/download
pipelines and other backends were not timed. Allocation removal is established
by the code path; RSS, device-memory peaks and cold-call behavior were not
measured.

| Survey | Shape / channels / ROI | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 64×3 / 4 / False | read | 0.942150 | 0.944150 | 1.016089 [0.992121, 1.041425] |
| 1 | 65×3 / 4 / False | read | 0.997750 | 0.979150 | 1.015029 [0.990598, 1.043627] |
| 1 | 1024×1024 / 4 / False | read | 3.507400 | 2.460700 | 1.404170 [1.359797, 1.445090] |
| 1 | 1921×1080 / 4 / False | read | 7.611200 | 5.370450 | 1.408447 [1.389517, 1.429676] |
| 1 | 3840×2160 / 4 / False | read | 23.764850 | 16.102100 | 1.477798 [1.389263, 1.559711] |
| 1 | 1024×1024 / 1 / False | read | 8.714100 | 7.756550 | 1.126732 [1.068103, 1.181442] |
| 1 | 1920×1080 / 2 / False | read | 11.737300 | 9.459300 | 1.230977 [1.162347, 1.312319] |
| 1 | 1920×1080 / 3 / True | read | 11.758250 | 9.503100 | 1.260758 [1.212279, 1.315642] |
| 2 | 64×3 / 4 / False | read | 0.864650 | 0.869050 | 1.015927 [0.992028, 1.041478] |
| 2 | 65×3 / 4 / False | read | 0.946750 | 0.940700 | 0.983925 [0.961064, 1.006812] |
| 2 | 1024×1024 / 4 / False | read | 3.437950 | 2.375100 | 1.393200 [1.303618, 1.459420] |
| 2 | 1921×1080 / 4 / False | read | 7.099800 | 5.174400 | 1.331711 [1.262422, 1.381752] |
| 2 | 3840×2160 / 4 / False | read | 22.276550 | 14.740100 | 1.522296 [1.433069, 1.616462] |
| 2 | 1024×1024 / 1 / False | read | 8.584000 | 7.594200 | 1.135458 [1.097319, 1.174090] |
| 2 | 1920×1080 / 2 / False | read | 11.500600 | 9.453650 | 1.274447 [1.222488, 1.341517] |
| 2 | 1920×1080 / 3 / True | read | 11.621900 | 9.842750 | 1.192341 [1.125606, 1.268284] |
