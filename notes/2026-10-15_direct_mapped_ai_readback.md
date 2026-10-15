# Direct mapped AI tensor readback

`GpuAiTensor::readback_f32` previously copied the complete mapped buffer into
an owned byte vector, then converted those bytes into the final `Vec<f32>`.
The method now uses the existing guarded mapping decoder to perform the same
native-endian conversion directly from mapped bytes into the owned float vector.

One framework payload allocation and copy of `4 * channels * width * height`
bytes is removed per successful download: 24,883,200 bytes (23.73046875 MiB)
for 1920x1080 with three channels, or 132,710,400 bytes (126.5625 MiB) for
3840x2160 with four channels. GPU staging and the final float allocation remain.
This is a payload allocation removed over construction; process peak memory
and backend staging allocation are not inferred from that count.

Device identity, staging allocation, copy/submit/map/wait and error strings
are unchanged. The shared decoder becomes visible only to sibling image
modules; its body and unmap guard do not change. The retained owned-byte
wrapper is compiled only on native targets, matching its remaining callers.
Mapped bytes remain live
through conversion, with the view dropped before unmapping on return or
unwinding. Transfer receipt recording still precedes float conversion, as in
the parent. No mapped borrow escapes in the returned owned float vector.

The exact `chunks_exact(4)` / `f32::from_ne_bytes` conversion is retained:
there is no alignment-dependent cast, numeric operation or scalar filtering.
CHW order, arbitrary NaN payloads, infinities, signed zeros and subnormals
are preserved. Normalization shaders, pipeline/dispatch planners, uploads,
pool ownership and public signatures are unchanged. The mapping remains live
longer than before, so mapped-memory/cache behavior can affect timing.

## Verification and measurement method

The complete parent tensor readback and its owned-byte mapping wrapper,
generic mapping body and unmap guard are frozen from merge
`edd0d11862b9ac1a98121d8c2db5c0617791f056`. Only test visibility/names change.
This preserves the original intermediate allocation/copy and exact baseline
cleanup instead of comparing against a modified byte reader.

Raw GPU-storage tests cover all channels at 1x1, 65x17 and 1024x257, using
special encodings plus deterministic mixed u32 words. Both download methods
must preserve every original bit, shape and repeated receipt. Inputs are
mutated/dropped before reads; returned vectors remain independent of later
reads, tensor recycling and pool clears. Public GPU CHW packing additionally
uses analytic dyadic normalization and independent CPU plane order. Wrong
devices reject with the frozen error and unchanged receipts; concurrent
callers retain independent live results. Existing image decoder panic/remap
and image-readback lifetime regressions remain in the selected suite.

The paired benchmark times complete public old/current tensor downloads:
staging, copy/submit/map/wait, receipt recording, conversion and owned return.
Image upload and GPU CHW preprocessing occur before timing. Output assertions
and output drop are outside the timer; old intermediate allocation/copy/drop
remain inside its public call. Every result is compared by original float
bits to exact CPU CHW words, and final old/current receipts must match.

Eight conditions cover three channels at 16x16, 224x224, 1024x1024 and
1920x1080; one/two/four channels at 1024x1024; and four channels at
3840x2160. Three warmups precede 30 alternating paired samples per condition.
Two isolated surveys are attempted with three-minute quiet scheduling
windows and before/during/after competing-work detection. Incomplete or
overlapping trials provide no timing evidence; unrelated jobs are not stopped.

Analysis retains all eight conditions per survey and all favorable, adverse
and inconclusive rows. Paired log old/new ratios use 10,000 bootstrap samples,
sorted indices 249/9749 for 95% intervals, with seed
2026101513 + 100*survey + 10*condition_index + JSON_field_index. Ratios above
one favor direct decoding; intervals crossing one are inconclusive. Zero
samples suppress a row's ratio. No allocation count alone establishes speed.

## Completed correctness checks

- All 116 native image/GPU regressions and four public facade tests passed
  (120 distinct local tests). The selected image suite includes the earlier
  mapped-decoder panic/remap and owned-image lifetime cases.
- The new raw-storage matrix made 10,570,960 exact float-word comparisons,
  including the special encodings above. Public CHW packing made 663,060
  analytic dyadic-normalization word comparisons. Repeated receipt, shape,
  wrong-runtime, source lifetime, live output, pool reuse and concurrent reads
  passed.
- All-feature/all-target GPU Clippy with warnings denied, CPU-only and WASM
  builds, formatting and diff checks passed.
- 353 source/executable hashes were captured after validation. The frozen
  tensor method and complete original byte-wrapper/mapping/guard code match
  the parent. Other than the tensor decoder, sibling visibility of the
  shared helper and native-only byte-wrapper compilation, 346 other preexisting
  GPU/image sources and all 97 WGSL files
  are unchanged. The embedded CHW normalization shader is also identical.
- The receipt stage remains before conversion; copying, map errors/wait,
  native-endian scalar conversion and public signatures retain their order
  and definitions. Adapter-name privacy scans found no matches.

## Completed-call measurements

One complete isolated survey succeeded on Vulkan / DiscreteGpu with 12
available CPU workers in a native release build. All eight conditions retained
30 alternating pairs after three warmups. The accepted run was survey 1,
attempt 4, with exit zero, one passing benchmark, empty stderr and no detected
competing work before, during or after the run. Survey 1 attempt 1 overlapped
other work and was cancelled; attempts 2/3 did not start a benchmark after
their final quiet-window checks. Survey 2 attempt 1 was also cancelled for
overlap. Its initial three-minute scheduling window then expired; one further
three-minute window found no quiet interval and started no benchmark. Every
interrupted run is discarded in full. Other projects' jobs were left running.

There is no completed second survey, so these latency results have not been
independently repeated. Seven paired intervals favor direct decoding within
the completed survey; the 16x16 case is inconclusive and has a slightly worse
new median. No unfavorable or inconclusive rows are removed. These are local
completed-download observations, not portable speed guarantees, GPU-kernel
speedups, or complete application/inference latency measurements. The host
allocation/copy removal is established separately by the source comparison.

| Survey | Shape / channels | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 16×16 / 3 | read | 0.872850 | 0.889250 | 0.980791 [0.951536, 1.007898] |
| 1 | 224×224 / 3 | read | 0.986450 | 0.957800 | 1.037399 [1.011595, 1.063994] |
| 1 | 1024×1024 / 3 | read | 10.201250 | 6.304000 | 1.592891 [1.561346, 1.623060] |
| 1 | 1920×1080 / 3 | read | 19.833500 | 12.665150 | 1.563334 [1.517706, 1.608170] |
| 1 | 1024×1024 / 1 | read | 8.114100 | 6.885650 | 1.181577 [1.141900, 1.222226] |
| 1 | 1024×1024 / 2 | read | 12.582150 | 10.001600 | 1.283119 [1.145352, 1.446511] |
| 1 | 1024×1024 / 4 | read | 19.975150 | 15.116650 | 1.298604 [1.176455, 1.422813] |
| 1 | 3840×2160 / 4 | read | 112.130250 | 77.416250 | 1.378779 [1.317464, 1.439780] |

For 1920x1080 / 3, the old/new medians are 19.833500 / 12.665150 ms
(36.1% lower new median in this survey). For 3840x2160 / 4 they are
112.130250 / 77.416250 ms (31.0% lower). Reported paired ratios are geometric
means of paired old/new ratios, rather than ratios of those medians. The
smallest case does not establish a speed benefit. There are no zero samples.

The measurement source/executable hash manifest was verified after the measurement
attempts: all 353 hashes matched, including the native benchmark executable.
The measured source is commit `ed5314c45787ef72f31d9a40d5b599b6d8495151`.
The final revision additionally gates the owned-byte wrapper to native targets:
CI's warnings-denied WASM build detected that its remaining histogram consumers
are native-only. This attribute changes no native function body or measured
native call path; the mapping and tensor decoder remain identical to the
measured version. The original measurement manifest and executable are retained
separately from the final source/executable hashes. Local WASM validation now
denies warnings for both the facade and CI's exact `gpu-image,gpu-aoso-staging`
GPU crate configuration. All 120 local tests, Clippy, CPU-only and both
warnings-denied WASM configurations passed after the fix. A separate final
353-file manifest was captured after validation. Its source audit confirms that
every measured source except the wrapper's native-target attribute is unchanged;
the archived measured executable still matches its original hash.

Raw attempts, the accepted eight-row survey,
bootstrap output, validation logs and hash manifest remain local under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/` with the
`ai-readback-` prefix. The published table contains every accepted condition
and no device model name. A second isolated survey remains follow-up evidence;
it is not counted as completed validation.
