# Vision 2 component baseline receipt — 2026-07-16

## Scope

Epic 112 adds measurement and documentation only; no production kernel changed.
The probe is packed RGB8 bilinear half-scale resize with caller-owned output.
Throughput uses input pixels. Python conversion is an explicit strided-to-packed
copy, allocation is an output-sized `numpy.empty`, and the native stage reuses
that output. Upload, device execution, and readback are reasoned
`not_applicable` stages for this CPU receipt.

The strict receipt contract and runner are in
`C:\Users\rsasa\Workspace\SpatialRust\bench\vision2_baseline\report.py` and
`C:\Users\rsasa\Workspace\SpatialRust\bench\vision2_baseline\run.py`.
The matched eight-workload Criterion mapping is in
`C:\Users\rsasa\Workspace\SpatialRust\bench\vision2_baseline\manifest.json`.

## Host and policy

- Host: Windows 11 10.0.26300, AMD64
- CPU: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel
- Logical CPUs: 12
- Batch policy: adaptive batches with at least 20 ms per sample, 10 samples
- Single mode: `RAYON_NUM_THREADS=1`, fresh process
- Default mode: `RAYON_NUM_THREADS` unset, fresh process; recorded worker policy 12

## Measured medians

| Profile | Mode | Python conversion | Allocation | Native kernel | Native MPix/s | Native ns/pixel | Accounted output bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VGA | single | 1.719167 ms | 0.000643 ms | 3.395943 ms | 90.46 | 11.055 | 230,400 |
| VGA | default | 1.830638 ms | 0.000650 ms | 3.296800 ms | 93.18 | 10.732 | 230,400 |
| 1080p | single | 11.365325 ms | 0.011915 ms | 24.013950 ms | 86.35 | 11.581 | 1,555,200 |
| 1080p | default | 11.472750 ms | 0.012083 ms | 22.285900 ms | 93.05 | 10.747 | 1,555,200 |
| 4K | single | 45.976950 ms | 0.024781 ms | 99.188100 ms | 83.62 | 11.958 | 6,220,800 |
| 4K | default | 45.306750 ms | 0.020701 ms | 97.071750 ms | 85.45 | 11.703 | 6,220,800 |

Peak caller workspace is 0 bytes for all six runs. `bytes_allocated` is an
accounted output-size metric rather than a process allocator sample.

## Attribution

The native resize kernel is the largest measured component on every profile,
at roughly twice the explicit Python packing copy. Output allocation is below
0.025 ms even at 4K and is not the present bottleneck. Default threading changes
native median throughput only from 90.46 to 93.18 MPix/s at VGA, 86.35 to
93.05 MPix/s at 1080p, and 83.62 to 85.45 MPix/s at 4K on this host. These
host-specific results do not imply a portable threading or OpenCV comparison.

## Reproduction and validation

```powershell
python bench/vision2_baseline/test_report.py
$env:RAYON_NUM_THREADS = "1"
python bench/vision2_baseline/run.py --profile vga --thread-mode single --output target/vision2-baseline/vga-single.json
Remove-Item Env:RAYON_NUM_THREADS
python bench/vision2_baseline/run.py --profile vga --thread-mode default --output target/vision2-baseline/vga-default.json
```

The same commands were run for `1080p` and `4k`. Generated JSON receipts remain
under `C:\Users\rsasa\Workspace\SpatialRust\target\vision2-baseline` and are
not committed. Contract tests reject missing/duplicate stages, non-finite
metrics, and derived ns/pixel values that disagree with the native median.
