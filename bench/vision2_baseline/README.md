# Vision 2 component baseline

This harness attributes Python conversion, output allocation, and native kernel
latency independently. The CPU baseline records upload, device execution, and
readback as explicitly not applicable; later GPU receipts must measure those
stages rather than silently folding transfers into a kernel result.

`manifest.json` maps all eight OpenCV Vision workloads to matched native
Criterion groups at VGA, 1080p, and 4K. The executable receipt currently uses
packed RGB8 bilinear half-scale resize as the stable component-attribution
probe. Pixel throughput is based on input pixels. `bytes_allocated` is the
caller-owned output size, and this resize path needs no caller workspace.

Run each thread policy in a fresh process because Rayon policy is process-wide:

```powershell
$env:RAYON_NUM_THREADS = "1"
python bench/vision2_baseline/run.py --profile vga --thread-mode single --output target/vision2-baseline/vga-single.json
Remove-Item Env:RAYON_NUM_THREADS
python bench/vision2_baseline/run.py --profile vga --thread-mode default --output target/vision2-baseline/vga-default.json
```

Repeat with `--profile 1080p` and `--profile 4k`. Receipts fail closed on
missing stages, duplicate stages, non-finite numbers, or inconsistent derived
ns/pixel values.

```powershell
python bench/vision2_baseline/test_report.py
```
