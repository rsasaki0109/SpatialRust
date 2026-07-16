# Vision 2 release receipt

Decision: **allowed**

The canonical values below are taken from the dated 2026-07-16 receipts on the
recorded Intel Core i7-9750H Windows host. They are release evidence for these
named workloads, not portable latency guarantees.

| Measurement | Observed | Ceiling |
| --- | ---: | ---: |
| Native RGB-to-gray allocate, 1080p | 648 us | 1,000 us |
| Native RGB-to-gray reuse, 1080p | 195 us | 400 us |
| Python RGB-to-gray allocate, 1080p | 825 us | 1,200 us |
| Python RGB-to-gray reuse, 1080p | 232 us | 400 us |
| Peak accounted CPU receipt bytes | 6,220,800 | 67,108,864 |
| Caller-output steady-state allocations | 0 | 0 |
| Default worker policy | 12 | 12 |
| 4K GPU-resident source upload | 33,177,600 bytes | 33,177,600 bytes |
| GPU-resident readback before request | 0 bytes | 0 bytes |

Required receipt families:

- [x] Vision 2 component baseline
- [x] resize and color
- [x] Gaussian and Sobel
- [x] morphology
- [x] Canny
- [x] GPU-resident chain

Reproduce the machine-checked decision and generated Markdown:

```powershell
cargo test -p spatialrust-platform vision2
cargo run -p spatialrust --no-default-features --features platform --example vision_2_release_gate
```
