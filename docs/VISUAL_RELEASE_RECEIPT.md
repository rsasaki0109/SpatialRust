# Visual release receipt

Decision: **allowed**

Candidate date: `2026-07-28` (Unix day `20662`)

Migration policy: `visual-1`

| Measurement | Observed | Ceiling |
| --- | ---: | ---: |
| headless pixel mismatches | 0 | 0 |
| headless maximum channel delta | 0 | 0 |
| canonical geometry upload (bytes) | 40 | 8,388,608 |
| render uniform upload (bytes) | 112 | 112 |
| unexpected readback (bytes) | 0 | 0 |
| screenshot readback (bytes) | 16,384 | 16,384 |
| peak LOD host memory (bytes) | 25,165,824 | 67,108,864 |
| peak LOD GPU memory (bytes) | 20,971,520 | 67,108,864 |
| in-flight LOD chunks | 4 | 8 |
| browser requested bytes | 2,097,152 | 8,388,608 |
| Python explicit copy (bytes) | 3,145,728 | 12,582,912 |
| state round-trip mismatches | 0 | 0 |

Fresh required receipts captured on Unix day `20662`:

- [x] `visual-viz-contracts`
- [x] `visual-wgpu-renderer`
- [x] `visual-native-debug`
- [x] `visual-scene-rgbd`
- [x] `visual-bounded-lod`
- [x] `visual-web-viewer`
- [x] `visual-python-jupyter`

Required conformance is recorded exactly once with Pass for Linux, Windows, and
macOS headless rendering; native viewer state; wasm32 and real-browser smoke;
Python 3.8 and current Python; executable Jupyter notebook; LOD budgets;
transfer ledgers; docs; and unsafe audit. The gate rejects missing, skipped,
duplicate, future-dated, older-than-30-day, over-budget, or wrong-migration
evidence.

Regenerate the machine-evaluated form with:

```text
cargo run -p spatialrust --no-default-features --features platform \
  --example visual_release_gate
```
