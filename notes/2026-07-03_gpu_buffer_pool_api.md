# GPU buffer pool public API（Epic 64 / 2026-07-03）

## 結論

Epic 46 の upload pool を **`GpuBufferPool` として public API 化**し、`WgpuRuntime` から明示的にアクセスできるようにした。

## Public API

| 型 / メソッド | 役割 |
| --- | --- |
| `GpuBufferPool` | バイト長 keyed の storage buffer 再利用プール |
| `WgpuRuntime::buffer_pool()` | ランタイムが所有するプールへの参照 |
| `WgpuRuntime::upload_pod_storage` / `upload_f32_storage` / `upload_u32_storage` | プール経由アップロード |
| `WgpuRuntime::recycle_storage` | バッファ返却 |
| `WgpuRuntime::clear_buffer_pool` | キャッシュ破棄 |
| `GpuBufferPool::cached_buffer_count` | 診断用 |

## 使用例

```rust
use spatialrust::gpu::{GpuBufferPool, WgpuRuntime};

let runtime = WgpuRuntime::shared()?;
let xs = vec![0.0_f32; 1024];
let buffer = runtime.upload_f32_storage("my-kernel-x", &xs)?;
// ... dispatch ...
runtime.recycle_storage(buffer.size(), buffer);

// またはプール直接:
let pool: &GpuBufferPool = runtime.buffer_pool();
pool.clear();
```

## テスト

```bash
cargo test -p spatialrust-gpu --features gpu-wgpu buffer_pool --release
```

## 次アクション

1. v1.0 API audit 仕上げ（Epic 62 checklist）
2. downsample 後 plane GPU 閾値チューニング（MVP Auto）
