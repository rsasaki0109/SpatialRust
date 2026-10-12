#![cfg(feature = "gpu-wgpu")]

use spatialrust::gpu::{build_voxel_segments_gpu_from_keys_buffer, WgpuRuntime};

#[test]
fn facade_large_voxel_segments_preserve_indices_and_scalar_transfer_boundary() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let count = 524289u32;
    let keys = vec![[-7i32, 0, 2, 0]; count as usize];
    let uploaded = runtime.upload_pod_storage("word-rank-smoke-keys", &keys).unwrap();
    let segments =
        build_voxel_segments_gpu_from_keys_buffer(&runtime, &uploaded, count, count).unwrap();
    assert_eq!(segments.cell_count(), 1);
    assert_eq!(segments.point_count(), count);
    assert_eq!(segments.construction_readback_bytes(), 4);
    let host = segments.to_voxel_segments(&runtime).unwrap();
    assert_eq!(host.keys, vec![(-7, 0, 2)]);
    assert_eq!(host.cell_starts, vec![0]);
    assert_eq!(host.cell_counts, vec![count]);
    assert_eq!(host.point_indices, (0..count).collect::<Vec<_>>());
    runtime.recycle_storage(u64::from(count) * 16, uploaded);
}
