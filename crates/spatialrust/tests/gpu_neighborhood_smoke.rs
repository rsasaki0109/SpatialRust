#![cfg(all(feature = "gpu-wgpu", not(target_arch = "wasm32")))]

use spatialrust::gpu::{GpuKdTree, GpuQueryBatch, WgpuRuntime};
use spatialrust::search::{KdTree, NearestNeighborIndex, RadiusSearchIndex};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_resident_neighborhoods_reuse_queries_and_reject_radius_overflow() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let x: Vec<_> = (0..257).map(|i| i as f32 * 0.5).collect();
    let yz = vec![0.0; x.len()];
    let tree = KdTree::from_slices(&x, &yz, &yz);
    let target = GpuKdTree::upload(&runtime, &tree).unwrap();
    let queries = GpuQueryBatch::upload(&runtime, &[[0.0; 3], [64.0, 0.0, 0.0]]).unwrap();
    let knn = target.nearest_k_resident(&runtime, &queries, 16).unwrap();
    assert_eq!(knn.execution_transfers().device_to_host_bytes(), 0);
    let rows = knn.download(&runtime).unwrap().into_complete_rows().unwrap();
    assert_eq!(rows, [tree.nearest_k(0.0, 0.0, 0.0, 16), tree.nearest_k(64.0, 0.0, 0.0, 16)]);
    let radius = target.radius_neighbors_resident(&runtime, &queries, 1.0, 1).unwrap();
    let downloaded = radius.download(&runtime).unwrap();
    assert_eq!(downloaded.total_counts(), [3, 5]);
    assert!(downloaded.complete_rows().is_err());
    let full = target
        .radius_neighbors_resident(&runtime, &queries, 1.0, 5)
        .unwrap()
        .download(&runtime)
        .unwrap();
    assert_eq!(
        full.complete_rows().unwrap(),
        [tree.radius_search(0.0, 0.0, 0.0, 1.0), tree.radius_search(64.0, 0.0, 0.0, 1.0)]
    );
    assert_eq!(full.transfers().device_to_host_bytes(), 2 * (8 + 5 * 8));
}
