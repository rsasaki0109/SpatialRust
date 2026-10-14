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

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_gpu_tree_gather_counts_device_bytes_and_keeps_owned_queries() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let x: Vec<_> = (0..257).map(|i| i as f32 * 0.5).collect();
    let yz = vec![0.0; x.len()];
    let tree = KdTree::from_slices(&x, &yz, &yz);
    let expected = [tree.nearest_k(0.0, 0.0, 0.0, 8), tree.nearest_k(128.0, 0.0, 0.0, 8)];
    let node_count = tree.layout().nodes().len();
    let target = GpuKdTree::upload_with_gpu_gather(&runtime, &tree).unwrap();
    assert_eq!(target.upload_transfers().host_to_device_bytes(), 257 * 16 + node_count as u64 * 32);
    assert_eq!(target.upload_transfers().device_to_device_bytes(), 257 * 16);
    assert_eq!(target.upload_transfers().device_to_host_bytes(), 0);
    drop(tree);
    drop(x);
    drop(yz);
    let queries = GpuQueryBatch::upload(&runtime, &[[0.0; 3], [128.0, 0.0, 0.0]]).unwrap();
    let result = target.nearest_k_resident(&runtime, &queries, 8).unwrap();
    drop(target);
    drop(queries);
    assert_eq!(result.download(&runtime).unwrap().into_complete_rows().unwrap(), expected);
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_shared_tree_queries_avoid_duplicate_host_upload() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let x: Vec<_> = (0..257).map(|i| i as f32 * 0.5).collect();
    let yz = vec![0.0; x.len()];
    let tree = KdTree::from_slices(&x, &yz, &yz);
    let expected: Vec<_> = x.iter().map(|&x| tree.nearest_k(x, 0.0, 0.0, 8)).collect();
    let nodes = tree.layout().nodes().len();
    let (target, queries) = GpuKdTree::upload_and_queries_with_gpu_gather(&runtime, &tree).unwrap();
    assert_eq!(queries.query_count(), 257);
    assert_eq!(target.upload_transfers().host_to_device_bytes(), 257 * 16 + nodes as u64 * 32);
    assert_eq!(queries.upload_transfers().host_to_device_bytes(), 0);
    assert_eq!(target.upload_transfers().device_to_device_bytes(), 257 * 16);
    assert_eq!(queries.upload_transfers().device_to_device_bytes(), 257 * 16);
    assert_eq!(queries.upload_transfers().device_to_host_bytes(), 0);
    drop(tree);
    drop(x);
    drop(yz);
    let result = target.nearest_k_resident(&runtime, &queries, 8).unwrap();
    drop(target);
    drop(queries);
    assert_eq!(result.download(&runtime).unwrap().into_complete_rows().unwrap(), expected);
}
