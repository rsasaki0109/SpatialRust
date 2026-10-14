#![cfg(all(feature = "gpu-wgpu", not(target_arch = "wasm32")))]

use spatialrust::gpu::{GpuGicpInput, GpuIcpInput, GpuPointToPlaneInput, WgpuRuntime};
use spatialrust::math::Isometry3;
use spatialrust::search::KdTree;

#[test]
#[ignore = "requires native wgpu adapter"]
fn facade_registration_gather_keeps_owned_equations_and_complete_receipts() {
    let r = WgpuRuntime::new_headless().unwrap();
    let source: Vec<_> = (0..65)
        .map(|i| [(i % 5) as f32 * 0.125, (i / 5 % 5) as f32 * 0.125, (i / 25) as f32 * 0.125])
        .collect();
    let axes: [Vec<_>; 3] = std::array::from_fn(|a| source.iter().map(|p| p[a]).collect());
    let tree = KdTree::from_slices(&axes[0], &axes[1], &axes[2]);
    let cov = vec![[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]; 65];
    let normals = vec![[0.0, 0.0, 2.0]; 65];
    let nodes = tree.layout().nodes().len() as u64;
    let old = GpuIcpInput::upload(&r, &source, &tree).unwrap();
    let new = GpuIcpInput::upload_with_gpu_gather(&r, &source, &tree).unwrap();
    let og = GpuGicpInput::upload(&r, &source, &cov, &tree, &cov).unwrap();
    let ng = GpuGicpInput::upload_with_gpu_gather(&r, &source, &cov, &tree, &cov).unwrap();
    let op = GpuPointToPlaneInput::upload(&r, &source, &tree, &normals).unwrap();
    let np = GpuPointToPlaneInput::upload_with_gpu_gather(&r, &source, &tree, &normals).unwrap();
    for (receipt, stride) in
        [(new.upload_transfers(), 32), (ng.upload_transfers(), 96), (np.upload_transfers(), 48)]
    {
        assert_eq!(receipt.host_to_device_bytes(), 65 * stride + nodes * 32);
        assert_eq!(receipt.device_to_device_bytes(), 65 * 32);
        assert_eq!(receipt.device_to_host_bytes(), 0);
    }
    let a = old.moments_resident(&r, Isometry3::identity(), 0.5).unwrap();
    let b = new.moments_resident(&r, Isometry3::identity(), 0.5).unwrap();
    let ga = og.equations_resident(&r, Isometry3::identity(), 0.5).unwrap();
    let gb = ng.equations_resident(&r, Isometry3::identity(), 0.5).unwrap();
    let pa = op.equations_resident(&r, Isometry3::identity(), 0.5).unwrap();
    let pb = np.equations_resident(&r, Isometry3::identity(), 0.5).unwrap();
    drop(old);
    drop(new);
    drop(og);
    drop(ng);
    drop(op);
    drop(np);
    drop(tree);
    drop(source);
    drop(normals);
    drop(cov);
    drop(axes);
    assert_eq!(a.download(&r).unwrap(), b.download(&r).unwrap());
    assert_eq!(ga.download(&r).unwrap(), gb.download(&r).unwrap());
    assert_eq!(pa.download(&r).unwrap(), pb.download(&r).unwrap());
}
