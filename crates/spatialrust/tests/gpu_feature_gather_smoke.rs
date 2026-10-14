#![cfg(all(feature = "gpu-wgpu", not(target_arch = "wasm32")))]

use spatialrust::gpu::{GpuFpfhInput, GpuIssInput, GpuMlsInput, WgpuRuntime};
use spatialrust::search::KdTree;

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_gathered_feature_inputs_keep_owned_outputs_and_complete_receipts() {
    let r = WgpuRuntime::new_headless().unwrap();
    let x: Vec<_> = (0..25).map(|i| (i % 5) as f32 * 0.05).collect();
    let y: Vec<_> = (0..25).map(|i| (i / 5) as f32 * 0.05).collect();
    let z = vec![0.0; 25];
    let normals = vec![[0.0, 0.0, 1.0]; 25];
    let tree = KdTree::from_slices(&x, &y, &z);
    let node_bytes = tree.layout().nodes().len() as u64 * 32;
    let of = GpuFpfhInput::upload(&r, &tree, &normals).unwrap();
    let f = GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &normals).unwrap();
    let oi = GpuIssInput::upload(&r, &tree).unwrap();
    let i = GpuIssInput::upload_with_gpu_gather(&r, &tree).unwrap();
    let om = GpuMlsInput::upload(&r, &tree).unwrap();
    let m = GpuMlsInput::upload_with_gpu_gather(&r, &tree).unwrap();
    for (receipt, stride, device_stride) in [
        (f.upload_transfers(), 28, 64),
        (i.upload_transfers(), 16, 32),
        (m.upload_transfers(), 16, 32),
    ] {
        assert_eq!(receipt.host_to_device_bytes(), 25 * stride + node_bytes);
        assert_eq!(receipt.device_to_device_bytes(), 25 * device_stride);
        assert_eq!(receipt.device_to_host_bytes(), 0);
    }
    drop(tree);
    drop(x);
    drop(y);
    drop(z);
    drop(normals);
    let ofn = of.radius_neighborhoods_resident(&r, 0.35, 25).unwrap();
    let fnn = f.radius_neighborhoods_resident(&r, 0.35, 25).unwrap();
    let ofo = of.descriptors_resident(&r, &ofn).unwrap();
    let fo = f.descriptors_resident(&r, &fnn).unwrap();
    let oin = oi.neighborhoods_resident(&r, 0.35, 0.3, 25, 25).unwrap();
    let inn = i.neighborhoods_resident(&r, 0.35, 0.3, 25, 25).unwrap();
    let oio = oi.keypoints_resident(&r, &oin, 0.975, 0.975, 5).unwrap();
    let io = i.keypoints_resident(&r, &inn, 0.975, 0.975, 5).unwrap();
    let omn = om.neighborhoods_resident(&r, 0.35, 25).unwrap();
    let mn = m.neighborhoods_resident(&r, 0.35, 25).unwrap();
    let omo = om.smooth_resident(&r, &omn, 2, 5).unwrap();
    let mo = m.smooth_resident(&r, &mn, 2, 5).unwrap();
    drop(of);
    drop(f);
    drop(oi);
    drop(i);
    drop(om);
    drop(m);
    assert_eq!(ofo.download(&r).unwrap(), fo.download(&r).unwrap());
    assert_eq!(oio.download(&r).unwrap(), io.download(&r).unwrap());
    let a = omo.download(&r).unwrap();
    let b = mo.download(&r).unwrap();
    assert_eq!(a.positions, b.positions);
    assert_eq!(a.modes, b.modes);
    assert_eq!(a.diagnostics, b.diagnostics);
    assert_eq!(a.transfers, b.transfers);
}
