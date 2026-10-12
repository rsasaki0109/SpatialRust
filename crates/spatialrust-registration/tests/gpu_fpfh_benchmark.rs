#![cfg(all(feature = "register-fpfh-gpu", not(target_arch = "wasm32")))]

#[path = "gpu_fpfh.rs"]
mod reference;

use spatialrust_core::HasNormals3;
use spatialrust_gpu::{GpuFpfhInput, WgpuRuntime};
use spatialrust_registration::{fpfh_descriptors, fpfh_descriptors_gpu};
use std::time::Instant;

#[test]
#[ignore = "completed FPFH benchmark; run this exact test alone in release mode"]
fn completed_fpfh_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!("GPU_FPFH_ENV backend={} device_type={}", info.backend, info.device_type);
    for n in [257, 4097, 16384, 65537] {
        for kind in [0, 1, 3] {
            let radius = 0.35;
            let capacity = 64;
            let cloud = reference::fixture(n, kind);
            let expected = fpfh_descriptors(&cloud, radius).unwrap();
            let tree = spatialrust_search::KdTree::from_point_cloud(&cloud).unwrap();
            let (nx, ny, nz) = cloud.normals3().unwrap();
            let normals: Vec<_> = (0..n).map(|i| [nx[i], ny[i], nz[i]]).collect();
            let input = GpuFpfhInput::upload(&runtime, &tree, &normals).unwrap();
            let neighborhoods =
                input.radius_neighborhoods_resident(&runtime, radius, capacity).unwrap();
            let cpu = || fpfh_descriptors(&cloud, radius).unwrap();
            let gpu =
                || fpfh_descriptors_gpu(&runtime, &cloud, radius, capacity).unwrap().descriptors;
            for _ in 0..3 {
                reference::compare(&cpu(), &expected, 0.002);
                reference::compare(&gpu(), &expected, 0.002);
                reference::compare(
                    &input
                        .descriptors_resident(&runtime, &neighborhoods)
                        .unwrap()
                        .download(&runtime)
                        .unwrap()
                        .0,
                    &expected,
                    0.002,
                );
            }
            let (mut cpu_ms, mut gpu_ms, mut resident_ms) = (Vec::new(), Vec::new(), Vec::new());
            for round in 0..20 {
                for offset in 0..3 {
                    let stage = (round + offset) % 3;
                    let start = Instant::now();
                    if stage == 2 {
                        let output = input.descriptors_resident(&runtime, &neighborhoods).unwrap();
                        runtime.wait_idle();
                        resident_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                        reference::compare(&output.download(&runtime).unwrap().0, &expected, 0.002);
                    } else {
                        let rows = if stage == 0 { cpu() } else { gpu() };
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        if stage == 0 {
                            cpu_ms.push(elapsed)
                        } else {
                            gpu_ms.push(elapsed)
                        };
                        reference::compare(&rows, &expected, 0.002);
                    }
                }
            }
            let workers = spatialrust_search::parallel_worker_count(n);
            println!("GPU_FPFH_BENCH n={n} kind={kind} workers={workers} cpu_ms={cpu_ms:?} gpu_ms={gpu_ms:?} resident_ms={resident_ms:?}");
        }
    }
}
