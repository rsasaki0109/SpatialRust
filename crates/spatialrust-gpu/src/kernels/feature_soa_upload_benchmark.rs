use super::*;
use std::time::Instant;

enum Input {
    Fpfh(GpuFpfhInput),
    Iss(GpuIssInput),
    Mls(GpuMlsInput),
}
impl Input {
    fn receipt(&self) -> TransferStats {
        match self {
            Self::Fpfh(v) => v.upload_transfers(),
            Self::Iss(v) => v.upload_transfers(),
            Self::Mls(v) => v.upload_transfers(),
        }
    }
    fn count(&self) -> u32 {
        match self {
            Self::Fpfh(v) => v.point_count(),
            Self::Iss(v) => v.point_count(),
            Self::Mls(v) => v.point_count(),
        }
    }
}

#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_old_and_direct_soa_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    println!(
        "FEATURE_SOA_ENV backend={} device_type={} available_workers={}",
        info.backend,
        info.device_type,
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    for (n, profile) in [
        (1, 0),
        (257, 0),
        (4097, 0),
        (65_537, 0),
        (131_073, 0),
        (262_145, 0),
        (262_145, 1),
        (262_145, 2),
    ] {
        let (tree, _points, normals) = fixture(n, profile);
        let nodes = tree.layout().nodes().len() as u64;
        let mut samples: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
        for round in 0..23 {
            for offset in 0..6 {
                let stage = (round + offset) % 6;
                let start = Instant::now();
                let input = match stage {
                    0 => Input::Fpfh(
                        fpfh::soa_upload_reference::previous_upload(&r, &tree, &normals).unwrap(),
                    ),
                    1 => Input::Fpfh(GpuFpfhInput::upload(&r, &tree, &normals).unwrap()),
                    2 => Input::Iss(iss::soa_upload_reference::previous_upload(&r, &tree).unwrap()),
                    3 => Input::Iss(GpuIssInput::upload(&r, &tree).unwrap()),
                    4 => Input::Mls(mls::soa_upload_reference::previous_upload(&r, &tree).unwrap()),
                    _ => Input::Mls(GpuMlsInput::upload(&r, &tree).unwrap()),
                };
                r.wait_idle();
                let receipt = input.receipt();
                let count = input.count();
                drop(input);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(count, n as u32);
                assert_eq!(
                    receipt.host_to_device_bytes(),
                    n as u64 * (if stage < 2 { 64 } else { 32 }) + nodes * 32
                );
                assert_eq!(receipt.device_to_host_bytes(), 0);
                if round >= 3 {
                    samples[stage].push(ms);
                }
            }
        }
        println!("FEATURE_SOA_RECORD {{\"points\":{n},\"profile\":{profile},\"nodes\":{nodes},\"removed_host_bytes\":{},\"fpfh_old_ms\":{:?},\"fpfh_new_ms\":{:?},\"iss_old_ms\":{:?},\"iss_new_ms\":{:?},\"mls_old_ms\":{:?},\"mls_new_ms\":{:?}}}",n*12,samples[0],samples[1],samples[2],samples[3],samples[4],samples[5]);
    }
}
