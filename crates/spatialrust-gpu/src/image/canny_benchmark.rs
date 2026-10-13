//! Completed CPU/API versus explicit uploaded/resident GPU comparisons.
use super::*;
use spatialrust_vision::{canny_into, CannyWorkspace};
use std::time::Instant;

fn measure(
    runtime: &WgpuRuntime,
    w: usize,
    h: usize,
    profile: u32,
    aperture_size: u32,
    l2_gradient: bool,
) {
    let image = fixture(w, h, profile);
    let o = GpuCannyOptions { aperture_size, l2_gradient, ..Default::default() };
    let expected = canny(image.view(), cpu_options(o)).unwrap();
    let mut into = Image::try_new(w, h, vec![0; w * h]).unwrap();
    let mut workspace = CannyWorkspace::new();
    let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
    for _ in 0..3 {
        assert_eq!(canny(image.view(), cpu_options(o)).unwrap(), expected);
        canny_into(image.view(), cpu_options(o), into.view_mut(), &mut workspace).unwrap();
        assert_eq!(into, expected);
        let mut result = canny_gpu(runtime, &source, o).unwrap();
        assert_eq!(result.readback_u8::<1>(runtime).unwrap(), expected);
        result.recycle(runtime);
        let uploaded = GpuImage::upload_u8(runtime, image.view()).unwrap();
        let mut result = canny_gpu(runtime, &uploaded, o).unwrap();
        assert_eq!(result.readback_u8::<1>(runtime).unwrap(), expected);
        uploaded.recycle(runtime);
        result.recycle(runtime);
    }
    let (mut cpu_ms, mut cpu_into_ms, mut full_gpu_ms, mut resident_ms) =
        (vec![], vec![], vec![], vec![]);
    let mut receipt = None;
    for round in 0..20 {
        for offset in 0..4 {
            let start = Instant::now();
            match (round + offset) % 4 {
                0 => {
                    let result = canny(image.view(), cpu_options(o)).unwrap();
                    cpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(result, expected);
                }
                1 => {
                    canny_into(image.view(), cpu_options(o), into.view_mut(), &mut workspace)
                        .unwrap();
                    cpu_into_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(into, expected);
                }
                2 => {
                    let uploaded = GpuImage::upload_u8(runtime, image.view()).unwrap();
                    let mut result = canny_gpu(runtime, &uploaded, o).unwrap();
                    let output = result.readback_u8::<1>(runtime).unwrap();
                    receipt = Some(result.receipt().clone());
                    uploaded.recycle(runtime);
                    result.recycle(runtime);
                    full_gpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(output, expected);
                }
                _ => {
                    let mut result = canny_gpu(runtime, &source, o).unwrap();
                    runtime.wait_idle();
                    resident_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    result.receipt().validate_resident_chain((w * h * 4) as u64).unwrap();
                    assert_eq!(result.readback_u8::<1>(runtime).unwrap(), expected);
                    result.recycle(runtime);
                }
            }
        }
    }
    let receipt = receipt.unwrap();
    assert_eq!(receipt.host_to_device_bytes(), (w * h * 4) as u64);
    assert_eq!(
        receipt.gpu_to_gpu_bytes(),
        (w * h * if aperture_size == 3 { 44 } else { 52 }) as u64
    );
    assert_eq!(receipt.device_to_host_bytes(), ((w * 4).div_ceil(256) * 256 * h) as u64);
    let edges = expected.as_slice().iter().filter(|&&v| v != 0).count();
    println!("GPU_CANNY_BENCH width={w} height={h} profile={profile} aperture={aperture_size} l2={} edges={edges} h2d={} g2g={} d2h={} cpu_ms={cpu_ms:?} cpu_into_ms={cpu_into_ms:?} full_gpu_ms={full_gpu_ms:?} resident_ms={resident_ms:?}",u8::from(l2_gradient),receipt.host_to_device_bytes(),receipt.gpu_to_gpu_bytes(),receipt.device_to_host_bytes());
    source.recycle(runtime);
}

#[test]
#[ignore = "run this exact completed-call survey alone in release mode"]
fn completed_canny_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!(
        "GPU_CANNY_ENV backend={} device_type={} cpu_workers={}",
        info.backend,
        info.device_type,
        rayon::current_num_threads()
    );
    measure(&runtime, 256, 192, 2, 3, false);
    measure(&runtime, 256, 192, 1, 7, true);
    for (profile, aperture, l2) in
        [(2, 3, false), (2, 3, true), (1, 3, false), (3, 3, true), (2, 5, false), (1, 7, true)]
    {
        measure(&runtime, 1920, 1080, profile, aperture, l2);
    }
    for (profile, aperture, l2) in [(2, 3, false), (1, 3, true), (2, 5, true), (3, 7, true)] {
        measure(&runtime, 3840, 2160, profile, aperture, l2);
    }
}
