use super::*;

fn measure<const CHANNELS: usize>(runtime: &WgpuRuntime) {
    for (width, height) in [(640, 480), (1920, 1080), (3840, 2160)] {
        let (source, stride, metadata) = fixture::<CHANNELS>(width, height);
        let input = view::<CHANNELS>(width, height, &source, stride, metadata);
        let expected: Vec<_> =
            (0..height).flat_map(|y| input.row(y).unwrap().iter().copied()).collect();
        let mut images = [
            GpuImage::previous_upload_u8(runtime, input).unwrap(),
            GpuImage::upload_u8(runtime, input).unwrap(),
        ];
        runtime.queue().submit(std::iter::empty());
        runtime.wait_idle();
        for readback in [false, true] {
            let mut times: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            for round in 0..23 {
                for offset in 0..2 {
                    let mode = (round + offset) % 2;
                    let start = std::time::Instant::now();
                    let elapsed;
                    if readback {
                        let output = if mode == 0 {
                            images[mode].previous_readback_u8::<CHANNELS>(runtime).unwrap()
                        } else {
                            images[mode].readback_u8::<CHANNELS>(runtime).unwrap()
                        };
                        elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        assert_eq!(output.as_slice(), expected);
                    } else {
                        let mut output = if mode == 0 {
                            GpuImage::previous_upload_u8(runtime, input).unwrap()
                        } else {
                            GpuImage::upload_u8(runtime, input).unwrap()
                        };
                        runtime.queue().submit(std::iter::empty());
                        runtime.wait_idle();
                        elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        assert_eq!(
                            output.readback_u8::<CHANNELS>(runtime).unwrap().as_slice(),
                            expected
                        );
                        output.recycle(runtime);
                    }
                    if round >= 3 {
                        times[mode].push(elapsed);
                    }
                }
            }
            eprintln!("PAIRED {{\"channels\":{CHANNELS},\"width\":{width},\"height\":{height},\"readback\":{readback},\"previous_ms\":{:?},\"current_ms\":{:?}}}",times[0],times[1]);
        }
        for image in images {
            image.recycle(runtime);
        }
    }
}

#[test]
#[ignore = "native paired completed image uploads/readbacks; run alone in release mode"]
fn paired_serial_current_gpu_image_transfers() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    eprintln!("CONDITIONS backend={} device_type={} available_workers={} warmups=3 pairs=20 boundary=completed_public_transfer upload_wait_included=true",runtime.adapter_info().backend,runtime.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    measure::<1>(&runtime);
    measure::<2>(&runtime);
    measure::<3>(&runtime);
    measure::<4>(&runtime);
}
