use super::*;
use std::{hint::black_box, time::Instant};

fn measure<const C: usize>(r: &WgpuRuntime, w: usize, h: usize, layout: usize) {
    let (source, stride, metadata) = fixture::<C>(w, h, layout);
    let input = view::<C>(w, h, layout, &source, stride, metadata);
    let height = input.height();
    let len = w * height * 4;
    let expected: Vec<_> =
        (0..height).flat_map(|y| input.row(y).unwrap().iter().copied()).collect();
    for upload in [false, true] {
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let mode = (round + offset) % 2;
                let start = Instant::now();
                let elapsed;
                if upload {
                    let mut output = if mode == 0 {
                        GpuImage::parent_upload_u8(r, black_box(input)).unwrap()
                    } else {
                        GpuImage::upload_u8(r, black_box(input)).unwrap()
                    };
                    r.queue().submit(std::iter::empty());
                    r.wait_idle();
                    elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(output.readback_u8::<C>(r).unwrap().as_slice(), expected);
                    assert_eq!(output.metadata(), metadata);
                    assert_eq!(output.receipt().host_to_device_bytes(), len as u64);
                    output.recycle(r);
                } else {
                    let payload = if mode == 0 {
                        Cow::Owned(pack_rgba8(black_box(input), black_box(len)))
                    } else {
                        upload_texels(black_box(input), black_box(len))
                    };
                    black_box(payload.as_ref());
                    elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(payload.len(), len);
                    assert_eq!(&*payload, pack_rgba8(input, len));
                }
                if round >= 3 {
                    samples[mode].push(elapsed);
                }
            }
        }
        let route = if upload { "upload" } else { "host" };
        eprintln!("PACKED_UPLOAD_RECORD {{\"width\":{w},\"height\":{height},\"channels\":{C},\"layout\":{layout},\"{route}_old_ms\":{:?},\"{route}_ms\":{:?}}}", samples[0], samples[1]);
    }
}

#[test]
#[ignore = "paired host preparation and completed public uploads; run alone in release mode"]
fn benchmark_packed_rgba_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    eprintln!("PACKED_UPLOAD_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 host_drop_excluded=true upload_submit_wait_included=true readback_assert_recycle_excluded=true", r.adapter_info().backend, r.adapter_info().device_type, std::thread::available_parallelism().map_or(1, |n| n.get()));
    for (w, h) in [(64, 64), (257, 129), (1024, 1024), (1920, 1080), (3840, 2160), (1023, 1025)] {
        measure::<4>(&r, w, h, 0);
    }
    measure::<4>(&r, 8191, 1, 2);
    measure::<4>(&r, 1920, 1080, 1);
    measure::<3>(&r, 1920, 1080, 0);
    measure::<1>(&r, 1024, 1024, 0);
}
