use super::*;
use crate::image::histogram::{histogram_u8_gpu, readback_bins_u32, GpuImageHistogram};
use crate::image::GpuImage;
use spatialrust_image::{ImageRegion, ImageView};
use wgpu::util::DeviceExt;

#[path = "mapped_readback_reference.rs"]
mod baseline;

fn raw_source(r: &WgpuRuntime, words: &[u32; 256]) -> wgpu::Buffer {
    // Explicit LE bytes also exercise decoding without native alignment casts.
    let bytes: Vec<_> = words
        .iter()
        .chain(std::iter::repeat(&0xdead_beef).take(16))
        .flat_map(|word| word.to_le_bytes())
        .collect();
    r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("histogram-mapped-readback-raw-source"),
        contents: &bytes,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    })
}

#[test]
fn mapped_histogram_readback_keeps_raw_u32_bits_prefix_and_owned_results() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for seed in 0..64_u32 {
        let expected = std::array::from_fn(|i| match (i + seed as usize) % 8 {
            0 => 0,
            1 => u32::MAX,
            2 => 0x0102_0304,
            3 => 0x8000_0001,
            _ => (i as u32).wrapping_mul(0x9e37_79b9).rotate_left(seed % 32),
        });
        let buffer = raw_source(&r, &expected);
        let live = readback_bins_u32(&r, &buffer).unwrap();
        assert_eq!(live, expected);
        assert_eq!(baseline::baseline_readback_bins_u32(&r, &buffer).unwrap(), expected);
        let changed: [u32; 256] = std::array::from_fn(|i| expected[i].reverse_bits());
        let bytes: Vec<_> = changed.iter().flat_map(|word| word.to_le_bytes()).collect();
        r.queue().write_buffer(&buffer, 0, &bytes);
        assert_eq!(readback_bins_u32(&r, &buffer).unwrap(), changed);
        assert_eq!(baseline::baseline_readback_bins_u32(&r, &buffer).unwrap(), changed);
        drop(buffer);
        r.clear_buffer_pool();
        assert_eq!(live, expected);
        checked += 5 * 256;
    }
    eprintln!("HISTOGRAM_MAPPED_RAW verified {checked} original/CPU u32 words, LE patterns, source prefix and owned results");
}

fn cumulative(bins: &[u32; 256]) -> [u32; 256] {
    let mut sum = 0_u64;
    std::array::from_fn(|i| {
        sum += u64::from(bins[i]);
        u32::try_from(sum).unwrap()
    })
}

fn check_public<const C: usize>(r: &WgpuRuntime) -> usize {
    let mut checked = 0;
    for (w, h) in [(1, 1), (17, 19), (257, 129)] {
        for roi in [false, true] {
            let stride = (w + 3) * C + 11;
            let mut data: Vec<_> =
                (0..stride * (h + 4)).map(|i| ((i * 37 + i / stride * 19) % 256) as u8).collect();
            let whole = ImageView::<u8, C>::new(w + 3, h + 4, stride, &data).unwrap();
            let view = whole.subview(ImageRegion::new(usize::from(roi), 2, w, h)).unwrap();
            let mut expected = [[0_u32; 256]; C];
            for y in 0..h {
                for pixel in view.row(y).unwrap().chunks_exact(C) {
                    for channel in 0..C {
                        expected[channel][pixel[channel] as usize] += 1;
                    }
                }
            }
            let source = GpuImage::upload_u8(r, view).unwrap();
            data.fill(231);
            for (channel, bins) in expected.iter().enumerate() {
                let mut current = histogram_u8_gpu(r, &source, channel as u32).unwrap();
                let mut prior = histogram_u8_gpu(r, &source, channel as u32).unwrap();
                let mut current_cdf = histogram_cdf_gpu(r, &current).unwrap();
                let mut prior_cdf = histogram_cdf_gpu(r, &prior).unwrap();
                let cdf_expected = cumulative(bins);
                let mut live = ([0; 256], [0; 256]);
                for round in 0..2 {
                    live = (current.readback_u32(r).unwrap(), current_cdf.readback_u32(r).unwrap());
                    assert_eq!(live.0, *bins);
                    assert_eq!(live.1, cdf_expected);
                    assert_eq!(prior.baseline_readback_u32(r).unwrap(), *bins);
                    assert_eq!(prior_cdf.baseline_readback_u32(r).unwrap(), cdf_expected);
                    assert_eq!(current.receipt(), prior.receipt());
                    assert_eq!(current_cdf.receipt(), prior_cdf.receipt());
                    assert_eq!(current.receipt().device_to_host_bytes(), (round + 1) * 1024);
                    assert_eq!(current_cdf.receipt().device_to_host_bytes(), (round + 1) * 1024);
                    assert_eq!(current.receipt().stages().last(), Some(&"readback_histogram_u32"));
                    assert_eq!(current_cdf.receipt().stages().last(), Some(&"readback_cdf_u32"));
                    checked += 4 * 256;
                }
                assert_eq!(current.pixel_count(), (w * h) as u32);
                assert_eq!(current_cdf.pixel_count(), (w * h) as u32);
                assert_eq!(current.channel(), channel as u32);
                assert_eq!(current_cdf.channel(), channel as u32);
                current.recycle(r);
                prior.recycle(r);
                current_cdf.recycle(r);
                prior_cdf.recycle(r);
                r.clear_buffer_pool();
                assert_eq!(live.0, *bins);
                assert_eq!(live.1, cdf_expected);
                checked += 2 * 256;
            }
            source.recycle(r);
        }
    }
    checked
}

#[test]
fn mapped_histogram_and_cdf_public_readbacks_preserve_receipts_roi_and_rejection() {
    let r = WgpuRuntime::new_headless().unwrap();
    let checked = check_public::<1>(&r)
        + check_public::<2>(&r)
        + check_public::<3>(&r)
        + check_public::<4>(&r);
    let other = WgpuRuntime::new_headless().unwrap();
    let host = spatialrust_image::Image::<u8, 1>::try_new(1, 1, vec![255]).unwrap();
    let source = GpuImage::upload_u8(&r, host.view()).unwrap();
    let mut histogram = histogram_u8_gpu(&r, &source, 0).unwrap();
    let mut cdf = histogram_cdf_gpu(&r, &histogram).unwrap();
    let receipts = (histogram.receipt().clone(), cdf.receipt().clone());
    let pools = (other.buffer_pool().stats(), other.scratch_buffer_pool_stats());
    assert_eq!(
        histogram.readback_u32(&other).unwrap_err().to_string(),
        histogram.baseline_readback_u32(&other).unwrap_err().to_string()
    );
    assert_eq!(
        cdf.readback_u32(&other).unwrap_err().to_string(),
        cdf.baseline_readback_u32(&other).unwrap_err().to_string()
    );
    assert_eq!(histogram.receipt(), &receipts.0);
    assert_eq!(cdf.receipt(), &receipts.1);
    assert_eq!(pools, (other.buffer_pool().stats(), other.scratch_buffer_pool_stats()));
    assert_eq!(other.initialized_image_pipeline_count(), 0);
    eprintln!("HISTOGRAM_MAPPED_PUBLIC verified {checked} original/CPU histogram/CDF words, channels 1-4, strided views, receipts, recycling and foreign-runtime rejection");
}

#[test]
fn mapped_histogram_and_cdf_concurrent_readbacks_keep_independent_owned_arrays() {
    let r = WgpuRuntime::new_headless().unwrap();
    let barrier = std::sync::Barrier::new(4);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = [0_u8, 19, 129, 255]
            .into_iter()
            .map(|value| {
                let r = &r;
                let barrier = &barrier;
                scope.spawn(move || {
                    let host = spatialrust_image::Image::<u8, 1>::try_new(
                        257,
                        129,
                        vec![value; 257 * 129],
                    )
                    .unwrap();
                    let source = GpuImage::upload_u8(r, host.view()).unwrap();
                    let mut histogram = histogram_u8_gpu(r, &source, 0).unwrap();
                    let mut cdf = histogram_cdf_gpu(r, &histogram).unwrap();
                    let mut bins = [0; 256];
                    bins[value as usize] = 257 * 129;
                    let expected_cdf = cumulative(&bins);
                    barrier.wait();
                    let live = (0..3)
                        .map(|_| {
                            let h = histogram.readback_u32(r).unwrap();
                            let c = cdf.readback_u32(r).unwrap();
                            assert_eq!(h, bins);
                            assert_eq!(c, expected_cdf);
                            (h, c)
                        })
                        .collect::<Vec<_>>();
                    histogram.recycle(r);
                    cdf.recycle(r);
                    source.recycle(r);
                    (live, bins, expected_cdf)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    r.clear_buffer_pool();
    r.clear_image_texture_pool();
    drop(r);
    let mut checked = 0;
    for (live, bins, expected_cdf) in results {
        for (h, c) in live {
            assert_eq!(h, bins);
            assert_eq!(c, expected_cdf);
            checked += 2 * 256;
        }
    }
    eprintln!("HISTOGRAM_MAPPED_CONCURRENT verified {checked} retained histogram/CDF words after four callers and runtime drop");
}
