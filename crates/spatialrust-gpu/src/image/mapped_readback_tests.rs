use super::*;
use spatialrust_image::{AlphaMode, ColorRange, ColorSpace, ImageRegion};

#[path = "mapped_readback_reference.rs"]
mod parent;

fn fixture<const C: usize>(w: usize, h: usize, roi: bool) -> (Vec<u8>, usize, ImageMetadata) {
    let stride = if roi { (w + 3) * C + 11 } else { w * C };
    let rows = if roi { h + 4 } else { h };
    let data = (0..stride * rows + 13).map(|i| ((i * 37 + i / stride * 19) % 256) as u8).collect();
    let metadata = ImageMetadata {
        color_space: match C {
            1 => ColorSpace::Gray,
            3 => ColorSpace::Bgr,
            4 => ColorSpace::Bgra,
            _ => ColorSpace::Unknown,
        },
        color_range: ColorRange::Limited,
        alpha_mode: if C == 4 { AlphaMode::Premultiplied } else { AlphaMode::None },
    };
    (data, stride, metadata)
}

fn view<const C: usize>(
    w: usize,
    h: usize,
    roi: bool,
    data: &[u8],
    stride: usize,
    metadata: ImageMetadata,
) -> ImageView<'_, u8, C> {
    if roi {
        ImageView::new_with_metadata(w + 3, h + 4, stride, data, metadata)
            .unwrap()
            .subview(ImageRegion::new(1, 2, w, h))
            .unwrap()
    } else {
        ImageView::new_with_metadata(w, h, stride, data, metadata).unwrap()
    }
}

fn check<const C: usize>(r: &WgpuRuntime) -> usize {
    let mut bytes = 0;
    for (w, h) in [(1, 1), (64, 3), (65, 3), (1023, 1025), (1024, 1024), (1024, 1025)] {
        for roi in [false, true] {
            let (mut new, mut old, expected, metadata) = {
                let (mut source, stride, metadata) = fixture::<C>(w, h, roi);
                let input = view::<C>(w, h, roi, &source, stride, metadata);
                let expected: Vec<u8> =
                    (0..h).flat_map(|y| input.row(y).unwrap().iter().copied()).collect();
                let new = GpuImage::upload_u8(r, input).unwrap();
                let old = GpuImage::upload_u8(r, input).unwrap();
                source.fill(231);
                (new, old, expected, metadata)
            };
            let mut live = new.readback_u8::<C>(r).unwrap();
            let prior = old.parent_readback_u8::<C>(r).unwrap();
            assert_eq!(live, prior);
            assert_eq!(live.as_slice(), expected);
            assert_eq!(live.metadata(), metadata);
            assert_eq!(new.receipt(), old.receipt());
            live[(0, 0)] = [231; C];
            let actual = new.readback_u8::<C>(r).unwrap();
            assert_eq!(actual, old.parent_readback_u8::<C>(r).unwrap());
            assert_eq!(actual.as_slice(), expected);
            assert_eq!(new.receipt(), old.receipt());
            assert_eq!(new.receipt().device_to_host_bytes(), (w * h * 8) as u64);
            assert_eq!(new.receipt().gpu_to_gpu_bytes(), 0);
            assert_eq!(
                new.receipt().stages(),
                &["upload_u8_texture", "readback_u8_texture", "readback_u8_texture"]
            );
            // Inspect all four physical components, including unused zero lanes.
            let mut physical = GpuImage::from_parts(
                r,
                w as u32,
                h as u32,
                4,
                new.texture().clone(),
                ImageMetadata::default(),
                GpuImageReceipt::default(),
            )
            .unwrap();
            let rgba = physical.readback_u8::<4>(r).unwrap();
            for (pixel, logical) in rgba.as_slice().chunks_exact(4).zip(expected.chunks_exact(C)) {
                assert_eq!(&pixel[..C], logical);
                assert!(pixel[C..].iter().all(|&b| b == 0));
            }
            bytes += expected.len() * 4 + rgba.as_slice().len();
            drop(physical);
            new.recycle(r);
            old.recycle(r);
            r.clear_image_texture_pool();
            assert_eq!(actual.as_slice(), expected);
            assert_eq!(live[(0, 0)], [231; C]);
        }
    }
    bytes
}

#[test]
fn direct_mapped_readback_preserves_parent_bytes_padding_metadata_and_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let bytes = check::<1>(&r) + check::<2>(&r) + check::<3>(&r) + check::<4>(&r);
    eprintln!("MAPPED_READBACK verified {bytes} physical/logical bytes across 48 fixtures");
}

#[test]
fn direct_mapped_readbacks_join_workers_before_unmapping_under_concurrent_reuse() {
    let r = WgpuRuntime::new_headless().unwrap();
    std::thread::scope(|scope| {
        for value in [0, 17, 129, 255] {
            let r = &r;
            scope.spawn(move || {
                let (mut a, mut b) = {
                    let host =
                        Image::<u8, 3>::try_new(1024, 1025, vec![value; 1024 * 1025 * 3]).unwrap();
                    (
                        GpuImage::upload_u8(r, host.view()).unwrap(),
                        GpuImage::upload_u8(r, host.view()).unwrap(),
                    )
                };
                let live = a.readback_u8::<3>(r).unwrap();
                for _ in 0..3 {
                    for image in [&mut a, &mut b] {
                        assert_eq!(image.readback_u8::<3>(r).unwrap(), live);
                    }
                }
                a.recycle(r);
                b.recycle(r);
                assert!(live.as_slice().iter().all(|&b| b == value));
            });
        }
    });
}

#[test]
fn mapped_decoder_limits_the_borrow_and_unmaps_after_success_error_and_panic() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let r = WgpuRuntime::new_headless().unwrap();
    let bytes: Vec<_> = (0..256).map(|i| i as u8).collect();
    let buffer = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("mapped-decoder-lifetime-test"),
        size: 256,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    r.queue().write_buffer(&buffer, 0, &bytes);
    r.queue().submit(std::iter::empty());
    struct OwnedSum(usize);
    let result = with_staging_bytes(r.device(), &buffer, 33, |data| {
        assert_eq!(data, &bytes[..33]);
        OwnedSum(data.iter().map(|&b| usize::from(b)).sum())
    })
    .unwrap();
    assert_eq!(result.0, 528);
    let result = with_staging_bytes(r.device(), &buffer, 0, |data| {
        assert!(data.is_empty());
        Err::<(), _>("decoder rejected data")
    })
    .unwrap();
    assert_eq!(result, Err("decoder rejected data"));
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ =
            with_staging_bytes(r.device(), &buffer, 256, |_| panic!("deliberate decoder panic"));
    }))
    .is_err());
    // Remapping the same buffer proves view-drop precedes unmapping on each exit.
    assert_eq!(read_staging_bytes(r.device(), &buffer, 256).unwrap(), bytes);
    assert_eq!(read_staging_bytes(r.device(), &buffer, 33).unwrap(), bytes[..33]);
}

#[path = "mapped_readback_benchmark.rs"]
mod benchmark;
