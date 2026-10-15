use super::*;
use spatialrust_image::{AlphaMode, ColorRange, ColorSpace, ImageRegion};

#[path = "packed_upload_reference.rs"]
mod parent;

fn fixture<const C: usize>(w: usize, h: usize, layout: usize) -> (Vec<u8>, usize, ImageMetadata) {
    let stride = if layout == 0 { w * C } else { (w + 5) * C + 11 };
    let rows = if layout == 0 { h } else { h + 4 };
    let source =
        (0..stride * rows + 13).map(|i| ((i * 37 + i / stride * 19) % 256) as u8).collect();
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
    (source, stride, metadata)
}

fn view<const C: usize>(
    w: usize,
    h: usize,
    layout: usize,
    data: &[u8],
    stride: usize,
    metadata: ImageMetadata,
) -> ImageView<'_, u8, C> {
    match layout {
        0 => ImageView::new_with_metadata(w, h, stride, data, metadata).unwrap(),
        1 => ImageView::new_with_metadata(w + 5, h + 4, stride, data, metadata)
            .unwrap()
            .subview(ImageRegion::new(2, 1, w, h))
            .unwrap(),
        2 => ImageView::new_with_metadata(w + 5, h + 4, stride, data, metadata)
            .unwrap()
            .subview(ImageRegion::new(2, 1, w, 1))
            .unwrap(),
        _ => unreachable!(),
    }
}

fn check_host<const C: usize>() {
    for layout in 0..3 {
        let (source, stride, metadata) = fixture::<C>(65, 7, layout);
        let input = view::<C>(65, 7, layout, &source, stride, metadata);
        let payload = upload_texels(input, input.width() * input.height() * 4);
        assert_eq!(&*payload, pack_rgba8(input, input.width() * input.height() * 4));
        match payload {
            Cow::Borrowed(pixels) => {
                assert_eq!(C, 4);
                assert!(layout == 0 || layout == 2);
                assert_eq!(pixels.as_ptr(), input.row(0).unwrap().as_ptr());
                assert_eq!(pixels.len(), input.width() * input.height() * 4);
            }
            Cow::Owned(_) => assert!(C != 4 || layout == 1),
        }
    }
}

#[test]
fn upload_payload_borrows_exact_dense_rgba_and_one_row_roi() {
    check_host::<1>();
    check_host::<2>();
    check_host::<3>();
    check_host::<4>();
    let data = [17; 17];
    let input = ImageView::<_, 4>::new(4, 1, usize::MAX, &data).unwrap();
    assert!(
        matches!(upload_texels(input, 16), Cow::Borrowed(p) if p.len() == 16 && p.as_ptr() == data.as_ptr())
    );
}

fn check_gpu<const C: usize>(r: &WgpuRuntime) -> usize {
    let mut bytes = 0;
    for (w, h) in [(1, 1), (65, 3), (1024, 1025)] {
        for layout in 0..3 {
            let (mut current, mut prior, expected, metadata, height) = {
                let (mut source, stride, metadata) = fixture::<C>(w, h, layout);
                let input = view::<C>(w, h, layout, &source, stride, metadata);
                let expected: Vec<u8> = (0..input.height())
                    .flat_map(|y| input.row(y).unwrap().iter().copied())
                    .collect();
                let current = GpuImage::upload_u8(r, input).unwrap();
                let prior = GpuImage::parent_upload_u8(r, input).unwrap();
                let height = input.height();
                source.fill(231);
                (current, prior, expected, metadata, height)
            };
            assert_eq!(current.receipt(), prior.receipt());
            assert_eq!(current.receipt().host_to_device_bytes(), (w * height * 4) as u64);
            assert_eq!(current.receipt().device_to_host_bytes(), 0);
            assert_eq!(current.receipt().stages(), &["upload_u8_texture"]);
            assert_eq!(current.metadata(), metadata);
            let mut physical = GpuImage::from_parts(
                r,
                w as u32,
                height as u32,
                4,
                current.texture().clone(),
                ImageMetadata::default(),
                GpuImageReceipt::default(),
            )
            .unwrap();
            let rgba = physical.readback_u8::<4>(r).unwrap();
            for (pixel, logical) in rgba.as_slice().chunks_exact(4).zip(expected.chunks_exact(C)) {
                assert_eq!(&pixel[..C], logical);
                assert!(pixel[C..].iter().all(|&v| v == 0));
            }
            for _ in 0..2 {
                let actual = current.readback_u8::<C>(r).unwrap();
                assert_eq!(actual, prior.readback_u8::<C>(r).unwrap());
                assert_eq!(actual.as_slice(), expected);
                assert_eq!(actual.metadata(), metadata);
                assert_eq!(current.receipt(), prior.receipt());
                bytes += expected.len() * 2;
            }
            bytes += rgba.as_slice().len();
            drop(physical);
            current.recycle(r);
            prior.recycle(r);
        }
    }
    bytes
}

#[test]
fn packed_upload_matches_parent_after_source_mutation_and_drop() {
    let r = WgpuRuntime::new_headless().unwrap();
    let bytes = check_gpu::<1>(&r) + check_gpu::<2>(&r) + check_gpu::<3>(&r) + check_gpu::<4>(&r);
    eprintln!("PACKED_UPLOAD verified {bytes} physical/logical bytes across 36 fixtures and repeated readbacks");
}

#[test]
fn concurrent_packed_uploads_retain_independent_live_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    std::thread::scope(|scope| {
        for value in [0, 17, 129, 255] {
            let r = &r;
            scope.spawn(move || {
                let (mut a, mut b) = {
                    let host = Image::<u8, 4>::try_new(129, 71, vec![value; 129 * 71 * 4]).unwrap();
                    (
                        GpuImage::upload_u8(r, host.view()).unwrap(),
                        GpuImage::upload_u8(r, host.view()).unwrap(),
                    )
                };
                for image in [&mut a, &mut b] {
                    assert!(image
                        .readback_u8::<4>(r)
                        .unwrap()
                        .as_slice()
                        .iter()
                        .all(|&v| v == value));
                }
                a.recycle(r);
                b.recycle(r);
            });
        }
    });
    assert!(r.image_texture_pool_stats().cached_bytes > 0);
}

#[path = "packed_upload_benchmark.rs"]
mod benchmark;
