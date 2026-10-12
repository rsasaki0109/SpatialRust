use super::*;
use spatialrust_image::{AlphaMode, ColorRange, ColorSpace, ImageRegion};

mod previous {
    use spatialrust_image::ImageView;
    include!("../../tests/fixtures/image_transfer_serial_packing.rs");
}
include!("../../tests/fixtures/image_transfer_serial_api.rs");

fn fixture<const CHANNELS: usize>(width: usize, height: usize) -> (Vec<u8>, usize, ImageMetadata) {
    let stride = (width + 5) * CHANNELS + 11;
    let source =
        (0..stride * (height + 7)).map(|i| ((i * 37 + i / stride * 19) % 251) as u8).collect();
    let metadata = ImageMetadata {
        color_space: match CHANNELS {
            1 => ColorSpace::Gray,
            3 => ColorSpace::Bgr,
            4 => ColorSpace::Bgra,
            _ => ColorSpace::Unknown,
        },
        color_range: ColorRange::Limited,
        alpha_mode: if CHANNELS == 4 { AlphaMode::Premultiplied } else { AlphaMode::None },
    };
    (source, stride, metadata)
}

fn view<const CHANNELS: usize>(
    width: usize,
    height: usize,
    source: &[u8],
    stride: usize,
    metadata: ImageMetadata,
) -> ImageView<'_, u8, CHANNELS> {
    ImageView::new_with_metadata(width + 5, height + 7, stride, source, metadata)
        .unwrap()
        .subview(ImageRegion::new(2, 3, width, height))
        .unwrap()
}

fn check<const CHANNELS: usize>(runtime: &WgpuRuntime) -> usize {
    let mut bytes = 0;
    for (width, height) in [(257, 19), (1297, 809)] {
        let (source, stride, metadata) = fixture::<CHANNELS>(width, height);
        let input = view::<CHANNELS>(width, height, &source, stride, metadata);
        let expected: Vec<_> =
            (0..height).flat_map(|y| input.row(y).unwrap().iter().copied()).collect();
        let mut current = GpuImage::upload_u8(runtime, input).unwrap();
        let mut prior = GpuImage::previous_upload_u8(runtime, input).unwrap();
        let initial = current.receipt().clone();
        assert_eq!(&initial, prior.receipt());
        assert_eq!(current.metadata(), metadata);
        let mut physical = GpuImage::from_parts(
            runtime,
            width as u32,
            height as u32,
            4,
            current.texture().clone(),
            ImageMetadata::default(),
            GpuImageReceipt::default(),
        )
        .unwrap();
        let physical_host = physical.readback_u8::<4>(runtime).unwrap();
        assert_eq!(physical_host.as_slice(), previous::pack_rgba8(input, width * height * 4));
        assert_eq!(
            previous::unpack_rgba8::<CHANNELS>(
                physical_host.as_slice(),
                width,
                height,
                width * 4,
                expected.len()
            ),
            expected
        );
        bytes += width * height * 4 + expected.len();
        drop(physical);
        for _ in 0..2 {
            let actual = current.readback_u8::<CHANNELS>(runtime).unwrap();
            let old = prior.previous_readback_u8::<CHANNELS>(runtime).unwrap();
            assert_eq!(actual.as_slice(), expected);
            assert_eq!(actual, old);
            assert_eq!(actual.metadata(), metadata);
            assert_eq!(current.receipt(), prior.receipt());
            bytes += expected.len() * 2;
        }
        current.recycle(runtime);
        prior.recycle(runtime);
    }
    bytes
}

#[test]
fn large_gpu_transfers_keep_frozen_rgba_logical_bytes_metadata_and_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let bytes =
        check::<1>(&runtime) + check::<2>(&runtime) + check::<3>(&runtime) + check::<4>(&runtime);
    eprintln!("TRANSFER_ROWS_GPU verified {bytes} frozen physical/logical bytes with shared current transfer planning and runtime pools.");
}

#[test]
fn concurrent_large_gpu_transfers_keep_live_outputs_and_pool_reuse() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    std::thread::scope(|scope| {
        for byte in 1..=4 {
            let runtime = &runtime;
            scope.spawn(move || {
                let source =
                    Image::<u8, 3>::try_new(1024, 1025, vec![byte; 1024 * 1025 * 3]).unwrap();
                let mut first = GpuImage::upload_u8(runtime, source.view()).unwrap();
                let mut second = GpuImage::upload_u8(runtime, source.view()).unwrap();
                assert_eq!(first.readback_u8::<3>(runtime).unwrap().as_slice(), source.as_slice());
                assert_eq!(second.readback_u8::<3>(runtime).unwrap().as_slice(), source.as_slice());
                first.recycle(runtime);
                second.recycle(runtime);
            });
        }
    });
    assert!(runtime.image_texture_pool_stats().cached_bytes > 0);
}

#[path = "transfer_packing_gpu_benchmark.rs"]
mod benchmark;
