use super::*;
use spatialrust_image::{AlphaMode, ColorRange, ColorSpace, ImageRegion};

fn generous_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_texture_dimension_2d: u32::MAX,
        max_buffer_size: u64::MAX,
        ..Default::default()
    }
}

#[test]
fn transfer_plans_match_independent_row_padding_and_channel_lengths() {
    let limits = generous_limits();
    let mut cases = 0;
    for width in 1usize..=1025 {
        for height in [1usize, 3, 257] {
            for channels in 1..=4 {
                let upload = upload_plan(width, height, channels, &limits).unwrap();
                let readback =
                    readback_plan(width as u32, height as u32, channels, &limits).unwrap();
                let row = width as u64 * 4;
                let padded = (row + 255) & !255;
                assert_eq!((upload.width, upload.height), (width as u32, height as u32));
                assert_eq!(u64::from(upload.row_bytes), row);
                assert_eq!(upload.storage_bytes, row * height as u64);
                assert_eq!(upload.packed_len as u64, row * height as u64);
                assert_eq!(u64::from(readback.row_bytes), row);
                assert_eq!(u64::from(readback.padded_row), padded);
                assert_eq!(readback.staging_bytes, padded * height as u64);
                assert_eq!(readback.staging_len as u64, padded * height as u64);
                assert_eq!(readback.output_len, width * height * channels);
                cases += 1;
            }
        }
    }
    eprintln!("Verified {cases} independent upload/readback layouts.");
}

#[test]
fn transfer_plans_reject_dimensions_overflow_and_exact_buffer_boundaries() {
    let limits = generous_limits();
    for (w, h, channels) in [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 5)] {
        assert!(upload_plan(w, h, channels, &limits).is_err());
        assert!(readback_plan(w as u32, h as u32, channels, &limits).is_err());
    }
    let mut small = limits.clone();
    small.max_texture_dimension_2d = 65;
    assert!(upload_plan(65, 65, 4, &small).is_ok());
    assert!(readback_plan(65, 65, 4, &small).is_ok());
    for (w, h) in [(66, 1), (1, 66)] {
        assert!(upload_plan(w, h, 1, &small).is_err());
        assert!(readback_plan(w as u32, h as u32, 1, &small).is_err());
    }
    small.max_texture_dimension_2d = 0;
    assert!(upload_plan(1, 1, 1, &small).is_err());
    assert!(readback_plan(1, 1, 1, &small).is_err());
    for width in [u32::MAX / 4 + 1, u32::MAX] {
        assert!(upload_plan(width as usize, 1, 1, &limits).is_err());
        assert!(readback_plan(width, 1, 1, &limits).is_err());
    }
    assert_eq!(
        upload_plan((u32::MAX / 4) as usize, 1, 4, &limits).is_ok(),
        host_length(u64::from(u32::MAX / 4) * 4).is_ok()
    );
    assert!(readback_plan(u32::MAX / 4, 1, 4, &limits).is_err());
    let width = (u32::MAX - 255) / 4;
    assert_eq!(
        readback_plan(width, 1, 4, &limits).is_ok(),
        host_length(u64::from(width) * 4).is_ok()
    );
    assert!(readback_plan(width + 1, 1, 4, &limits).is_err());
    assert!(upload_plan(usize::MAX, 1, 1, &limits).is_err());
    assert!(upload_plan(1, usize::MAX, 1, &limits).is_err());
    assert_eq!(host_length(isize::MAX as u64).unwrap(), isize::MAX as usize);
    assert!(host_length(isize::MAX as u64 + 1).is_err());
    assert!(host_length(u64::MAX).is_err());
    assert!(texture_bytes(u32::MAX, u32::MAX).is_err());
    assert_eq!(texture_bytes(u32::MAX, 1).unwrap(), u64::from(u32::MAX) * 4);
    // A padded 65x3 readback needs 1,536 bytes, not its 780 physical texel bytes.
    small = limits.clone();
    small.max_buffer_size = 1536;
    assert!(readback_plan(65, 3, 3, &small).is_ok());
    small.max_buffer_size = 1535;
    assert!(readback_plan(65, 3, 3, &small).is_err());
    // Uploads allocate no caller-visible GPU buffer; compute/storage binding
    // limits do not constrain either texture transfer.
    small.max_buffer_size = 0;
    small.max_storage_buffer_binding_size = 0;
    small.max_compute_workgroups_per_dimension = 0;
    assert!(upload_plan(65, 3, 3, &small).is_ok());
    assert!(readback_plan(65, 3, 3, &small).is_err());
    small.max_buffer_size = 1536;
    assert!(readback_plan(65, 3, 3, &small).is_ok());
    // Large plans are arithmetic probes; no giant image allocation occurs.
    assert!(upload_plan((u32::MAX / 4) as usize, u32::MAX as usize, 4, &limits).is_err());
    assert!(readback_plan(width, u32::MAX, 4, &limits).is_err());
}

fn roundtrip_cases<const CHANNELS: usize>(runtime: &WgpuRuntime) -> usize {
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
    let mut cases = 0;
    for (width, height) in [
        (1, 1),
        (1, 17),
        (17, 1),
        (63, 3),
        (64, 5),
        (65, 7),
        (127, 9),
        (128, 11),
        (129, 13),
        (257, 19),
    ] {
        for mode in 0..3 {
            let (parent_w, parent_h, ox, oy) =
                if mode == 2 { (width + 5, height + 7, 2, 3) } else { (width, height, 0, 0) };
            let stride = parent_w * CHANNELS + if mode == 0 { 0 } else { 11 };
            let mut data = vec![253; stride * parent_h];
            let value = |x: usize, y: usize, c: usize| ((x * 37 + y * 19 + c * 53) % 251) as u8;
            for y in 0..parent_h {
                for x in 0..parent_w {
                    for c in 0..CHANNELS {
                        data[y * stride + x * CHANNELS + c] = value(x, y, c);
                    }
                }
            }
            let view = ImageView::<u8, CHANNELS>::new_with_metadata(
                parent_w, parent_h, stride, &data, metadata,
            )
            .unwrap()
            .subview(ImageRegion::new(ox, oy, width, height))
            .unwrap();
            let mut logical = Vec::new();
            let mut rgba = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    for c in 0..4 {
                        let byte = if c < CHANNELS { value(x + ox, y + oy, c) } else { 0 };
                        rgba.push(byte);
                        if c < CHANNELS {
                            logical.push(byte);
                        }
                    }
                }
            }
            let plan = upload_plan(width, height, CHANNELS, &runtime.device().limits()).unwrap();
            assert_eq!(pack_rgba8(view, plan.packed_len), rgba);
            let mut gpu = GpuImage::upload_u8(runtime, view).unwrap();
            assert_eq!(gpu.metadata(), metadata);
            assert_eq!(gpu.receipt().host_to_device_bytes(), (width * height * 4) as u64);
            assert_eq!(gpu.receipt().device_to_host_bytes(), 0);
            let mut physical = GpuImage::from_parts(
                runtime,
                width as u32,
                height as u32,
                4,
                gpu.texture().clone(),
                ImageMetadata::default(),
                GpuImageReceipt::default(),
            )
            .unwrap();
            assert_eq!(physical.readback_u8::<4>(runtime).unwrap().as_slice(), rgba);
            drop(physical); // The alias must never enter the texture pool.
            for read in 1..=2 {
                let host = gpu.readback_u8::<CHANNELS>(runtime).unwrap();
                assert_eq!(host.as_slice(), logical);
                assert_eq!(host.metadata(), metadata);
                assert_eq!(
                    gpu.receipt().device_to_host_bytes(),
                    (read * width * height * 4) as u64
                );
                assert_eq!(gpu.receipt().stages().len(), read + 1);
            }
            gpu.recycle(runtime);
            cases += 1;
        }
    }
    cases
}

#[test]
fn packed_strided_roi_transfers_preserve_all_rgba_bytes_metadata_and_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cases = roundtrip_cases::<4>(&runtime)
        + roundtrip_cases::<3>(&runtime)
        + roundtrip_cases::<2>(&runtime)
        + roundtrip_cases::<1>(&runtime);
    assert_eq!(cases, 120);
    eprintln!("Verified {cases} independent packed/strided/ROI full-RGBA transfer cases.");
}

#[test]
fn invalid_public_transfers_leave_pools_receipts_and_validation_scopes_unchanged() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let image = Image::<u8, 1>::try_new(1, 1, vec![73]).unwrap();
    let mut source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let before = runtime.image_texture_pool_stats();
    let receipt = source.receipt().clone();
    runtime.device().push_error_scope(wgpu::ErrorFilter::Validation);
    for (w, h) in [(0, 1), (1, 0)] {
        let empty = ImageView::<u8, 1>::new(w, h, w, &[]).unwrap();
        assert!(GpuImage::upload_u8(&runtime, empty).is_err());
    }
    let invalid_channels = Image::<u8, 5>::try_new(1, 1, vec![0; 5]).unwrap();
    assert!(GpuImage::upload_u8(&runtime, invalid_channels.view()).is_err());
    let too_large = runtime.device().limits().max_texture_dimension_2d as usize + 1;
    for (w, h) in [(too_large, 1), (1, too_large)] {
        let image = Image::<u8, 1>::try_new(w, h, vec![0; w * h]).unwrap();
        assert!(GpuImage::upload_u8(&runtime, image.view()).is_err());
    }
    assert!(source.readback_u8::<1>(&other).is_err());
    assert!(source.readback_u8::<3>(&runtime).is_err());
    assert!(GpuImage::from_parts(
        &runtime,
        u32::MAX,
        u32::MAX,
        1,
        source.texture().clone(),
        ImageMetadata::default(),
        GpuImageReceipt::default(),
    )
    .is_err());
    // Metadata-only negative probes retain the tiny backing texture. They
    // must fail before staging/copy commands and are restored before recycling.
    for (w, h) in [(too_large as u32, 1), (1, too_large as u32), (0, 1)] {
        source.width = w;
        source.height = h;
        assert!(source.readback_u8::<1>(&runtime).is_err());
        assert_eq!(source.receipt(), &receipt);
    }
    source.width = 1;
    source.height = 1;
    assert!(pollster::block_on(runtime.device().pop_error_scope()).is_none());
    assert_eq!(runtime.image_texture_pool_stats(), before);
    assert_eq!(other.image_texture_pool_stats().allocation_count, 0);
    assert_eq!(runtime.initialized_image_pipeline_count(), 0);
    assert_eq!(source.readback_u8::<1>(&runtime).unwrap().as_slice(), &[73]);
    source.recycle(&runtime);
}

#[test]
fn concurrent_uploads_and_live_readbacks_survive_reuse_runtime_move_and_zero_budget() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4_u8)
            .map(|byte| {
                let runtime = &runtime;
                scope.spawn(move || {
                    let image = Image::<u8, 3>::try_new(65, 7, vec![byte; 65 * 7 * 3]).unwrap();
                    GpuImage::upload_u8(runtime, image.view()).unwrap()
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect::<Vec<_>>()
    });
    let moved = Box::new(runtime);
    for (byte, output) in outputs.iter_mut().enumerate() {
        assert_eq!(
            output.readback_u8::<3>(&moved).unwrap().as_slice(),
            vec![byte as u8; 65 * 7 * 3]
        );
    }
    for output in outputs.drain(..) {
        output.recycle(&moved);
    }
    let before = moved.image_texture_pool_stats();
    for byte in 1..=12 {
        let image = Image::<u8, 1>::try_new(65, 7, vec![byte; 65 * 7]).unwrap();
        let mut gpu = GpuImage::upload_u8(&moved, image.view()).unwrap();
        assert_eq!(gpu.readback_u8::<1>(&moved).unwrap().as_slice(), image.as_slice());
        gpu.recycle(&moved);
    }
    assert_eq!(moved.image_texture_pool_stats().allocation_count, before.allocation_count);
    assert!(moved.image_texture_pool_stats().reuse_count >= before.reuse_count + 12);
    let image = Image::<u8, 1>::try_new(65, 7, vec![91; 65 * 7]).unwrap();
    let mut live = GpuImage::upload_u8(&moved, image.view()).unwrap();
    moved.set_image_texture_pool_budget(0);
    assert_eq!(live.readback_u8::<1>(&moved).unwrap().as_slice(), image.as_slice());
    live.recycle(&moved);
    assert_eq!(moved.cached_image_texture_count(), 0);
}
