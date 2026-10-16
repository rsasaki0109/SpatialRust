use super::*;
use spatialrust_image::{Image, ImageMetadata, ImageRegion, ImageView};
use spatialrust_vision::{median_blur, BorderMode};

fn border<const C: usize>(b: GpuImageBorder) -> BorderMode<u8, C> {
    match b {
        GpuImageBorder::Replicate => BorderMode::Replicate,
        GpuImageBorder::ConstantZero => BorderMode::Constant([0; C]),
    }
}

fn fixture<const C: usize>(w: usize, h: usize, profile: usize) -> Image<u8, C> {
    Image::try_new(
        w,
        h,
        (0..w * h * C)
            .map(|i| {
                let x = i / C % w;
                let y = i / (C * w);
                match profile {
                    0 => (i as u32)
                        .wrapping_mul(747_796_405)
                        .wrapping_add(2_891_336_453)
                        .rotate_left(13) as u8,
                    1 => 0,
                    2 => 255,
                    3 => {
                        if (x + y) % 2 == 0 {
                            0
                        } else {
                            255
                        }
                    }
                    4 => {
                        if (x + 3 * y) % 19 == 0 {
                            255
                        } else {
                            17
                        }
                    }
                    5 => (i % 256) as u8,
                    _ => [19, 73, 129, 251][i % C],
                }
            })
            .collect(),
    )
    .unwrap()
}

fn selected(values: &[u8], pairs: &[Comparator]) -> u8 {
    let mut sorted = [255; 32];
    sorted[..values.len()].copy_from_slice(values);
    for &Comparator { a, b, ascending } in pairs {
        let (low, high) = (sorted[a].min(sorted[b]), sorted[a].max(sorted[b]));
        (sorted[a], sorted[b]) = if ascending { (low, high) } else { (high, low) };
    }
    sorted[values.len() / 2]
}

#[test]
fn median_network_matches_exhaustive_3x3_binary_and_independent_byte_sort() {
    let pairs = network(3);
    for mask in 0..512_u32 {
        let values: [u8; 9] = std::array::from_fn(|i| if mask & (1 << i) == 0 { 0 } else { 255 });
        assert_eq!(selected(&values, &pairs), if mask.count_ones() >= 5 { 255 } else { 0 });
    }
    let mut cases = 0;
    for kernel in [3, 5] {
        let n = (kernel * kernel) as usize;
        let pairs = network(kernel);
        for seed in 0..100_000_u32 {
            let mut state = seed.wrapping_add(2_891_336_453);
            let mut values: Vec<_> = (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state as u8
                })
                .collect();
            let actual = selected(&values, &pairs);
            values.sort_unstable();
            assert_eq!(actual, values[n / 2]);
            cases += 1;
        }
        for value in 0..=255 {
            assert_eq!(selected(&vec![value; n], &pairs), value);
        }
        println!("median_network kernel={kernel} comparisons={}", pairs.len());
    }
    println!(
        "median_network_random_windows={cases} exhaustive_binary_windows=512 constant_windows=512"
    );
}

#[test]
fn median_plan_bounds_textures_coordinates_dispatch_and_uniform_before_allocation() {
    let limits = wgpu::Limits::default();
    assert_eq!(std::mem::size_of::<Params>(), 16);
    assert_eq!(plan(257, 259, 3, &limits).unwrap(), 257 * 259 * 4);
    for kernel in [0, 1, 2, 4, 7, u32::MAX] {
        assert!(plan(1, 1, kernel, &limits).is_err());
    }
    for (w, h) in [(0, 1), (1, 0), (8193, 1), (1, 8193), (u32::MAX, u32::MAX)] {
        assert!(plan(w, h, 5, &limits).is_err());
    }
    for field in 0..12 {
        let mut bad = limits.clone();
        match field {
            0 => bad.max_compute_workgroup_size_x = 15,
            1 => bad.max_compute_workgroup_size_y = 15,
            2 => bad.max_compute_workgroup_size_z = 0,
            3 => bad.max_compute_invocations_per_workgroup = 255,
            4 => bad.max_compute_workgroups_per_dimension = 0,
            5 => bad.max_sampled_textures_per_shader_stage = 0,
            6 => bad.max_storage_textures_per_shader_stage = 0,
            7 => bad.max_uniform_buffers_per_shader_stage = 0,
            8 => bad.max_uniform_buffer_binding_size = 15,
            9 => bad.max_buffer_size = 15,
            10 => bad.max_bind_groups = 0,
            _ => bad.max_bindings_per_bind_group = 2,
        }
        assert!(plan(1, 1, 3, &bad).is_err(), "field={field}");
    }
    let mut exact = limits.clone();
    exact.max_compute_workgroups_per_dimension = 2;
    assert!(plan(32, 32, 5, &exact).is_ok());
    assert!(plan(33, 32, 5, &exact).is_err());
    assert!(plan(32, 33, 5, &exact).is_err());
    exact = limits.clone();
    exact.max_buffer_size = 16;
    exact.max_uniform_buffer_binding_size = 16;
    exact.max_storage_buffer_binding_size = 0;
    exact.max_storage_buffers_per_shader_stage = 0;
    assert!(plan(8192, 8192, 5, &exact).is_ok());
    exact.max_texture_dimension_2d = u32::MAX;
    exact.max_compute_workgroups_per_dimension = u32::MAX;
    let largest = i32::MAX as u32 - 1;
    assert_eq!(plan(largest, largest, 5, &exact).unwrap(), u64::from(largest).pow(2) * 4);
    assert!(plan(largest + 1, 1, 5, &exact).is_err());
    assert!(plan(1, largest + 1, 5, &exact).is_err());
}

fn compare<const C: usize>(runtime: &WgpuRuntime) -> (usize, usize) {
    let mut calls = 0;
    let mut bytes = 0;
    for (w, h) in [(1, 1), (1, 9), (9, 1), (2, 3), (3, 2), (17, 11), (65, 33), (129, 127)] {
        for profile in 0..7 {
            let image = fixture::<C>(w, h, profile);
            let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
            for kernel in [3, 5] {
                for b in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                    let expected = median_blur(image.view(), kernel as usize, border(b)).unwrap();
                    let mut output = median_blur_gpu(runtime, &source, kernel, b).unwrap();
                    assert_eq!(output.channels(), C as u32);
                    assert_eq!(output.metadata(), image.metadata());
                    assert_eq!(output.receipt().host_to_device_bytes(), (w * h * 4) as u64);
                    assert_eq!(output.receipt().gpu_to_gpu_bytes(), (w * h * 4) as u64);
                    output.receipt().validate_resident_chain((w * h * 4) as u64).unwrap();
                    assert_eq!(output.receipt().stages(), ["upload_u8_texture", "median_blur_gpu"]);
                    assert_eq!(
                        output.readback_u8::<C>(runtime).unwrap(),
                        expected,
                        "{w}x{h} C={C} kernel={kernel} profile={profile} border={b:?}"
                    );
                    assert_eq!(output.receipt().device_to_host_bytes(), (w * h * 4) as u64);
                    let mut physical = GpuImage::from_parts(
                        runtime,
                        w as u32,
                        h as u32,
                        4,
                        output.texture().clone(),
                        Default::default(),
                        Default::default(),
                    )
                    .unwrap();
                    let rgba = physical.readback_u8::<4>(runtime).unwrap();
                    assert!(rgba
                        .as_slice()
                        .chunks_exact(4)
                        .all(|p| p[C..].iter().all(|&v| v == 0)));
                    drop(physical);
                    output.recycle(runtime);
                    calls += 1;
                    bytes += w * h * C;
                }
            }
            source.recycle(runtime);
        }
    }
    (calls, bytes)
}

#[test]
#[ignore = "requires native GPU adapter"]
fn median_logical_channels_borders_impulses_and_strip_sizes_match_cpu_bytes() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cases = [
        compare::<1>(&runtime),
        compare::<2>(&runtime),
        compare::<3>(&runtime),
        compare::<4>(&runtime),
    ];
    println!(
        "GPU_MEDIAN_CPU calls={} exact_bytes={}",
        cases.iter().map(|p| p.0).sum::<usize>(),
        cases.iter().map(|p| p.1).sum::<usize>()
    );
}

#[test]
#[ignore = "requires native GPU adapter"]
fn median_strided_roi_metadata_chain_and_owned_outputs_survive_input_drop() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let (w, h, stride) = (23, 19, 23 * 4 + 17);
    let mut data = vec![211; stride * h];
    let image = fixture::<4>(w, h, 0);
    for (row, src) in data.chunks_mut(stride).zip(image.as_slice().chunks(w * 4)) {
        row[..w * 4].copy_from_slice(src);
    }
    let metadata = ImageMetadata {
        color_space: spatialrust_image::ColorSpace::Bgra,
        color_range: spatialrust_image::ColorRange::Limited,
        alpha_mode: spatialrust_image::AlphaMode::Straight,
    };
    let view = ImageView::<u8, 4>::new_with_metadata(w, h, stride, &data, metadata).unwrap();
    let roi = view.subview(ImageRegion::new(3, 2, 17, 13)).unwrap();
    let source = GpuImage::upload_u8(&runtime, roi).unwrap();
    let mut outputs = Vec::new();
    for kernel in [3, 5] {
        for b in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
            let output = median_blur_gpu(&runtime, &source, kernel, b).unwrap();
            outputs.push((output, median_blur(roi, kernel as usize, border(b)).unwrap()));
        }
    }
    source.recycle(&runtime);
    for (mut output, expected) in outputs {
        assert_eq!(output.metadata(), metadata);
        assert_eq!(output.readback_u8::<4>(&runtime).unwrap(), expected);
        output.recycle(&runtime);
    }
    let image = fixture::<3>(65, 33, 4);
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let filtered = median_blur_gpu(&runtime, &source, 5, GpuImageBorder::Replicate).unwrap();
    let gray = crate::rgb_to_gray_gpu(&runtime, &filtered).unwrap();
    let edges = crate::sobel_gpu(&runtime, &gray).unwrap();
    let tensor = crate::pack_ai_chw_gpu(&runtime, &edges, 1.0 / 255.0, [0.0; 4], [1.0; 4]).unwrap();
    tensor.receipt().validate_resident_chain(65 * 33 * 4).unwrap();
    assert_eq!(tensor.receipt().device_to_host_bytes(), 0);
}

#[path = "median_benchmark.rs"]
mod benchmark;

#[test]
#[ignore = "requires native GPU adapter"]
fn median_errors_do_not_allocate_and_cold_kernels_are_concurrent() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let image = fixture::<3>(65, 33, 0);
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let pool = runtime.image_texture_pool_stats();
    let receipt = source.receipt().clone();
    for kernel in [0, 1, 2, 4, 7, u32::MAX] {
        assert!(median_blur_gpu(&runtime, &source, kernel, GpuImageBorder::Replicate).is_err());
    }
    assert!(median_blur_gpu(&other, &source, 3, GpuImageBorder::Replicate).is_err());
    assert_eq!(runtime.image_texture_pool_stats(), pool);
    assert_eq!(runtime.initialized_image_pipeline_count(), 0);
    assert_eq!(other.initialized_image_pipeline_count(), 0);
    assert_eq!(source.receipt(), &receipt);
    let outputs = std::thread::scope(|scope| {
        let handles: Vec<_> = [
            (3, GpuImageBorder::Replicate),
            (5, GpuImageBorder::Replicate),
            (3, GpuImageBorder::ConstantZero),
            (5, GpuImageBorder::ConstantZero),
        ]
        .into_iter()
        .map(|(kernel, b)| {
            let runtime = &runtime;
            let source = &source;
            scope.spawn(move || (median_blur_gpu(runtime, source, kernel, b).unwrap(), kernel, b))
        })
        .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    assert_eq!(runtime.initialized_image_pipeline_count(), 1);
    for (mut output, kernel, b) in outputs {
        assert_eq!(
            output.readback_u8::<3>(&runtime).unwrap(),
            median_blur(image.view(), kernel as usize, border(b)).unwrap()
        );
        output.recycle(&runtime);
    }
}

#[path = "median_nine_tests.rs"]
mod nine;

#[path = "median_five_tests.rs"]
mod five;
