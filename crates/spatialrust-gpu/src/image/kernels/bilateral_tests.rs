use super::*;
use crate::image::gpu_image::read_staging_bytes;
use spatialrust_image::{Image, ImageMetadata, ImageRegion, ImageView};
use spatialrust_vision::{bilateral_filter, BorderMode};
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

fn border<const C: usize>(b: GpuImageBorder) -> BorderMode<u8, C> {
    match b {
        GpuImageBorder::Replicate => BorderMode::Replicate,
        GpuImageBorder::ConstantZero => BorderMode::Constant([0; C]),
    }
}
#[test]
fn bilateral_control_coefficients_match_cpu_fma_exp_and_reject_limits() {
    let l = wgpu::Limits::default();
    for c in 1..=4 {
        for diameter in 1..=7 {
            for (sc, ss) in [(40.0, 3.0), (80.0, 1.0), (f64::MAX, f64::MAX), (1.0, 0.01)] {
                let options = GpuBilateralOptions {
                    diameter,
                    sigma_color: sc,
                    sigma_space: ss,
                    ..Default::default()
                };
                let (data, taps) = control(c, options, &l).unwrap();
                let cf = -0.5 / (sc * sc);
                let sf = -0.5 / (ss * ss);
                for tap in 0..taps as usize {
                    let [dx, dy] = data[2 * tap];
                    let (dx, dy) = (dx as i32, dy as i32);
                    let d = f64::from(dx).mul_add(f64::from(dx), f64::from(dy) * f64::from(dy));
                    let row = data[2 * tap + 1][0] as usize;
                    for color in 0..=255 * c {
                        let expected = if d == 0.0 {
                            if color == 0 {
                                1.0
                            } else {
                                0.0
                            }
                        } else {
                            let delta = f64::from(color);
                            (delta * delta).mul_add(cf, d * sf).exp()
                        };
                        let actual = u64::from(data[row + color as usize][0])
                            | (u64::from(data[row + color as usize][1]) << 32);
                        assert_eq!(actual, expected.to_bits());
                    }
                }
            }
        }
    }
    for c in [0, 5, u32::MAX] {
        assert!(control(c, Default::default(), &l).is_err());
    }
    for options in [
        GpuBilateralOptions { diameter: 0, ..Default::default() },
        GpuBilateralOptions { diameter: 8, ..Default::default() },
        GpuBilateralOptions { sigma_color: 0.0, ..Default::default() },
        GpuBilateralOptions { sigma_space: f64::NAN, ..Default::default() },
        GpuBilateralOptions { sigma_color: f64::INFINITY, ..Default::default() },
        GpuBilateralOptions { sigma_color: 1e-200, ..Default::default() },
        GpuBilateralOptions { sigma_color: 20.0, ..Default::default() },
    ] {
        assert!(control(4, options, &l).is_err());
    }
    let (data, _) = control(4, Default::default(), &l).unwrap();
    let bytes = (data.len() * 8) as u64;
    for index in 0..14 {
        let mut weak = l.clone();
        match index {
            0 => weak.max_buffer_size = bytes - 1,
            1 => weak.max_storage_buffer_binding_size = bytes as u32 - 1,
            2 => weak.max_storage_buffers_per_shader_stage = 0,
            3 => weak.max_uniform_buffer_binding_size = 31,
            4 => weak.max_bind_groups = 0,
            5 => weak.max_bindings_per_bind_group = 3,
            6 => weak.max_uniform_buffers_per_shader_stage = 0,
            7 => weak.max_sampled_textures_per_shader_stage = 0,
            8 => weak.max_storage_textures_per_shader_stage = 0,
            9 => weak.max_compute_workgroup_size_x = 15,
            10 => weak.max_compute_workgroup_size_y = 15,
            11 => weak.max_compute_workgroup_size_z = 0,
            12 => weak.max_compute_invocations_per_workgroup = 255,
            _ => weak.max_buffer_size = 31,
        };
        assert!(capabilities(bytes, &weak).is_err(), "limit {index}");
    }
    let mut limits = l.clone();
    limits.max_compute_workgroups_per_dimension = 1;
    assert!(dimensions(16, 16, 3, &limits).is_ok());
    assert!(dimensions(17, 1, 3, &limits).is_err());
    assert!(dimensions(1, 17, 3, &limits).is_err());
    assert!(dimensions(0, 1, 3, &l).is_err());
}
#[test]
#[ignore = "requires native GPU adapter"]
fn bilateral_exact_binary64_quotient_byte_boundaries_and_mantissa_products() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    let mut seed = 19u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed
    };
    let mut push = |n: f64, d: f64| {
        let nb = n.to_bits();
        let db = d.to_bits();
        inputs.extend([nb as u32, (nb >> 32) as u32, db as u32, (db >> 32) as u32]);
        expected.push((n / d).round().clamp(0.0, 255.0) as u32);
    };
    for k in 0..255 {
        for d in [
            1.0,
            3.0,
            7.0,
            127.0,
            255.0,
            1.0 + f64::EPSILON,
            2.0_f64.powi(-1000),
            2.0_f64.powi(900),
        ] {
            let center = (k as f64 + 0.5) * d;
            for offset in -3i64..=3 {
                push(f64::from_bits((center.to_bits() as i64 + offset) as u64), d);
            }
        }
    }
    for _ in 0..65536 {
        let exponent = (next() % 1800 + 123) as u64;
        let d = f64::from_bits(
            (exponent << 52) | (u64::from(next() & 0xfffff) << 32) | u64::from(next()),
        );
        let k = next() % 255;
        let center = (f64::from(k) + 0.5) * d;
        let offset = i64::from(next() % 5) - 2;
        push(f64::from_bits((center.to_bits() as i64 + offset) as u64), d);
    }
    for (n, d) in [
        (0.0, 1.0),
        (1.0, 2.0),
        (5.0, 2.0),
        (9.0, 2.0),
        (f64::MAX, f64::MIN_POSITIVE),
        (f64::MIN_POSITIVE, f64::MAX),
    ] {
        push(n, d);
    }
    let shader = format!(
        "{}\n{}\n{}\n{}",
        include_str!("../rounding_u32.wgsl"),
        include_str!("../binary64_positive.wgsl"),
        include_str!("bilateral_round.wgsl"),
        r#"
@group(0) @binding(0) var<storage,read> test_inputs:array<vec4<u32>>;
@group(0) @binding(1) var<storage,read_write> test_outputs:array<u32>;
@compute @workgroup_size(256)
fn proof(@builtin(global_invocation_id) id:vec3<u32>) {if id.x>=arrayLength(&test_inputs) {return;} let v=test_inputs[id.x];test_outputs[id.x]=bilateral_byte(fp_decode(v.xy),fp_decode(v.zw));}
"#
    );
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bilateral-round-proof"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("proof"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (expected.len() * 4) as u64;
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((expected.len() as u32).div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
    runtime.queue().submit(Some(encoder.finish()));
    let actual = read_staging_bytes(runtime.device(), &staging, bytes as usize).unwrap();
    for (index, (actual, expected)) in actual.chunks_exact(4).zip(&expected).enumerate() {
        assert_eq!(
            u32::from_le_bytes(actual.try_into().unwrap()),
            *expected,
            "row {index} input {:?}",
            &inputs[index * 4..index * 4 + 4]
        );
    }
    println!("GPU_BILATERAL_ROUND exact_ratios={}", expected.len());
}
fn images<const C: usize>(runtime: &WgpuRuntime) -> (usize, usize) {
    let frozen = candidate_tests::frozen_pipeline(runtime);
    let mut calls = 0;
    let mut bytes = 0;
    for diameter in [1, 2, 3, 5, 7] {
        for (sc, ss) in [(40.0, 3.0), (f64::MAX, f64::MAX), (1.0, 0.01)] {
            for b in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                let options =
                    GpuBilateralOptions { diameter, sigma_color: sc, sigma_space: ss, border: b };
                let plan = GpuBilateralFilter::prepare(runtime, C as u32, options).unwrap();
                for (w, h) in [(1, 1), (1, 17), (2, 3), (17, 11), (33, 17)] {
                    for p in 0..5 {
                        let image = fixture::<C>(w, h, p);
                        let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
                        let expected =
                            bilateral_filter(image.view(), diameter as usize, sc, ss, border(b))
                                .unwrap();
                        let mut result = plan.filter(runtime, &source).unwrap();
                        let mut previous =
                            candidate_tests::filter_frozen(&plan, runtime, &source, &frozen)
                                .unwrap();
                        let receipt = result.receipt().clone();
                        assert_eq!(result.readback_u8::<C>(runtime).unwrap(),expected,"C={C} d={diameter} sigmas={sc}/{ss} border={b:?} shape={w}x{h} profile={p}");
                        assert_eq!(previous.readback_u8::<C>(runtime).unwrap(), expected);
                        assert_eq!(receipt.host_to_device_bytes(), (w * h * 4) as u64);
                        assert_eq!(receipt.gpu_to_gpu_bytes(), (w * h * 4) as u64);
                        assert_eq!(result.receipt().device_to_host_bytes(), (w * h * 4) as u64);
                        source.recycle(runtime);
                        result.recycle(runtime);
                        previous.recycle(runtime);
                        calls += 1;
                        bytes += w * h * C;
                    }
                }
            }
        }
    }
    (calls, bytes)
}
#[test]
#[ignore = "requires native GPU adapter"]
fn bilateral_channels_disks_even_diameters_weights_and_borders_match_cpu() {
    let r = WgpuRuntime::new_headless().unwrap();
    let a = images::<1>(&r);
    let b = images::<2>(&r);
    let c = images::<3>(&r);
    let d = images::<4>(&r);
    println!(
        "GPU_BILATERAL_IMAGES calls={} exact_bytes={}",
        a.0 + b.0 + c.0 + d.0,
        a.1 + b.1 + c.1 + d.1
    );
}
#[test]
#[ignore = "requires native GPU adapter"]
fn bilateral_roi_receipts_cold_concurrency_and_ownership() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let shape = (19, 17);
    let stride = shape.0 * 4 + 13;
    let packed = fixture::<4>(shape.0, shape.1, 0);
    let mut data = vec![211; stride * shape.1];
    for (row, src) in data.chunks_mut(stride).zip(packed.as_slice().chunks(shape.0 * 4)) {
        row[..shape.0 * 4].copy_from_slice(src);
    }
    let meta = ImageMetadata {
        color_space: spatialrust_image::ColorSpace::Bgra,
        color_range: spatialrust_image::ColorRange::Limited,
        alpha_mode: spatialrust_image::AlphaMode::Straight,
    };
    let view = ImageView::<u8, 4>::new_with_metadata(shape.0, shape.1, stride, &data, meta)
        .unwrap()
        .subview(ImageRegion::new(1, 2, 17, 13))
        .unwrap();
    let source = GpuImage::upload_u8(&r, view).unwrap();
    let plan = GpuBilateralFilter::prepare(&r, 4, Default::default()).unwrap();
    assert_eq!(r.initialized_image_pipeline_count(), 0);
    assert_eq!(plan.receipt().device_to_host_bytes(), 0);
    assert_eq!(plan.receipt().host_to_device_bytes(), 32880);
    assert!(plan.filter(&other, &source).is_err());
    let wrong = GpuImage::upload_u8(&r, fixture::<3>(2, 3, 0).view()).unwrap();
    assert!(plan.filter(&r, &wrong).is_err());
    assert_eq!(r.initialized_image_pipeline_count(), 0);
    let expected = bilateral_filter(view, 5, 40.0, 3.0, BorderMode::Replicate).unwrap();
    let outputs = std::thread::scope(|scope| {
        let mut tasks = Vec::new();
        for _ in 0..4 {
            tasks.push(scope.spawn(|| plan.filter(&r, &source).unwrap()));
        }
        tasks.into_iter().map(|t| t.join().unwrap()).collect::<Vec<_>>()
    });
    assert_eq!(r.initialized_image_pipeline_count(), 1);
    source.recycle(&r);
    for mut out in outputs {
        assert_eq!(out.readback_u8::<4>(&r).unwrap(), expected);
        out.recycle(&r);
    }
    let rgb = fixture::<3>(17, 13, 0);
    let source = GpuImage::upload_u8(&r, rgb.view()).unwrap();
    let rgb_plan = GpuBilateralFilter::prepare(&r, 3, Default::default()).unwrap();
    let color = rgb_plan.filter(&r, &source).unwrap();
    let gray = crate::rgb_to_gray_gpu(&r, &color).unwrap();
    let edges = crate::sobel_gpu(&r, &gray).unwrap();
    let tensor = crate::pack_ai_chw_gpu(&r, &edges, 1.0 / 255.0, [0.0; 4], [1.0; 4]).unwrap();
    tensor.receipt().validate_resident_chain(17 * 13 * 4).unwrap();
    for bad in [
        GpuBilateralOptions { sigma_color: 1e-200, ..Default::default() },
        GpuBilateralOptions { diameter: 8, ..Default::default() },
    ] {
        let count = r.initialized_image_pipeline_count();
        let stats = r.image_texture_pool_stats();
        assert!(GpuBilateralFilter::prepare(&r, 4, bad).is_err());
        assert_eq!(r.initialized_image_pipeline_count(), count);
        assert_eq!(r.image_texture_pool_stats(), stats);
    }
}

#[test]
#[ignore = "requires native GPU adapter"]
fn bilateral_production_accumulators_match_frozen_cpu_bits() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let code = format!(
        "{SHADER}\n{}",
        r#"
@group(0) @binding(4) var<storage,read_write> raw:array<vec2<u32>>;
@compute @workgroup_size(16,16)
fn prove_sums(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.width || id.y>=params.height {return;}
    let values=bilateral_accumulate(vec2<i32>(id.xy));let row=5u*(id.y*params.width+id.x);
    raw[row]=fp_encode(values.r);raw[row+1u]=fp_encode(values.g);raw[row+2u]=fp_encode(values.b);raw[row+3u]=fp_encode(values.a);raw[row+4u]=fp_encode(values.total);
}
"#
    );
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bilateral-accumulation-proof"),
        source: wgpu::ShaderSource::Wgsl(code.into()),
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("prove_sums"),
        compilation_options: Default::default(),
        cache: None,
    });
    let image = fixture::<4>(17, 11, 0);
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let mut count = 0;
    for diameter in [2, 3, 5, 7] {
        for (sc, ss) in [(40.0, 3.0), (f64::MAX, f64::MAX)] {
            for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                let options =
                    GpuBilateralOptions { diameter, sigma_color: sc, sigma_space: ss, border };
                let plan = GpuBilateralFilter::prepare(&runtime, 4, options).unwrap();
                let params = Params {
                    width: 17,
                    height: 11,
                    channels: 4,
                    taps: plan.taps,
                    border: u32::from(border == GpuImageBorder::ConstantZero),
                    padding: [0; 3],
                };
                let uniform =
                    runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let bytes = 17 * 11 * 40;
                let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&source.view()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: plan.coefficients.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
                    ],
                });
                let mut enc = runtime.device().create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(2, 1, 1);
                }
                enc.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
                runtime.queue().submit(Some(enc.finish()));
                let raw = read_staging_bytes(runtime.device(), &staging, bytes as usize).unwrap();
                let mut expected = Vec::new();
                let radius = (diameter / 2) as i32;
                let cf = -0.5 / (sc * sc);
                let sf = -0.5 / (ss * ss);
                for y in 0..11 {
                    for x in 0..17 {
                        let center = image[(x, y)];
                        let mut sums = [0.0; 4];
                        let mut total = 0.0;
                        for dy in -radius..=radius {
                            for dx in -radius..=radius {
                                let spatial = f64::from(dx)
                                    .mul_add(f64::from(dx), f64::from(dy) * f64::from(dy));
                                if spatial > f64::from(radius * radius) {
                                    continue;
                                }
                                let px = x as i32 + dx;
                                let py = y as i32 + dy;
                                let p = if border == GpuImageBorder::ConstantZero
                                    && (px < 0 || py < 0 || px >= 17 || py >= 11)
                                {
                                    [0; 4]
                                } else {
                                    image[(px.clamp(0, 16) as usize, py.clamp(0, 10) as usize)]
                                };
                                let distance = (0..4)
                                    .map(|c| f64::from(p[c]) - f64::from(center[c]))
                                    .map(f64::abs)
                                    .sum::<f64>();
                                let weight = (distance * distance).mul_add(cf, spatial * sf).exp();
                                for c in 0..4 {
                                    sums[c] += f64::from(p[c]) * weight;
                                }
                                total += weight;
                            }
                        }
                        for v in sums.into_iter().chain(std::iter::once(total)) {
                            expected.push(v.to_bits());
                        }
                    }
                }
                for (i, (actual, expected)) in raw.chunks_exact(8).zip(expected).enumerate() {
                    assert_eq!(
                        u64::from_le_bytes(actual.try_into().unwrap()),
                        expected,
                        "d={diameter} sigma={sc}/{ss} border={border:?} point {} component {}",
                        i / 5,
                        i % 5
                    );
                    count += 1;
                }
            }
        }
    }
    println!("GPU_BILATERAL_SUM exact_binary64_words={count}");
}
#[path = "bilateral_benchmark.rs"]
mod benchmark;

#[path = "bilateral_candidate_tests.rs"]
mod candidate_tests;
