//! Explicit native diagnostic: old/new lookup passes on the same device/input.
use super::{reference, *};
use crate::WgpuPowerPreference;
use spatialrust_image::Image;

const SERIAL_SHADER: &str = concat!(
    include_str!("rounding_u32.wgsl"),
    include_str!("clahe_params.wgsl"),
    include_str!("../../tests/fixtures/clahe_lookup_serial.wgsl")
);

fn expected_lookups(image: &Image<u8, 1>, p: &Params) -> Vec<u32> {
    let reflect = |position: u32, length: u32| {
        if length == 1 {
            return 0;
        }
        let period = 2 * (length - 1);
        let value = position % period;
        if value < length {
            value
        } else {
            period - value
        }
    };
    let mut expected = Vec::new();
    for tile in 0..p.tile_count {
        let mut counts = [0; 256];
        let ox = tile % p.tiles_x * p.tile_width;
        let oy = tile / p.tiles_x * p.tile_height;
        for y in 0..p.tile_height {
            for x in 0..p.tile_width {
                let i = (reflect(oy + y, p.height) * p.width + reflect(ox + x, p.width)) as usize;
                counts[image.as_slice()[i] as usize] += 1;
            }
        }
        expected.extend(reference(&counts, p.clip_count).chunks_exact(2).map(|v| v[1]));
    }
    expected
}

// Not a CI performance assertion: explicit local run, GPU pass timestamps only.
// The serial shader is the frozen PR #131 implementation and is test-only.
#[test]
#[ignore = "native paired GPU timestamp benchmark; requires explicit invocation"]
fn paired_serial_parallel_clahe_lookup_timestamps() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    assert!(runtime.timestamp_queries_enabled(), "paired benchmark requires timestamp support");
    eprintln!("adapter={:?}; 4 warmups + 40 alternating paired rounds; timestamps exclude uploads, allocation, command setup and readback", runtime.adapter_info());
    let pipelines = ClahePipelines::new(runtime.device());
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("clahe-serial-baseline"),
        source: wgpu::ShaderSource::Wgsl(SERIAL_SHADER.into()),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&pipelines.layouts[1]],
        push_constant_ranges: &[],
    });
    let serial = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("clahe-serial-baseline"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("clahe_lookup"),
        compilation_options: Default::default(),
        cache: None,
    });
    for (width, height, x, y) in [
        (17, 19, 8, 8),
        (640, 480, 8, 8),
        (640, 480, 64, 64),
        (3840, 2160, 8, 8),
        (3840, 2160, 32, 32),
    ] {
        for hot in [false, true] {
            let image = Image::<u8, 1>::try_new(
                width as usize,
                height as usize,
                (0..width as usize * height as usize)
                    .map(|i| if hot { 43 } else { ((i * 37 + i / 997) % 129 + 32) as u8 })
                    .collect(),
            )
            .unwrap();
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            let view = source.view();
            for clip in [0.0, 2.0, 40.0] {
                let p = plan(width, height, clip, x, y, &runtime.device().limits()).unwrap();
                let uniform =
                    runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::bytes_of(&p.params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let outputs: [wgpu::Buffer; 2] = std::array::from_fn(|_| {
                    runtime.device().create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: p.lookup_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                });
                let groups: [wgpu::BindGroup; 2] = std::array::from_fn(|i| {
                    runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipelines.layouts[1],
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: outputs[i].as_entire_binding(),
                            },
                        ],
                    })
                });
                let variants = [&serial, &pipelines.pipelines[1]];
                let labels = ["serial_lookup", "parallel_lookup"];
                let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                for round in 0..44 {
                    let mut profiler = GpuStageProfiler::new(&runtime, 2).unwrap();
                    let mut encoder = runtime.device().create_command_encoder(&Default::default());
                    for offset in 0..2 {
                        let i = (round + offset) % 2;
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(labels[i]),
                            timestamp_writes: crate::profiling::timestamp_writes(
                                &mut Some(&mut profiler),
                                labels[i],
                            ),
                        });
                        pass.set_pipeline(variants[i]);
                        pass.set_bind_group(0, &groups[i], &[]);
                        pass.dispatch_workgroups(p.params.tile_groups, 1, 1);
                    }
                    runtime.queue().submit(Some(encoder.finish()));
                    let report = profiler.collect(&runtime).unwrap();
                    assert_eq!(report.device_to_host_bytes, 32);
                    if round >= 4 {
                        for s in report.stages {
                            samples[usize::from(s.stage == labels[1])].push(s.elapsed_ns / 1000.0);
                        }
                    }
                }
                let expected = expected_lookups(&image, &p.params);
                for output in &outputs {
                    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: p.lookup_bytes,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });
                    let mut encoder = runtime.device().create_command_encoder(&Default::default());
                    encoder.copy_buffer_to_buffer(output, 0, &staging, 0, p.lookup_bytes);
                    runtime.queue().submit(Some(encoder.finish()));
                    let bytes =
                        read_staging_bytes(runtime.device(), &staging, p.lookup_bytes as usize)
                            .unwrap();
                    for (i, (actual, &expected)) in bytes.chunks_exact(4).zip(&expected).enumerate()
                    {
                        assert_eq!(
                            u32::from_le_bytes(actual.try_into().unwrap()),
                            expected,
                            "{width}x{height} grid{x}x{y} hot{hot} clip{clip}, LUT word {i}"
                        );
                    }
                }
                eprintln!("PAIRED {{\"width\":{width},\"height\":{height},\"tiles_x\":{x},\"tiles_y\":{y},\"hot\":{hot},\"clip\":{clip},\"serial_us\":{:?},\"parallel_us\":{:?}}}",samples[0],samples[1]);
            }
            source.recycle(&runtime);
        }
    }
}
