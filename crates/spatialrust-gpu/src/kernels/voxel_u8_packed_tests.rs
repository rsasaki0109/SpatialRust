use super::*;
use crate::kernels::{voxel_gather, voxel_reduce};
use wgpu::util::DeviceExt;

#[path = "voxel_u8_packed_reference.rs"]
mod baseline;

fn invoke_baseline(
    r: &WgpuRuntime,
    f: &Fixture,
    attributes: &[&[f32]],
    bytes: &[&[u8]],
    mode: usize,
) -> SpatialResult<Output> {
    let function = match mode {
        0 => baseline::baseline_gather_voxel_first_xyz_and_multi_gpu,
        1 => baseline::baseline_gather_voxel_first_xyz_and_average_multi_gpu,
        2 => baseline::baseline_reduce_voxel_centroids_xyz_and_average_multi_gpu,
        3 => baseline::baseline_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
        _ => unreachable!(),
    };
    function(r, &f.xyz[0], &f.xyz[1], &f.xyz[2], attributes, bytes, &f.segments)
}

#[test]
fn mapped_voxel_packed_u8_keeps_float_bits_byte_order_padding_and_unmap_on_panic() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut comparisons = 0;
    for n in [0, 1, 2, 3, 4, 5, 17, 255, 256, 257, 65_537, 262_145] {
        for a in [0, 4] {
            for u in [0, 1, 3] {
                let mut word_input = words(n * (3 + a));
                let mut packed_input = word_input.clone();
                for channel in 0..u {
                    let bytes: Vec<_> =
                        (0..n).map(|i| ((i * 37 + channel * 71) % 256) as u8).collect();
                    word_input.extend(bytes.iter().map(|&b| b as u32 | 0xa5a5_5a00));
                    // Nonzero unused bytes prove that the mapped decoder ignores tail padding.
                    for chunk in bytes.chunks(4) {
                        let mut word = 0xa5a5_a5a5;
                        for (lane, &value) in chunk.iter().enumerate() {
                            let shift = lane * 8;
                            word = (word & !(255 << shift)) | ((value as u32) << shift);
                        }
                        packed_input.push(word);
                    }
                }
                word_input.extend([0xdead_beef; 16]);
                packed_input.extend([0xdead_beef; 16]);
                let word_buffer = staging(&r, &word_input);
                let packed_buffer = staging(&r, &packed_input);
                let expected = crate::readback::read_staging_xyz_and_attributes(
                    r.device(),
                    &word_buffer,
                    n,
                    a,
                    u,
                )
                .unwrap();
                let mut live = crate::readback::read_staging_xyz_and_packed_attributes(
                    r.device(),
                    &packed_buffer,
                    n,
                    a,
                    u,
                )
                .unwrap();
                assert_output(&live, &expected);
                assert_capacity(&live, n, a, u, false);
                assert_eq!(
                    crate::readback::packed_u8_output_staging_bytes(n, u),
                    n.div_ceil(4) * 4 * u
                );
                if n > 0 {
                    live.0[0] = f32::from_bits(0xdead_beef);
                    if u > 0 {
                        live.4[0][0] ^= 255;
                    }
                }
                let next = crate::readback::read_staging_xyz_and_packed_attributes(
                    r.device(),
                    &packed_buffer,
                    n,
                    a,
                    u,
                )
                .unwrap();
                drop(word_buffer);
                drop(packed_buffer);
                assert_output(&next, &expected);
                if n > 0 {
                    assert_eq!(live.0[0].to_bits(), 0xdead_beef);
                    if u > 0 {
                        assert_eq!(live.4[0][0], expected.4[0][0] ^ 255);
                    }
                }
                comparisons += 2 * n * (3 + a + u);
            }
        }
    }
    let buffer = staging(&r, &words(72));
    for (n, a, u) in [(25, 0, 0), (16, 1, 3)] {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::readback::read_staging_xyz_and_packed_attributes(r.device(), &buffer, n, a, u)
                .unwrap();
        }))
        .is_err());
        let valid =
            crate::readback::read_staging_xyz_and_packed_attributes(r.device(), &buffer, 16, 1, 2)
                .unwrap();
        assert_capacity(&valid, 16, 1, 2, false);
    }
    eprintln!("VOXEL_U8_RAW verified {comparisons} exact float/byte elements");
}

fn read_words(r: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| sender.send(result).unwrap());
    r.device().poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    buffer.unmap();
    words
}

fn run_kernel(
    r: &WgpuRuntime,
    values: &[u8],
    indices: &[u32],
    starts: &[u32],
    reduce: bool,
    packed: bool,
) -> Vec<u32> {
    let n = starts.len();
    let output_words = if packed { n.div_ceil(4) } else { n };
    let initial = vec![0xdead_beef_u32; output_words + 8];
    let output = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-u8-sentinel-output"),
        contents: bytemuck::cast_slice(&initial),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let values_buffer = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-u8-packed-source"),
        contents: &crate::readback::pad_u8_for_gpu_storage(if values.is_empty() {
            &[0]
        } else {
            values
        }),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let upload = |name, data: &[u32]| {
        r.upload_u32_storage(name, if data.is_empty() { &[0] } else { data }).unwrap()
    };
    let indices_buffer = upload("voxel-u8-packed-indices", indices);
    let starts_buffer = upload("voxel-u8-packed-starts", starts);
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-u8-packed-kernel-staging"),
        size: output.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let record = match (reduce, packed) {
        (false, false) => voxel_gather::record_voxel_gather_u8_pass,
        (false, true) => voxel_gather::record_voxel_gather_u8_packed_pass,
        (true, false) => voxel_reduce::record_voxel_reduce_u8_pass,
        (true, true) => voxel_reduce::record_voxel_reduce_u8_packed_pass,
    };
    record(
        &mut encoder,
        r,
        &values_buffer,
        &indices_buffer,
        &starts_buffer,
        n as u32,
        indices.len() as u32,
        &output,
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output.size());
    r.queue().submit(Some(encoder.finish()));
    let result = read_words(r, &staging);
    assert_eq!(&result[output_words..], &[0xdead_beef; 8]);
    result[..output_words].to_vec()
}

#[test]
fn packed_voxel_u8_kernels_match_legacy_words_cpu_values_tails_and_guard_words() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    let mut ties = 0;
    let mut seen = [false; 256];
    for n in [0, 1, 2, 3, 4, 5, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 65_537] {
        let counts: Vec<_> = (0..n).map(|cell| [0, 1, 2, 3, 4, 7, 17][cell % 7]).collect();
        let mut starts = Vec::with_capacity(n);
        let mut points = 0;
        for count in &counts {
            starts.push(points as u32);
            points += count;
        }
        let indices: Vec<_> = (0..points).rev().map(|i| i as u32).collect();
        for variant in 0..3 {
            let values: Vec<u8> =
                (0..points).map(|i| ((i * 37 + variant * 71) % 256) as u8).collect();
            for value in &values {
                seen[*value as usize] = true;
            }
            for reduce in [false, true] {
                let old = run_kernel(&r, &values, &indices, &starts, reduce, false);
                let packed = run_kernel(&r, &values, &indices, &starts, reduce, true);
                let mut expected_words = vec![0_u32; n.div_ceil(4)];
                for cell in 0..n {
                    let count = counts[cell];
                    let start = starts[cell] as usize;
                    let expected = if count == 0 {
                        0
                    } else if !reduce {
                        values[indices[start] as usize] as u32
                    } else {
                        let sum: u32 = indices[start..start + count]
                            .iter()
                            .map(|&i| values[i as usize] as u32)
                            .sum();
                        let divisor = count as u32;
                        let (quotient, remainder) = (sum / divisor, sum % divisor);
                        if remainder * 2 == divisor {
                            // Keep the legacy GPU's tie result; non-ties have an exact integer oracle.
                            assert!(old[cell] == quotient || old[cell] == quotient + 1);
                            ties += 1;
                            old[cell]
                        } else {
                            quotient + u32::from(remainder * 2 > divisor)
                        }
                    };
                    assert_eq!(old[cell], expected, "cell={cell} reduce={reduce}");
                    expected_words[cell / 4] |= expected << ((cell % 4) * 8);
                    compared += 1;
                }
                assert_eq!(packed, expected_words);
            }
        }
    }
    assert!(seen.into_iter().all(|v| v));
    assert!(ties > 0);
    eprintln!("VOXEL_U8_KERNEL verified {compared} cell results; {ties} legacy-exact half ties; all 256 byte codes");
}

#[test]
fn public_voxel_packed_u8_matches_complete_parent_calls_across_modes_and_boundaries() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for (n, occupancy) in [
        (1, 1),
        (2, 4),
        (3, 1),
        (4, 4),
        (5, 1),
        (255, 4),
        (256, 1),
        (257, 4),
        (65_537, 4),
        (1_048_577, 1),
    ] {
        for (a, u) in [(0, 1), (4, 3)] {
            let f = fixture(&r, n, a, u, occupancy);
            let attributes: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
            let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
            for mode in 0..4 {
                let old = invoke_baseline(&r, &f, &attributes, &bytes, mode).unwrap();
                let new = invoke(&r, &f, &attributes, &bytes, mode, false).unwrap();
                assert_output(&new, &old);
                assert_output(&new, &expected(&f, mode));
                assert_capacity(&new, n, a, u, false);
                assert_capacity(&old, n, a, u, false);
                compared += 2 * n * (3 + a + u);
            }
        }
    }
    eprintln!("VOXEL_U8_PUBLIC verified {compared} parent/CPU exact elements");
}

#[test]
fn concurrent_voxel_packed_u8_calls_share_lazy_pipelines_and_keep_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let barrier = std::sync::Barrier::new(3);
    std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        for cells in [1_025, 4_097, 16_385] {
            let (r, barrier) = (&r, &barrier);
            jobs.push(scope.spawn(move || {
                let f = fixture(r, cells, 4, 3, 4);
                barrier.wait();
                let live = call(r, &f, 0, false);
                for mode in 0..4 {
                    assert_output(&call(r, &f, mode, false), &expected(&f, mode));
                }
                let expected = expected(&f, 0);
                drop(f);
                r.clear_buffer_pool();
                assert_output(&live, &expected);
                (
                    r.pipelines().voxel_gather.packed_u8_pipeline(r.device()) as *const _ as usize,
                    r.pipelines().voxel_reduce.packed_u8_pipeline(r.device()) as *const _ as usize,
                )
            }));
        }
        let pointers: Vec<_> = jobs.into_iter().map(|j| j.join().unwrap()).collect();
        assert!(pointers.iter().all(|p| *p == pointers[0]));
    });
}

#[path = "voxel_u8_packed_benchmark.rs"]
mod benchmark;
