use super::*;
use std::borrow::Cow;
use wgpu::util::DeviceExt;
#[path = "voxel_borrowed_u8_reference.rs"]
mod baseline;

fn invoke_current_parent(
    r: &WgpuRuntime,
    f: &Fixture,
    attrs: &[&[f32]],
    bytes: &[&[u8]],
    mode: usize,
    old: bool,
) -> SpatialResult<Output> {
    let function = match (mode, old) {
        (0, false) => gather_voxel_first_xyz_and_multi_gpu,
        (0, true) => baseline::baseline_gather_voxel_first_xyz_and_multi_gpu,
        (1, false) => gather_voxel_first_xyz_and_average_multi_gpu,
        (1, true) => baseline::baseline_gather_voxel_first_xyz_and_average_multi_gpu,
        (2, false) => reduce_voxel_centroids_xyz_and_average_multi_gpu,
        (2, true) => baseline::baseline_reduce_voxel_centroids_xyz_and_average_multi_gpu,
        (3, false) => reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
        (3, true) => baseline::baseline_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
        _ => unreachable!(),
    };
    function(r, &f.xyz[0], &f.xyz[1], &f.xyz[2], attrs, bytes, &f.segments)
}

fn complete_call(r: &WgpuRuntime, f: &Fixture, mode: usize, old: bool) -> Output {
    invoke_current_parent(
        r,
        f,
        &f.floats[3..].iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &f.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        mode,
        old,
    )
    .unwrap()
}

#[test]
fn borrowed_voxel_u8_payloads_preserve_all_remainders_offsets_bytes_and_zero_padding() {
    let mut lengths: Vec<_> = (0..=1025).collect();
    lengths.extend([65_535, 65_536, 65_537, 1_048_575, 1_048_576, 1_048_577, 4_194_304]);
    let mut checked = 0;
    let mut saved_bytes = 0;
    let mut saved_allocations = 0;
    for len in lengths {
        for offset in 0..4 {
            let storage: Vec<_> = (0..len + offset + 8).map(|i| (i * 37 + 19) as u8).collect();
            let source = &storage[offset..offset + len];
            let payload = crate::readback::u8_storage_upload_contents(source);
            let old = crate::readback::pad_u8_for_gpu_storage(source);
            assert_eq!(payload.as_ref(), old);
            assert_eq!(payload.len(), len.div_ceil(4) * 4);
            assert_eq!(&payload[..len], source);
            assert!(payload[len..].iter().all(|&b| b == 0));
            if len % 4 == 0 {
                assert!(matches!(&payload, Cow::Borrowed(_)));
                assert_eq!(payload.as_ptr(), source.as_ptr());
                if len != 0 {
                    saved_bytes += len;
                    saved_allocations += 1;
                }
            } else {
                assert!(matches!(&payload, Cow::Owned(_)));
                assert_ne!(payload.as_ptr(), source.as_ptr());
            }
            checked += payload.len() * 2;
        }
    }
    eprintln!("VOXEL_BORROWED_U8_HOST verified {checked} exact old/independent payload bytes across all length remainders and source offsets; avoided_preparation_bytes={saved_bytes} avoided_nonempty_payload_allocations={saved_allocations}");
}

fn read_raw(r: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (send, recv) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| send.send(v).unwrap());
    r.device().poll(wgpu::Maintain::Wait);
    recv.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let out = view.to_vec();
    drop(view);
    buffer.unmap();
    out
}

#[test]
fn borrowed_voxel_u8_gpu_uploads_snapshot_odd_address_inputs_before_mutation_and_submit() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for len in [1_usize, 2, 3, 4, 255, 256, 257, 65_536, 65_537, 1_048_576, 1_048_577] {
        let mut source: Vec<_> = (0..len + 9).map(|i| (i * 37 + 19) as u8).collect();
        let old = crate::readback::pad_u8_for_gpu_storage(&source[1..len + 1]);
        let current_buffer = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("borrowed-voxel-u8-upload"),
            contents: &crate::readback::u8_storage_upload_contents(&source[1..len + 1]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let old_buffer = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("owned-voxel-u8-upload-parent"),
            contents: &old,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let words = vec![0xdead_beef; (old.len() * 2 + 32) / 4];
        let target = staging(&r, &words);
        let mut encoder = r.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&current_buffer, 0, &target, 16, old.len() as u64);
        encoder.copy_buffer_to_buffer(
            &old_buffer,
            0,
            &target,
            (16 + old.len()) as u64,
            old.len() as u64,
        );
        source.fill(0xa5);
        drop(source);
        r.queue().submit(Some(encoder.finish()));
        let got = read_raw(&r, &target);
        assert_eq!(&got[..16], &bytemuck::cast_slice::<u32, u8>(&words)[..16]);
        assert_eq!(&got[16..16 + old.len()], old);
        assert_eq!(&got[16 + old.len()..16 + 2 * old.len()], old);
        assert_eq!(
            &got[16 + 2 * old.len()..],
            &bytemuck::cast_slice::<u32, u8>(&words)[16 + 2 * old.len()..]
        );
        checked += old.len() * 2;
    }
    eprintln!("VOXEL_BORROWED_U8_RAW verified {checked} GPU upload bytes, all tails, odd host addresses, source mutation/drop before submit and destination guards");
}

#[test]
fn borrowed_voxel_u8_four_public_modes_match_current_parents_cpu_and_owned_capacities() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, attrs, bytes, occupancy) in [
        (0, 7, 3, 4),
        (1, 0, 1, 1),
        (2, 1, 1, 1),
        (3, 4, 3, 1),
        (4, 4, 3, 1),
        (17, 0, 3, 1),
        (18, 0, 3, 1),
        (19, 0, 3, 1),
        (20, 0, 3, 1),
        (257, 7, 7, 4),
        (65_537, 4, 3, 1),
        (1_048_576, 7, 3, 1),
        (1_048_577, 0, 7, 1),
    ] {
        let mut f = fixture(&r, cells, attrs, bytes, occupancy);
        if cells == 257 {
            f.segments = crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys(
                &r,
                &(0..cells * occupancy).map(|i| ((i % cells) as i64, 0, 0)).collect::<Vec<_>>(),
            )
            .unwrap();
            assert_eq!(f.segments.construction_readback_bytes(), 4);
        }
        let mut retained = Vec::new();
        for mode in 0..4 {
            let live = complete_call(&r, &f, mode, false);
            let old = complete_call(&r, &f, mode, true);
            let cpu = expected(&f, mode);
            assert_output(&live, &old);
            assert_output(&live, &cpu);
            assert_capacity(&live, cells, attrs, bytes, false);
            assert_capacity(&old, cells, attrs, bytes, false);
            retained.push((live, cpu));
            checked += cells * (3 + attrs + bytes) * 2;
        }
        drop(f);
        r.clear_buffer_pool();
        for (live, cpu) in retained {
            assert_output(&live, &cpu);
        }
    }
    eprintln!("VOXEL_BORROWED_U8_PUBLIC verified {checked} complete four-mode current-parent/CPU elements, all point-count remainders, scalar/batched/many-channel byte paths, sorted segments and owned lifetimes");
}

#[test]
fn borrowed_voxel_u8_sparse_rounding_and_validation_keep_complete_parent_behavior() {
    let r = WgpuRuntime::new_headless().unwrap();
    for bytes in [1, 3, 7] {
        let mut f = fixture(&r, 257, 0, bytes, 2);
        let indices: Vec<_> = (0..514).rev().map(|i| i as u32).collect();
        let starts: Vec<_> = (0..257).map(|i| (i / 2 * 2) as u32).collect();
        let upload = |label, values: &[u32]| r.upload_u32_storage(label, values).unwrap();
        f.segments = GpuVoxelSegments::new(
            257,
            514,
            upload("borrowed-u8-keys", &[0]),
            upload("borrowed-u8-indices", &indices),
            upload("borrowed-u8-starts", &starts),
            0,
        );
        for mode in 0..4 {
            let live = complete_call(&r, &f, mode, false);
            assert_output(&live, &complete_call(&r, &f, mode, true));
            for (plane, source) in live.4.iter().zip(&f.bytes) {
                for cell in 0..257 {
                    let start = starts[cell] as usize;
                    let end = starts.get(cell + 1).copied().unwrap_or(514) as usize;
                    let actual = u32::from(plane[cell]);
                    if start == end {
                        assert_eq!(actual, 0);
                    } else if mode == 0 || mode == 3 {
                        assert_eq!(actual, u32::from(source[indices[start] as usize]));
                    } else {
                        let sum: u32 = indices[start..end]
                            .iter()
                            .map(|&i| u32::from(source[i as usize]))
                            .sum();
                        let count = (end - start) as u32;
                        let (q, rem) = (sum / count, sum % count);
                        if rem * 2 == count {
                            assert!(actual == q || actual == q + 1);
                        } else {
                            assert_eq!(actual, q + u32::from(rem * 2 > count));
                        }
                    }
                }
            }
        }
    }
    let f = fixture(&r, 17, 7, 3, 4);
    let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
    let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
    let empty = fixture(&r, 0, 7, 3, 1);
    for mode in 0..4 {
        for (index, found) in [(0, 67), (1, 66), (2, 65)] {
            let mut bad = bytes.clone();
            bad[index] = &f.bytes[index][..found];
            if index < 2 {
                bad[2] = &f.bytes[2][..1];
            }
            let live = invoke_current_parent(&r, &f, &attrs, &bad, mode, false).unwrap_err();
            assert_eq!(live, invoke_current_parent(&r, &f, &attrs, &bad, mode, true).unwrap_err());
            assert_eq!(live, SpatialError::BufferLengthMismatch { expected: 68, found });
        }
        let mut bad_attrs = attrs.clone();
        bad_attrs[3] = &f.floats[6][..66];
        let mut bad_bytes = bytes.clone();
        bad_bytes[0] = &f.bytes[0][..1];
        let live = invoke_current_parent(&r, &f, &bad_attrs, &bad_bytes, mode, false).unwrap_err();
        assert_eq!(
            live,
            invoke_current_parent(&r, &f, &bad_attrs, &bad_bytes, mode, true).unwrap_err()
        );
        assert_eq!(live, SpatialError::BufferLengthMismatch { expected: 68, found: 66 });
        let stats = r.buffer_pool().stats();
        let live = invoke_current_parent(&r, &empty, &[&[1.0]], &[&[1_u8]], mode, false).unwrap();
        assert_output(
            &live,
            &invoke_current_parent(&r, &empty, &[&[1.0]], &[&[1_u8]], mode, true).unwrap(),
        );
        assert_capacity(&live, 0, 1, 1, false);
        assert_eq!(r.buffer_pool().stats(), stats);
    }
}

#[test]
fn borrowed_voxel_u8_concurrent_aligned_and_padded_calls_keep_prior_outputs_after_mutation() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|mode| {
            let r = r.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let mut f = fixture(&r, 65_536 + mode, [0, 4, 7, 1][mode], [1, 3, 4, 7][mode], 1);
                gate.wait();
                let live = complete_call(&r, &f, mode, false);
                let old = complete_call(&r, &f, mode, true);
                assert_output(&live, &old);
                assert_output(&live, &expected(&f, mode));
                for plane in &mut f.bytes {
                    plane.fill(77);
                }
                let changed = complete_call(&r, &f, mode, false);
                assert_output(&changed, &expected(&f, mode));
                assert!(changed.4.iter().flatten().all(|&b| b == 77));
                drop(f);
                assert_output(&live, &old);
                (live, old)
            })
        })
        .collect();
    let retained: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    r.clear_buffer_pool();
    for (live, old) in retained {
        assert_output(&live, &old);
    }
}

#[path = "voxel_borrowed_u8_benchmark.rs"]
mod benchmark;
