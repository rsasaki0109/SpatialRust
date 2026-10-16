use super::read_staging_f32;
use crate::WgpuRuntime;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[path = "readback_scalar_reference.rs"]
mod parent;

fn words(len: usize, seed: u32) -> Vec<u32> {
    const SPECIAL: [u32; 16] = [
        0,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0x7f80_0001,
        0xffc0_9876,
        0x7fff_ffff,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x3f80_0000,
        0xbf80_0000,
        0x3eaa_aaab,
    ];
    (0..len)
        .map(|i| {
            if i < SPECIAL.len() {
                SPECIAL[i]
            } else {
                (i as u32).wrapping_mul(0x9e37_79b9).rotate_left(11) ^ seed
            }
        })
        .collect()
}

fn upload(r: &WgpuRuntime, buffer: &wgpu::Buffer, words: &[u32]) {
    r.queue().write_buffer(buffer, 0, bytemuck::cast_slice(words));
    r.queue().submit([]);
}

fn staging(r: &WgpuRuntime, words: &[u32]) -> wgpu::Buffer {
    let buffer = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("scalar-readback-fixture"),
        size: std::mem::size_of_val(words) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    upload(r, &buffer, words);
    buffer
}

fn assert_bits(values: &[f32], words: &[u32]) {
    assert_eq!(values.len(), words.len());
    for (i, (value, word)) in values.iter().zip(words).enumerate() {
        assert_eq!(value.to_bits(), *word, "word {i}");
    }
}

#[test]
fn voxel_scalar_readback_preserves_prefix_bits_capacities_and_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for len in [0, 1, 3, 17, 65_537, 1_048_577] {
        let input = words(len + 16, 0x85eb_ca6b);
        let buffer = staging(&r, &input);
        let mut live = read_staging_f32(r.device(), &buffer, len).unwrap();
        let old = parent::parent_read_staging_f32(r.device(), &buffer, len).unwrap();
        assert_bits(&live, &input[..len]);
        assert_bits(&old, &input[..len]);
        assert_eq!(live.capacity(), len);
        assert_eq!(old.capacity(), len);
        // Reuploading also proves that the successful read released its map.
        let changed = words(len + 16, 0x1234_5678);
        upload(&r, &buffer, &changed);
        let next = read_staging_f32(r.device(), &buffer, changed.len()).unwrap();
        assert_bits(&next, &changed);
        if let Some(first) = live.first_mut() {
            *first = f32::from_bits(0xdead_beef);
        }
        let again = read_staging_f32(r.device(), &buffer, len).unwrap();
        drop(buffer);
        assert_bits(&again, &changed[..len]);
        assert_bits(&next, &changed);
        assert_bits(&old, &input[..len]);
        if !live.is_empty() {
            assert_eq!(live[0].to_bits(), 0xdead_beef);
        }
        checked += len * 4 + changed.len() * 2;
    }
    eprintln!("SCALAR_READBACK_BITS verified {checked} exact prefix/parent/changed/owned words");
}

#[test]
fn voxel_scalar_readback_unmaps_on_bounds_panic_and_can_map_and_upload_again() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut input = words(64, 0x85eb_ca6b);
    let buffer = staging(&r, &input);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        parent::parent_read_staging_f32(r.device(), &buffer, input.len() + 1).unwrap();
    }))
    .is_err());
    // The frozen parent leaves this successful map active after its bounds panic.
    // Inspect it without requesting a second map, then clean up the control case.
    {
        let data = buffer.slice(..).get_mapped_range();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&data), input);
    }
    buffer.unmap();
    for invalid in [input.len() + 1, usize::MAX, input.len() + 17] {
        assert!(catch_unwind(AssertUnwindSafe(|| {
            read_staging_f32(r.device(), &buffer, invalid).unwrap();
        }))
        .is_err());
        for len in [0, 1, input.len()] {
            let values = read_staging_f32(r.device(), &buffer, len).unwrap();
            assert_bits(&values, &input[..len]);
        }
        input.rotate_left(1);
        upload(&r, &buffer, &input);
        let values = read_staging_f32(r.device(), &buffer, input.len()).unwrap();
        assert_bits(&values, &input);
        let old = parent::parent_read_staging_f32(r.device(), &buffer, input.len()).unwrap();
        assert_bits(&old, &input);
    }
    eprintln!("SCALAR_READBACK_UNWIND verified frozen parent map retention and three panic/map/upload cycles");
}

#[test]
fn voxel_scalar_readback_concurrent_buffers_recover_and_keep_owned_results() {
    let r = WgpuRuntime::new_headless().unwrap();
    let barrier = std::sync::Barrier::new(4);
    let outputs = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|worker| {
                let r = &r;
                let barrier = &barrier;
                scope.spawn(move || {
                    let input = words(65_537 + worker * 17, worker as u32);
                    let buffer = staging(r, &input);
                    barrier.wait();
                    assert!(catch_unwind(AssertUnwindSafe(|| {
                        read_staging_f32(r.device(), &buffer, input.len() + 1).unwrap();
                    }))
                    .is_err());
                    let values = read_staging_f32(r.device(), &buffer, input.len()).unwrap();
                    let changed: Vec<_> = input.iter().map(|word| word ^ 0xffff_ffff).collect();
                    upload(r, &buffer, &changed);
                    let next = read_staging_f32(r.device(), &buffer, input.len()).unwrap();
                    assert_bits(&values, &input);
                    assert_bits(&next, &changed);
                    (values, input, next, changed)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    r.wait_idle();
    drop(r);
    let mut checked = 0;
    for (values, input, next, changed) in outputs {
        assert_bits(&values, &input);
        assert_bits(&next, &changed);
        checked += values.len() + next.len();
    }
    eprintln!("SCALAR_READBACK_CONCURRENT verified {checked} words after four independent panic/reuse cycles and runtime drop");
}
