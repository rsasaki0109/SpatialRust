use super::voxel_reduce::record_voxel_reduce_f32_pass;
use crate::{
    gather_voxel_first_f32_multi_gpu, reduce_voxel_average_f32_multi_gpu, GpuVoxelSegments,
    WgpuRuntime,
};
use spatialrust_core::{SpatialError, SpatialResult};

#[path = "voxel_channel_readback_reference.rs"]
mod parent;

type Channels = Vec<Vec<f32>>;

fn assert_bits(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (channel, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.len(), b.len());
        for (cell, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "channel {channel}, cell {cell}");
        }
    }
}

fn capacity(values: &[Vec<f32>]) -> usize {
    values.iter().map(Vec::capacity).sum()
}

fn words(len: usize) -> Vec<u32> {
    const SPECIAL: [u32; 12] = [
        0,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0x7f80_0001,
        0xffc0_9876,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
    ];
    (0..len)
        .map(|i| {
            if i < SPECIAL.len() {
                SPECIAL[i]
            } else {
                (i as u32).wrapping_mul(0x9e37_79b9).rotate_left(11) ^ 0x85eb_ca6b
            }
        })
        .collect()
}

fn staging(r: &WgpuRuntime, words: &[u32]) -> wgpu::Buffer {
    let buffer = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-channel-staging-fixture"),
        size: std::mem::size_of_val(words) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    r.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(words));
    r.queue().submit([]);
    buffer
}

#[test]
fn mapped_voxel_channels_preserve_float_bits_lengths_capacities_and_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut comparisons = 0;
    for channels in [0, 1, 2, 3, 4, 7, 11] {
        for cells in [0, 1, 17, 65_537, 262_145] {
            let input = words(cells * channels + 16);
            let buffer = staging(&r, &input);
            let expected: Channels = (0..channels)
                .map(|c| {
                    input[c * cells..(c + 1) * cells].iter().map(|&x| f32::from_bits(x)).collect()
                })
                .collect();
            let mut live =
                crate::readback::read_staging_f32_channels(r.device(), &buffer, channels, cells)
                    .unwrap();
            let flat =
                parent::parent_read_staging_f32(r.device(), &buffer, cells * channels).unwrap();
            let old = parent::parent_split_channel_blocks(flat, channels, cells);
            assert_bits(&live, &expected);
            assert_bits(&old, &expected);
            assert_eq!(capacity(&live), cells * channels);
            assert_eq!(capacity(&old), if channels == 0 { 0 } else { cells * (2 * channels - 1) });
            if channels > 0 && cells > 0 {
                live[0][0] = f32::from_bits(0xdead_beef);
            }
            let next =
                crate::readback::read_staging_f32_channels(r.device(), &buffer, channels, cells)
                    .unwrap();
            drop(buffer);
            assert_bits(&next, &expected);
            if channels > 0 && cells > 0 {
                assert_eq!(live[0][0].to_bits(), 0xdead_beef);
            }
            comparisons += cells * channels * 3;
        }
    }
    eprintln!("VOXEL_CHANNEL_RAW verified {comparisons} exact float words");
}

#[test]
fn mapped_voxel_channels_unmap_after_panic_and_allow_subsequent_maps() {
    let r = WgpuRuntime::new_headless().unwrap();
    let input = words(48);
    let buffer = staging(&r, &input);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::readback::read_staging_f32_channels(r.device(), &buffer, 4, 13).unwrap();
    }));
    assert!(panic.is_err());
    let output = crate::readback::read_staging_f32_channels(r.device(), &buffer, 4, 12).unwrap();
    for (value, word) in output.iter().flatten().zip(&input) {
        assert_eq!(value.to_bits(), *word);
    }
    assert_eq!(parent::parent_read_staging_f32(r.device(), &buffer, 48).unwrap().len(), 48);
}

struct Fixture {
    source: Channels,
    segments: GpuVoxelSegments,
    first: Channels,
    mean: Channels,
}

fn fixture(r: &WgpuRuntime, cells: usize, channels: usize, occupancy: usize) -> Fixture {
    let points = cells * occupancy;
    let source: Channels = (0..channels)
        .map(|c| (0..points).map(|i| ((i * 17 + c * 29) % 4096) as f32 * 0.25 - 511.5).collect())
        .collect();
    let first = source.iter().map(|v| v[..cells].to_vec()).collect();
    let mean = source
        .iter()
        .map(|v| {
            (0..cells)
                .map(|cell| {
                    let sum: f32 = (0..occupancy).map(|lane| v[lane * cells + cell]).sum();
                    sum * (1.0 / occupancy as f32)
                })
                .collect()
        })
        .collect();
    let indices: Vec<u32> = (0..cells)
        .flat_map(|cell| (0..occupancy).map(move |lane| (lane * cells + cell) as u32))
        .collect();
    let starts: Vec<u32> = (0..cells).map(|cell| (cell * occupancy) as u32).collect();
    let keys: Vec<u32> = (0..cells).flat_map(|cell| [cell as u32, 0, 0, 0]).collect();
    let upload = |name, data: &[u32]| {
        r.upload_u32_storage(name, if data.is_empty() { &[0] } else { data }).unwrap()
    };
    let segments = GpuVoxelSegments::new(
        cells as u32,
        points as u32,
        upload("voxel-channel-keys", &keys),
        upload("voxel-channel-indices", &indices),
        upload("voxel-channel-starts", &starts),
        0,
    );
    Fixture { source, segments, first, mean }
}

fn call(
    r: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    reduce: bool,
    old: bool,
) -> SpatialResult<Channels> {
    match (reduce, old) {
        (false, false) => gather_voxel_first_f32_multi_gpu(r, channels, segments),
        (false, true) => parent::parent_gather_voxel_first_f32_multi_gpu(r, channels, segments),
        (true, false) => reduce_voxel_average_f32_multi_gpu(r, channels, segments),
        (true, true) => parent::parent_reduce_voxel_average_f32_multi_gpu(r, channels, segments),
    }
}

fn parent_capacity(r: &WgpuRuntime, cells: usize, channels: usize, reduce: bool) -> usize {
    if channels == 0 {
        return 0;
    }
    let batches = if reduce { 1 } else { channels.div_ceil(r.max_gather_channels() as usize) };
    cells * (2 * channels - batches)
}

#[test]
fn public_voxel_channels_keep_frozen_analytic_batched_results_validation_and_lifetimes() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut comparisons = 0;
    for cells in [0, 1, 17, 257, 65_537] {
        for channels in [0, 1, 2, 3, 4, 7] {
            for occupancy in [1, 4] {
                let mut f = fixture(&r, cells, channels, occupancy);
                if cells == 257 {
                    let keys: Vec<_> =
                        (0..cells * occupancy).map(|i| ((i % cells) as i64, 0, 0)).collect();
                    f.segments =
                        super::voxel_sort::build_voxel_segments_gpu_from_keys(&r, &keys).unwrap();
                    assert_eq!(f.segments.cell_count(), cells as u32);
                    assert_eq!(f.segments.construction_readback_bytes(), 4);
                }
                let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
                for reduce in [false, true] {
                    let expected = if reduce { &f.mean } else { &f.first };
                    let pool = r.buffer_pool().stats();
                    let mut new = call(&r, &refs, &f.segments, reduce, false).unwrap();
                    let old = call(&r, &refs, &f.segments, reduce, true).unwrap();
                    assert_bits(&new, &old);
                    assert_bits(&new, expected);
                    assert_eq!(capacity(&new), cells * channels);
                    assert_eq!(capacity(&old), parent_capacity(&r, cells, channels, reduce));
                    if cells == 0 || channels == 0 {
                        assert_eq!(r.buffer_pool().stats(), pool);
                    } else {
                        new[0][0] = f32::from_bits(0xdead_beef);
                    }
                    let next = call(&r, &refs, &f.segments, reduce, false).unwrap();
                    r.clear_buffer_pool();
                    assert_bits(&next, expected);
                    if cells > 0 && channels > 0 {
                        assert_eq!(new[0][0].to_bits(), 0xdead_beef);
                    }
                    comparisons += cells * channels * 3;
                }
                let live = call(&r, &refs, &f.segments, false, false).unwrap();
                let expected = f.first.clone();
                drop(refs);
                for source in &mut f.source {
                    source.fill(0.0);
                }
                let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
                let zeros = vec![vec![0.0; cells]; channels];
                assert_bits(&call(&r, &refs, &f.segments, false, false).unwrap(), &zeros);
                drop(refs);
                drop(f);
                r.clear_buffer_pool();
                assert_bits(&live, &expected);
            }
        }
    }
    let f = fixture(&r, 17, 4, 4);
    let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
    let mut bad = refs.clone();
    bad[2] = &f.source[2][..67];
    let pool = r.buffer_pool().stats();
    for reduce in [false, true] {
        let error = call(&r, &bad, &f.segments, reduce, false).unwrap_err();
        assert_eq!(error, SpatialError::BufferLengthMismatch { expected: 68, found: 67 });
        assert_eq!(error, call(&r, &bad, &f.segments, reduce, true).unwrap_err());
        assert_eq!(r.buffer_pool().stats(), pool);
        assert_bits(
            &call(&r, &refs, &f.segments, reduce, false).unwrap(),
            if reduce { &f.mean } else { &f.first },
        );
    }
    let empty = fixture(&r, 0, 0, 1);
    for reduce in [false, true] {
        assert_bits(&call(&r, &[&[123.0]], &empty.segments, reduce, false).unwrap(), &[Vec::new()]);
        assert_bits(&call(&r, &[&[123.0]], &empty.segments, reduce, true).unwrap(), &[Vec::new()]);
    }
    eprintln!("VOXEL_CHANNEL_PUBLIC verified {comparisons} frozen/analytic/repeated words; gather_batch_channels={}",r.max_gather_channels());
}

#[test]
fn concurrent_voxel_channels_keep_owned_results_independent() {
    let r = WgpuRuntime::new_headless().unwrap();
    std::thread::scope(|scope| {
        for channels in [2, 4, 7] {
            let r = &r;
            scope.spawn(move || {
                let f = fixture(r, 16_385, channels, 4);
                let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
                let live = call(r, &refs, &f.segments, false, false).unwrap();
                for _ in 0..3 {
                    assert_bits(&call(r, &refs, &f.segments, false, false).unwrap(), &f.first);
                    assert_bits(&call(r, &refs, &f.segments, true, false).unwrap(), &f.mean);
                }
                let expected = f.first.clone();
                drop(refs);
                drop(f);
                r.clear_buffer_pool();
                assert_bits(&live, &expected);
            });
        }
    });
}

#[path = "voxel_channel_readback_benchmark.rs"]
mod benchmark;
