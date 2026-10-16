#[cfg(not(target_arch = "wasm32"))]
const MIN_PARALLEL_CELLS: usize = 1_048_576;
#[cfg(not(target_arch = "wasm32"))]
const MAX_WORKERS: usize = 4;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaneCopyPlan {
    pub workers: usize,
    pub channels_per_worker: usize,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn plan_plane_copy(
    cells: usize,
    channels: usize,
    available: usize,
) -> Option<PlaneCopyPlan> {
    if cells < MIN_PARALLEL_CELLS || channels < 2 {
        return None;
    }
    let bytes = cells.checked_mul(channels)?.checked_mul(std::mem::size_of::<f32>())?;
    if bytes > isize::MAX as usize {
        return None;
    }
    let workers = available.min(MAX_WORKERS).min(channels);
    if workers < 2 {
        return None;
    }
    Some(PlaneCopyPlan { workers, channels_per_worker: channels.div_ceil(workers) })
}

pub(crate) fn copy_owned_f32_planes(
    values: &[f32],
    cells: usize,
    channels: usize,
) -> Vec<Vec<f32>> {
    let words = cells.checked_mul(channels).expect("float plane extent overflow");
    let values = &values[..words];
    #[cfg(not(target_arch = "wasm32"))]
    if cells >= MIN_PARALLEL_CELLS && channels >= 2 {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        if let Some(plan) = plan_plane_copy(cells, channels, available) {
            return copy_with_plan(values, cells, channels, plan);
        }
    }
    (0..channels).map(|c| values[c * cells..(c + 1) * cells].to_vec()).collect()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn copy_with_plan(
    values: &[f32],
    cells: usize,
    channels: usize,
    plan: PlaneCopyPlan,
) -> Vec<Vec<f32>> {
    // Bounds are checked before workers borrow the mapped data.
    let words = cells.checked_mul(channels).expect("float plane extent overflow");
    let values = &values[..words];
    let mut output: Vec<Vec<f32>> = (0..channels).map(|_| Vec::new()).collect();
    debug_assert!(channels.div_ceil(plan.channels_per_worker) <= plan.workers);
    std::thread::scope(|scope| {
        for (index, band) in output.chunks_mut(plan.channels_per_worker).enumerate() {
            let first = index * plan.channels_per_worker;
            let source = &values[first * cells..(first + band.len()) * cells];
            scope.spawn(move || {
                for (plane, source) in band.iter_mut().zip(source.chunks_exact(cells)) {
                    *plane = source.to_vec();
                }
            });
        }
    });
    output
}

pub(crate) fn copy_owned_xyz_planes(
    values: &[f32],
    cells: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let words = cells.checked_mul(3).expect("XYZ plane extent overflow");
    let values = &values[..words];
    #[cfg(not(target_arch = "wasm32"))]
    if cells >= MIN_PARALLEL_CELLS {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        if let Some(plan) = plan_plane_copy(cells, 3, available) {
            return copy_xyz_with_workers(values, cells, plan.workers);
        }
    }
    (values[..cells].to_vec(), values[cells..cells * 2].to_vec(), values[cells * 2..].to_vec())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn copy_xyz_with_workers(
    values: &[f32],
    cells: usize,
    workers: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!((2..=3).contains(&workers), "XYZ copies require two or three workers");
    // Validate the complete source before scoped workers borrow mapped planes.
    let words = cells.checked_mul(3).expect("XYZ plane extent overflow");
    let values = &values[..words];
    let (x, rest) = values.split_at(cells);
    let (y, z) = rest.split_at(cells);
    std::thread::scope(|scope| {
        // The caller copies Z; no outer output Vec or point-sized scratch is needed.
        if workers == 2 {
            let xy = scope.spawn(|| (x.to_vec(), y.to_vec()));
            let z = z.to_vec();
            let (x, y) = xy.join().expect("XYZ copy worker panicked");
            (x, y, z)
        } else {
            let x = scope.spawn(|| x.to_vec());
            let y = scope.spawn(|| y.to_vec());
            let z = z.to_vec();
            (
                x.join().expect("X copy worker panicked"),
                y.join().expect("Y copy worker panicked"),
                z,
            )
        }
    })
}

#[cfg(not(target_arch = "wasm32"))]
const MIN_PARALLEL_U8_CELLS: usize = MIN_PARALLEL_CELLS * std::mem::size_of::<u32>();

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn plan_packed_u8_copy(
    cells: usize,
    channels: usize,
    available: usize,
) -> Option<PlaneCopyPlan> {
    if cells < MIN_PARALLEL_U8_CELLS {
        return None;
    }
    // Keep the existing four-MiB minimum per plane and bound the packed source extent.
    plan_plane_copy(cells.div_ceil(4), channels, available)
}

fn copy_packed_u8_plane(words: &[u32], cells: usize) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        bytemuck::cast_slice::<u32, u8>(words)[..cells].to_vec()
    }
    #[cfg(target_endian = "big")]
    {
        (0..cells).map(|cell| ((words[cell / 4] >> ((cell % 4) * 8)) & 0xFF) as u8).collect()
    }
}

pub(crate) fn copy_owned_packed_u8_planes(
    words: &[u32],
    cells: usize,
    channels: usize,
) -> Vec<Vec<u8>> {
    let words_per_channel = cells.div_ceil(4);
    let extent =
        words_per_channel.checked_mul(channels).expect("packed byte plane extent overflow");
    let words = &words[..extent];
    #[cfg(not(target_arch = "wasm32"))]
    if cells >= MIN_PARALLEL_U8_CELLS && channels >= 2 {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        if let Some(plan) = plan_packed_u8_copy(cells, channels, available) {
            return copy_packed_u8_with_plan(words, cells, channels, plan);
        }
    }
    (0..channels)
        .map(|channel| {
            copy_packed_u8_plane(
                &words[channel * words_per_channel..(channel + 1) * words_per_channel],
                cells,
            )
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn copy_packed_u8_with_plan(
    words: &[u32],
    cells: usize,
    channels: usize,
    plan: PlaneCopyPlan,
) -> Vec<Vec<u8>> {
    let words_per_channel = cells.div_ceil(4);
    let extent =
        words_per_channel.checked_mul(channels).expect("packed byte plane extent overflow");
    let words = &words[..extent];
    let mut output: Vec<Vec<u8>> = (0..channels).map(|_| Vec::new()).collect();
    debug_assert!(channels.div_ceil(plan.channels_per_worker) <= plan.workers);
    std::thread::scope(|scope| {
        for (index, band) in output.chunks_mut(plan.channels_per_worker).enumerate() {
            let first = index * plan.channels_per_worker;
            let source =
                &words[first * words_per_channel..(first + band.len()) * words_per_channel];
            scope.spawn(move || {
                for (plane, source) in band.iter_mut().zip(source.chunks_exact(words_per_channel)) {
                    *plane = copy_packed_u8_plane(source, cells);
                }
            });
        }
    });
    output
}
