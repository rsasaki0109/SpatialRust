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
