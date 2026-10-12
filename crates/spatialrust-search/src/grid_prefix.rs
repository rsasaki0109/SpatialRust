#[inline]
fn serial_prefix(values: &mut [u32]) -> u32 {
    let mut sum = 0;
    for value in values {
        sum += *value;
        *value = sum;
    }
    sum
}

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
const MIN_PARALLEL_ENTRIES: usize = 8_388_608;
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
const WORKER_CHUNK_ENTRIES: usize = 262_144;

#[inline]
pub(super) fn inclusive_grid_prefix(values: &mut [u32], point_count: usize) {
    #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
    let _ = point_count;
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    if values.len() >= MIN_PARALLEL_ENTRIES && point_count <= values.len() / 1_024 {
        let workers = crate::parallel_worker_count_with_chunk(values.len(), WORKER_CHUNK_ENTRIES);
        if workers > 1 {
            parallel_prefix(values, workers);
            return;
        }
    }
    serial_prefix(values);
}

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
#[cold]
#[inline(never)]
fn parallel_prefix(values: &mut [u32], workers: usize) {
    let workers = workers.max(1).min(values.len());
    if workers <= 1 {
        serial_prefix(values);
        return;
    }
    let chunk_size = values.len().div_ceil(workers);
    // Joins complete before bases are computed or offsets are added. A worker
    // panic propagates without leaving any other worker blocked on a barrier.
    let mut bases = std::thread::scope(|scope| {
        let handles: Vec<_> = values
            .chunks_mut(chunk_size)
            .map(|chunk| scope.spawn(move || serial_prefix(chunk)))
            .collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
    });
    let mut sum = 0;
    for base in &mut bases {
        let count = *base;
        *base = sum;
        sum += count;
    }
    std::thread::scope(|scope| {
        for (chunk, base) in values.chunks_mut(chunk_size).zip(bases) {
            if base != 0 {
                scope.spawn(move || {
                    for value in chunk {
                        *value += base;
                    }
                });
            }
        }
    });
}

#[cfg(test)]
#[path = "grid_prefix_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "grid_prefix_benchmark.rs"]
mod benchmark;
