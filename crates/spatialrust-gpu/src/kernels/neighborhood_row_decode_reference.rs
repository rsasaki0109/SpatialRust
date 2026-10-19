//! Frozen decoder from parent 331adc0f776aa3b4fbcbda72d42a428147a2f269.

//! Bounded CPU conversion borrowed from an explicit, already completed GPU map.

use super::{GpuNeighborhoodKind, Neighbor, SpatialError, SpatialResult};

type Decoded = (Vec<Vec<Neighbor>>, Vec<u32>);
const MIN_ROWS: usize = 65_536;
const MIN_NEIGHBORS: usize = 524_288;
const PROBE_ROWS: usize = 64;
const MAX_WORKERS: usize = 4;
const MIN_COUNT_ROWS: usize = 65_536;

pub(super) fn decode(
    words: &[u32],
    stride: usize,
    capacity: u32,
    kind: GpuNeighborhoodKind,
) -> SpatialResult<Decoded> {
    if capacity == 0 && words.len() / 2 >= MIN_COUNT_ROWS {
        return counts(words, kind);
    }
    if !eligible(words, stride, capacity) {
        return serial(words, stride, capacity, kind);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    #[cfg(target_arch = "wasm32")]
    let available = 1;
    decode_with_workers(words, stride, capacity, kind, available)
}

fn counts(words: &[u32], kind: GpuNeighborhoodKind) -> SpatialResult<Decoded> {
    // Capacity zero has exactly two header words per row and no neighbors.
    // Validate every header before allocating output, including unused totals
    // for nearest-k. Radius totals may contain any u32 value.
    let invalid = match kind {
        GpuNeighborhoodKind::NearestK => words.iter().fold(0, |bits, &word| bits | word),
        GpuNeighborhoodKind::Radius => words.chunks_exact(2).fold(0, |bits, row| bits | row[0]),
    };
    if invalid != 0 {
        return Err(SpatialError::InvalidArgument(
            "GPU neighborhood metadata is invalid".to_owned(),
        ));
    }
    let n = words.len() / 2;
    let rows = (0..n).map(|_| Vec::new()).collect();
    let totals = match kind {
        GpuNeighborhoodKind::NearestK => vec![0; n],
        GpuNeighborhoodKind::Radius => words.chunks_exact(2).map(|row| row[1]).collect(),
    };
    Ok((rows, totals))
}

fn eligible(words: &[u32], stride: usize, capacity: u32) -> bool {
    let rows = words.len() / stride;
    if cfg!(target_arch = "wasm32") || rows < MIN_ROWS || capacity == 0 {
        return false;
    }
    // A bounded sample only selects execution policy. Every header is still
    // validated by the selected decoder, including headers outside the sample.
    let step = rows / PROBE_ROWS;
    let mut bands = [0u64; MAX_WORKERS];
    for i in 0..PROBE_ROWS {
        bands[i / (PROBE_ROWS / MAX_WORKERS)] += u64::from(words[i * step * stride].min(capacity));
    }
    let needed = MIN_NEIGHBORS.div_ceil(rows) as u64;
    bands.iter().sum::<u64>() / PROBE_ROWS as u64 >= needed
        && bands.iter().all(|&sum| sum / (PROBE_ROWS / MAX_WORKERS) as u64 >= needed.div_ceil(2))
}

fn workers(rows: usize, available: usize) -> usize {
    available.clamp(1, MAX_WORKERS).min((rows / (MIN_ROWS / MAX_WORKERS)).max(1))
}

fn decode_with_workers(
    words: &[u32],
    stride: usize,
    capacity: u32,
    kind: GpuNeighborhoodKind,
    available: usize,
) -> SpatialResult<Decoded> {
    let count =
        if !cfg!(target_arch = "wasm32") { workers(words.len() / stride, available) } else { 1 };
    #[cfg(not(target_arch = "wasm32"))]
    if count > 1 {
        let n = words.len() / stride;
        // Initialize final top-level slots safely. No row payload is copied to
        // a worker vector, and the caller is included in the thread bound.
        let mut rows: Vec<Vec<Neighbor>> = (0..n).map(|_| Vec::new()).collect();
        let mut totals = vec![0; n];
        let band_size = n.div_ceil(count);
        std::thread::scope(|scope| -> SpatialResult<()> {
            let mut bands = rows.chunks_mut(band_size).zip(totals.chunks_mut(band_size));
            let (first_rows, first_totals) = bands.next().expect("parallel rows are nonempty");
            let mut handles = Vec::with_capacity(count - 1);
            for (i, (rows, totals)) in bands.enumerate() {
                let start = (i + 1) * band_size * stride;
                let source = &words[start..start + rows.len() * stride];
                handles
                    .push(scope.spawn(move || fill(source, stride, capacity, kind, rows, totals)));
            }
            let mut result = fill(
                &words[..first_rows.len() * stride],
                stride,
                capacity,
                kind,
                first_rows,
                first_totals,
            );
            // Join every band even when validation failed. All mapped borrows
            // end before the caller releases and unmaps the staging allocation.
            for handle in handles {
                let band = handle.join().expect("neighborhood decode worker panicked");
                if result.is_ok() {
                    result = band;
                }
            }
            result
        })?;
        return Ok((rows, totals));
    }
    let _ = count;
    serial(words, stride, capacity, kind)
}

#[cfg(not(target_arch = "wasm32"))]
fn row(words: &[u32], capacity: u32, kind: GpuNeighborhoodKind) -> SpatialResult<Vec<Neighbor>> {
    if words[0] > capacity
        || words[1] < words[0]
        || (kind == GpuNeighborhoodKind::NearestK && words[0] != words[1])
    {
        return Err(SpatialError::InvalidArgument(
            "GPU neighborhood metadata is invalid".to_owned(),
        ));
    }
    Ok((0..words[0] as usize)
        .map(|j| Neighbor {
            index: words[2 + 2 * j] as usize,
            distance_squared: f32::from_bits(words[3 + 2 * j]),
        })
        .collect())
}

#[cfg(not(target_arch = "wasm32"))]
fn fill(
    words: &[u32],
    stride: usize,
    capacity: u32,
    kind: GpuNeighborhoodKind,
    rows: &mut [Vec<Neighbor>],
    totals: &mut [u32],
) -> SpatialResult<()> {
    for ((words, output), total) in words.chunks_exact(stride).zip(rows).zip(totals) {
        *output = row(words, capacity, kind)?;
        *total = words[1];
    }
    Ok(())
}

#[inline]
fn serial(
    words: &[u32],
    stride: usize,
    capacity: u32,
    kind: GpuNeighborhoodKind,
) -> SpatialResult<Decoded> {
    let n = words.len() / stride;
    let mut rows = Vec::with_capacity(n);
    let mut totals = Vec::with_capacity(n);
    // Retain the original in-loop collector. Returning a Result<Vec<_>> for
    // every row adds measurable overhead even when no worker is started.
    for row in words.chunks_exact(stride) {
        if row[0] > capacity
            || row[1] < row[0]
            || (kind == GpuNeighborhoodKind::NearestK && row[0] != row[1])
        {
            return Err(SpatialError::InvalidArgument(
                "GPU neighborhood metadata is invalid".to_owned(),
            ));
        }
        rows.push(
            (0..row[0] as usize)
                .map(|j| Neighbor {
                    index: row[2 + 2 * j] as usize,
                    distance_squared: f32::from_bits(row[3 + 2 * j]),
                })
                .collect(),
        );
        totals.push(row[1]);
    }
    Ok((rows, totals))
}
