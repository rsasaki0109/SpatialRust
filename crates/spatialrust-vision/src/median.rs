//! Generic exact median selection with bounded independent output bands.
use super::validate_odd_size;
#[cfg(not(target_arch = "wasm32"))]
use crate::dispatch::ROW_PARALLEL_COMPONENTS;
use crate::{border::fetch, BorderMode, PixelComponent, VisionError, VisionResult};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use spatialrust_image::{Image, ImageView};

pub(super) fn run<T: PixelComponent, const C: usize>(
    input: ImageView<'_, T, C>,
    kernel: usize,
    border: BorderMode<T, C>,
) -> VisionResult<Image<T, C>> {
    run_with_workers(input, kernel, border, None)
}

fn worker_count(len: usize, height: usize, available: usize) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::dispatch::bounded_workers(len, height, ROW_PARALLEL_COMPONENTS, available)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (len, height, available);
        1
    }
}

fn run_with_workers<T: PixelComponent, const C: usize>(
    input: ImageView<'_, T, C>,
    kernel: usize,
    border: BorderMode<T, C>,
    available: Option<usize>,
) -> VisionResult<Image<T, C>> {
    validate_odd_size(kernel, "median")?;
    let area = kernel
        .checked_mul(kernel)
        .ok_or_else(|| VisionError::InvalidParameter("median kernel area overflows".into()))?;
    let row_len = input
        .width()
        .checked_mul(C)
        .ok_or_else(|| VisionError::InvalidDimensions("median output row size overflows".into()))?;
    let len = row_len
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("median output size overflows".into()))?;
    if len == 0 {
        return Ok(Image::try_new_with_metadata(
            input.width(),
            input.height(),
            Vec::new(),
            input.metadata(),
        )?);
    }
    // Do not initialize/query the global pool for small or single-row calls.
    #[cfg(not(target_arch = "wasm32"))]
    let available = if len >= ROW_PARALLEL_COMPONENTS && input.height() > 1 {
        available.unwrap_or_else(rayon::current_num_threads)
    } else {
        1
    };
    #[cfg(target_arch = "wasm32")]
    let available = {
        let _ = available;
        1
    };
    let workers = worker_count(len, input.height(), available);
    let radius = (kernel / 2) as isize;
    #[cfg(not(target_arch = "wasm32"))]
    if workers > 1 {
        // Copy a real component to initialize safe final storage. Avoid extra
        // scalar conversions or a second result-sized merge vector.
        let initial = input.get(0, 0).expect("nonempty median input")[0];
        let mut output = vec![initial; len];
        let rows = crate::dispatch::items_per_worker(input.height(), workers);
        output.par_chunks_mut(rows * row_len).enumerate().for_each(|(band, output)| {
            let mut offset = 0;
            scan_rows(input, radius, area, border, band * rows, output.len() / row_len, |value| {
                output[offset] = value;
                offset += 1;
            });
        });
        return Ok(Image::try_new_with_metadata(
            input.width(),
            input.height(),
            output,
            input.metadata(),
        )?);
    }
    let _ = workers;
    let mut output = Vec::with_capacity(len);
    scan_rows(input, radius, area, border, 0, input.height(), |value| output.push(value));
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

#[allow(clippy::too_many_arguments)]
fn scan_rows<T: PixelComponent, const C: usize>(
    input: ImageView<'_, T, C>,
    radius: isize,
    area: usize,
    border: BorderMode<T, C>,
    start: usize,
    rows: usize,
    mut emit: impl FnMut(T),
) {
    let mut samples: Vec<_> = (0..C).map(|_| Vec::<f64>::with_capacity(area)).collect();
    for y in start..start + rows {
        for x in 0..input.width() {
            for channel in &mut samples {
                channel.clear();
            }
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let pixel = fetch(input, x as isize + dx, y as isize + dy, border);
                    for channel in 0..C {
                        samples[channel].push(pixel[channel].to_f64());
                    }
                }
            }
            for channel in &mut samples {
                // Keep the library's optimized tiny sort for 1x1/3x3. Rank
                // selection pays off on larger windows, but regresses tiny ones.
                let median = if area <= 9 {
                    channel.sort_by(f64::total_cmp);
                    channel[area / 2]
                } else {
                    *channel.select_nth_unstable_by(area / 2, f64::total_cmp).1
                };
                emit(T::from_f64(median));
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "median_tests.rs"]
mod tests;
