//! Exact generic bilateral arithmetic with bounded native output bands.
use super::validate_positive_finite;
use crate::{border::fetch, BorderMode, PixelComponent, VisionError, VisionResult};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use spatialrust_image::{Image, ImageView};

pub(super) fn run<T: PixelComponent, const C: usize>(
    input: ImageView<'_, T, C>,
    diameter: usize,
    sigma_color: f64,
    sigma_space: f64,
    border: BorderMode<T, C>,
) -> VisionResult<Image<T, C>> {
    run_with_workers(input, diameter, sigma_color, sigma_space, border, None)
}
fn run_with_workers<T: PixelComponent, const C: usize>(
    input: ImageView<'_, T, C>,
    diameter: usize,
    sigma_color: f64,
    sigma_space: f64,
    border: BorderMode<T, C>,
    available: Option<usize>,
) -> VisionResult<Image<T, C>> {
    if diameter == 0 {
        return Err(VisionError::InvalidParameter("bilateral diameter must be non-zero".into()));
    }
    validate_positive_finite(sigma_color, "sigma_color")?;
    validate_positive_finite(sigma_space, "sigma_space")?;
    let row_len = input
        .width()
        .checked_mul(C)
        .ok_or_else(|| VisionError::InvalidDimensions("bilateral output row overflows".into()))?;
    let len = row_len
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("bilateral output size overflows".into()))?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::dispatch::{bounded_workers, items_per_worker, ROW_PARALLEL_COMPONENTS};
        if len >= ROW_PARALLEL_COMPONENTS && input.height() > 1 {
            let workers = bounded_workers(
                len,
                input.height(),
                ROW_PARALLEL_COMPONENTS,
                available.unwrap_or_else(rayon::current_num_threads),
            );
            if workers > 1 {
                let radius = (diameter / 2) as isize;
                let color_factor = -0.5 / (sigma_color * sigma_color);
                let space_factor = -0.5 / (sigma_space * sigma_space);
                let rows = items_per_worker(input.height(), workers);
                let mut output = vec![input.get(0, 0).expect("nonempty bilateral input")[0]; len];
                output.par_chunks_mut(rows * row_len).enumerate().for_each(|(band, values)| {
                    for (dy, row) in values.chunks_exact_mut(row_len).enumerate() {
                        let y = band * rows + dy;
                        for (x, out) in row.chunks_exact_mut(C).enumerate() {
                            out.copy_from_slice(&pixel(
                                input,
                                x,
                                y,
                                radius,
                                color_factor,
                                space_factor,
                                border,
                            ));
                        }
                    }
                });
                return Ok(Image::try_new_with_metadata(
                    input.width(),
                    input.height(),
                    output,
                    input.metadata(),
                )?);
            }
        }
    }
    let _ = (available, len);
    // Keep the original serial loop, including its code path for tiny/one-row calls.
    serial(input, diameter, sigma_color, sigma_space, border)
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn pixel<T: PixelComponent, const C: usize>(
    input: ImageView<'_, T, C>,
    x: usize,
    y: usize,
    radius: isize,
    color_factor: f64,
    space_factor: f64,
    border: BorderMode<T, C>,
) -> [T; C] {
    let center = fetch(input, x as isize, y as isize, border);
    let mut sums = [0.0; C];
    let mut total_weight = 0.0;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let dx_f64 = dx as f64;
            let dy_f64 = dy as f64;
            let spatial_distance = dx_f64.mul_add(dx_f64, dy_f64 * dy_f64);
            if spatial_distance > (radius as f64) * (radius as f64) {
                continue;
            }
            let pixel = fetch(input, x as isize + dx, y as isize + dy, border);
            let color_distance = (0..C)
                .map(|channel| pixel[channel].to_f64() - center[channel].to_f64())
                .map(f64::abs)
                .sum::<f64>();
            let weight = ((color_distance * color_distance)
                .mul_add(color_factor, spatial_distance * space_factor))
            .exp();
            for channel in 0..C {
                sums[channel] += pixel[channel].to_f64() * weight;
            }
            total_weight += weight;
        }
    }
    sums.map(|sum| T::from_f64(sum / total_weight))
}
fn serial<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    diameter: usize,
    sigma_color: f64,
    sigma_space: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    if diameter == 0 {
        return Err(VisionError::InvalidParameter("bilateral diameter must be non-zero".into()));
    }
    validate_positive_finite(sigma_color, "sigma_color")?;
    validate_positive_finite(sigma_space, "sigma_space")?;
    let radius = (diameter / 2) as isize;
    let color_factor = -0.5 / (sigma_color * sigma_color);
    let space_factor = -0.5 / (sigma_space * sigma_space);
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            let center = fetch(input, x as isize, y as isize, border);
            let mut sums = [0.0; CHANNELS];
            let mut total_weight = 0.0;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let dx_f64 = dx as f64;
                    let dy_f64 = dy as f64;
                    let spatial_distance = dx_f64.mul_add(dx_f64, dy_f64 * dy_f64);
                    if spatial_distance > (radius as f64) * (radius as f64) {
                        continue;
                    }
                    let pixel = fetch(input, x as isize + dx, y as isize + dy, border);
                    let color_distance = (0..CHANNELS)
                        .map(|channel| pixel[channel].to_f64() - center[channel].to_f64())
                        .map(f64::abs)
                        .sum::<f64>();
                    let weight = ((color_distance * color_distance)
                        .mul_add(color_factor, spatial_distance * space_factor))
                    .exp();
                    for channel in 0..CHANNELS {
                        sums[channel] += pixel[channel].to_f64() * weight;
                    }
                    total_weight += weight;
                }
            }
            output.extend(sums.map(|sum| T::from_f64(sum / total_weight)));
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "bilateral_tests.rs"]
mod tests;
