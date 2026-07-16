//! Non-linear filters, image derivatives, and Gaussian pyramids.

use pulp::Arch;
use rayon::prelude::*;
use spatialrust_image::{Image, ImageView};

use crate::border::fetch;
use crate::{
    filter2d_f32, separable_filter, separable_filter_f32, BorderMode, Kernel1D, Kernel2D,
    PixelComponent, VisionError, VisionResult,
};

/// Applies a per-channel median filter with an odd square aperture.
pub fn median_blur<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel_size: usize,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    validate_odd_size(kernel_size, "median")?;
    let radius = (kernel_size / 2) as isize;
    let area = kernel_size
        .checked_mul(kernel_size)
        .ok_or_else(|| VisionError::InvalidParameter("median kernel area overflows".into()))?;
    let mut samples = (0..CHANNELS).map(|_| Vec::<f64>::with_capacity(area)).collect::<Vec<_>>();
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            for channel in &mut samples {
                channel.clear();
            }
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let pixel = fetch(input, x as isize + dx, y as isize + dy, border);
                    for channel in 0..CHANNELS {
                        samples[channel].push(pixel[channel].to_f64());
                    }
                }
            }
            for channel in &mut samples {
                channel.sort_by(f64::total_cmp);
                output.push(T::from_f64(channel[area / 2]));
            }
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Applies an edge-preserving bilateral filter.
///
/// Color distance is the squared sum of per-channel absolute differences,
/// matching OpenCV's CPU implementation. The operation is out-of-place and
/// never performs a device transfer.
pub fn bilateral_filter<T: PixelComponent, const CHANNELS: usize>(
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

/// Computes a Sobel derivative as signed `f32` values.
pub fn sobel<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    dx: usize,
    dy: usize,
    kernel_size: usize,
    scale: f64,
    delta: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<f32, CHANNELS>> {
    validate_derivative(dx, dy, kernel_size, scale, delta)?;
    let mut kernel_x = derivative_kernel(kernel_size, dx)?;
    let kernel_y = derivative_kernel(kernel_size, dy)?;
    kernel_x =
        Kernel1D::try_new(kernel_x.coefficients().iter().map(|value| value * scale).collect())?;
    separable_filter_f32(input, &kernel_x, &kernel_y, delta, border)
}

/// Computes a first-order 3x3 Sobel derivative directly into signed `f32` output.
///
/// This specialization accepts grayscale `u8` input, derivative order `(1, 0)`
/// or `(0, 1)`, and Replicate or Reflect101 borders. It avoids the generic
/// separable filter's full-image `f64` intermediate while preserving the same
/// coefficients, scale, delta, metadata, and strided-input behavior.
pub fn sobel_3x3_u8(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    dy: usize,
    scale: f64,
    delta: f64,
    border: BorderMode<u8, 1>,
) -> VisionResult<Image<f32, 1>> {
    let len = input
        .width()
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("Sobel output size overflows".into()))?;
    let mut output = vec![0.0; len];
    sobel_3x3_u8_into(input, dx, dy, scale, delta, border, &mut output)?;
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Computes a first-order 3x3 Sobel derivative into caller-owned packed output.
pub fn sobel_3x3_u8_into(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    dy: usize,
    scale: f64,
    delta: f64,
    border: BorderMode<u8, 1>,
    output: &mut [f32],
) -> VisionResult<()> {
    validate_derivative(dx, dy, 3, scale, delta)?;
    if dx + dy != 1 {
        return Err(VisionError::InvalidParameter(
            "specialized 3x3 Sobel requires derivative order (1, 0) or (0, 1)".into(),
        ));
    }
    if !matches!(border, BorderMode::Replicate | BorderMode::Reflect101) {
        return Err(VisionError::InvalidParameter(
            "specialized 3x3 Sobel supports only Replicate and Reflect101 borders".into(),
        ));
    }
    let len = input
        .width()
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("Sobel output size overflows".into()))?;
    if output.len() != len {
        return Err(VisionError::ShapeMismatch(format!(
            "Sobel output needs {len} elements, found {}",
            output.len()
        )));
    }
    if len == 0 {
        return Ok(());
    }

    if scale == 1.0 && delta == 0.0 {
        sobel_3x3_identity_into(input, dx, border, output);
        return Ok(());
    }

    let width = input.width();
    let height = input.height();
    let arch = Arch::new();
    if len >= 262_144 && height > 1 {
        let workers = rayon::current_num_threads().min(height);
        let rows_per_worker = height.div_ceil(workers);
        output.par_chunks_mut(rows_per_worker * width).enumerate().for_each(|(chunk, output)| {
            arch.dispatch(|| {
                sobel_3x3_rows_dispatch(
                    input,
                    dx,
                    scale,
                    delta,
                    border,
                    chunk * rows_per_worker,
                    output,
                );
            });
        });
    } else {
        arch.dispatch(|| {
            sobel_3x3_rows_dispatch(input, dx, scale, delta, border, 0, output);
        });
    }
    Ok(())
}

/// Computes an exact absolute first-order 3x3 Sobel response as saturated `u8`.
///
/// This fuses signed derivative calculation, absolute value, and saturation,
/// avoiding the signed intermediate image required by a two-stage pipeline.
pub fn sobel_abs_3x3_u8(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    dy: usize,
    border: BorderMode<u8, 1>,
) -> VisionResult<Image<u8, 1>> {
    let len = input.width().checked_mul(input.height()).ok_or_else(|| {
        VisionError::InvalidDimensions("absolute Sobel output size overflows".into())
    })?;
    let mut output = vec![0; len];
    sobel_abs_3x3_u8_into(input, dx, dy, border, &mut output)?;
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Computes an exact absolute first-order 3x3 Sobel response into packed output.
pub fn sobel_abs_3x3_u8_into(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    dy: usize,
    border: BorderMode<u8, 1>,
    output: &mut [u8],
) -> VisionResult<()> {
    validate_derivative(dx, dy, 3, 1.0, 0.0)?;
    if dx + dy != 1 {
        return Err(VisionError::InvalidParameter(
            "absolute 3x3 Sobel requires derivative order (1, 0) or (0, 1)".into(),
        ));
    }
    if !matches!(border, BorderMode::Replicate | BorderMode::Reflect101) {
        return Err(VisionError::InvalidParameter(
            "absolute 3x3 Sobel supports only Replicate and Reflect101 borders".into(),
        ));
    }
    let len = input.width().checked_mul(input.height()).ok_or_else(|| {
        VisionError::InvalidDimensions("absolute Sobel output size overflows".into())
    })?;
    if output.len() != len {
        return Err(VisionError::ShapeMismatch(format!(
            "absolute Sobel output needs {len} elements, found {}",
            output.len()
        )));
    }
    if len == 0 {
        return Ok(());
    }

    sobel_abs_or_threshold_3x3_into(input, dx, border, None, output);
    Ok(())
}

/// Computes a binary edge mask from absolute 3x3 Sobel response.
///
/// Pixels whose saturated absolute response is strictly greater than
/// `threshold` become 255; all others become zero. This matches OpenCV's
/// `Sobel(CV_16S)` → `convertScaleAbs` → `THRESH_BINARY` pipeline.
pub fn sobel_threshold_3x3_u8(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    dy: usize,
    threshold: u8,
    border: BorderMode<u8, 1>,
) -> VisionResult<Image<u8, 1>> {
    let len = input
        .width()
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("Sobel mask size overflows".into()))?;
    let mut output = vec![0; len];
    sobel_threshold_3x3_u8_into(input, dx, dy, threshold, border, &mut output)?;
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Computes a binary absolute-Sobel edge mask into caller-owned packed output.
pub fn sobel_threshold_3x3_u8_into(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    dy: usize,
    threshold: u8,
    border: BorderMode<u8, 1>,
    output: &mut [u8],
) -> VisionResult<()> {
    validate_derivative(dx, dy, 3, 1.0, 0.0)?;
    if dx + dy != 1 {
        return Err(VisionError::InvalidParameter(
            "Sobel threshold requires derivative order (1, 0) or (0, 1)".into(),
        ));
    }
    if !matches!(border, BorderMode::Replicate | BorderMode::Reflect101) {
        return Err(VisionError::InvalidParameter(
            "Sobel threshold supports only Replicate and Reflect101 borders".into(),
        ));
    }
    let len = input
        .width()
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("Sobel mask size overflows".into()))?;
    if output.len() != len {
        return Err(VisionError::ShapeMismatch(format!(
            "Sobel mask needs {len} elements, found {}",
            output.len()
        )));
    }
    if len == 0 {
        return Ok(());
    }
    sobel_abs_or_threshold_3x3_into(input, dx, border, Some(threshold), output);
    Ok(())
}

fn sobel_abs_or_threshold_3x3_into(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    border: BorderMode<u8, 1>,
    threshold: Option<u8>,
    output: &mut [u8],
) {
    let width = input.width();
    let height = input.height();
    let len = output.len();
    let workers =
        if len >= 262_144 && height > 1 { rayon::current_num_threads().min(height) } else { 1 };
    let rows_per_worker = height.div_ceil(workers);
    let mut scratch = vec![0_i16; workers * 3 * width];
    scratch
        .par_chunks_mut(3 * width)
        .zip(output.par_chunks_mut(rows_per_worker * width))
        .enumerate()
        .for_each(|(chunk, (scratch, output))| {
            sobel_abs_3x3_stripe(
                input,
                dx,
                border,
                threshold,
                chunk * rows_per_worker,
                scratch,
                output,
            );
        });
}

fn sobel_abs_3x3_stripe(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    border: BorderMode<u8, 1>,
    threshold: Option<u8>,
    start_y: usize,
    scratch: &mut [i16],
    output: &mut [u8],
) {
    let width = input.width();
    let height = input.height();
    let mut slots = [0, 1, 2];
    let (top_y, bottom_y) = gradient_neighbors(start_y, height, border);
    sobel_horizontal_row(input, top_y, dx, border, ring_row_mut(scratch, width, slots[0]));
    sobel_horizontal_row(input, start_y, dx, border, ring_row_mut(scratch, width, slots[1]));
    sobel_horizontal_row(input, bottom_y, dx, border, ring_row_mut(scratch, width, slots[2]));

    let stripe_rows = output.len() / width;
    for (local_y, output) in output.chunks_mut(width).enumerate() {
        let y = start_y + local_y;
        let top = ring_row(scratch, width, slots[0]);
        let middle = ring_row(scratch, width, slots[1]);
        let bottom = ring_row(scratch, width, slots[2]);
        if dx == 1 {
            for (((output, &top), &middle), &bottom) in
                output.iter_mut().zip(top).zip(middle).zip(bottom)
            {
                *output = threshold_sobel(top + 2 * middle + bottom, threshold);
            }
        } else {
            for ((output, &top), &bottom) in output.iter_mut().zip(top).zip(bottom) {
                *output = threshold_sobel(bottom - top, threshold);
            }
        }
        if local_y + 1 < stripe_rows {
            slots.rotate_left(1);
            let (_, next_bottom_y) = gradient_neighbors(y + 1, height, border);
            sobel_horizontal_row(
                input,
                next_bottom_y,
                dx,
                border,
                ring_row_mut(scratch, width, slots[2]),
            );
        }
    }
}

#[inline(always)]
fn saturating_abs_u8(value: i16) -> u8 {
    value.unsigned_abs().min(u16::from(u8::MAX)) as u8
}

#[inline(always)]
fn threshold_sobel(value: i16, threshold: Option<u8>) -> u8 {
    let value = saturating_abs_u8(value);
    threshold.map_or(value, |threshold| u8::from(value > threshold) * u8::MAX)
}

fn sobel_3x3_identity_into(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    border: BorderMode<u8, 1>,
    output: &mut [f32],
) {
    let width = input.width();
    let height = input.height();
    let len = width * height;
    let workers =
        if len >= 262_144 && height > 1 { rayon::current_num_threads().min(height) } else { 1 };
    let rows_per_worker = height.div_ceil(workers);
    let mut scratch = vec![0_i16; workers * 3 * width];
    scratch
        .par_chunks_mut(3 * width)
        .zip(output.par_chunks_mut(rows_per_worker * width))
        .enumerate()
        .for_each(|(chunk, (scratch, output))| {
            let start_y = chunk * rows_per_worker;
            sobel_3x3_identity_stripe(input, dx, border, start_y, scratch, output);
        });
}

fn sobel_3x3_identity_stripe(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    border: BorderMode<u8, 1>,
    start_y: usize,
    scratch: &mut [i16],
    output: &mut [f32],
) {
    let width = input.width();
    let height = input.height();
    let mut slots = [0, 1, 2];
    let (top_y, bottom_y) = gradient_neighbors(start_y, height, border);
    sobel_horizontal_row(input, top_y, dx, border, ring_row_mut(scratch, width, slots[0]));
    sobel_horizontal_row(input, start_y, dx, border, ring_row_mut(scratch, width, slots[1]));
    sobel_horizontal_row(input, bottom_y, dx, border, ring_row_mut(scratch, width, slots[2]));

    let stripe_rows = output.len() / width;
    for (local_y, output) in output.chunks_mut(width).enumerate() {
        let y = start_y + local_y;
        let top = ring_row(scratch, width, slots[0]);
        let middle = ring_row(scratch, width, slots[1]);
        let bottom = ring_row(scratch, width, slots[2]);
        if dx == 1 {
            for (((output, &top), &middle), &bottom) in
                output.iter_mut().zip(top).zip(middle).zip(bottom)
            {
                *output = (top + 2 * middle + bottom) as f32;
            }
        } else {
            for ((output, &top), &bottom) in output.iter_mut().zip(top).zip(bottom) {
                *output = (bottom - top) as f32;
            }
        }

        if local_y + 1 < stripe_rows {
            slots.rotate_left(1);
            let next_y = y + 1;
            let (_, next_bottom_y) = gradient_neighbors(next_y, height, border);
            sobel_horizontal_row(
                input,
                next_bottom_y,
                dx,
                border,
                ring_row_mut(scratch, width, slots[2]),
            );
        }
    }
}

fn sobel_horizontal_row(
    input: ImageView<'_, u8, 1>,
    y: usize,
    dx: usize,
    border: BorderMode<u8, 1>,
    output: &mut [i16],
) {
    let width = input.width();
    let row = input.row(y).expect("Sobel horizontal row in bounds");
    if width == 1 {
        output[0] = if dx == 1 { 0 } else { 4 * i16::from(row[0]) };
        return;
    }
    let (left, _) = gradient_neighbors(0, width, border);
    output[0] = if dx == 1 {
        i16::from(row[1]) - i16::from(row[left])
    } else {
        i16::from(row[left]) + 2 * i16::from(row[0]) + i16::from(row[1])
    };
    if dx == 1 {
        for (output, (&left, &right)) in
            output[1..width - 1].iter_mut().zip(row[..width - 2].iter().zip(&row[2..]))
        {
            *output = i16::from(right) - i16::from(left);
        }
    } else {
        for (output, ((&left, &center), &right)) in output[1..width - 1]
            .iter_mut()
            .zip(row[..width - 2].iter().zip(&row[1..width - 1]).zip(&row[2..]))
        {
            *output = i16::from(left) + 2 * i16::from(center) + i16::from(right);
        }
    }
    let x = width - 1;
    let (_, right) = gradient_neighbors(x, width, border);
    output[x] = if dx == 1 {
        i16::from(row[right]) - i16::from(row[x - 1])
    } else {
        i16::from(row[x - 1]) + 2 * i16::from(row[x]) + i16::from(row[right])
    };
}

fn ring_row(scratch: &[i16], width: usize, slot: usize) -> &[i16] {
    &scratch[slot * width..(slot + 1) * width]
}

fn ring_row_mut(scratch: &mut [i16], width: usize, slot: usize) -> &mut [i16] {
    &mut scratch[slot * width..(slot + 1) * width]
}

#[inline]
fn sobel_3x3_rows_dispatch(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    scale: f64,
    delta: f64,
    border: BorderMode<u8, 1>,
    start_y: usize,
    output: &mut [f32],
) {
    if scale == 1.0 && delta == 0.0 {
        if dx == 1 {
            sobel_x_3x3_rows(input, border, start_y, output);
        } else {
            sobel_y_3x3_rows(input, border, start_y, output);
        }
    } else {
        sobel_3x3_rows(input, dx, scale, delta, border, start_y, output);
    }
}

fn sobel_x_3x3_rows(
    input: ImageView<'_, u8, 1>,
    border: BorderMode<u8, 1>,
    start_y: usize,
    output: &mut [f32],
) {
    let width = input.width();
    let height = input.height();
    for (local_y, output) in output.chunks_mut(width).enumerate() {
        let y = start_y + local_y;
        let (top_y, bottom_y) = gradient_neighbors(y, height, border);
        let top = input.row(top_y).expect("Sobel X row in bounds");
        let middle = input.row(y).expect("Sobel X row in bounds");
        let bottom = input.row(bottom_y).expect("Sobel X row in bounds");
        if width == 1 {
            output[0] = 0.0;
            continue;
        }
        let (left, _) = gradient_neighbors(0, width, border);
        output[0] = sobel_x_value(top, middle, bottom, left, 1) as f32;
        for (
            ((output, (top_left, top_right)), (middle_left, middle_right)),
            (bottom_left, bottom_right),
        ) in output[1..width - 1]
            .iter_mut()
            .zip(top[..width - 2].iter().zip(&top[2..]))
            .zip(middle[..width - 2].iter().zip(&middle[2..]))
            .zip(bottom[..width - 2].iter().zip(&bottom[2..]))
        {
            *output = (i16::from(*top_right) - i16::from(*top_left)
                + 2 * (i16::from(*middle_right) - i16::from(*middle_left))
                + i16::from(*bottom_right)
                - i16::from(*bottom_left)) as f32;
        }
        let x = width - 1;
        let (_, right) = gradient_neighbors(x, width, border);
        output[x] = sobel_x_value(top, middle, bottom, x - 1, right) as f32;
    }
}

fn sobel_y_3x3_rows(
    input: ImageView<'_, u8, 1>,
    border: BorderMode<u8, 1>,
    start_y: usize,
    output: &mut [f32],
) {
    let width = input.width();
    let height = input.height();
    for (local_y, output) in output.chunks_mut(width).enumerate() {
        let y = start_y + local_y;
        let (top_y, bottom_y) = gradient_neighbors(y, height, border);
        let top = input.row(top_y).expect("Sobel Y row in bounds");
        let bottom = input.row(bottom_y).expect("Sobel Y row in bounds");
        if width == 1 {
            output[0] = (4 * (i16::from(bottom[0]) - i16::from(top[0]))) as f32;
            continue;
        }
        let (left, _) = gradient_neighbors(0, width, border);
        output[0] = sobel_y_value(top, bottom, left, 0, 1) as f32;
        for (
            (output, ((top_left, top_center), top_right)),
            ((bottom_left, bottom_center), bottom_right),
        ) in output[1..width - 1]
            .iter_mut()
            .zip(top[..width - 2].iter().zip(&top[1..width - 1]).zip(&top[2..]))
            .zip(bottom[..width - 2].iter().zip(&bottom[1..width - 1]).zip(&bottom[2..]))
        {
            *output = (i16::from(*bottom_left) - i16::from(*top_left)
                + 2 * (i16::from(*bottom_center) - i16::from(*top_center))
                + i16::from(*bottom_right)
                - i16::from(*top_right)) as f32;
        }
        let x = width - 1;
        let (_, right) = gradient_neighbors(x, width, border);
        output[x] = sobel_y_value(top, bottom, x - 1, x, right) as f32;
    }
}

#[inline(always)]
fn sobel_x_value(top: &[u8], middle: &[u8], bottom: &[u8], left: usize, right: usize) -> i16 {
    i16::from(top[right]) - i16::from(top[left])
        + 2 * (i16::from(middle[right]) - i16::from(middle[left]))
        + i16::from(bottom[right])
        - i16::from(bottom[left])
}

#[inline(always)]
fn sobel_y_value(top: &[u8], bottom: &[u8], left: usize, center: usize, right: usize) -> i16 {
    i16::from(bottom[left]) - i16::from(top[left])
        + 2 * (i16::from(bottom[center]) - i16::from(top[center]))
        + i16::from(bottom[right])
        - i16::from(top[right])
}

fn sobel_3x3_rows(
    input: ImageView<'_, u8, 1>,
    dx: usize,
    scale: f64,
    delta: f64,
    border: BorderMode<u8, 1>,
    start_y: usize,
    output: &mut [f32],
) {
    let width = input.width();
    let height = input.height();
    for (local_y, output) in output.chunks_mut(width).enumerate() {
        let y = start_y + local_y;
        let (top_y, bottom_y) = gradient_neighbors(y, height, border);
        let top = input.row(top_y).expect("Sobel row in bounds");
        let middle = input.row(y).expect("Sobel row in bounds");
        let bottom = input.row(bottom_y).expect("Sobel row in bounds");
        if width == 1 {
            output[0] = scaled_sobel_3x3_pixel(top, middle, bottom, 0, 0, 0, dx, scale, delta);
            continue;
        }
        let (left, _) = gradient_neighbors(0, width, border);
        output[0] = scaled_sobel_3x3_pixel(top, middle, bottom, left, 0, 1, dx, scale, delta);
        for (x, value) in output.iter_mut().enumerate().take(width - 1).skip(1) {
            *value = scaled_sobel_3x3_pixel(top, middle, bottom, x - 1, x, x + 1, dx, scale, delta);
        }
        let x = width - 1;
        let (_, right) = gradient_neighbors(x, width, border);
        output[x] = scaled_sobel_3x3_pixel(top, middle, bottom, x - 1, x, right, dx, scale, delta);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn scaled_sobel_3x3_pixel(
    top: &[u8],
    middle: &[u8],
    bottom: &[u8],
    left: usize,
    center: usize,
    right: usize,
    dx: usize,
    scale: f64,
    delta: f64,
) -> f32 {
    let value = if dx == 1 {
        i16::from(top[right]) + 2 * i16::from(middle[right]) + i16::from(bottom[right])
            - i16::from(top[left])
            - 2 * i16::from(middle[left])
            - i16::from(bottom[left])
    } else {
        i16::from(bottom[left]) + 2 * i16::from(bottom[center]) + i16::from(bottom[right])
            - i16::from(top[left])
            - 2 * i16::from(top[center])
            - i16::from(top[right])
    };
    f64::from(value).mul_add(scale, delta) as f32
}

/// Computes exact 3×3 Sobel X/Y gradients together for grayscale `u8` input.
///
/// This matches OpenCV `spatialGradient`: outputs are signed `i16`, the two
/// first derivatives share one source traversal, and only replicated or
/// Reflect101 borders are accepted. CPU storage remains caller-owned and no
/// device transfer is performed.
pub fn spatial_gradient_u8(
    input: ImageView<'_, u8, 1>,
    border: BorderMode<u8, 1>,
) -> VisionResult<(Image<i16, 1>, Image<i16, 1>)> {
    let len = input
        .width()
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("spatial gradient size overflows".into()))?;
    let mut gradient_x = vec![0; len];
    let mut gradient_y = vec![0; len];
    spatial_gradient_u8_into(input, border, &mut gradient_x, &mut gradient_y)?;
    Ok((
        Image::try_new_with_metadata(input.width(), input.height(), gradient_x, input.metadata())?,
        Image::try_new_with_metadata(input.width(), input.height(), gradient_y, input.metadata())?,
    ))
}

/// Computes exact paired 3×3 Sobel gradients into caller-owned packed output.
///
/// Both output slices must contain exactly `input.width() * input.height()`
/// elements and must not overlap each other. Safe Rust borrowing prevents the
/// `u8` input from aliasing either `i16` output.
pub fn spatial_gradient_u8_into(
    input: ImageView<'_, u8, 1>,
    border: BorderMode<u8, 1>,
    gradient_x: &mut [i16],
    gradient_y: &mut [i16],
) -> VisionResult<()> {
    if !matches!(border, BorderMode::Replicate | BorderMode::Reflect101) {
        return Err(VisionError::InvalidParameter(
            "spatial gradient supports only Replicate and Reflect101 borders".into(),
        ));
    }
    let len = input
        .width()
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("spatial gradient size overflows".into()))?;
    validate_gradient_output(gradient_x, len, "gradient_x")?;
    validate_gradient_output(gradient_y, len, "gradient_y")?;
    if len == 0 {
        return Ok(());
    }

    let width = input.width();
    let height = input.height();
    let arch = Arch::new();
    if len >= 1_000_000 && height > 1 {
        let workers = rayon::current_num_threads().min(height);
        let rows_per_worker = height.div_ceil(workers);
        gradient_x
            .par_chunks_mut(rows_per_worker * width)
            .zip(gradient_y.par_chunks_mut(rows_per_worker * width))
            .enumerate()
            .for_each(|(chunk, (gradient_x, gradient_y))| {
                arch.dispatch(|| {
                    spatial_gradient_rows(
                        input,
                        border,
                        chunk * rows_per_worker,
                        gradient_x,
                        gradient_y,
                    );
                });
            });
    } else {
        arch.dispatch(|| spatial_gradient_rows(input, border, 0, gradient_x, gradient_y));
    }
    Ok(())
}

/// Computes exact 3×3 Sobel L1 magnitude (`|Gx| + |Gy|`) for grayscale `u8`.
///
/// The non-negative signed `i16` output ranges from 0 through 2040. Fusing the
/// paired derivatives and magnitude avoids materializing two gradient images.
pub fn sobel_l1_magnitude_u8(
    input: ImageView<'_, u8, 1>,
    border: BorderMode<u8, 1>,
) -> VisionResult<Image<i16, 1>> {
    let len = input
        .width()
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("Sobel magnitude size overflows".into()))?;
    let mut magnitude = vec![0; len];
    sobel_l1_magnitude_u8_into(input, border, &mut magnitude)?;
    Ok(Image::try_new_with_metadata(input.width(), input.height(), magnitude, input.metadata())?)
}

/// Computes exact fused 3×3 Sobel L1 magnitude into caller-owned packed output.
pub fn sobel_l1_magnitude_u8_into(
    input: ImageView<'_, u8, 1>,
    border: BorderMode<u8, 1>,
    magnitude: &mut [i16],
) -> VisionResult<()> {
    if !matches!(border, BorderMode::Replicate | BorderMode::Reflect101) {
        return Err(VisionError::InvalidParameter(
            "Sobel L1 magnitude supports only Replicate and Reflect101 borders".into(),
        ));
    }
    let len = input
        .width()
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("Sobel magnitude size overflows".into()))?;
    validate_gradient_output(magnitude, len, "magnitude")?;
    if len == 0 {
        return Ok(());
    }
    let width = input.width();
    let height = input.height();
    let arch = Arch::new();
    if len >= 1_000_000 && height > 1 {
        let workers = rayon::current_num_threads().min(height);
        let rows_per_worker = height.div_ceil(workers);
        magnitude.par_chunks_mut(rows_per_worker * width).enumerate().for_each(
            |(chunk, magnitude)| {
                arch.dispatch(|| {
                    sobel_l1_rows(input, border, chunk * rows_per_worker, magnitude);
                });
            },
        );
    } else {
        arch.dispatch(|| sobel_l1_rows(input, border, 0, magnitude));
    }
    Ok(())
}

fn validate_gradient_output(output: &[i16], len: usize, name: &str) -> VisionResult<()> {
    if output.len() != len {
        return Err(VisionError::ShapeMismatch(format!(
            "{name} needs {len} elements, found {}",
            output.len()
        )));
    }
    Ok(())
}

fn spatial_gradient_rows(
    input: ImageView<'_, u8, 1>,
    border: BorderMode<u8, 1>,
    start_y: usize,
    gradient_x: &mut [i16],
    gradient_y: &mut [i16],
) {
    let width = input.width();
    let height = input.height();
    for (local_y, (gradient_x, gradient_y)) in
        gradient_x.chunks_mut(width).zip(gradient_y.chunks_mut(width)).enumerate()
    {
        let y = start_y + local_y;
        let (top_y, bottom_y) = gradient_neighbors(y, height, border);
        let top = input.row(top_y).expect("gradient row in bounds");
        let middle = input.row(y).expect("gradient row in bounds");
        let bottom = input.row(bottom_y).expect("gradient row in bounds");
        if width == 1 {
            write_spatial_gradient_pixel(top, middle, bottom, 0, 0, 0, gradient_x, gradient_y);
            continue;
        }
        let (left, _) = gradient_neighbors(0, width, border);
        write_spatial_gradient_pixel(top, middle, bottom, left, 0, 1, gradient_x, gradient_y);
        for ((((top, middle), bottom), gradient_x), gradient_y) in top
            .windows(3)
            .zip(middle.windows(3))
            .zip(bottom.windows(3))
            .zip(gradient_x[1..width - 1].iter_mut())
            .zip(gradient_y[1..width - 1].iter_mut())
        {
            let top_left = i16::from(top[0]);
            let top_middle = i16::from(top[1]);
            let top_right = i16::from(top[2]);
            let middle_left = i16::from(middle[0]);
            let middle_right = i16::from(middle[2]);
            let bottom_left = i16::from(bottom[0]);
            let bottom_middle = i16::from(bottom[1]);
            let bottom_right = i16::from(bottom[2]);
            *gradient_x = top_right + 2 * middle_right + bottom_right
                - top_left
                - 2 * middle_left
                - bottom_left;
            *gradient_y = bottom_left + 2 * bottom_middle + bottom_right
                - top_left
                - 2 * top_middle
                - top_right;
        }
        let x = width - 1;
        let (_, right) = gradient_neighbors(x, width, border);
        write_spatial_gradient_pixel(top, middle, bottom, x - 1, x, right, gradient_x, gradient_y);
    }
}

fn sobel_l1_rows(
    input: ImageView<'_, u8, 1>,
    border: BorderMode<u8, 1>,
    start_y: usize,
    magnitude: &mut [i16],
) {
    let width = input.width();
    let height = input.height();
    for (local_y, magnitude) in magnitude.chunks_mut(width).enumerate() {
        let y = start_y + local_y;
        let (top_y, bottom_y) = gradient_neighbors(y, height, border);
        let top = input.row(top_y).expect("Sobel magnitude row in bounds");
        let middle = input.row(y).expect("Sobel magnitude row in bounds");
        let bottom = input.row(bottom_y).expect("Sobel magnitude row in bounds");
        if width == 1 {
            magnitude[0] = sobel_l1_pixel(top, middle, bottom, 0, 0, 0);
            continue;
        }
        let (left, _) = gradient_neighbors(0, width, border);
        magnitude[0] = sobel_l1_pixel(top, middle, bottom, left, 0, 1);
        for (((top, middle), bottom), magnitude) in top
            .windows(3)
            .zip(middle.windows(3))
            .zip(bottom.windows(3))
            .zip(magnitude[1..width - 1].iter_mut())
        {
            *magnitude = sobel_l1_window(top, middle, bottom);
        }
        let x = width - 1;
        let (_, right) = gradient_neighbors(x, width, border);
        magnitude[x] = sobel_l1_pixel(top, middle, bottom, x - 1, x, right);
    }
}

#[inline(always)]
fn sobel_l1_window(top: &[u8], middle: &[u8], bottom: &[u8]) -> i16 {
    sobel_l1_values(top[0], top[1], top[2], middle[0], middle[2], bottom[0], bottom[1], bottom[2])
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn sobel_l1_pixel(
    top: &[u8],
    middle: &[u8],
    bottom: &[u8],
    left: usize,
    center: usize,
    right: usize,
) -> i16 {
    sobel_l1_values(
        top[left],
        top[center],
        top[right],
        middle[left],
        middle[right],
        bottom[left],
        bottom[center],
        bottom[right],
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn sobel_l1_values(
    top_left: u8,
    top_middle: u8,
    top_right: u8,
    middle_left: u8,
    middle_right: u8,
    bottom_left: u8,
    bottom_middle: u8,
    bottom_right: u8,
) -> i16 {
    let gradient_x = i16::from(top_right) + 2 * i16::from(middle_right) + i16::from(bottom_right)
        - i16::from(top_left)
        - 2 * i16::from(middle_left)
        - i16::from(bottom_left);
    let gradient_y =
        i16::from(bottom_left) + 2 * i16::from(bottom_middle) + i16::from(bottom_right)
            - i16::from(top_left)
            - 2 * i16::from(top_middle)
            - i16::from(top_right);
    gradient_x.abs() + gradient_y.abs()
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn write_spatial_gradient_pixel(
    top: &[u8],
    middle: &[u8],
    bottom: &[u8],
    left: usize,
    center: usize,
    right: usize,
    gradient_x: &mut [i16],
    gradient_y: &mut [i16],
) {
    let top_left = i16::from(top[left]);
    let top_middle = i16::from(top[center]);
    let top_right = i16::from(top[right]);
    let middle_left = i16::from(middle[left]);
    let middle_right = i16::from(middle[right]);
    let bottom_left = i16::from(bottom[left]);
    let bottom_middle = i16::from(bottom[center]);
    let bottom_right = i16::from(bottom[right]);
    gradient_x[center] =
        top_right + 2 * middle_right + bottom_right - top_left - 2 * middle_left - bottom_left;
    gradient_y[center] =
        bottom_left + 2 * bottom_middle + bottom_right - top_left - 2 * top_middle - top_right;
}

fn gradient_neighbors(index: usize, length: usize, border: BorderMode<u8, 1>) -> (usize, usize) {
    if length <= 1 {
        return (0, 0);
    }
    let left = if index > 0 {
        index - 1
    } else if matches!(border, BorderMode::Reflect101) {
        1
    } else {
        0
    };
    let right = if index + 1 < length {
        index + 1
    } else if matches!(border, BorderMode::Reflect101) {
        length - 2
    } else {
        length - 1
    };
    (left, right)
}

/// Computes a 3×3 Scharr first derivative as signed `f32` values.
pub fn scharr<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    dx: usize,
    dy: usize,
    scale: f64,
    delta: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<f32, CHANNELS>> {
    if dx + dy != 1 || dx > 1 || dy > 1 {
        return Err(VisionError::InvalidParameter(
            "Scharr requires derivative order (1, 0) or (0, 1)".into(),
        ));
    }
    if !scale.is_finite() || !delta.is_finite() {
        return Err(VisionError::InvalidParameter("scale and delta must be finite".into()));
    }
    let derivative = Kernel1D::try_new(vec![-scale, 0.0, scale])?;
    let smoothing = Kernel1D::try_new(vec![3.0, 10.0, 3.0])?;
    if dx == 1 {
        separable_filter_f32(input, &derivative, &smoothing, delta, border)
    } else {
        separable_filter_f32(input, &smoothing, &derivative, delta, border)
    }
}

/// Computes a Laplacian response as signed `f32` values.
pub fn laplacian<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel_size: usize,
    scale: f64,
    delta: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<f32, CHANNELS>> {
    if kernel_size == 1 {
        if !scale.is_finite() || !delta.is_finite() {
            return Err(VisionError::InvalidParameter("scale and delta must be finite".into()));
        }
        let kernel = Kernel2D::try_new(
            3,
            3,
            vec![0.0, scale, 0.0, scale, -4.0 * scale, scale, 0.0, scale, 0.0],
        )?;
        return filter2d_f32(input, &kernel, delta, border);
    }
    validate_derivative(2, 0, kernel_size, scale, delta)?;
    let dxx = sobel(input, 2, 0, kernel_size, scale, 0.0, border)?;
    let dyy = sobel(input, 0, 2, kernel_size, scale, 0.0, border)?;
    let output = dxx
        .as_slice()
        .iter()
        .zip(dyy.as_slice())
        .map(|(&x, &y)| (f64::from(x) + f64::from(y) + delta) as f32)
        .collect();
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Reduces each dimension by approximately two using the canonical 5×5 kernel.
pub fn pyr_down<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    let kernel =
        Kernel1D::try_new(vec![1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0])?;
    let blurred = separable_filter(input, &kernel, &kernel, 0.0, border)?;
    let width = input.width().div_ceil(2);
    let height = input.height().div_ceil(2);
    let mut output = Vec::with_capacity(width * height * CHANNELS);
    for y in 0..height {
        for x in 0..width {
            output.extend_from_slice(blurred.get(x * 2, y * 2).expect("pyramid sample in bounds"));
        }
    }
    Ok(Image::try_new_with_metadata(width, height, output, input.metadata())?)
}

/// Doubles each dimension using zero insertion and the canonical 5×5 kernel.
pub fn pyr_up<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    let width = input
        .width()
        .checked_mul(2)
        .ok_or_else(|| VisionError::InvalidDimensions("pyramid width overflow".into()))?;
    let height = input
        .height()
        .checked_mul(2)
        .ok_or_else(|| VisionError::InvalidDimensions("pyramid height overflow".into()))?;
    let mut expanded =
        Image::<T, CHANNELS>::from_pixel(width, height, std::array::from_fn(|_| T::from_f64(0.0)))?;
    expanded.set_metadata(input.metadata())?;
    for y in 0..input.height() {
        for x in 0..input.width() {
            *expanded.get_mut(x * 2, y * 2).expect("expanded coordinate in bounds") =
                *input.get(x, y).expect("source coordinate in bounds");
        }
    }
    let kernel = Kernel1D::try_new(vec![1.0 / 8.0, 4.0 / 8.0, 6.0 / 8.0, 4.0 / 8.0, 1.0 / 8.0])?;
    separable_filter(expanded.view(), &kernel, &kernel, 0.0, border)
}

/// Builds a packed Gaussian pyramid including the input as level zero.
pub fn build_gaussian_pyramid<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    levels: usize,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Vec<Image<T, CHANNELS>>> {
    if levels == 0 {
        return Err(VisionError::InvalidParameter("pyramid levels must be non-zero".into()));
    }
    let mut packed = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            packed.extend_from_slice(input.get(x, y).expect("input coordinate in bounds"));
        }
    }
    let first =
        Image::try_new_with_metadata(input.width(), input.height(), packed, input.metadata())?;
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(first);
    while pyramid.len() < levels {
        let next = pyr_down(pyramid.last().expect("level zero exists").view(), border)?;
        pyramid.push(next);
    }
    Ok(pyramid)
}

fn validate_odd_size(size: usize, name: &str) -> VisionResult<()> {
    if size == 0 || size % 2 == 0 {
        return Err(VisionError::InvalidParameter(format!(
            "{name} kernel size must be positive and odd"
        )));
    }
    Ok(())
}

fn validate_positive_finite(value: f64, name: &str) -> VisionResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(VisionError::InvalidParameter(format!("{name} must be finite and positive")));
    }
    Ok(())
}

fn validate_derivative(
    dx: usize,
    dy: usize,
    kernel_size: usize,
    scale: f64,
    delta: f64,
) -> VisionResult<()> {
    if !matches!(kernel_size, 3 | 5 | 7) {
        return Err(VisionError::InvalidParameter("Sobel kernel size must be 3, 5, or 7".into()));
    }
    if dx + dy == 0 || dx > 2 || dy > 2 || dx >= kernel_size || dy >= kernel_size {
        return Err(VisionError::InvalidParameter(
            "Sobel derivative orders must total at least one and each be at most two".into(),
        ));
    }
    if !scale.is_finite() || !delta.is_finite() {
        return Err(VisionError::InvalidParameter("scale and delta must be finite".into()));
    }
    Ok(())
}

fn derivative_kernel(size: usize, order: usize) -> VisionResult<Kernel1D> {
    let mut polynomial = vec![1.0];
    for _ in 0..order {
        polynomial = convolve_coefficients(&polynomial, &[-1.0, 1.0]);
    }
    for _ in 0..(size - 1 - order) {
        polynomial = convolve_coefficients(&polynomial, &[1.0, 1.0]);
    }
    Kernel1D::try_new(polynomial)
}

fn convolve_coefficients(left: &[f64], right: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; left.len() + right.len() - 1];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            output[i + j] += a * b;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        bilateral_filter, build_gaussian_pyramid, laplacian, median_blur, pyr_down, scharr, sobel,
        sobel_3x3_u8, sobel_3x3_u8_into, sobel_abs_3x3_u8, sobel_abs_3x3_u8_into,
        sobel_l1_magnitude_u8, sobel_l1_magnitude_u8_into, sobel_threshold_3x3_u8,
        sobel_threshold_3x3_u8_into, spatial_gradient_u8, spatial_gradient_u8_into,
    };
    use crate::BorderMode;
    use spatialrust_image::{Image, ImageRegion};

    #[test]
    fn median_removes_impulse_and_accepts_strided_roi() {
        let parent =
            Image::<u8, 1>::try_new(5, 3, vec![0, 0, 0, 0, 0, 0, 10, 255, 10, 0, 0, 10, 10, 10, 0])
                .unwrap();
        let roi = parent.view().subview(ImageRegion::new(1, 0, 3, 3)).unwrap();
        let output = median_blur(roi, 3, BorderMode::Replicate).unwrap();
        assert_eq!(output[(1, 1)][0], 10);
    }

    #[test]
    fn bilateral_preserves_sharp_constant_regions() {
        let image = Image::<u8, 1>::try_new(5, 1, vec![0, 0, 0, 255, 255]).unwrap();
        let output = bilateral_filter(image.view(), 3, 5.0, 1.0, BorderMode::Replicate).unwrap();
        assert_eq!(output.as_slice(), image.as_slice());
    }

    #[test]
    fn sobel_and_scharr_detect_horizontal_ramp() {
        let image =
            Image::<u8, 1>::try_new(5, 3, (0..3).flat_map(|_| [0, 10, 20, 30, 40]).collect())
                .unwrap();
        let sx = sobel(image.view(), 1, 0, 3, 1.0, 0.0, BorderMode::Reflect101).unwrap();
        let sy = sobel(image.view(), 0, 1, 3, 1.0, 0.0, BorderMode::Reflect101).unwrap();
        let scharr_x = scharr(image.view(), 1, 0, 1.0, 0.0, BorderMode::Reflect101).unwrap();
        assert_eq!(sx[(2, 1)][0], 80.0);
        assert_eq!(sy[(2, 1)][0], 0.0);
        assert_eq!(scharr_x[(2, 1)][0], 320.0);
    }

    #[test]
    fn paired_spatial_gradient_matches_independent_sobel_for_strided_input() {
        let parent = Image::<u8, 1>::try_new(
            9,
            6,
            (0..54).map(|index| ((index * 37 + 11) % 256) as u8).collect(),
        )
        .unwrap();
        let input = parent.view().subview(ImageRegion::new(1, 1, 7, 4)).unwrap();
        for border in [BorderMode::Replicate, BorderMode::Reflect101] {
            let (gradient_x, gradient_y) = spatial_gradient_u8(input, border).unwrap();
            let expected_x = sobel(input, 1, 0, 3, 1.0, 0.0, border).unwrap();
            let expected_y = sobel(input, 0, 1, 3, 1.0, 0.0, border).unwrap();
            assert_eq!(
                gradient_x.as_slice(),
                expected_x.as_slice().iter().map(|value| *value as i16).collect::<Vec<_>>()
            );
            assert_eq!(
                gradient_y.as_slice(),
                expected_y.as_slice().iter().map(|value| *value as i16).collect::<Vec<_>>()
            );
            assert_eq!(gradient_x.metadata(), input.metadata());
            assert_eq!(gradient_y.metadata(), input.metadata());
        }
    }

    #[test]
    fn specialized_sobel_matches_generic_for_strided_input() {
        let parent = Image::<u8, 1>::try_new(
            11,
            8,
            (0..88).map(|index| ((index * 41 + 19) % 256) as u8).collect(),
        )
        .unwrap();
        let input = parent.view().subview(ImageRegion::new(1, 1, 9, 6)).unwrap();
        for border in [BorderMode::Replicate, BorderMode::Reflect101] {
            for (dx, dy) in [(1, 0), (0, 1)] {
                let expected = sobel(input, dx, dy, 3, 0.75, -2.5, border).unwrap();
                let actual = sobel_3x3_u8(input, dx, dy, 0.75, -2.5, border).unwrap();
                assert_eq!(actual, expected);
                assert_eq!(actual.metadata(), input.metadata());
            }
        }
    }

    #[test]
    fn specialized_sobel_into_validates_contract() {
        let image = Image::<u8, 1>::try_new(3, 2, vec![0, 1, 2, 3, 4, 5]).unwrap();
        let mut output = vec![0.0; 6];
        sobel_3x3_u8_into(image.view(), 1, 0, 1.0, 0.0, BorderMode::Reflect101, &mut output)
            .unwrap();
        assert!(sobel_3x3_u8_into(
            image.view(),
            1,
            0,
            1.0,
            0.0,
            BorderMode::Reflect101,
            &mut output[..5],
        )
        .is_err());
        assert!(sobel_3x3_u8_into(
            image.view(),
            1,
            1,
            1.0,
            0.0,
            BorderMode::Reflect101,
            &mut output,
        )
        .is_err());
        assert!(sobel_3x3_u8_into(image.view(), 1, 0, 1.0, 0.0, BorderMode::Wrap, &mut output,)
            .is_err());
    }

    #[test]
    fn absolute_sobel_matches_saturated_specialized_derivative() {
        let parent = Image::<u8, 1>::try_new(
            12,
            9,
            (0..108).map(|index| ((index * 67 + 13) % 256) as u8).collect(),
        )
        .unwrap();
        let input = parent.view().subview(ImageRegion::new(1, 1, 10, 7)).unwrap();
        for border in [BorderMode::Replicate, BorderMode::Reflect101] {
            for (dx, dy) in [(1, 0), (0, 1)] {
                let derivative = sobel_3x3_u8(input, dx, dy, 1.0, 0.0, border).unwrap();
                let expected = derivative
                    .as_slice()
                    .iter()
                    .map(|value| value.abs().min(255.0) as u8)
                    .collect::<Vec<_>>();
                let actual = sobel_abs_3x3_u8(input, dx, dy, border).unwrap();
                assert_eq!(actual.as_slice(), expected);
                let mut reused = vec![0; input.width() * input.height()];
                sobel_abs_3x3_u8_into(input, dx, dy, border, &mut reused).unwrap();
                assert_eq!(reused, expected);
            }
        }
    }

    #[test]
    fn thresholded_sobel_matches_absolute_response() {
        let image = Image::<u8, 1>::try_new(
            17,
            13,
            (0..221).map(|index| ((index * 29 + 7) % 256) as u8).collect(),
        )
        .unwrap();
        for (dx, dy) in [(1, 0), (0, 1)] {
            let absolute = sobel_abs_3x3_u8(image.view(), dx, dy, BorderMode::Reflect101).unwrap();
            let expected = absolute
                .as_slice()
                .iter()
                .map(|&value| u8::from(value > 96) * u8::MAX)
                .collect::<Vec<_>>();
            let actual =
                sobel_threshold_3x3_u8(image.view(), dx, dy, 96, BorderMode::Reflect101).unwrap();
            assert_eq!(actual.as_slice(), expected);
            let mut reused = vec![0; expected.len()];
            sobel_threshold_3x3_u8_into(
                image.view(),
                dx,
                dy,
                96,
                BorderMode::Reflect101,
                &mut reused,
            )
            .unwrap();
            assert_eq!(reused, expected);
        }
    }

    #[test]
    fn paired_spatial_gradient_into_validates_outputs_and_borders() {
        let image = Image::<u8, 1>::try_new(3, 2, vec![0, 1, 2, 3, 4, 5]).unwrap();
        let mut gradient_x = vec![0; 6];
        let mut gradient_y = vec![0; 6];
        spatial_gradient_u8_into(
            image.view(),
            BorderMode::Reflect101,
            &mut gradient_x,
            &mut gradient_y,
        )
        .unwrap();
        assert!(spatial_gradient_u8_into(
            image.view(),
            BorderMode::Reflect101,
            &mut gradient_x[..5],
            &mut gradient_y,
        )
        .is_err());
        assert!(spatial_gradient_u8_into(
            image.view(),
            BorderMode::Wrap,
            &mut gradient_x,
            &mut gradient_y,
        )
        .is_err());
    }

    #[test]
    fn fused_sobel_l1_matches_paired_gradients_for_strided_input() {
        let parent = Image::<u8, 1>::try_new(
            10,
            7,
            (0..70).map(|index| ((index * 53 + 7) % 256) as u8).collect(),
        )
        .unwrap();
        let input = parent.view().subview(ImageRegion::new(1, 1, 8, 5)).unwrap();
        for border in [BorderMode::Replicate, BorderMode::Reflect101] {
            let (gradient_x, gradient_y) = spatial_gradient_u8(input, border).unwrap();
            let magnitude = sobel_l1_magnitude_u8(input, border).unwrap();
            let expected = gradient_x
                .as_slice()
                .iter()
                .zip(gradient_y.as_slice())
                .map(|(&x, &y)| x.abs() + y.abs())
                .collect::<Vec<_>>();
            assert_eq!(magnitude.as_slice(), expected);
            assert_eq!(magnitude.metadata(), input.metadata());
        }
    }

    #[test]
    fn fused_sobel_l1_into_validates_output() {
        let image = Image::<u8, 1>::try_new(3, 2, vec![0, 1, 2, 3, 4, 5]).unwrap();
        let mut output = vec![0; 6];
        sobel_l1_magnitude_u8_into(image.view(), BorderMode::Replicate, &mut output).unwrap();
        assert!(sobel_l1_magnitude_u8_into(image.view(), BorderMode::Replicate, &mut output[..5])
            .is_err());
    }

    #[test]
    fn laplacian_of_constant_is_zero() {
        let image = Image::<u16, 1>::from_pixel(7, 5, [42]).unwrap();
        for size in [1, 3, 5, 7] {
            let output = laplacian(image.view(), size, 1.0, 0.0, BorderMode::Reflect101).unwrap();
            assert!(output.as_slice().iter().all(|value| value.abs() < f32::EPSILON));
        }
    }

    #[test]
    fn pyramid_dimensions_follow_ceil_halving() {
        let image = Image::<f32, 3>::from_pixel(7, 5, [1.0, 2.0, 3.0]).unwrap();
        let down = pyr_down(image.view(), BorderMode::Reflect101).unwrap();
        assert_eq!((down.width(), down.height()), (4, 3));
        let pyramid = build_gaussian_pyramid(image.view(), 4, BorderMode::Reflect101).unwrap();
        assert_eq!(
            pyramid.iter().map(|level| (level.width(), level.height())).collect::<Vec<_>>(),
            vec![(7, 5), (4, 3), (2, 2), (1, 1)]
        );
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let image = Image::<u8, 1>::try_new(1, 1, vec![0]).unwrap();
        assert!(median_blur(image.view(), 2, BorderMode::Replicate).is_err());
        assert!(bilateral_filter(image.view(), 0, 1.0, 1.0, BorderMode::Replicate).is_err());
        assert!(sobel(image.view(), 0, 0, 3, 1.0, 0.0, BorderMode::Replicate).is_err());
    }
}
