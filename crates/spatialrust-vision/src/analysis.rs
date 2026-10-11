//! Thresholding, histograms, contrast enhancement, and integral images.

use rayon::prelude::*;
use spatialrust_image::{Image, ImageView};

use crate::border::fetch;
use crate::dispatch::{
    bounded_workers, items_per_worker, should_parallelize, LARGE_PARALLEL_COMPONENTS,
    LIGHT_PARALLEL_COMPONENTS,
};
use crate::{BorderMode, PixelComponent, VisionError, VisionResult};

/// Point-wise threshold transformation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThresholdType {
    /// `max_value` above the threshold, zero otherwise.
    Binary,
    /// Zero above the threshold, `max_value` otherwise.
    BinaryInv,
    /// Clamp values above the threshold to the threshold.
    Truncate,
    /// Preserve values above the threshold, zero otherwise.
    ToZero,
    /// Zero values above the threshold, preserve the others.
    ToZeroInv,
}

/// Adaptive neighborhood statistic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdaptiveThresholdMethod {
    /// Uniform neighborhood mean.
    Mean,
    /// Gaussian-weighted neighborhood mean.
    Gaussian,
}

/// Applies a fixed threshold independently to every component.
#[inline]
pub fn threshold<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    threshold: f64,
    max_value: f64,
    threshold_type: ThresholdType,
) -> VisionResult<Image<T, CHANNELS>> {
    if !threshold.is_finite() || !max_value.is_finite() {
        return Err(VisionError::InvalidParameter("threshold and max_value must be finite".into()));
    }
    let workers = analysis_workers(
        input.width() * input.height() * CHANNELS,
        input.height(),
        LIGHT_PARALLEL_COMPONENTS,
    );
    if workers == 1 {
        return threshold_scalar(input, threshold, max_value, threshold_type);
    }
    // Select the mode outside the hot loop and convert constant outputs once.
    // In particular, integer binary thresholds need no per-component rounding.
    let zero = T::from_f64(0.0);
    match threshold_type {
        ThresholdType::Binary => {
            let max = T::from_f64(max_value);
            map_rows(input, workers, |value| if value.to_f64() > threshold { max } else { zero })
        }
        ThresholdType::BinaryInv => {
            let max = T::from_f64(max_value);
            map_rows(input, workers, |value| if value.to_f64() > threshold { zero } else { max })
        }
        ThresholdType::Truncate => {
            map_rows(input, workers, |value| T::from_f64(value.to_f64().min(threshold)))
        }
        ThresholdType::ToZero => map_rows(input, workers, |value| {
            let value = value.to_f64();
            if value > threshold {
                T::from_f64(value)
            } else {
                zero
            }
        }),
        ThresholdType::ToZeroInv => map_rows(input, workers, |value| {
            let value = value.to_f64();
            if value > threshold {
                zero
            } else {
                T::from_f64(value)
            }
        }),
    }
}

fn map_rows<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    workers: usize,
    transform: impl Fn(T) -> T + Sync,
) -> VisionResult<Image<T, CHANNELS>> {
    let row_len = input.width() * CHANNELS;
    let len = row_len * input.height();
    let mut output = vec![T::from_f64(0.0); len];
    let rows = items_per_worker(input.height(), workers);
    output.par_chunks_mut(rows * row_len).enumerate().for_each(|(band, output)| {
        for (row, target) in output.chunks_mut(row_len).enumerate() {
            let source = input.row(band * rows + row).expect("validated image row");
            for (to, &from) in target.iter_mut().zip(source) {
                *to = transform(from);
            }
        }
    });
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

#[inline]
fn threshold_scalar<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    threshold: f64,
    max_value: f64,
    threshold_type: ThresholdType,
) -> VisionResult<Image<T, CHANNELS>> {
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            let pixel = input.get(x, y).expect("input coordinate in bounds");
            output.extend(std::array::from_fn::<_, CHANNELS, _>(|channel| {
                T::from_f64(apply_threshold(
                    pixel[channel].to_f64(),
                    threshold,
                    max_value,
                    threshold_type,
                ))
            }));
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

fn analysis_workers(components: usize, rows: usize, threshold: usize) -> usize {
    if should_parallelize(components, rows, threshold) {
        bounded_workers(components, rows, threshold, rayon::current_num_threads())
    } else {
        // Keep small/empty inputs from initializing the global worker pool.
        1
    }
}

/// Selects an Otsu threshold for an 8-bit grayscale image and applies it.
pub fn otsu_threshold_u8(
    input: ImageView<'_, u8, 1>,
    max_value: u8,
    threshold_type: ThresholdType,
) -> VisionResult<(u8, Image<u8, 1>)> {
    let histogram = histogram_u8(input);
    let selected = otsu_from_histogram(&histogram, input.width() * input.height()) as u8;
    let output = threshold(input, f64::from(selected), f64::from(max_value), threshold_type)?;
    Ok((selected, output))
}

/// Selects an Otsu threshold for a 16-bit grayscale image and applies it.
pub fn otsu_threshold_u16(
    input: ImageView<'_, u16, 1>,
    max_value: u16,
    threshold_type: ThresholdType,
) -> VisionResult<(u16, Image<u16, 1>)> {
    let mut histogram = vec![0_u64; 65_536];
    for y in 0..input.height() {
        for x in 0..input.width() {
            histogram[input.get(x, y).expect("coordinate in bounds")[0] as usize] += 1;
        }
    }
    let selected = otsu_from_histogram(&histogram, input.width() * input.height()) as u16;
    let output = threshold(input, f64::from(selected), f64::from(max_value), threshold_type)?;
    Ok((selected, output))
}

/// Applies adaptive binary thresholding to an 8-bit grayscale image.
pub fn adaptive_threshold(
    input: ImageView<'_, u8, 1>,
    max_value: u8,
    method: AdaptiveThresholdMethod,
    threshold_type: ThresholdType,
    block_size: usize,
    c: f64,
    border: BorderMode<u8, 1>,
) -> VisionResult<Image<u8, 1>> {
    if !matches!(threshold_type, ThresholdType::Binary | ThresholdType::BinaryInv) {
        return Err(VisionError::InvalidParameter(
            "adaptive threshold supports only binary and binary-inverse output".into(),
        ));
    }
    if block_size <= 1 || block_size % 2 == 0 {
        return Err(VisionError::InvalidParameter(
            "adaptive block size must be odd and greater than one".into(),
        ));
    }
    if !c.is_finite() {
        return Err(VisionError::InvalidParameter("adaptive constant must be finite".into()));
    }
    let weights = match method {
        AdaptiveThresholdMethod::Mean => vec![1.0 / (block_size * block_size) as f64; block_size],
        AdaptiveThresholdMethod::Gaussian => gaussian_weights(block_size),
    };
    let radius = (block_size / 2) as isize;
    let mut output = Vec::with_capacity(input.width() * input.height());
    for y in 0..input.height() {
        for x in 0..input.width() {
            let mut local = 0.0;
            for ky in 0..block_size {
                for kx in 0..block_size {
                    let pixel = fetch(
                        input,
                        x as isize + kx as isize - radius,
                        y as isize + ky as isize - radius,
                        border,
                    )[0];
                    let weight = match method {
                        AdaptiveThresholdMethod::Mean => weights[kx],
                        AdaptiveThresholdMethod::Gaussian => weights[kx] * weights[ky],
                    };
                    local += f64::from(pixel) * weight;
                }
            }
            let source = i64::from(input.get(x, y).expect("coordinate in bounds")[0]);
            let local = cv_round(local).clamp(0, 255);
            let delta = match threshold_type {
                ThresholdType::Binary => c.ceil() as i64,
                ThresholdType::BinaryInv => c.floor() as i64,
                _ => unreachable!("adaptive threshold type checked above"),
            };
            let selected = match threshold_type {
                ThresholdType::Binary => source - local > -delta,
                ThresholdType::BinaryInv => source - local <= -delta,
                _ => unreachable!("adaptive threshold type checked above"),
            };
            output.push(if selected { max_value } else { 0 });
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Computes a configurable one-channel histogram, optionally through a mask.
pub fn histogram<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    channel: usize,
    bins: usize,
    range: (f64, f64),
    mask: Option<ImageView<'_, u8, 1>>,
) -> VisionResult<Vec<u64>> {
    if channel >= CHANNELS {
        return Err(VisionError::InvalidParameter(format!(
            "histogram channel {channel} is outside {CHANNELS} channels"
        )));
    }
    if bins == 0 {
        return Err(VisionError::InvalidParameter("histogram bins must be non-zero".into()));
    }
    if !range.0.is_finite() || !range.1.is_finite() || range.1 <= range.0 {
        return Err(VisionError::InvalidParameter(
            "histogram range must be finite and increasing".into(),
        ));
    }
    if let Some(mask) = mask {
        if mask.width() != input.width() || mask.height() != input.height() {
            return Err(VisionError::ShapeMismatch(
                "histogram mask dimensions must match the image".into(),
            ));
        }
    }
    let mut result = vec![0_u64; bins];
    let scale = bins as f64 / (range.1 - range.0);
    for y in 0..input.height() {
        for x in 0..input.width() {
            if mask.is_some_and(|mask| mask.get(x, y).expect("mask coordinate in bounds")[0] == 0) {
                continue;
            }
            let value = input.get(x, y).expect("coordinate in bounds")[channel].to_f64();
            if value >= range.0 && value < range.1 {
                let bin = ((value - range.0) * scale).floor() as usize;
                result[bin.min(bins - 1)] += 1;
            }
        }
    }
    Ok(result)
}

/// Computes the exact 256-bin histogram of an 8-bit grayscale image.
#[must_use]
pub fn histogram_u8(input: ImageView<'_, u8, 1>) -> Vec<u64> {
    let workers =
        analysis_workers(input.width() * input.height(), input.height(), LARGE_PARALLEL_COMPONENTS);
    if workers > 1 {
        let rows = items_per_worker(input.height(), workers);
        // One fixed-size histogram per bounded band; no shared counters.
        let partials: Vec<[u64; 256]> = (0..input.height().div_ceil(rows))
            .into_par_iter()
            .map(|band| {
                let mut histogram = [0_u64; 256];
                for y in band * rows..((band + 1) * rows).min(input.height()) {
                    for &value in input.row(y).expect("validated image row") {
                        histogram[usize::from(value)] += 1;
                    }
                }
                histogram
            })
            .collect();
        let mut result = vec![0_u64; 256];
        for partial in partials {
            for (count, value) in result.iter_mut().zip(partial) {
                *count += value;
            }
        }
        return result;
    }
    let mut result = vec![0_u64; 256];
    for y in 0..input.height() {
        for &value in input.row(y).expect("validated image row") {
            result[usize::from(value)] += 1;
        }
    }
    result
}

/// Equalizes an 8-bit grayscale histogram.
pub fn equalize_histogram(input: ImageView<'_, u8, 1>) -> VisionResult<Image<u8, 1>> {
    let histogram = histogram_u8(input);
    let total = input.width() * input.height();
    if total == 0 {
        return Ok(Image::try_new_with_metadata(0, 0, Vec::new(), input.metadata())?);
    }
    let first = histogram.iter().position(|&count| count != 0).expect("non-empty image");
    if histogram[first] as usize == total {
        return Ok(Image::try_new_with_metadata(
            input.width(),
            input.height(),
            vec![first as u8; total],
            input.metadata(),
        )?);
    }
    let scale = 255.0 / (total as f64 - histogram[first] as f64);
    let mut cumulative = 0_u64;
    let mut lookup = [0_u8; 256];
    for (value, &count) in histogram.iter().enumerate() {
        cumulative += count;
        lookup[value] = if value <= first {
            0
        } else {
            cv_round((cumulative - histogram[first]) as f64 * scale).clamp(0, 255) as u8
        };
    }
    map_u8(input, &lookup)
}

/// Applies contrast-limited adaptive histogram equalization to grayscale u8.
pub fn clahe(
    input: ImageView<'_, u8, 1>,
    clip_limit: f64,
    tiles_x: usize,
    tiles_y: usize,
) -> VisionResult<Image<u8, 1>> {
    if !clip_limit.is_finite() || clip_limit < 0.0 {
        return Err(VisionError::InvalidParameter(
            "CLAHE clip limit must be finite and non-negative".into(),
        ));
    }
    if tiles_x == 0 || tiles_y == 0 {
        return Err(VisionError::InvalidParameter(
            "CLAHE tile grid dimensions must be non-zero".into(),
        ));
    }
    if input.width() == 0 || input.height() == 0 {
        return Ok(Image::try_new_with_metadata(0, 0, Vec::new(), input.metadata())?);
    }
    let tile_width = input.width().div_ceil(tiles_x);
    let tile_height = input.height().div_ceil(tiles_y);
    let tile_area = tile_width * tile_height;
    let clip_count = if clip_limit > 0.0 {
        ((clip_limit * tile_area as f64 / 256.0) as usize).max(1)
    } else {
        0
    };
    let tile_count = tiles_x
        .checked_mul(tiles_y)
        .ok_or_else(|| VisionError::InvalidDimensions("CLAHE tile grid overflows".into()))?;
    if tile_count > isize::MAX as usize / std::mem::size_of::<[u8; 256]>() {
        return Err(VisionError::InvalidDimensions("CLAHE lookup allocation overflows".into()));
    }
    let mut lookups = vec![[0_u8; 256]; tile_count];
    let workers =
        analysis_workers(input.width() * input.height(), tile_count, LARGE_PARALLEL_COMPONENTS);
    let fill = |start: usize, lookups: &mut [[u8; 256]]| {
        for (offset, lookup) in lookups.iter_mut().enumerate() {
            let tile = start + offset;
            clahe_tile_lookup(
                input,
                tile % tiles_x,
                tile / tiles_x,
                tile_width,
                tile_height,
                clip_count,
                lookup,
            );
        }
    };
    if workers > 1 {
        let tiles = items_per_worker(tile_count, workers);
        lookups
            .par_chunks_mut(tiles)
            .enumerate()
            .for_each(|(band, lookups)| fill(band * tiles, lookups));
    } else {
        fill(0, &mut lookups);
    }
    interpolate_clahe(input, &lookups, tiles_x, tiles_y, tile_width, tile_height)
}

#[allow(clippy::too_many_arguments)]
fn clahe_tile_lookup(
    input: ImageView<'_, u8, 1>,
    tile_x: usize,
    tile_y: usize,
    tile_width: usize,
    tile_height: usize,
    clip_count: usize,
    lookup: &mut [u8; 256],
) {
    let mut histogram = [0_usize; 256];
    for local_y in 0..tile_height {
        for local_x in 0..tile_width {
            let x = tile_x * tile_width + local_x;
            let y = tile_y * tile_height + local_y;
            let value = fetch(input, x as isize, y as isize, BorderMode::Reflect101)[0];
            histogram[value as usize] += 1;
        }
    }
    if clip_count > 0 {
        clip_histogram(&mut histogram, clip_count);
    }
    let scale = 255.0 / (tile_width * tile_height) as f64;
    let mut cumulative = 0_usize;
    for (value, &count) in histogram.iter().enumerate() {
        cumulative += count;
        lookup[value] = cv_round(cumulative as f64 * scale).clamp(0, 255) as u8;
    }
}

/// Summed-area table with a zero top row and left column.
#[derive(Clone, Debug, PartialEq)]
pub struct IntegralImage {
    source_width: usize,
    source_height: usize,
    data: Vec<f64>,
}

impl IntegralImage {
    /// Source image width, excluding the zero border.
    #[must_use]
    pub const fn source_width(&self) -> usize {
        self.source_width
    }

    /// Source image height, excluding the zero border.
    #[must_use]
    pub const fn source_height(&self) -> usize {
        self.source_height
    }

    /// Integral table width (`source_width + 1`).
    #[must_use]
    pub const fn width(&self) -> usize {
        self.source_width + 1
    }

    /// Integral table height (`source_height + 1`).
    #[must_use]
    pub const fn height(&self) -> usize {
        self.source_height + 1
    }

    /// Row-major integral values.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Returns the sum over half-open source rectangle `[x0,x1) × [y0,y1)`.
    pub fn sum_region(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> VisionResult<f64> {
        if x0 > x1 || y0 > y1 || x1 > self.source_width || y1 > self.source_height {
            return Err(VisionError::InvalidParameter(
                "integral region is outside source dimensions".into(),
            ));
        }
        let stride = self.width();
        Ok(self.data[y1 * stride + x1] - self.data[y0 * stride + x1] - self.data[y1 * stride + x0]
            + self.data[y0 * stride + x0])
    }
}

/// Computes a summed-area table for one selected channel.
pub fn integral_image<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    channel: usize,
) -> VisionResult<IntegralImage> {
    if channel >= CHANNELS {
        return Err(VisionError::InvalidParameter(format!(
            "integral channel {channel} is outside {CHANNELS} channels"
        )));
    }
    let width = input
        .width()
        .checked_add(1)
        .ok_or_else(|| VisionError::InvalidDimensions("integral width overflows".into()))?;
    let height = input
        .height()
        .checked_add(1)
        .ok_or_else(|| VisionError::InvalidDimensions("integral height overflows".into()))?;
    let len = width
        .checked_mul(height)
        .ok_or_else(|| VisionError::InvalidDimensions("integral allocation overflows".into()))?;
    let mut data = vec![0.0; len];
    for y in 0..input.height() {
        let mut row_sum = 0.0;
        for x in 0..input.width() {
            row_sum += input.get(x, y).expect("coordinate in bounds")[channel].to_f64();
            data[(y + 1) * width + x + 1] = data[y * width + x + 1] + row_sum;
        }
    }
    Ok(IntegralImage { source_width: input.width(), source_height: input.height(), data })
}

fn apply_threshold(
    value: f64,
    threshold: f64,
    max_value: f64,
    threshold_type: ThresholdType,
) -> f64 {
    match threshold_type {
        ThresholdType::Binary => {
            if value > threshold {
                max_value
            } else {
                0.0
            }
        }
        ThresholdType::BinaryInv => {
            if value > threshold {
                0.0
            } else {
                max_value
            }
        }
        ThresholdType::Truncate => value.min(threshold),
        ThresholdType::ToZero => {
            if value > threshold {
                value
            } else {
                0.0
            }
        }
        ThresholdType::ToZeroInv => {
            if value > threshold {
                0.0
            } else {
                value
            }
        }
    }
}

fn otsu_from_histogram(histogram: &[u64], total: usize) -> usize {
    if total == 0 {
        return 0;
    }
    let global_mean = histogram
        .iter()
        .enumerate()
        .map(|(value, &count)| value as f64 * count as f64)
        .sum::<f64>();
    let mut background_count = 0_u64;
    let mut background_sum = 0.0;
    let mut best_variance = -1.0;
    let mut best = 0;
    for (value, &count) in histogram.iter().enumerate() {
        background_count += count;
        if background_count == 0 {
            continue;
        }
        let foreground_count = total as u64 - background_count;
        if foreground_count == 0 {
            break;
        }
        background_sum += value as f64 * count as f64;
        let background_mean = background_sum / background_count as f64;
        let foreground_mean = (global_mean - background_sum) / foreground_count as f64;
        let difference = background_mean - foreground_mean;
        let variance = background_count as f64 * foreground_count as f64 * difference * difference;
        if variance > best_variance {
            best_variance = variance;
            best = value;
        }
    }
    best
}

fn gaussian_weights(size: usize) -> Vec<f64> {
    let exact: Option<&[f64]> = match size {
        1 => Some(&[1.0]),
        3 => Some(&[0.25, 0.5, 0.25]),
        5 => Some(&[0.0625, 0.25, 0.375, 0.25, 0.0625]),
        7 => Some(&[0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]),
        9 => Some(&[
            0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375, 0.19921875, 0.1171875,
            0.05078125, 0.015625,
        ]),
        _ => None,
    };
    if let Some(exact) = exact {
        return exact.to_vec();
    }
    let sigma = 0.3 * ((size as f64 - 1.0) * 0.5 - 1.0) + 0.8;
    let center = (size / 2) as f64;
    let mut weights = (0..size)
        .map(|index| {
            let offset = index as f64 - center;
            (-(offset * offset) / (2.0 * sigma * sigma)).exp()
        })
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    weights.iter_mut().for_each(|weight| *weight /= sum);
    weights
}

fn map_u8(input: ImageView<'_, u8, 1>, lookup: &[u8; 256]) -> VisionResult<Image<u8, 1>> {
    let workers =
        analysis_workers(input.width() * input.height(), input.height(), LIGHT_PARALLEL_COMPONENTS);
    if workers > 1 {
        return map_rows(input, workers, |value| lookup[usize::from(value)]);
    }
    let mut output = Vec::with_capacity(input.width() * input.height());
    for y in 0..input.height() {
        for x in 0..input.width() {
            output.push(lookup[input.get(x, y).expect("coordinate in bounds")[0] as usize]);
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

fn clip_histogram(histogram: &mut [usize; 256], clip_limit: usize) {
    let mut clipped = 0_usize;
    for count in histogram.iter_mut() {
        if *count > clip_limit {
            clipped += *count - clip_limit;
            *count = clip_limit;
        }
    }
    let batch = clipped / 256;
    if batch > 0 {
        histogram.iter_mut().for_each(|count| *count += batch);
    }
    let residual = clipped - batch * 256;
    let step = 256_usize.checked_div(residual).unwrap_or(1).max(1);
    for index in (0..256).step_by(step).take(residual) {
        histogram[index] += 1;
    }
}

// Horizontal geometry is independent of the image row. Share one compact
// table across bounded output bands, preserving the original f64 operation order.
struct ClaheHorizontal {
    left: usize,
    right: usize,
    weight: f64,
}

fn interpolate_clahe(
    input: ImageView<'_, u8, 1>,
    lookups: &[[u8; 256]],
    tiles_x: usize,
    tiles_y: usize,
    tile_width: usize,
    tile_height: usize,
) -> VisionResult<Image<u8, 1>> {
    let workers =
        analysis_workers(input.width() * input.height(), input.height(), LARGE_PARALLEL_COMPONENTS);
    if workers == 1 {
        return interpolate_clahe_scalar(input, lookups, tiles_x, tiles_y, tile_width, tile_height);
    }
    let horizontal: Vec<_> = (0..input.width())
        .map(|x| {
            let tx = x as f64 / tile_width as f64 - 0.5;
            let raw = tx.floor() as isize;
            ClaheHorizontal {
                left: raw.clamp(0, tiles_x as isize - 1) as usize,
                right: (raw + 1).clamp(0, tiles_x as isize - 1) as usize,
                weight: tx - tx.floor(),
            }
        })
        .collect();
    let rows = items_per_worker(input.height(), workers);
    let mut output = vec![0; input.width() * input.height()];
    output.par_chunks_mut(rows * input.width()).enumerate().for_each(|(band, output)| {
        for (row, target) in output.chunks_mut(input.width()).enumerate() {
            let y = band * rows + row;
            let ty = y as f64 / tile_height as f64 - 0.5;
            let raw = ty.floor() as isize;
            let ya = ty - ty.floor();
            let y0 = raw.clamp(0, tiles_y as isize - 1) as usize;
            let y1 = (raw + 1).clamp(0, tiles_y as isize - 1) as usize;
            let source = input.row(y).expect("validated image row");
            for ((to, &from), geometry) in target.iter_mut().zip(source).zip(&horizontal) {
                let value = usize::from(from);
                let xa = geometry.weight;
                let top = f64::from(lookups[y0 * tiles_x + geometry.left][value]) * (1.0 - xa)
                    + f64::from(lookups[y0 * tiles_x + geometry.right][value]) * xa;
                let bottom = f64::from(lookups[y1 * tiles_x + geometry.left][value]) * (1.0 - xa)
                    + f64::from(lookups[y1 * tiles_x + geometry.right][value]) * xa;
                *to = cv_round(top * (1.0 - ya) + bottom * ya).clamp(0, 255) as u8;
            }
        }
    });
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

fn interpolate_clahe_scalar(
    input: ImageView<'_, u8, 1>,
    lookups: &[[u8; 256]],
    tiles_x: usize,
    tiles_y: usize,
    tile_width: usize,
    tile_height: usize,
) -> VisionResult<Image<u8, 1>> {
    let mut output = Vec::with_capacity(input.width() * input.height());
    for y in 0..input.height() {
        let ty = y as f64 / tile_height as f64 - 0.5;
        let y0_raw = ty.floor() as isize;
        let ya = ty - ty.floor();
        let y0 = y0_raw.clamp(0, tiles_y as isize - 1) as usize;
        let y1 = (y0_raw + 1).clamp(0, tiles_y as isize - 1) as usize;
        for x in 0..input.width() {
            let tx = x as f64 / tile_width as f64 - 0.5;
            let x0_raw = tx.floor() as isize;
            let xa = tx - tx.floor();
            let x0 = x0_raw.clamp(0, tiles_x as isize - 1) as usize;
            let x1 = (x0_raw + 1).clamp(0, tiles_x as isize - 1) as usize;
            let value = input.get(x, y).expect("coordinate in bounds")[0] as usize;
            let top = f64::from(lookups[y0 * tiles_x + x0][value]) * (1.0 - xa)
                + f64::from(lookups[y0 * tiles_x + x1][value]) * xa;
            let bottom = f64::from(lookups[y1 * tiles_x + x0][value]) * (1.0 - xa)
                + f64::from(lookups[y1 * tiles_x + x1][value]) * xa;
            output.push(cv_round(top * (1.0 - ya) + bottom * ya).clamp(0, 255) as u8);
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

fn cv_round(value: f64) -> i64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor as i64
    } else if fraction > 0.5 {
        floor as i64 + 1
    } else {
        let base = floor as i64;
        if base % 2 == 0 {
            base
        } else {
            base + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        adaptive_threshold, clahe, equalize_histogram, histogram, integral_image,
        otsu_threshold_u8, threshold, AdaptiveThresholdMethod, ThresholdType,
    };
    use crate::BorderMode;
    use spatialrust_image::{Image, ImageRegion};

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()
    }

    const MODES: [ThresholdType; 5] = [
        ThresholdType::Binary,
        ThresholdType::BinaryInv,
        ThresholdType::Truncate,
        ThresholdType::ToZero,
        ThresholdType::ToZeroInv,
    ];

    #[test]
    fn parallel_threshold_preserves_strided_roi_channels_and_metadata() {
        let image = Image::<u16, 3>::try_new(
            514,
            262,
            (0..514 * 262 * 3).map(|i| (i * 37 % 65_536) as u16).collect(),
        )
        .unwrap();
        let roi = image.view().subview(ImageRegion::new(1, 2, 511, 257)).unwrap();
        let serial = pool(1);
        for threads in [2, 6] {
            let parallel = pool(threads);
            for mode in MODES {
                let expected = serial.install(|| threshold(roi, 32_768.0, 66_000.0, mode).unwrap());
                let actual = parallel.install(|| threshold(roi, 32_768.0, 66_000.0, mode).unwrap());
                let reference: Vec<u16> = (0..roi.height())
                    .flat_map(|y| roi.row(y).unwrap())
                    .map(|&value| {
                        <u16 as crate::PixelComponent>::from_f64(super::apply_threshold(
                            f64::from(value),
                            32_768.0,
                            66_000.0,
                            mode,
                        ))
                    })
                    .collect();
                assert!(expected.as_slice() == reference, "original definition, mode={mode:?}");
                assert!(
                    actual.as_slice() == expected.as_slice(),
                    "threads={threads}, mode={mode:?}"
                );
                assert_eq!(actual.metadata(), roi.metadata());
                assert_eq!((actual.width(), actual.height()), (511, 257));
            }
        }
    }

    #[test]
    fn parallel_float_threshold_preserves_nan_infinity_and_signed_zero_bits() {
        let (width, height, stride) = (129, 261, 129 * 4 + 7);
        let mut data = vec![12345_f32; stride * height];
        let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, 0.5, 1.0, -7.0];
        for y in 0..height {
            for i in 0..width * 4 {
                data[y * stride + i] = values[(y + i) % values.len()];
            }
        }
        let input =
            spatialrust_image::ImageView::<f32, 4>::new(width, height, stride, &data).unwrap();
        let serial = pool(1);
        let parallel = pool(6);
        for mode in MODES {
            let expected = serial.install(|| threshold(input, 0.5, 1.0, mode).unwrap());
            let actual = parallel.install(|| threshold(input, 0.5, 1.0, mode).unwrap());
            let reference: Vec<f32> = (0..height)
                .flat_map(|y| input.row(y).unwrap())
                .map(|&value| super::apply_threshold(f64::from(value), 0.5, 1.0, mode) as f32)
                .collect();
            assert!(
                expected.as_slice().iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                "original definition, mode={mode:?}"
            );
            assert!(
                actual
                    .as_slice()
                    .iter()
                    .zip(expected.as_slice())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "mode={mode:?}"
            );
        }
    }

    #[test]
    fn parallel_histogram_excludes_padding_and_has_exact_integer_counts() {
        let (width, height, stride) = (1025, 1027, 1042);
        let mut data = vec![255; stride * height];
        let mut expected = vec![0_u64; 256];
        for y in 0..height {
            for x in 0..width {
                let value = ((x + y * 7) % 255) as u8;
                data[y * stride + x] = value;
                expected[usize::from(value)] += 1;
            }
        }
        let input =
            spatialrust_image::ImageView::<u8, 1>::new(width, height, stride, &data).unwrap();
        for threads in [1, 2, 6] {
            let actual = pool(threads).install(|| super::histogram_u8(input));
            assert_eq!(actual, expected);
            assert_eq!(actual[255], 0);
            assert_eq!(actual.iter().sum::<u64>(), (width * height) as u64);
        }
    }

    #[test]
    fn analysis_handles_empty_single_row_and_small_inputs_with_multiple_workers() {
        let parallel = pool(6);
        for (width, height) in [(0, 0), (1, 1), (31, 3), (100_003, 1)] {
            let image = Image::<u8, 1>::try_new(width, height, vec![7; width * height]).unwrap();
            let output = parallel
                .install(|| threshold(image.view(), 7.0, 255.0, ThresholdType::BinaryInv).unwrap());
            assert!(output.as_slice().iter().all(|&value| value == 255));
            let histogram = parallel.install(|| super::histogram_u8(image.view()));
            assert_eq!(histogram[7], (width * height) as u64);
        }
    }

    #[test]
    fn fixed_threshold_types_match_definitions_on_roi() {
        let image = Image::<u16, 1>::try_new(5, 1, vec![0, 10, 20, 30, 40]).unwrap();
        let roi = image.view().subview(ImageRegion::new(1, 0, 3, 1)).unwrap();
        assert_eq!(
            threshold(roi, 20.0, 100.0, ThresholdType::Binary).unwrap().as_slice(),
            &[0, 0, 100]
        );
        assert_eq!(
            threshold(roi, 20.0, 100.0, ThresholdType::Truncate).unwrap().as_slice(),
            &[10, 20, 20]
        );
    }

    #[test]
    fn otsu_separates_bimodal_values() {
        let image = Image::<u8, 1>::try_new(6, 1, vec![10, 10, 10, 200, 200, 200]).unwrap();
        let (selected, output) =
            otsu_threshold_u8(image.view(), 255, ThresholdType::Binary).unwrap();
        assert_eq!(selected, 10);
        assert_eq!(output.as_slice(), &[0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn adaptive_threshold_rejects_even_blocks() {
        let image = Image::<u8, 1>::from_pixel(3, 3, [10]).unwrap();
        assert!(adaptive_threshold(
            image.view(),
            255,
            AdaptiveThresholdMethod::Mean,
            ThresholdType::Binary,
            2,
            0.0,
            BorderMode::Replicate
        )
        .is_err());
    }

    #[test]
    fn histogram_honors_channel_range_and_mask() {
        let image =
            Image::<f32, 2>::try_new(2, 2, vec![0.0, 0.2, 0.5, 0.7, 0.9, 1.1, 1.0, 0.4]).unwrap();
        let mask = Image::<u8, 1>::try_new(2, 2, vec![1, 0, 1, 1]).unwrap();
        assert_eq!(
            histogram(image.view(), 0, 2, (0.0, 1.0), Some(mask.view())).unwrap(),
            vec![1, 1]
        );
    }

    #[test]
    fn equalization_spreads_low_contrast_values() {
        let image = Image::<u8, 1>::try_new(4, 1, vec![10, 10, 20, 30]).unwrap();
        assert_eq!(equalize_histogram(image.view()).unwrap().as_slice(), &[0, 0, 128, 255]);
    }

    #[test]
    fn clahe_preserves_dimensions_and_constant_input() {
        let image = Image::<u8, 1>::from_pixel(7, 5, [42]).unwrap();
        let output = clahe(image.view(), 2.0, 3, 2).unwrap();
        assert_eq!((output.width(), output.height()), (7, 5));
    }

    #[test]
    fn parallel_equalization_preserves_padded_roi_and_metadata() {
        use spatialrust_image::{ColorRange, ColorSpace, ImageMetadata, ImageView};
        let (width, height, stride) = (1029, 1029, 1043);
        let mut data = vec![255; stride * height];
        for y in 0..height {
            for x in 0..width {
                data[y * stride + x] = ((x * 17 + y * 31) % 129 + 32) as u8;
            }
        }
        let metadata = ImageMetadata {
            color_space: ColorSpace::Gray,
            color_range: ColorRange::Full,
            ..Default::default()
        };
        let input = ImageView::<u8, 1>::new_with_metadata(width, height, stride, &data, metadata)
            .unwrap()
            .subview(ImageRegion::new(2, 1, 1025, 1027))
            .unwrap();
        let expected = pool(1).install(|| equalize_histogram(input).unwrap());
        for workers in [2, 6] {
            let actual = pool(workers).install(|| equalize_histogram(input).unwrap());
            assert!(actual.as_slice() == expected.as_slice(), "workers={workers}");
            assert_eq!(actual.metadata(), metadata);
            assert_eq!((actual.width(), actual.height()), (1025, 1027));
        }
    }

    #[test]
    fn parallel_clahe_matches_scalar_for_odd_roi_tiles_and_clip_limits() {
        use spatialrust_image::{ColorRange, ColorSpace, ImageMetadata};
        let metadata = ImageMetadata {
            color_space: ColorSpace::Gray,
            color_range: ColorRange::Limited,
            ..Default::default()
        };
        let image = Image::<u8, 1>::try_new_with_metadata(
            1029,
            1031,
            (0..1029 * 1031).map(|i| ((i * 37 + i / 1029 * 17) % 251) as u8).collect(),
            metadata,
        )
        .unwrap();
        let input = image.view().subview(ImageRegion::new(2, 3, 1025, 1027)).unwrap();
        let serial = pool(1);
        let two = pool(2);
        let six = pool(6);
        for (tiles_x, tiles_y) in [(8, 8), (7, 5), (1, 1)] {
            for clip in [0.0, 0.01, 2.0, 1e100] {
                let expected = serial.install(|| clahe(input, clip, tiles_x, tiles_y).unwrap());
                for parallel in [&two, &six] {
                    let actual = parallel.install(|| clahe(input, clip, tiles_x, tiles_y).unwrap());
                    assert!(
                        actual.as_slice() == expected.as_slice(),
                        "workers={}, grid={tiles_x}x{tiles_y}, clip={clip}",
                        parallel.current_num_threads()
                    );
                    assert_eq!(actual.metadata(), metadata);
                    assert_eq!((actual.width(), actual.height()), (1025, 1027));
                }
            }
        }
    }

    #[test]
    fn parallel_clahe_interpolation_keeps_original_rounding_at_narrow_borders() {
        let lookups: Vec<[u8; 256]> = (0..4)
            .map(|tile| {
                std::array::from_fn(|value| match tile {
                    0 => value as u8,
                    1 => (255 - value) as u8,
                    2 => (value / 2) as u8,
                    _ => (value / 2 + 128) as u8,
                })
            })
            .collect();
        let parallel = pool(6);
        for (width, height) in [(1, 1_000_003), (2, 500_003), (1_000_003, 1)] {
            let image = Image::<u8, 1>::try_new(
                width,
                height,
                (0..width * height).map(|i| (i % 256) as u8).collect(),
            )
            .unwrap();
            let tile_width = width.div_ceil(2);
            let tile_height = height.div_ceil(2);
            let expected = super::interpolate_clahe_scalar(
                image.view(),
                &lookups,
                2,
                2,
                tile_width,
                tile_height,
            )
            .unwrap();
            let actual = parallel.install(|| {
                super::interpolate_clahe(image.view(), &lookups, 2, 2, tile_width, tile_height)
                    .unwrap()
            });
            assert!(actual.as_slice() == expected.as_slice(), "dimensions={width}x{height}");
        }
    }

    #[test]
    fn contrast_fallbacks_constant_images_and_rounding_fixture_remain_exact() {
        let serial = pool(1);
        let parallel = pool(6);
        for (width, height) in [(0, 0), (1, 1), (7, 5), (100_003, 1), (1, 100_003)] {
            let image = Image::<u8, 1>::from_pixel(width, height, [42]).unwrap();
            let expected = serial.install(|| clahe(image.view(), 2.0, 13, 11).unwrap());
            let actual = parallel.install(|| clahe(image.view(), 2.0, 13, 11).unwrap());
            assert!(actual.as_slice() == expected.as_slice());
            assert!(parallel
                .install(|| equalize_histogram(image.view()).unwrap())
                .as_slice()
                .iter()
                .all(|&value| value == 42));
        }
        let constant = Image::<u8, 1>::from_pixel(1025, 1027, [42]).unwrap();
        let actual = parallel.install(|| clahe(constant.view(), 0.0, 7, 5).unwrap());
        assert!(actual.as_slice().iter().all(|&value| value == 255));
        let fixture = Image::<u8, 1>::try_new(3, 2, vec![0, 40, 80, 120, 160, 200]).unwrap();
        assert_eq!(
            clahe(fixture.view(), 0.0, 1, 1).unwrap().as_slice(),
            &[42, 85, 128, 170, 212, 255]
        );
    }

    #[test]
    fn clahe_rejects_overflowing_lookup_grids_before_allocation() {
        let image = Image::<u8, 1>::from_pixel(1, 1, [42]).unwrap();
        for (x, y) in [(usize::MAX, 2), (isize::MAX as usize / 256 + 1, 1)] {
            assert!(matches!(
                clahe(image.view(), 2.0, x, y),
                Err(crate::VisionError::InvalidDimensions(_))
            ));
        }
        for clip in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(clahe(image.view(), clip, 8, 8).is_err());
        }
        assert!(clahe(image.view(), 2.0, 0, 8).is_err());
        assert!(clahe(image.view(), 2.0, 8, 0).is_err());
    }

    #[test]
    fn integral_region_matches_direct_sum() {
        let image = Image::<f32, 1>::try_new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let integral = integral_image(image.view(), 0).unwrap();
        assert_eq!((integral.width(), integral.height()), (4, 3));
        assert_eq!(integral.sum_region(1, 0, 3, 2).unwrap(), 16.0);
    }
}
