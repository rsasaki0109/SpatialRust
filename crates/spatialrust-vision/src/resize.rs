use rayon::prelude::*;
use spatialrust_image::{ColorSpace, Image, ImageMetadata, ImageView, ImageViewMut, PlanarImage};

use crate::{PixelComponent, VisionError, VisionResult};

/// Sampling filter used by image resampling and geometric warps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Interpolation {
    /// Closest source pixel using half-pixel coordinate mapping.
    Nearest,
    /// Bilinear interpolation using half-pixel coordinate mapping.
    #[default]
    Bilinear,
    /// Bicubic interpolation with OpenCV-compatible cubic coefficient `-0.75`.
    Bicubic,
    /// Pixel-area integration when shrinking; bilinear when enlarging.
    Area,
}

const BILINEAR_WEIGHT_BITS: u32 = 11;
const BILINEAR_WEIGHT_SCALE: u32 = 1 << BILINEAR_WEIGHT_BITS;
const PARALLEL_RESIZE_COMPONENTS: usize = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BilinearAxisSample {
    lower: usize,
    upper: usize,
    upper_weight: u16,
}

/// Reusable source coordinates and fixed-point coefficients for `u8` bilinear resize.
///
/// Constructing a plan performs all half-pixel coordinate mapping once. Execution
/// never allocates and accepts packed or explicitly strided input/output views.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BilinearResizeU8Plan {
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    x_samples: Vec<BilinearAxisSample>,
    y_samples: Vec<BilinearAxisSample>,
}

impl BilinearResizeU8Plan {
    /// Builds a plan for the named input and output dimensions.
    pub fn new(
        input_width: usize,
        input_height: usize,
        output_width: usize,
        output_height: usize,
    ) -> VisionResult<Self> {
        if output_width != 0 && output_height != 0 && (input_width == 0 || input_height == 0) {
            return Err(VisionError::InvalidDimensions(
                "cannot resize an empty input to a non-empty output".to_owned(),
            ));
        }
        let half_scale = input_width == output_width.saturating_mul(2)
            && input_height == output_height.saturating_mul(2);
        Ok(Self {
            input_width,
            input_height,
            output_width,
            output_height,
            x_samples: if half_scale {
                Vec::new()
            } else {
                bilinear_axis_samples(input_width, output_width)
            },
            y_samples: if half_scale {
                Vec::new()
            } else {
                bilinear_axis_samples(input_height, output_height)
            },
        })
    }

    /// Returns the planned input dimensions as `(width, height)`.
    #[must_use]
    pub const fn input_dimensions(&self) -> (usize, usize) {
        (self.input_width, self.input_height)
    }

    /// Returns the planned output dimensions as `(width, height)`.
    #[must_use]
    pub const fn output_dimensions(&self) -> (usize, usize) {
        (self.output_width, self.output_height)
    }

    /// Resizes into a newly allocated image while preserving input metadata.
    pub fn resize<const CHANNELS: usize>(
        &self,
        input: ImageView<'_, u8, CHANNELS>,
    ) -> VisionResult<Image<u8, CHANNELS>> {
        let len = self
            .output_width
            .checked_mul(self.output_height)
            .and_then(|pixels| pixels.checked_mul(CHANNELS))
            .ok_or_else(|| {
                VisionError::InvalidDimensions("resize output is too large".to_owned())
            })?;
        let mut output = Image::try_new_with_metadata(
            self.output_width,
            self.output_height,
            vec![0; len],
            input.metadata(),
        )?;
        self.resize_into(input, output.view_mut())?;
        Ok(output)
    }

    /// Resizes into caller-owned storage without allocating.
    pub fn resize_into<const CHANNELS: usize>(
        &self,
        input: ImageView<'_, u8, CHANNELS>,
        mut output: ImageViewMut<'_, u8, CHANNELS>,
    ) -> VisionResult<()> {
        if (input.width(), input.height()) != self.input_dimensions() {
            return Err(VisionError::InvalidDimensions(format!(
                "resize plan expects input {}x{}, found {}x{}",
                self.input_width,
                self.input_height,
                input.width(),
                input.height()
            )));
        }
        if (output.width(), output.height()) != self.output_dimensions() {
            return Err(VisionError::InvalidDimensions(format!(
                "resize plan expects output {}x{}, found {}x{}",
                self.output_width,
                self.output_height,
                output.width(),
                output.height()
            )));
        }
        output.set_metadata(input.metadata())?;
        if self.output_width == 0 || self.output_height == 0 {
            return Ok(());
        }

        let row_stride = output.row_stride();
        let output_components = self.output_width * self.output_height * CHANNELS;
        let rows = output.as_mut_slice().par_chunks_mut(row_stride).enumerate();
        if self.input_width == self.output_width.saturating_mul(2)
            && self.input_height == self.output_height.saturating_mul(2)
        {
            if output_components >= PARALLEL_RESIZE_COMPONENTS {
                rows.for_each(|(y, row)| resize_half_row(input, row, y, self.output_width));
            } else {
                output
                    .as_mut_slice()
                    .chunks_mut(row_stride)
                    .enumerate()
                    .for_each(|(y, row)| resize_half_row(input, row, y, self.output_width));
            }
        } else if output_components >= PARALLEL_RESIZE_COMPONENTS {
            rows.for_each(|(y, row)| self.resize_bilinear_row(input, row, y));
        } else {
            output
                .as_mut_slice()
                .chunks_mut(row_stride)
                .enumerate()
                .for_each(|(y, row)| self.resize_bilinear_row(input, row, y));
        }
        Ok(())
    }

    /// Fuses bilinear RGB resize and BT.601 grayscale conversion into one pass.
    ///
    /// The result is bit-exact with this plan's RGB [`Self::resize`] followed by
    /// SpatialRust's Q14 RGB-to-gray conversion, while avoiding the intermediate
    /// three-channel image.
    pub fn resize_rgb_to_gray(&self, input: ImageView<'_, u8, 3>) -> VisionResult<Image<u8, 1>> {
        let len = self.output_width.checked_mul(self.output_height).ok_or_else(|| {
            VisionError::InvalidDimensions("fused resize-to-gray output is too large".to_owned())
        })?;
        let metadata = ImageMetadata { color_space: ColorSpace::Gray, ..input.metadata() };
        let mut output = Image::try_new_with_metadata(
            self.output_width,
            self.output_height,
            vec![0; len],
            metadata,
        )?;
        self.resize_rgb_to_gray_into(input, output.view_mut())?;
        Ok(output)
    }

    /// Fuses bilinear RGB resize and BT.601 grayscale conversion into caller-owned storage.
    pub fn resize_rgb_to_gray_into(
        &self,
        input: ImageView<'_, u8, 3>,
        mut output: ImageViewMut<'_, u8, 1>,
    ) -> VisionResult<()> {
        if (input.width(), input.height()) != self.input_dimensions() {
            return Err(VisionError::InvalidDimensions(format!(
                "resize plan expects input {}x{}, found {}x{}",
                self.input_width,
                self.input_height,
                input.width(),
                input.height()
            )));
        }
        if (output.width(), output.height()) != self.output_dimensions() {
            return Err(VisionError::InvalidDimensions(format!(
                "resize plan expects output {}x{}, found {}x{}",
                self.output_width,
                self.output_height,
                output.width(),
                output.height()
            )));
        }
        output.set_metadata(ImageMetadata { color_space: ColorSpace::Gray, ..input.metadata() })?;
        if self.output_width == 0 || self.output_height == 0 {
            return Ok(());
        }

        let row_stride = output.row_stride();
        let half_scale = self.input_width == self.output_width.saturating_mul(2)
            && self.input_height == self.output_height.saturating_mul(2);
        let run_row = |y: usize, row: &mut [u8]| {
            if half_scale {
                resize_half_rgb_to_gray_row(input, row, y, self.output_width);
            } else {
                self.resize_bilinear_rgb_to_gray_row(input, row, y);
            }
        };
        if self.output_width * self.output_height >= PARALLEL_RESIZE_COMPONENTS {
            if self.output_height >= 2_000 {
                const ROWS_PER_TASK: usize = 8;
                let block_stride = row_stride.checked_mul(ROWS_PER_TASK).ok_or_else(|| {
                    VisionError::InvalidDimensions("fused resize row block is too large".to_owned())
                })?;
                output.as_mut_slice().par_chunks_mut(block_stride).enumerate().for_each(
                    |(block, rows)| {
                        for (row, target) in rows.chunks_mut(row_stride).enumerate() {
                            run_row(block * ROWS_PER_TASK + row, target);
                        }
                    },
                );
            } else {
                output
                    .as_mut_slice()
                    .par_chunks_mut(row_stride)
                    .enumerate()
                    .for_each(|(y, row)| run_row(y, row));
            }
        } else {
            output
                .as_mut_slice()
                .chunks_mut(row_stride)
                .enumerate()
                .for_each(|(y, row)| run_row(y, row));
        }
        Ok(())
    }

    /// Fuses bilinear RGB resize, per-channel normalization, and CHW packing.
    ///
    /// The result is bit-exact with this plan's RGB [`Self::resize`] followed by
    /// SpatialRust's planar normalization for the same parameters, while avoiding
    /// the intermediate resized RGB image.
    pub fn resize_rgb_to_chw(
        &self,
        input: ImageView<'_, u8, 3>,
        scale: f32,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> VisionResult<PlanarImage<f32, 3>> {
        let plane_len = self.output_width.checked_mul(self.output_height).ok_or_else(|| {
            VisionError::InvalidDimensions("fused resize-to-CHW plane is too large".to_owned())
        })?;
        let len = plane_len.checked_mul(3).ok_or_else(|| {
            VisionError::InvalidDimensions("fused resize-to-CHW output is too large".to_owned())
        })?;
        let mut output = vec![0.0_f32; len];
        self.resize_rgb_to_chw_into(input, scale, mean, std, &mut output)?;
        Ok(PlanarImage::try_new_with_metadata(
            self.output_width,
            self.output_height,
            output,
            input.metadata(),
        )?)
    }

    /// Fuses bilinear RGB resize, normalization, and CHW packing into a reusable slice.
    pub fn resize_rgb_to_chw_into(
        &self,
        input: ImageView<'_, u8, 3>,
        scale: f32,
        mean: [f32; 3],
        std: [f32; 3],
        output: &mut [f32],
    ) -> VisionResult<()> {
        validate_rgb_chw_normalization(scale, std)?;
        if (input.width(), input.height()) != self.input_dimensions() {
            return Err(VisionError::InvalidDimensions(format!(
                "resize plan expects input {}x{}, found {}x{}",
                self.input_width,
                self.input_height,
                input.width(),
                input.height()
            )));
        }
        let plane_len = self.output_width.checked_mul(self.output_height).ok_or_else(|| {
            VisionError::InvalidDimensions("fused resize-to-CHW plane is too large".to_owned())
        })?;
        let required = plane_len.checked_mul(3).ok_or_else(|| {
            VisionError::InvalidDimensions("fused resize-to-CHW output is too large".to_owned())
        })?;
        if output.len() != required {
            return Err(VisionError::ShapeMismatch(format!(
                "CHW output needs {required} elements, found {}",
                output.len()
            )));
        }
        if plane_len == 0 {
            return Ok(());
        }

        let half_scale = self.input_width == self.output_width.saturating_mul(2)
            && self.input_height == self.output_height.saturating_mul(2);
        let run_row = |plane_row: usize, target: &mut [f32]| {
            let channel = plane_row / self.output_height;
            let y = plane_row % self.output_height;
            if half_scale {
                resize_half_rgb_to_chw_row(
                    input,
                    target,
                    y,
                    channel,
                    scale,
                    mean[channel],
                    std[channel],
                );
            } else {
                self.resize_bilinear_rgb_to_chw_row(
                    input,
                    target,
                    y,
                    channel,
                    scale,
                    mean[channel],
                    std[channel],
                );
            }
        };
        if required >= PARALLEL_RESIZE_COMPONENTS {
            output
                .par_chunks_mut(self.output_width)
                .enumerate()
                .for_each(|(plane_row, target)| run_row(plane_row, target));
        } else {
            output
                .chunks_mut(self.output_width)
                .enumerate()
                .for_each(|(plane_row, target)| run_row(plane_row, target));
        }
        Ok(())
    }

    fn resize_bilinear_row<const CHANNELS: usize>(
        &self,
        input: ImageView<'_, u8, CHANNELS>,
        output: &mut [u8],
        y: usize,
    ) {
        let y_sample = self.y_samples[y];
        let top = input.row(y_sample.lower).expect("planned source row");
        let bottom = input.row(y_sample.upper).expect("planned source row");
        let wy = u32::from(y_sample.upper_weight);
        let inv_wy = BILINEAR_WEIGHT_SCALE - wy;
        let round = 1 << (BILINEAR_WEIGHT_BITS * 2 - 1);
        for (x, x_sample) in self.x_samples.iter().copied().enumerate() {
            let wx = u32::from(x_sample.upper_weight);
            let inv_wx = BILINEAR_WEIGHT_SCALE - wx;
            let lower = x_sample.lower * CHANNELS;
            let upper = x_sample.upper * CHANNELS;
            let destination = x * CHANNELS;
            for channel in 0..CHANNELS {
                let horizontal_top =
                    u32::from(top[lower + channel]) * inv_wx + u32::from(top[upper + channel]) * wx;
                let horizontal_bottom = u32::from(bottom[lower + channel]) * inv_wx
                    + u32::from(bottom[upper + channel]) * wx;
                output[destination + channel] =
                    ((horizontal_top * inv_wy + horizontal_bottom * wy + round)
                        >> (BILINEAR_WEIGHT_BITS * 2)) as u8;
            }
        }
    }

    fn resize_bilinear_rgb_to_gray_row(
        &self,
        input: ImageView<'_, u8, 3>,
        output: &mut [u8],
        y: usize,
    ) {
        let y_sample = self.y_samples[y];
        let top = input.row(y_sample.lower).expect("planned source row");
        let bottom = input.row(y_sample.upper).expect("planned source row");
        let wy = u32::from(y_sample.upper_weight);
        let inv_wy = BILINEAR_WEIGHT_SCALE - wy;
        for (x, x_sample) in self.x_samples.iter().copied().enumerate() {
            let wx = u32::from(x_sample.upper_weight);
            let inv_wx = BILINEAR_WEIGHT_SCALE - wx;
            let lower = x_sample.lower * 3;
            let upper = x_sample.upper * 3;
            let pixel = std::array::from_fn(|channel| {
                bilinear_u8_component(
                    top[lower + channel],
                    top[upper + channel],
                    bottom[lower + channel],
                    bottom[upper + channel],
                    inv_wx,
                    wx,
                    inv_wy,
                    wy,
                )
            });
            output[x] = rgb_luma_q14(pixel);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn resize_bilinear_rgb_to_chw_row(
        &self,
        input: ImageView<'_, u8, 3>,
        output: &mut [f32],
        y: usize,
        channel: usize,
        scale: f32,
        mean: f32,
        std: f32,
    ) {
        let y_sample = self.y_samples[y];
        let top = input.row(y_sample.lower).expect("planned source row");
        let bottom = input.row(y_sample.upper).expect("planned source row");
        let wy = u32::from(y_sample.upper_weight);
        let inv_wy = BILINEAR_WEIGHT_SCALE - wy;
        for (x, x_sample) in self.x_samples.iter().copied().enumerate() {
            let wx = u32::from(x_sample.upper_weight);
            let inv_wx = BILINEAR_WEIGHT_SCALE - wx;
            let lower = x_sample.lower * 3 + channel;
            let upper = x_sample.upper * 3 + channel;
            let value = bilinear_u8_component(
                top[lower],
                top[upper],
                bottom[lower],
                bottom[upper],
                inv_wx,
                wx,
                inv_wy,
                wy,
            );
            output[x] = (f32::from(value) * scale - mean) / std;
        }
    }
}

fn validate_rgb_chw_normalization(scale: f32, std: [f32; 3]) -> VisionResult<()> {
    if !scale.is_finite() || std.iter().any(|value| !value.is_finite() || *value == 0.0) {
        return Err(VisionError::InvalidParameter(
            "normalization scale/std must be finite and std non-zero".to_owned(),
        ));
    }
    Ok(())
}

#[inline(always)]
fn bilinear_u8_component(
    top_left: u8,
    top_right: u8,
    bottom_left: u8,
    bottom_right: u8,
    inv_wx: u32,
    wx: u32,
    inv_wy: u32,
    wy: u32,
) -> u8 {
    let top = u32::from(top_left) * inv_wx + u32::from(top_right) * wx;
    let bottom = u32::from(bottom_left) * inv_wx + u32::from(bottom_right) * wx;
    let round = 1 << (BILINEAR_WEIGHT_BITS * 2 - 1);
    ((top * inv_wy + bottom * wy + round) >> (BILINEAR_WEIGHT_BITS * 2)) as u8
}

#[inline(always)]
fn rgb_luma_q14(pixel: [u8; 3]) -> u8 {
    ((4_899_u32 * u32::from(pixel[0])
        + 9_617_u32 * u32::from(pixel[1])
        + 1_868_u32 * u32::from(pixel[2])
        + 8_192)
        >> 14) as u8
}

fn bilinear_axis_samples(input_len: usize, output_len: usize) -> Vec<BilinearAxisSample> {
    if input_len == 0 {
        return Vec::new();
    }
    (0..output_len)
        .map(|output| {
            let coordinate = half_pixel_coordinate(output, input_len, output_len);
            let base = coordinate.floor() as isize;
            BilinearAxisSample {
                lower: clamped_index(base, input_len),
                upper: clamped_index(base + 1, input_len),
                upper_weight: ((coordinate - coordinate.floor()) * f64::from(BILINEAR_WEIGHT_SCALE))
                    .round()
                    .clamp(0.0, f64::from(BILINEAR_WEIGHT_SCALE))
                    as u16,
            }
        })
        .collect()
}

fn resize_half_row<const CHANNELS: usize>(
    input: ImageView<'_, u8, CHANNELS>,
    output: &mut [u8],
    y: usize,
    output_width: usize,
) {
    let top = input.row(y * 2).expect("half-scale source row");
    let bottom = input.row(y * 2 + 1).expect("half-scale source row");
    for x in 0..output_width {
        let source = x * 2 * CHANNELS;
        let destination = x * CHANNELS;
        for channel in 0..CHANNELS {
            let sum = u16::from(top[source + channel])
                + u16::from(top[source + CHANNELS + channel])
                + u16::from(bottom[source + channel])
                + u16::from(bottom[source + CHANNELS + channel]);
            output[destination + channel] = ((sum + 2) >> 2) as u8;
        }
    }
}

fn resize_half_rgb_to_gray_row(
    input: ImageView<'_, u8, 3>,
    output: &mut [u8],
    y: usize,
    output_width: usize,
) {
    let top = input.row(y * 2).expect("half-scale source row");
    let bottom = input.row(y * 2 + 1).expect("half-scale source row");
    for (x, target) in output.iter_mut().take(output_width).enumerate() {
        let source = x * 6;
        let pixel = std::array::from_fn(|channel| {
            let sum = u16::from(top[source + channel])
                + u16::from(top[source + 3 + channel])
                + u16::from(bottom[source + channel])
                + u16::from(bottom[source + 3 + channel]);
            ((sum + 2) >> 2) as u8
        });
        *target = rgb_luma_q14(pixel);
    }
}

#[allow(clippy::too_many_arguments)]
fn resize_half_rgb_to_chw_row(
    input: ImageView<'_, u8, 3>,
    output: &mut [f32],
    y: usize,
    channel: usize,
    scale: f32,
    mean: f32,
    std: f32,
) {
    let top = input.row(y * 2).expect("half-scale source row");
    let bottom = input.row(y * 2 + 1).expect("half-scale source row");
    for (x, target) in output.iter_mut().enumerate() {
        let source = x * 6 + channel;
        let sum = u16::from(top[source])
            + u16::from(top[source + 3])
            + u16::from(bottom[source])
            + u16::from(bottom[source + 3]);
        let value = ((sum + 2) >> 2) as u8;
        *target = (f32::from(value) * scale - mean) / std;
    }
}

/// Resizes an interleaved image while preserving semantic metadata.
pub fn resize<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    output_width: usize,
    output_height: usize,
    interpolation: Interpolation,
) -> VisionResult<Image<T, CHANNELS>> {
    if output_width == 0 || output_height == 0 {
        return Image::try_new_with_metadata(
            output_width,
            output_height,
            Vec::new(),
            input.metadata(),
        )
        .map_err(Into::into);
    }
    if input.width() == 0 || input.height() == 0 {
        return Err(VisionError::InvalidDimensions(
            "cannot resize an empty input to a non-empty output".to_owned(),
        ));
    }

    let mut output = Vec::with_capacity(output_width * output_height * CHANNELS);
    let area_downsample = interpolation == Interpolation::Area
        && (output_width < input.width() || output_height < input.height());
    for y in 0..output_height {
        for x in 0..output_width {
            let pixel = resized_pixel(
                input,
                x,
                y,
                output_width,
                output_height,
                interpolation,
                area_downsample,
            );
            output.extend_from_slice(&pixel);
        }
    }
    Ok(Image::try_new_with_metadata(output_width, output_height, output, input.metadata())?)
}

/// Resizes into caller-owned storage without allocating.
///
/// The output dimensions select the requested size. Packed and strided output
/// views are accepted, and semantic metadata is replaced with the input
/// metadata after channel validation.
pub fn resize_into<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    mut output: ImageViewMut<'_, T, CHANNELS>,
    interpolation: Interpolation,
) -> VisionResult<()> {
    let output_width = output.width();
    let output_height = output.height();
    output.set_metadata(input.metadata())?;
    if output_width == 0 || output_height == 0 {
        return Ok(());
    }
    if input.width() == 0 || input.height() == 0 {
        return Err(VisionError::InvalidDimensions(
            "cannot resize an empty input to a non-empty output".to_owned(),
        ));
    }
    let area_downsample = interpolation == Interpolation::Area
        && (output_width < input.width() || output_height < input.height());
    for y in 0..output_height {
        let row = output.row_mut(y).expect("output row in bounds");
        for x in 0..output_width {
            let pixel = resized_pixel(
                input,
                x,
                y,
                output_width,
                output_height,
                interpolation,
                area_downsample,
            );
            row[x * CHANNELS..(x + 1) * CHANNELS].copy_from_slice(&pixel);
        }
    }
    Ok(())
}

fn resized_pixel<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: usize,
    y: usize,
    output_width: usize,
    output_height: usize,
    interpolation: Interpolation,
    area_downsample: bool,
) -> [T; CHANNELS] {
    if area_downsample {
        sample_area(input, x, y, output_width, output_height)
    } else {
        let sx = half_pixel_coordinate(x, input.width(), output_width);
        let sy = half_pixel_coordinate(y, input.height(), output_height);
        match interpolation {
            Interpolation::Nearest => sample_nearest(input, sx, sy),
            Interpolation::Bilinear | Interpolation::Area => sample_bilinear(input, sx, sy),
            Interpolation::Bicubic => sample_bicubic(input, sx, sy),
        }
    }
}

fn half_pixel_coordinate(output: usize, input_len: usize, output_len: usize) -> f64 {
    (output as f64 + 0.5) * input_len as f64 / output_len as f64 - 0.5
}

fn clamped_index(value: isize, length: usize) -> usize {
    value.clamp(0, length.saturating_sub(1) as isize) as usize
}

pub(crate) fn sample_nearest<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: f64,
    y: f64,
) -> [T; CHANNELS] {
    let ix = clamped_index(x.round() as isize, input.width());
    let iy = clamped_index(y.round() as isize, input.height());
    *input.get(ix, iy).expect("clamped resize coordinate")
}

pub(crate) fn sample_bilinear<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: f64,
    y: f64,
) -> [T; CHANNELS] {
    let x0_raw = x.floor() as isize;
    let y0_raw = y.floor() as isize;
    let x1_raw = x0_raw + 1;
    let y1_raw = y0_raw + 1;
    let wx = x - x.floor();
    let wy = y - y.floor();
    let x0 = clamped_index(x0_raw, input.width());
    let x1 = clamped_index(x1_raw, input.width());
    let y0 = clamped_index(y0_raw, input.height());
    let y1 = clamped_index(y1_raw, input.height());
    let p00 = input.get(x0, y0).expect("clamped resize coordinate");
    let p10 = input.get(x1, y0).expect("clamped resize coordinate");
    let p01 = input.get(x0, y1).expect("clamped resize coordinate");
    let p11 = input.get(x1, y1).expect("clamped resize coordinate");
    std::array::from_fn(|channel| {
        let top = p00[channel].to_f64().mul_add(1.0 - wx, p10[channel].to_f64() * wx);
        let bottom = p01[channel].to_f64().mul_add(1.0 - wx, p11[channel].to_f64() * wx);
        T::from_f64(top.mul_add(1.0 - wy, bottom * wy))
    })
}

fn cubic_weight(distance: f64) -> f64 {
    let x = distance.abs();
    const A: f64 = -0.75;
    if x <= 1.0 {
        (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        A * x * x * x - 5.0 * A * x * x + 8.0 * A * x - 4.0 * A
    } else {
        0.0
    }
}

fn sample_bicubic<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: f64,
    y: f64,
) -> [T; CHANNELS] {
    let base_x = x.floor() as isize;
    let base_y = y.floor() as isize;
    let mut sums = [0.0_f64; CHANNELS];
    let mut total_weight = 0.0;
    for dy in -1..=2 {
        let wy = cubic_weight(y - (base_y + dy) as f64);
        let iy = clamped_index(base_y + dy, input.height());
        for dx in -1..=2 {
            let weight = wy * cubic_weight(x - (base_x + dx) as f64);
            let ix = clamped_index(base_x + dx, input.width());
            let pixel = input.get(ix, iy).expect("clamped resize coordinate");
            for channel in 0..CHANNELS {
                sums[channel] += pixel[channel].to_f64() * weight;
            }
            total_weight += weight;
        }
    }
    std::array::from_fn(|channel| T::from_f64(sums[channel] / total_weight))
}

fn sample_area<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    output_x: usize,
    output_y: usize,
    output_width: usize,
    output_height: usize,
) -> [T; CHANNELS] {
    let scale_x = input.width() as f64 / output_width as f64;
    let scale_y = input.height() as f64 / output_height as f64;
    let start_x = output_x as f64 * scale_x;
    let end_x = (output_x + 1) as f64 * scale_x;
    let start_y = output_y as f64 * scale_y;
    let end_y = (output_y + 1) as f64 * scale_y;
    let mut sums = [0.0_f64; CHANNELS];
    let mut total_weight = 0.0;
    for iy in start_y.floor() as usize..end_y.ceil().min(input.height() as f64) as usize {
        let overlap_y = (end_y.min(iy as f64 + 1.0) - start_y.max(iy as f64)).max(0.0);
        for ix in start_x.floor() as usize..end_x.ceil().min(input.width() as f64) as usize {
            let overlap_x = (end_x.min(ix as f64 + 1.0) - start_x.max(ix as f64)).max(0.0);
            let weight = overlap_x * overlap_y;
            let pixel = input.get(ix, iy).expect("area sample within image");
            for channel in 0..CHANNELS {
                sums[channel] += pixel[channel].to_f64() * weight;
            }
            total_weight += weight;
        }
    }
    std::array::from_fn(|channel| T::from_f64(sums[channel] / total_weight))
}

#[cfg(test)]
mod tests {
    use super::{resize, resize_into, BilinearResizeU8Plan, Interpolation};
    use spatialrust_image::{Image, ImageView, ImageViewMut};

    #[test]
    fn nearest_repeats_pixels() {
        let input = Image::<u8, 1>::try_new(2, 1, vec![10, 20]).unwrap();
        let output = resize(input.view(), 4, 1, Interpolation::Nearest).unwrap();
        assert_eq!(output.as_slice(), &[10, 10, 20, 20]);
    }

    #[test]
    fn bilinear_interpolates_center() {
        let input = Image::<f32, 1>::try_new(2, 2, vec![0.0, 10.0, 20.0, 30.0]).unwrap();
        let output = resize(input.view(), 3, 3, Interpolation::Bilinear).unwrap();
        assert!((output[(1, 1)][0] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn area_computes_block_average() {
        let input = Image::<u8, 1>::try_new(2, 2, vec![0, 10, 20, 30]).unwrap();
        let output = resize(input.view(), 1, 1, Interpolation::Area).unwrap();
        assert_eq!(output.as_slice(), &[15]);
    }

    #[test]
    fn identity_resize_is_exact_for_all_filters() {
        let input = Image::<u8, 3>::try_new(2, 2, (0..12).collect()).unwrap();
        for filter in [
            Interpolation::Nearest,
            Interpolation::Bilinear,
            Interpolation::Bicubic,
            Interpolation::Area,
        ] {
            assert_eq!(resize(input.view(), 2, 2, filter).unwrap(), input);
        }
    }

    #[test]
    fn resize_into_reuses_strided_output_and_preserves_padding() {
        let input = Image::<u8, 1>::try_new(2, 2, vec![0, 10, 20, 30]).unwrap();
        let mut storage = vec![99_u8; 15];
        let output = ImageViewMut::<u8, 1>::new(3, 3, 6, &mut storage).unwrap();
        resize_into(input.view(), output, Interpolation::Bilinear).unwrap();
        let expected = resize(input.view(), 3, 3, Interpolation::Bilinear).unwrap();
        for y in 0..3 {
            assert_eq!(&storage[y * 6..y * 6 + 3], expected.view().row(y).unwrap());
            if y < 2 {
                assert_eq!(&storage[y * 6 + 3..(y + 1) * 6], &[99; 3]);
            }
        }
    }

    #[test]
    fn bilinear_u8_plan_matches_generic_with_bounded_rounding() {
        let input =
            Image::<u8, 3>::try_new(7, 5, (0..105).map(|value| (value * 37 % 256) as u8).collect())
                .unwrap();
        let plan = BilinearResizeU8Plan::new(7, 5, 11, 8).unwrap();
        let actual = plan.resize(input.view()).unwrap();
        let expected = resize(input.view(), 11, 8, Interpolation::Bilinear).unwrap();
        assert!(actual
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .all(|(&left, &right)| left.abs_diff(right) <= 1));
    }

    #[test]
    fn bilinear_u8_plan_half_scale_matches_generic_exactly() {
        let input =
            Image::<u8, 3>::try_new(8, 6, (0..144).map(|value| (value * 53 % 256) as u8).collect())
                .unwrap();
        let plan = BilinearResizeU8Plan::new(8, 6, 4, 3).unwrap();
        let actual = plan.resize(input.view()).unwrap();
        let expected = resize(input.view(), 4, 3, Interpolation::Bilinear).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn bilinear_u8_plan_preserves_strided_padding_and_validates_shape() {
        let input = Image::<u8, 1>::try_new(4, 4, (0..16).collect()).unwrap();
        let plan = BilinearResizeU8Plan::new(4, 4, 2, 2).unwrap();
        let mut storage = vec![99_u8; 7];
        let output = ImageViewMut::<u8, 1>::new(2, 2, 5, &mut storage).unwrap();
        plan.resize_into(input.view(), output).unwrap();
        assert_eq!(&storage, &[3, 5, 99, 99, 99, 11, 13]);

        let wrong = BilinearResizeU8Plan::new(5, 4, 2, 2).unwrap();
        let mut output = Image::<u8, 1>::try_new(2, 2, vec![0; 4]).unwrap();
        assert!(wrong.resize_into(input.view(), output.view_mut()).is_err());
    }

    #[test]
    fn bilinear_u8_plan_accepts_strided_rgb_input() {
        let mut storage = vec![231_u8; 54];
        for y in 0..4 {
            for x in 0..12 {
                storage[y * 14 + x] = (y * 41 + x * 13) as u8;
            }
        }
        let input = ImageView::<u8, 3>::new(4, 4, 14, &storage).unwrap();
        let plan = BilinearResizeU8Plan::new(4, 4, 2, 2).unwrap();
        let actual = plan.resize(input).unwrap();
        let expected = resize(input, 2, 2, Interpolation::Bilinear).unwrap();
        assert_eq!(actual, expected);
    }
}
