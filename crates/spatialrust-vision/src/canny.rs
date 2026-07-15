//! Canny edge detection with inspectable CPU intermediates.

use std::collections::VecDeque;

use rayon::prelude::*;
use spatialrust_image::{Image, ImageView, ImageViewMut};

use crate::{
    sobel, spatial_gradient_u8, spatial_gradient_u8_into, BorderMode, VisionError, VisionResult,
};

/// Validated Canny thresholds and gradient settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CannyOptions {
    /// Lower hysteresis threshold.
    pub low_threshold: f64,
    /// Upper strong-edge threshold.
    pub high_threshold: f64,
    /// Sobel aperture size: 3, 5, or 7.
    pub aperture_size: usize,
    /// Use Euclidean gradient magnitude instead of the L1 approximation.
    pub l2_gradient: bool,
}

impl Default for CannyOptions {
    fn default() -> Self {
        Self { low_threshold: 100.0, high_threshold: 200.0, aperture_size: 3, l2_gradient: false }
    }
}

impl CannyOptions {
    fn validate(self) -> VisionResult<Self> {
        if !self.low_threshold.is_finite()
            || !self.high_threshold.is_finite()
            || self.low_threshold < 0.0
            || self.high_threshold < 0.0
        {
            return Err(VisionError::InvalidParameter(
                "Canny thresholds must be finite and non-negative".into(),
            ));
        }
        if !matches!(self.aperture_size, 3 | 5 | 7) {
            return Err(VisionError::InvalidParameter(
                "Canny aperture size must be 3, 5, or 7".into(),
            ));
        }
        Ok(self)
    }
}

/// Edge map and numerical stages produced by Canny.
#[derive(Clone, Debug, PartialEq)]
pub struct CannyResult {
    /// Final binary edge map containing 0 or 255.
    pub edges: Image<u8, 1>,
    /// Horizontal signed Sobel derivative.
    pub gradient_x: Image<f32, 1>,
    /// Vertical signed Sobel derivative.
    pub gradient_y: Image<f32, 1>,
    /// L1 or Euclidean gradient magnitude before suppression.
    pub magnitude: Image<f32, 1>,
    /// Magnitude retained by directional non-maximum suppression.
    pub suppressed: Image<f32, 1>,
}

/// Reusable caller-owned storage for the allocation-light Canny path.
///
/// The workspace grows to the largest image seen and retains that capacity.
/// It contains CPU buffers only and never performs an implicit device copy.
#[derive(Clone, Debug, Default)]
pub struct CannyWorkspace {
    gradient_x: Vec<i16>,
    gradient_y: Vec<i16>,
    comparison_magnitude: Vec<i32>,
    states: Vec<u8>,
    strong: Vec<usize>,
}

impl CannyWorkspace {
    /// Creates an empty workspace that allocates on its first call.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            gradient_x: Vec::new(),
            gradient_y: Vec::new(),
            comparison_magnitude: Vec::new(),
            states: Vec::new(),
            strong: Vec::new(),
        }
    }

    /// Returns the reusable pixel capacity without counting the edge stack.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.gradient_x
            .capacity()
            .min(self.gradient_y.capacity())
            .min(self.comparison_magnitude.capacity())
            .min(self.states.capacity())
    }

    fn resize(&mut self, len: usize) {
        self.gradient_x.resize(len, 0);
        self.gradient_y.resize(len, 0);
        self.comparison_magnitude.resize(len, 0);
        self.states.resize(len, 1);
        self.strong.clear();
    }
}

/// Finds edges in a single-channel u8 image.
pub fn canny(input: ImageView<'_, u8, 1>, options: CannyOptions) -> VisionResult<Image<u8, 1>> {
    let len = checked_len(input.width(), input.height())?;
    let mut output = Image::try_new_with_metadata(
        input.width(),
        input.height(),
        vec![0; len],
        input.metadata(),
    )?;
    let mut workspace = CannyWorkspace::new();
    canny_into(input, options, output.view_mut(), &mut workspace)?;
    Ok(output)
}

/// Finds edges into caller-owned output using reusable CPU workspace.
///
/// The output may be packed or strided, must match the input dimensions, and
/// receives only binary values (`0` or `255`). The specialized allocation-light
/// path applies to the common 3x3 aperture; larger apertures retain the exact
/// inspectable implementation.
pub fn canny_into(
    input: ImageView<'_, u8, 1>,
    options: CannyOptions,
    mut output: ImageViewMut<'_, u8, 1>,
    workspace: &mut CannyWorkspace,
) -> VisionResult<()> {
    let options = options.validate()?;
    if output.width() != input.width() || output.height() != input.height() {
        return Err(VisionError::ShapeMismatch(format!(
            "Canny output must be {}x{}, found {}x{}",
            input.width(),
            input.height(),
            output.width(),
            output.height()
        )));
    }
    if options.aperture_size != 3 {
        let result = canny_with_intermediates(input, options)?;
        for y in 0..input.height() {
            let start = y * input.width();
            output
                .row_mut(y)
                .expect("validated Canny output row")
                .copy_from_slice(&result.edges.as_slice()[start..start + input.width()]);
        }
        return Ok(());
    }

    canny_3x3_into(input, options, &mut output, workspace)
}

fn checked_len(width: usize, height: usize) -> VisionResult<usize> {
    width
        .checked_mul(height)
        .ok_or_else(|| VisionError::InvalidDimensions("Canny image dimensions overflow".into()))
}

fn canny_3x3_into(
    input: ImageView<'_, u8, 1>,
    options: CannyOptions,
    output: &mut ImageViewMut<'_, u8, 1>,
    workspace: &mut CannyWorkspace,
) -> VisionResult<()> {
    let width = input.width();
    let height = input.height();
    let len = checked_len(width, height)?;
    workspace.resize(len);
    spatial_gradient_u8_into(
        input,
        BorderMode::Replicate,
        &mut workspace.gradient_x,
        &mut workspace.gradient_y,
    )?;

    if options.l2_gradient {
        if len >= 1_000_000 {
            workspace.comparison_magnitude.par_iter_mut().enumerate().for_each(
                |(index, magnitude)| {
                    let x = i32::from(workspace.gradient_x[index]);
                    let y = i32::from(workspace.gradient_y[index]);
                    *magnitude = x * x + y * y;
                },
            );
        } else {
            for ((magnitude, &x), &y) in workspace
                .comparison_magnitude
                .iter_mut()
                .zip(&workspace.gradient_x)
                .zip(&workspace.gradient_y)
            {
                let x = i32::from(x);
                let y = i32::from(y);
                *magnitude = x * x + y * y;
            }
        }
    } else {
        for ((magnitude, &x), &y) in workspace
            .comparison_magnitude
            .iter_mut()
            .zip(&workspace.gradient_x)
            .zip(&workspace.gradient_y)
        {
            *magnitude = i32::from(x).abs() + i32::from(y).abs();
        }
    }

    let (low, high) = canny_thresholds(options);
    workspace.states.fill(1);
    if len >= 1_000_000 {
        let gradient_x = &workspace.gradient_x;
        let gradient_y = &workspace.gradient_y;
        let magnitudes = &workspace.comparison_magnitude;
        let strong = workspace
            .states
            .par_iter_mut()
            .enumerate()
            .fold(Vec::new, |mut strong, (index, state)| {
                let magnitude = magnitudes[index];
                if i64::from(magnitude) > low
                    && is_directional_maximum_i32(
                        index % width,
                        index / width,
                        width,
                        height,
                        i32::from(gradient_x[index]),
                        i32::from(gradient_y[index]),
                        magnitude,
                        magnitudes,
                    )
                {
                    if i64::from(magnitude) > high {
                        *state = 2;
                        strong.push(index);
                    } else {
                        *state = 0;
                    }
                }
                strong
            })
            .reduce(Vec::new, |mut left, mut right| {
                left.append(&mut right);
                left
            });
        workspace.strong = strong;
    } else {
        for y in 0..height {
            let row = y * width;
            for x in 0..width {
                let index = row + x;
                let magnitude = workspace.comparison_magnitude[index];
                if i64::from(magnitude) <= low
                    || !is_directional_maximum_i32(
                        x,
                        y,
                        width,
                        height,
                        i32::from(workspace.gradient_x[index]),
                        i32::from(workspace.gradient_y[index]),
                        magnitude,
                        &workspace.comparison_magnitude,
                    )
                {
                    continue;
                }
                if i64::from(magnitude) > high {
                    workspace.states[index] = 2;
                    workspace.strong.push(index);
                } else {
                    workspace.states[index] = 0;
                }
            }
        }
    }

    while let Some(index) = workspace.strong.pop() {
        let x = index % width;
        let y = index / width;
        let x0 = x.saturating_sub(1);
        let x1 = (x + 1).min(width.saturating_sub(1));
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(height.saturating_sub(1));
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                let neighbor = ny * width + nx;
                if workspace.states[neighbor] == 0 {
                    workspace.states[neighbor] = 2;
                    workspace.strong.push(neighbor);
                }
            }
        }
    }

    if len >= 1_000_000 && output.row_stride() == width {
        output
            .as_mut_slice()
            .par_iter_mut()
            .zip(&workspace.states)
            .for_each(|(pixel, &state)| *pixel = if state == 2 { 255 } else { 0 });
    } else {
        for y in 0..height {
            let start = y * width;
            let target = output.row_mut(y).expect("validated Canny output row");
            for (pixel, &state) in target.iter_mut().zip(&workspace.states[start..start + width]) {
                *pixel = if state == 2 { 255 } else { 0 };
            }
        }
    }
    Ok(())
}

fn canny_thresholds(options: CannyOptions) -> (i64, i64) {
    let (mut low, mut high) = (options.low_threshold, options.high_threshold);
    if options.aperture_size == 7 {
        low /= 16.0;
        high /= 16.0;
    }
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }
    if options.l2_gradient {
        (
            low.min(32767.0).mul_add(low.min(32767.0), 0.0).floor() as i64,
            high.min(32767.0).mul_add(high.min(32767.0), 0.0).floor() as i64,
        )
    } else {
        (low.floor() as i64, high.floor() as i64)
    }
}

/// Runs Canny and retains gradient, magnitude, and suppression stages.
pub fn canny_with_intermediates(
    input: ImageView<'_, u8, 1>,
    options: CannyOptions,
) -> VisionResult<CannyResult> {
    let options = options.validate()?;
    let width = input.width();
    let height = input.height();
    let len = width
        .checked_mul(height)
        .ok_or_else(|| VisionError::InvalidDimensions("Canny image dimensions overflow".into()))?;
    let (gradient_x, gradient_y, gx, gy) = if options.aperture_size == 3 {
        let (gradient_x_i16, gradient_y_i16) = spatial_gradient_u8(input, BorderMode::Replicate)?;
        let gx = gradient_x_i16.as_slice().iter().map(|&value| i32::from(value)).collect();
        let gy = gradient_y_i16.as_slice().iter().map(|&value| i32::from(value)).collect();
        let gradient_x = gradient_x_i16.as_slice().iter().map(|&value| f32::from(value)).collect();
        let gradient_y = gradient_y_i16.as_slice().iter().map(|&value| f32::from(value)).collect();
        (
            Image::try_new_with_metadata(width, height, gradient_x, input.metadata())?,
            Image::try_new_with_metadata(width, height, gradient_y, input.metadata())?,
            gx,
            gy,
        )
    } else {
        let scale = if options.aperture_size == 7 { 1.0 / 16.0 } else { 1.0 };
        let gradient_x =
            sobel(input, 1, 0, options.aperture_size, scale, 0.0, BorderMode::Replicate)?;
        let gradient_y =
            sobel(input, 0, 1, options.aperture_size, scale, 0.0, BorderMode::Replicate)?;
        let gx = gradient_x
            .as_slice()
            .iter()
            // OpenCV's CV_16S Sobel path uses cvRound, whose supported CPU paths
            // round half-way values to even. This matters for aperture 7's 1/16 scale.
            .map(|&value| round_i16_ties_even(value))
            .collect::<Vec<_>>();
        let gy = gradient_y
            .as_slice()
            .iter()
            .map(|&value| round_i16_ties_even(value))
            .collect::<Vec<_>>();
        (gradient_x, gradient_y, gx, gy)
    };
    let comparison_magnitude = gx
        .iter()
        .zip(&gy)
        .map(|(&x, &y)| {
            if options.l2_gradient {
                i64::from(x) * i64::from(x) + i64::from(y) * i64::from(y)
            } else {
                i64::from(x.abs() + y.abs())
            }
        })
        .collect::<Vec<_>>();
    let magnitude_values = comparison_magnitude
        .iter()
        .map(|&value| if options.l2_gradient { (value as f64).sqrt() as f32 } else { value as f32 })
        .collect::<Vec<_>>();

    let (mut low, mut high) = (options.low_threshold, options.high_threshold);
    if options.aperture_size == 7 {
        low /= 16.0;
        high /= 16.0;
    }
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }
    let (low, high) = if options.l2_gradient {
        (
            low.min(32767.0).mul_add(low.min(32767.0), 0.0).floor() as i64,
            high.min(32767.0).mul_add(high.min(32767.0), 0.0).floor() as i64,
        )
    } else {
        (low.floor() as i64, high.floor() as i64)
    };

    let mut states = vec![1_u8; len];
    let mut suppressed_values = vec![0.0_f32; len];
    let mut strong = VecDeque::new();
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let magnitude = comparison_magnitude[index];
            if magnitude <= low
                || !is_directional_maximum(
                    x,
                    y,
                    width,
                    height,
                    gx[index],
                    gy[index],
                    magnitude,
                    &comparison_magnitude,
                )
            {
                continue;
            }
            suppressed_values[index] = magnitude_values[index];
            if magnitude > high {
                states[index] = 2;
                strong.push_back(index);
            } else {
                states[index] = 0;
            }
        }
    }

    while let Some(index) = strong.pop_front() {
        let x = index % width;
        let y = index / width;
        for dy in -1_isize..=1 {
            for dx in -1_isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    continue;
                }
                let neighbor = ny as usize * width + nx as usize;
                if states[neighbor] == 0 {
                    states[neighbor] = 2;
                    strong.push_back(neighbor);
                }
            }
        }
    }
    let edges = states.into_iter().map(|state| if state == 2 { 255 } else { 0 }).collect();
    let metadata = input.metadata();
    Ok(CannyResult {
        edges: Image::try_new_with_metadata(width, height, edges, metadata)?,
        gradient_x,
        gradient_y,
        magnitude: Image::try_new_with_metadata(width, height, magnitude_values, metadata)?,
        suppressed: Image::try_new_with_metadata(width, height, suppressed_values, metadata)?,
    })
}

fn round_i16_ties_even(value: f32) -> i32 {
    let value = value.clamp(f32::from(i16::MIN), f32::from(i16::MAX));
    let lower = value.floor();
    let fraction = value - lower;
    if fraction < 0.5 {
        lower as i32
    } else if fraction > 0.5 {
        lower as i32 + 1
    } else {
        let lower = lower as i32;
        if lower & 1 == 0 {
            lower
        } else {
            lower + 1
        }
    }
}

fn is_directional_maximum(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    gradient_x: i32,
    gradient_y: i32,
    magnitude: i64,
    magnitudes: &[i64],
) -> bool {
    const TG22: i64 = 13_573;
    let abs_x = i64::from(gradient_x.abs());
    let abs_y_scaled = i64::from(gradient_y.abs()) << 15;
    let tg22_x = abs_x * TG22;
    let get = |offset_x: isize, offset_y: isize| {
        let nx = x as isize + offset_x;
        let ny = y as isize + offset_y;
        if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
            0
        } else {
            magnitudes[ny as usize * width + nx as usize]
        }
    };
    if abs_y_scaled < tg22_x {
        magnitude > get(-1, 0) && magnitude >= get(1, 0)
    } else if abs_y_scaled > tg22_x + (abs_x << 16) {
        magnitude > get(0, -1) && magnitude >= get(0, 1)
    } else {
        let sign = if (gradient_x ^ gradient_y) < 0 { -1 } else { 1 };
        magnitude > get(-sign, -1) && magnitude > get(sign, 1)
    }
}

#[inline]
fn is_directional_maximum_i32(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    gradient_x: i32,
    gradient_y: i32,
    magnitude: i32,
    magnitudes: &[i32],
) -> bool {
    const TG22: i64 = 13_573;
    let abs_x = i64::from(gradient_x.abs());
    let abs_y_scaled = i64::from(gradient_y.abs()) << 15;
    let tg22_x = abs_x * TG22;
    let get = |offset_x: isize, offset_y: isize| {
        let nx = x as isize + offset_x;
        let ny = y as isize + offset_y;
        if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
            0
        } else {
            magnitudes[ny as usize * width + nx as usize]
        }
    };
    if abs_y_scaled < tg22_x {
        magnitude > get(-1, 0) && magnitude >= get(1, 0)
    } else if abs_y_scaled > tg22_x + (abs_x << 16) {
        magnitude > get(0, -1) && magnitude >= get(0, 1)
    } else {
        let sign = if (gradient_x ^ gradient_y) < 0 { -1 } else { 1 };
        magnitude > get(-sign, -1) && magnitude > get(sign, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::{canny, canny_into, canny_with_intermediates, CannyOptions, CannyWorkspace};
    use spatialrust_image::{Image, ImageRegion, ImageViewMut};

    #[test]
    fn finds_both_sides_of_bright_bar_on_strided_roi() {
        let mut data = vec![0_u8; 9 * 7];
        for y in 1..6 {
            for x in 3..6 {
                data[y * 9 + x] = 255;
            }
        }
        let image = Image::<u8, 1>::try_new(9, 7, data).unwrap();
        let roi = image.view().subview(ImageRegion::new(1, 1, 7, 5)).unwrap();
        let edges = canny(
            roi,
            CannyOptions { low_threshold: 50.0, high_threshold: 100.0, ..Default::default() },
        )
        .unwrap();
        assert!(edges.as_slice().iter().filter(|&&value| value == 255).count() >= 6);
        assert!(edges.as_slice().iter().all(|&value| value == 0 || value == 255));
    }

    #[test]
    fn intermediates_preserve_shape_and_signed_gradients() {
        let image =
            Image::<u8, 1>::try_new(5, 3, (0..3).flat_map(|_| [0, 10, 20, 30, 40]).collect())
                .unwrap();
        let result = canny_with_intermediates(image.view(), CannyOptions::default()).unwrap();
        assert_eq!((result.edges.width(), result.edges.height()), (5, 3));
        assert!(result.gradient_x.as_slice().iter().any(|&value| value > 0.0));
        assert!(result.gradient_y.as_slice().iter().all(|&value| value == 0.0));
    }

    #[test]
    fn flat_and_empty_images_have_no_edges() {
        for image in [
            Image::<u8, 1>::from_pixel(4, 3, [42]).unwrap(),
            Image::<u8, 1>::try_new(0, 0, Vec::new()).unwrap(),
        ] {
            assert!(canny(image.view(), CannyOptions::default())
                .unwrap()
                .as_slice()
                .iter()
                .all(|&value| value == 0));
        }
    }

    #[test]
    fn validates_thresholds_and_aperture() {
        let image = Image::<u8, 1>::from_pixel(1, 1, [0]).unwrap();
        assert!(
            canny(image.view(), CannyOptions { aperture_size: 4, ..Default::default() }).is_err()
        );
        assert!(canny(image.view(), CannyOptions { low_threshold: -1.0, ..Default::default() })
            .is_err());
    }

    #[test]
    fn fast_path_matches_inspectable_path_for_l1_and_l2() {
        let data = (0..31 * 19).map(|index| ((index * 37 + index / 31 * 17) & 255) as u8).collect();
        let image = Image::<u8, 1>::try_new(31, 19, data).unwrap();
        for l2_gradient in [false, true] {
            let options = CannyOptions {
                low_threshold: 80.0,
                high_threshold: 160.0,
                l2_gradient,
                ..Default::default()
            };
            assert_eq!(
                canny(image.view(), options).unwrap(),
                canny_with_intermediates(image.view(), options).unwrap().edges
            );
        }
    }

    #[test]
    fn reusable_path_supports_strided_output_without_touching_padding() {
        let image = Image::<u8, 1>::try_new(
            7,
            5,
            (0..35).map(|index| ((index * 53) & 255) as u8).collect(),
        )
        .unwrap();
        let expected = canny(image.view(), CannyOptions::default()).unwrap();
        let mut storage = vec![17_u8; 5 * 11];
        let output = ImageViewMut::<u8, 1>::new(7, 5, 11, &mut storage).unwrap();
        let mut workspace = CannyWorkspace::new();
        canny_into(image.view(), CannyOptions::default(), output, &mut workspace).unwrap();
        for y in 0..5 {
            assert_eq!(&storage[y * 11..y * 11 + 7], &expected.as_slice()[y * 7..y * 7 + 7]);
            if y < 4 {
                assert_eq!(&storage[y * 11 + 7..(y + 1) * 11], &[17; 4]);
            }
        }
    }
}
