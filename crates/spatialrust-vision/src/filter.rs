//! Linear CPU image filters with explicit border and kernel contracts.

use pulp::Arch;
use rayon::prelude::*;
use spatialrust_image::{Image, ImageView};

use crate::border::{fetch, map_index};
use crate::{BorderMode, PixelComponent, VisionError, VisionResult};

/// Validated two-dimensional correlation kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel2D {
    width: usize,
    height: usize,
    anchor_x: usize,
    anchor_y: usize,
    coefficients: Vec<f64>,
}

impl Kernel2D {
    /// Creates a kernel with its anchor at `(width / 2, height / 2)`.
    pub fn try_new(width: usize, height: usize, coefficients: Vec<f64>) -> VisionResult<Self> {
        Self::try_new_with_anchor(width, height, width / 2, height / 2, coefficients)
    }

    /// Creates a kernel with an explicit in-kernel anchor.
    pub fn try_new_with_anchor(
        width: usize,
        height: usize,
        anchor_x: usize,
        anchor_y: usize,
        coefficients: Vec<f64>,
    ) -> VisionResult<Self> {
        if width == 0 || height == 0 {
            return Err(VisionError::InvalidParameter("kernel dimensions must be non-zero".into()));
        }
        let expected = width
            .checked_mul(height)
            .ok_or_else(|| VisionError::InvalidParameter("kernel dimensions overflow".into()))?;
        if coefficients.len() != expected {
            return Err(VisionError::ShapeMismatch(format!(
                "kernel needs {expected} coefficients, found {}",
                coefficients.len()
            )));
        }
        if anchor_x >= width || anchor_y >= height {
            return Err(VisionError::InvalidParameter(format!(
                "kernel anchor ({anchor_x}, {anchor_y}) is outside {width}x{height}"
            )));
        }
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Err(VisionError::InvalidParameter("kernel coefficients must be finite".into()));
        }
        Ok(Self { width, height, anchor_x, anchor_y, coefficients })
    }

    /// Kernel width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Kernel height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Kernel anchor `(x, y)`.
    #[must_use]
    pub const fn anchor(&self) -> (usize, usize) {
        (self.anchor_x, self.anchor_y)
    }

    /// Row-major coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Returns a kernel reversed in both axes for true convolution.
    #[must_use]
    pub fn reversed(&self) -> Self {
        let mut coefficients = self.coefficients.clone();
        coefficients.reverse();
        Self {
            width: self.width,
            height: self.height,
            anchor_x: self.width - 1 - self.anchor_x,
            anchor_y: self.height - 1 - self.anchor_y,
            coefficients,
        }
    }
}

/// Validated one-dimensional correlation kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel1D {
    anchor: usize,
    coefficients: Vec<f64>,
}

impl Kernel1D {
    /// Creates a kernel anchored at `coefficients.len() / 2`.
    pub fn try_new(coefficients: Vec<f64>) -> VisionResult<Self> {
        let anchor = coefficients.len() / 2;
        Self::try_new_with_anchor(coefficients, anchor)
    }

    /// Creates a one-dimensional kernel with an explicit anchor.
    pub fn try_new_with_anchor(coefficients: Vec<f64>, anchor: usize) -> VisionResult<Self> {
        if coefficients.is_empty() {
            return Err(VisionError::InvalidParameter("kernel must not be empty".into()));
        }
        if anchor >= coefficients.len() {
            return Err(VisionError::InvalidParameter(format!(
                "kernel anchor {anchor} is outside length {}",
                coefficients.len()
            )));
        }
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Err(VisionError::InvalidParameter("kernel coefficients must be finite".into()));
        }
        Ok(Self { anchor, coefficients })
    }

    /// Kernel length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Returns whether the kernel is empty. Valid kernels always return false.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Anchor index.
    #[must_use]
    pub const fn anchor(&self) -> usize {
        self.anchor
    }

    /// Kernel coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

/// Applies OpenCV-style correlation and converts the result back to the input dtype.
pub fn filter2d<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel: &Kernel2D,
    delta: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    validate_delta(delta)?;
    let accumulators = correlate(input, kernel, delta, border);
    let output = accumulators.into_iter().map(T::from_f64).collect();
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Applies OpenCV-style correlation and preserves signed/fractional results as `f32`.
pub fn filter2d_f32<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel: &Kernel2D,
    delta: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<f32, CHANNELS>> {
    validate_delta(delta)?;
    let output =
        correlate(input, kernel, delta, border).into_iter().map(|value| value as f32).collect();
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Applies true convolution by reversing the supplied kernel around its anchor.
pub fn convolve2d<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel: &Kernel2D,
    delta: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    filter2d(input, &kernel.reversed(), delta, border)
}

/// Applies horizontal and vertical kernels using an `f64` intermediate buffer.
pub fn separable_filter<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel_x: &Kernel1D,
    kernel_y: &Kernel1D,
    delta: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    let output = separable_accumulators(input, kernel_x, kernel_y, delta, border)?
        .into_iter()
        .map(T::from_f64)
        .collect();
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Applies separable kernels and preserves signed/fractional output as `f32`.
pub fn separable_filter_f32<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel_x: &Kernel1D,
    kernel_y: &Kernel1D,
    delta: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<f32, CHANNELS>> {
    let output = separable_accumulators(input, kernel_x, kernel_y, delta, border)?
        .into_iter()
        .map(|value| value as f32)
        .collect();
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Applies a normalized rectangular box filter.
pub fn box_blur<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel_width: usize,
    kernel_height: usize,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    if kernel_width == 0 || kernel_height == 0 {
        return Err(VisionError::InvalidParameter("box kernel dimensions must be non-zero".into()));
    }
    let x = Kernel1D::try_new(vec![1.0 / kernel_width as f64; kernel_width])?;
    let y = Kernel1D::try_new(vec![1.0 / kernel_height as f64; kernel_height])?;
    separable_filter(input, &x, &y, 0.0, border)
}

/// Applies a separable Gaussian blur with explicit odd sizes and standard deviations.
pub fn gaussian_blur<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel_width: usize,
    kernel_height: usize,
    sigma_x: f64,
    sigma_y: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    let x = gaussian_kernel(kernel_width, sigma_x)?;
    let y = gaussian_kernel(kernel_height, sigma_y)?;
    separable_filter(input, &x, &y, 0.0, border)
}

/// Reusable scratch storage and fixed-point kernel cache for [`gaussian_blur_u8_into`].
///
/// The workspace makes the intermediate allocation and kernel construction explicit. A
/// workspace may be reused across calls, but must not be shared by concurrent operations.
#[derive(Clone, Debug, Default)]
pub struct GaussianBlurU8Workspace {
    horizontal: Vec<u16>,
    high_precision_horizontal: Vec<u32>,
    kernel_x: Vec<u16>,
    kernel_y: Vec<u16>,
    kernel_x_key: Option<(usize, u64, u8)>,
    kernel_y_key: Option<(usize, u64, u8)>,
}

impl GaussianBlurU8Workspace {
    /// Creates an empty workspace that grows on first use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            horizontal: Vec::new(),
            high_precision_horizontal: Vec::new(),
            kernel_x: Vec::new(),
            kernel_y: Vec::new(),
            kernel_x_key: None,
            kernel_y_key: None,
        }
    }

    /// Current intermediate-buffer capacity in scalar channel elements.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.horizontal.capacity().max(self.high_precision_horizontal.capacity())
    }
}

/// Applies a specialized 3×3, 5×5, or 7×7 Gaussian blur to interleaved `u8` input.
///
/// The 3×3 and 5×5 paths use normalized Q8 kernels and a `u16` horizontal intermediate. Calls
/// with a 7×7 axis use Q15 kernels, a `u32` horizontal intermediate, and a rounded `u64` final
/// accumulation. OpenCV comparison is gated to a maximum difference of two `u8` levels.
pub fn gaussian_blur_u8<const CHANNELS: usize>(
    input: ImageView<'_, u8, CHANNELS>,
    kernel_width: usize,
    kernel_height: usize,
    sigma_x: f64,
    sigma_y: f64,
    border: BorderMode<u8, CHANNELS>,
) -> VisionResult<Image<u8, CHANNELS>> {
    validate_specialized_gaussian_size(kernel_width)?;
    validate_specialized_gaussian_size(kernel_height)?;
    validate_gaussian_sigma(sigma_x)?;
    validate_gaussian_sigma(sigma_y)?;
    let len = gaussian_output_len::<CHANNELS>(input.width(), input.height())?;
    let mut output = vec![0; len];
    let mut workspace = GaussianBlurU8Workspace::new();
    gaussian_blur_u8_into(
        input,
        kernel_width,
        kernel_height,
        sigma_x,
        sigma_y,
        border,
        &mut output,
        &mut workspace,
    )?;
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Applies the specialized `u8` Gaussian blur into caller-owned packed output and workspace.
#[allow(clippy::too_many_arguments)]
pub fn gaussian_blur_u8_into<const CHANNELS: usize>(
    input: ImageView<'_, u8, CHANNELS>,
    kernel_width: usize,
    kernel_height: usize,
    sigma_x: f64,
    sigma_y: f64,
    border: BorderMode<u8, CHANNELS>,
    output: &mut [u8],
    workspace: &mut GaussianBlurU8Workspace,
) -> VisionResult<()> {
    let len = gaussian_output_len::<CHANNELS>(input.width(), input.height())?;
    if output.len() != len {
        return Err(VisionError::ShapeMismatch(format!(
            "Gaussian output needs {len} elements, found {}",
            output.len()
        )));
    }
    validate_specialized_gaussian_size(kernel_width)?;
    validate_specialized_gaussian_size(kernel_height)?;
    validate_gaussian_sigma(sigma_x)?;
    validate_gaussian_sigma(sigma_y)?;
    if len == 0 {
        return Ok(());
    }
    if kernel_width == 7 || kernel_height == 7 {
        return gaussian_blur_u8_high_precision_into(
            input,
            kernel_width,
            kernel_height,
            sigma_x,
            sigma_y,
            border,
            output,
            workspace,
        );
    }

    prepare_fixed_gaussian_kernel(
        kernel_width,
        sigma_x,
        &mut workspace.kernel_x_key,
        &mut workspace.kernel_x,
    );
    prepare_fixed_gaussian_kernel(
        kernel_height,
        sigma_y,
        &mut workspace.kernel_y_key,
        &mut workspace.kernel_y,
    );
    workspace.horizontal.resize(len, 0);
    let row_len = input.width() * CHANNELS;
    let height = input.height();
    let horizontal = &mut workspace.horizontal;
    let kernel_x = workspace.kernel_x.as_slice();
    let arch = Arch::new();
    if len >= 1_000_000 {
        let workers = rayon::current_num_threads().min(height.max(1));
        let rows_per_worker = height.div_ceil(workers);
        horizontal.par_chunks_mut(rows_per_worker * row_len).enumerate().for_each(
            |(chunk, rows)| {
                let start_y = chunk * rows_per_worker;
                for (offset, row) in rows.chunks_mut(row_len).enumerate() {
                    arch.dispatch(|| {
                        gaussian_horizontal_row(input, start_y + offset, kernel_x, border, row);
                    });
                }
            },
        );
    } else {
        for (y, row) in horizontal.chunks_mut(row_len).enumerate() {
            arch.dispatch(|| gaussian_horizontal_row(input, y, kernel_x, border, row));
        }
    }

    let kernel_y = workspace.kernel_y.as_slice();
    if len >= 1_000_000 {
        let workers = rayon::current_num_threads().min(height.max(1));
        let rows_per_worker = height.div_ceil(workers);
        output.par_chunks_mut(rows_per_worker * row_len).enumerate().for_each(|(chunk, rows)| {
            let start_y = chunk * rows_per_worker;
            for (offset, row) in rows.chunks_mut(row_len).enumerate() {
                arch.dispatch(|| {
                    gaussian_vertical_row(
                        horizontal,
                        input.width(),
                        height,
                        start_y + offset,
                        kernel_y,
                        border,
                        row,
                    );
                });
            }
        });
    } else {
        for (y, row) in output.chunks_mut(row_len).enumerate() {
            arch.dispatch(|| {
                gaussian_vertical_row(
                    horizontal,
                    input.width(),
                    input.height(),
                    y,
                    kernel_y,
                    border,
                    row,
                );
            });
        }
    }
    Ok(())
}

fn gaussian_output_len<const CHANNELS: usize>(width: usize, height: usize) -> VisionResult<usize> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(CHANNELS))
        .ok_or_else(|| VisionError::InvalidDimensions("Gaussian output size overflows".into()))
}

fn validate_specialized_gaussian_size(size: usize) -> VisionResult<()> {
    if !matches!(size, 3 | 5 | 7) {
        return Err(VisionError::InvalidParameter(
            "specialized Gaussian kernel size must be 3, 5, or 7".into(),
        ));
    }
    Ok(())
}

fn validate_gaussian_sigma(sigma: f64) -> VisionResult<()> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "Gaussian sigma must be finite and positive".into(),
        ));
    }
    Ok(())
}

const HIGH_PRECISION_GAUSSIAN_BITS: u8 = 15;
const HIGH_PRECISION_GAUSSIAN_SCALE: u32 = 1 << HIGH_PRECISION_GAUSSIAN_BITS;

#[allow(clippy::too_many_arguments)]
fn gaussian_blur_u8_high_precision_into<const CHANNELS: usize>(
    input: ImageView<'_, u8, CHANNELS>,
    kernel_width: usize,
    kernel_height: usize,
    sigma_x: f64,
    sigma_y: f64,
    border: BorderMode<u8, CHANNELS>,
    output: &mut [u8],
    workspace: &mut GaussianBlurU8Workspace,
) -> VisionResult<()> {
    prepare_high_precision_gaussian_kernel(
        kernel_width,
        sigma_x,
        &mut workspace.kernel_x_key,
        &mut workspace.kernel_x,
    );
    prepare_high_precision_gaussian_kernel(
        kernel_height,
        sigma_y,
        &mut workspace.kernel_y_key,
        &mut workspace.kernel_y,
    );

    let row_len = input.width() * CHANNELS;
    let height = input.height();
    workspace.high_precision_horizontal.resize(output.len(), 0);
    let horizontal = &mut workspace.high_precision_horizontal;
    let kernel_x = workspace.kernel_x.as_slice();
    if output.len() >= 1_000_000 {
        let workers = rayon::current_num_threads().min(height.max(1));
        let rows_per_worker = height.div_ceil(workers);
        horizontal.par_chunks_mut(rows_per_worker * row_len).enumerate().for_each(
            |(chunk, rows)| {
                let start_y = chunk * rows_per_worker;
                for (offset, row) in rows.chunks_mut(row_len).enumerate() {
                    gaussian_horizontal_high_precision_row(
                        input,
                        start_y + offset,
                        kernel_x,
                        border,
                        row,
                    );
                }
            },
        );
    } else {
        for (y, row) in horizontal.chunks_mut(row_len).enumerate() {
            gaussian_horizontal_high_precision_row(input, y, kernel_x, border, row);
        }
    }

    let kernel_y = workspace.kernel_y.as_slice();
    if output.len() >= 1_000_000 {
        let workers = rayon::current_num_threads().min(height.max(1));
        let rows_per_worker = height.div_ceil(workers);
        output.par_chunks_mut(rows_per_worker * row_len).enumerate().for_each(|(chunk, rows)| {
            let start_y = chunk * rows_per_worker;
            for (offset, row) in rows.chunks_mut(row_len).enumerate() {
                gaussian_vertical_high_precision_row(
                    horizontal,
                    input.width(),
                    height,
                    start_y + offset,
                    kernel_y,
                    border,
                    row,
                );
            }
        });
    } else {
        for (y, row) in output.chunks_mut(row_len).enumerate() {
            gaussian_vertical_high_precision_row(
                horizontal,
                input.width(),
                height,
                y,
                kernel_y,
                border,
                row,
            );
        }
    }
    Ok(())
}

fn prepare_high_precision_gaussian_kernel(
    size: usize,
    sigma: f64,
    cached_key: &mut Option<(usize, u64, u8)>,
    cached_kernel: &mut Vec<u16>,
) {
    let key = (size, sigma.to_bits(), HIGH_PRECISION_GAUSSIAN_BITS);
    if *cached_key == Some(key) {
        return;
    }
    let center = (size / 2) as f64;
    let denominator = 2.0 * sigma * sigma;
    let mut fixed = (0..size)
        .map(|index| {
            let offset = index as f64 - center;
            (-(offset * offset) / denominator).exp()
        })
        .collect::<Vec<_>>();
    let sum = fixed.iter().sum::<f64>();
    let mut fixed = fixed
        .drain(..)
        .map(|weight| (weight * f64::from(HIGH_PRECISION_GAUSSIAN_SCALE) / sum).round() as i32)
        .collect::<Vec<_>>();
    let adjustment = HIGH_PRECISION_GAUSSIAN_SCALE as i32 - fixed.iter().sum::<i32>();
    fixed[size / 2] += adjustment;
    cached_kernel.clear();
    cached_kernel.extend(fixed.into_iter().map(|weight| weight as u16));
    *cached_key = Some(key);
}

fn gaussian_horizontal_high_precision_row<const CHANNELS: usize>(
    input: ImageView<'_, u8, CHANNELS>,
    y: usize,
    kernel: &[u16],
    border: BorderMode<u8, CHANNELS>,
    output: &mut [u32],
) {
    let width = input.width();
    let radius = kernel.len() / 2;
    let source = input.row(y).expect("Gaussian source row in bounds");
    let constant = match border {
        BorderMode::Constant(pixel) => pixel,
        _ => [0; CHANNELS],
    };
    for x in 0..width {
        let interior = x >= radius && x + radius < width;
        for channel in 0..CHANNELS {
            let mut sum = 0_u32;
            for (tap, &weight) in kernel.iter().enumerate() {
                let value = if interior {
                    source[(x + tap - radius) * CHANNELS + channel]
                } else {
                    let source_x = x as isize + tap as isize - radius as isize;
                    map_index(source_x, width, border)
                        .map_or(constant[channel], |mapped| source[mapped * CHANNELS + channel])
                };
                sum += u32::from(value) * u32::from(weight);
            }
            output[x * CHANNELS + channel] = sum;
        }
    }
}

fn gaussian_vertical_high_precision_row<const CHANNELS: usize>(
    horizontal: &[u32],
    width: usize,
    height: usize,
    y: usize,
    kernel: &[u16],
    border: BorderMode<u8, CHANNELS>,
    output: &mut [u8],
) {
    let row_len = width * CHANNELS;
    let radius = kernel.len() / 2;
    let interior = y >= radius && y + radius < height;
    let constant = match border {
        BorderMode::Constant(pixel) => {
            pixel.map(|value| u32::from(value) * HIGH_PRECISION_GAUSSIAN_SCALE)
        }
        _ => [0; CHANNELS],
    };
    let round = 1_u64 << (u32::from(HIGH_PRECISION_GAUSSIAN_BITS) * 2 - 1);
    let shift = u32::from(HIGH_PRECISION_GAUSSIAN_BITS) * 2;
    for scalar_x in 0..row_len {
        let channel = scalar_x % CHANNELS;
        let mut sum = 0_u64;
        for (tap, &weight) in kernel.iter().enumerate() {
            let value = if interior {
                horizontal[(y + tap - radius) * row_len + scalar_x]
            } else {
                let source_y = y as isize + tap as isize - radius as isize;
                map_index(source_y, height, border)
                    .map_or(constant[channel], |mapped| horizontal[mapped * row_len + scalar_x])
            };
            sum += u64::from(value) * u64::from(weight);
        }
        output[scalar_x] = ((sum + round) >> shift).min(255) as u8;
    }
}

fn prepare_fixed_gaussian_kernel(
    size: usize,
    sigma: f64,
    cached_key: &mut Option<(usize, u64, u8)>,
    cached_kernel: &mut Vec<u16>,
) {
    let key = (size, sigma.to_bits(), 8);
    if *cached_key == Some(key) {
        return;
    }
    let center = (size / 2) as f64;
    let denominator = 2.0 * sigma * sigma;
    let mut weights = (0..size)
        .map(|index| {
            let offset = index as f64 - center;
            (-(offset * offset) / denominator).exp()
        })
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    let mut fixed =
        weights.drain(..).map(|weight| (weight * 256.0 / sum).round() as i32).collect::<Vec<_>>();
    let adjustment = 256 - fixed.iter().sum::<i32>();
    fixed[size / 2] += adjustment;
    cached_kernel.clear();
    cached_kernel.extend(fixed.into_iter().map(|weight| weight as u16));
    *cached_key = Some(key);
}

fn gaussian_horizontal_row<const CHANNELS: usize>(
    input: ImageView<'_, u8, CHANNELS>,
    y: usize,
    kernel: &[u16],
    border: BorderMode<u8, CHANNELS>,
    output: &mut [u16],
) {
    let width = input.width();
    let radius = kernel.len() / 2;
    let source = input.row(y).expect("Gaussian source row in bounds");
    let constant = match border {
        BorderMode::Constant(pixel) => pixel,
        _ => [0; CHANNELS],
    };
    if width <= radius * 2 {
        for x in 0..width {
            gaussian_horizontal_border_pixel(source, width, x, kernel, border, constant, output);
        }
        return;
    }
    for x in 0..radius {
        gaussian_horizontal_border_pixel(source, width, x, kernel, border, constant, output);
    }
    for x in width - radius..width {
        gaussian_horizontal_border_pixel(source, width, x, kernel, border, constant, output);
    }

    let start = radius * CHANNELS;
    let end = (width - radius) * CHANNELS;
    match kernel {
        [outer, center, _] => {
            let (outer, center) = (u32::from(*outer), u32::from(*center));
            for index in start..end {
                output[index] = (u32::from(source[index]) * center
                    + (u32::from(source[index - CHANNELS]) + u32::from(source[index + CHANNELS]))
                        * outer) as u16;
            }
        }
        [outer, inner, center, _, _] => {
            let (outer, inner, center) = (u32::from(*outer), u32::from(*inner), u32::from(*center));
            for index in start..end {
                output[index] = (u32::from(source[index]) * center
                    + (u32::from(source[index - CHANNELS]) + u32::from(source[index + CHANNELS]))
                        * inner
                    + (u32::from(source[index - 2 * CHANNELS])
                        + u32::from(source[index + 2 * CHANNELS]))
                        * outer) as u16;
            }
        }
        [outer, middle, inner, center, _, _, _] => {
            let (outer, middle, inner, center) =
                (u32::from(*outer), u32::from(*middle), u32::from(*inner), u32::from(*center));
            for index in start..end {
                output[index] = (u32::from(source[index]) * center
                    + (u32::from(source[index - CHANNELS]) + u32::from(source[index + CHANNELS]))
                        * inner
                    + (u32::from(source[index - 2 * CHANNELS])
                        + u32::from(source[index + 2 * CHANNELS]))
                        * middle
                    + (u32::from(source[index - 3 * CHANNELS])
                        + u32::from(source[index + 3 * CHANNELS]))
                        * outer) as u16;
            }
        }
        _ => unreachable!("specialized Gaussian kernel is validated"),
    }
}

fn gaussian_horizontal_border_pixel<const CHANNELS: usize>(
    source: &[u8],
    width: usize,
    x: usize,
    kernel: &[u16],
    border: BorderMode<u8, CHANNELS>,
    constant: [u8; CHANNELS],
    output: &mut [u16],
) {
    let radius = kernel.len() / 2;
    for channel in 0..CHANNELS {
        let mut sum = 0_u32;
        for (tap, &weight) in kernel.iter().enumerate() {
            let source_x = x as isize + tap as isize - radius as isize;
            let value = map_index(source_x, width, border)
                .map_or(constant[channel], |mapped| source[mapped * CHANNELS + channel]);
            sum += u32::from(value) * u32::from(weight);
        }
        output[x * CHANNELS + channel] = sum as u16;
    }
}

#[inline(always)]
fn gaussian_round_u8(sum: u32) -> u8 {
    ((sum + 32_768) >> 16).min(255) as u8
}

fn gaussian_vertical_border_row<const CHANNELS: usize>(
    horizontal: &[u16],
    width: usize,
    height: usize,
    y: usize,
    kernel: &[u16],
    border: BorderMode<u8, CHANNELS>,
    output: &mut [u8],
) {
    let row_len = width * CHANNELS;
    let radius = kernel.len() / 2;
    let constant = match border {
        BorderMode::Constant(pixel) => pixel.map(|value| u32::from(value) * 256),
        _ => [0; CHANNELS],
    };
    for scalar_x in 0..row_len {
        let channel = scalar_x % CHANNELS;
        let mut sum = 0_u32;
        for (tap, &weight) in kernel.iter().enumerate() {
            let source_y = y as isize + tap as isize - radius as isize;
            let value = map_index(source_y, height, border).map_or(constant[channel], |mapped| {
                u32::from(horizontal[mapped * row_len + scalar_x])
            });
            sum += value * u32::from(weight);
        }
        output[scalar_x] = gaussian_round_u8(sum);
    }
}

#[allow(clippy::too_many_arguments)]
fn gaussian_vertical_row<const CHANNELS: usize>(
    horizontal: &[u16],
    width: usize,
    height: usize,
    y: usize,
    kernel: &[u16],
    border: BorderMode<u8, CHANNELS>,
    output: &mut [u8],
) {
    let row_len = width * CHANNELS;
    let radius = kernel.len() / 2;
    if y < radius || y + radius >= height {
        gaussian_vertical_border_row(horizontal, width, height, y, kernel, border, output);
        return;
    }
    match kernel {
        [outer, center, _] => {
            let (outer, center) = (u32::from(*outer), u32::from(*center));
            let above = (y - 1) * row_len;
            let current = y * row_len;
            let below = (y + 1) * row_len;
            for index in 0..row_len {
                output[index] = gaussian_round_u8(
                    u32::from(horizontal[current + index]) * center
                        + (u32::from(horizontal[above + index])
                            + u32::from(horizontal[below + index]))
                            * outer,
                );
            }
        }
        [outer, inner, center, _, _] => {
            let (outer, inner, center) = (u32::from(*outer), u32::from(*inner), u32::from(*center));
            let row0 = (y - 2) * row_len;
            let row1 = (y - 1) * row_len;
            let row2 = y * row_len;
            let row3 = (y + 1) * row_len;
            let row4 = (y + 2) * row_len;
            for index in 0..row_len {
                output[index] = gaussian_round_u8(
                    u32::from(horizontal[row2 + index]) * center
                        + (u32::from(horizontal[row1 + index])
                            + u32::from(horizontal[row3 + index]))
                            * inner
                        + (u32::from(horizontal[row0 + index])
                            + u32::from(horizontal[row4 + index]))
                            * outer,
                );
            }
        }
        [outer, middle, inner, center, _, _, _] => {
            let (outer, middle, inner, center) =
                (u32::from(*outer), u32::from(*middle), u32::from(*inner), u32::from(*center));
            let row0 = (y - 3) * row_len;
            let row1 = (y - 2) * row_len;
            let row2 = (y - 1) * row_len;
            let row3 = y * row_len;
            let row4 = (y + 1) * row_len;
            let row5 = (y + 2) * row_len;
            let row6 = (y + 3) * row_len;
            for index in 0..row_len {
                output[index] = gaussian_round_u8(
                    u32::from(horizontal[row3 + index]) * center
                        + (u32::from(horizontal[row2 + index])
                            + u32::from(horizontal[row4 + index]))
                            * inner
                        + (u32::from(horizontal[row1 + index])
                            + u32::from(horizontal[row5 + index]))
                            * middle
                        + (u32::from(horizontal[row0 + index])
                            + u32::from(horizontal[row6 + index]))
                            * outer,
                );
            }
        }
        _ => unreachable!("specialized Gaussian kernel is validated"),
    }
}

fn gaussian_kernel(size: usize, sigma: f64) -> VisionResult<Kernel1D> {
    if size == 0 || size % 2 == 0 {
        return Err(VisionError::InvalidParameter(
            "Gaussian kernel size must be positive and odd".into(),
        ));
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "Gaussian sigma must be finite and positive".into(),
        ));
    }
    let center = (size / 2) as f64;
    let denominator = 2.0 * sigma * sigma;
    let mut coefficients = (0..size)
        .map(|index| {
            let offset = index as f64 - center;
            (-(offset * offset) / denominator).exp()
        })
        .collect::<Vec<_>>();
    let sum = coefficients.iter().sum::<f64>();
    for value in &mut coefficients {
        *value /= sum;
    }
    Kernel1D::try_new(coefficients)
}

fn validate_delta(delta: f64) -> VisionResult<()> {
    if !delta.is_finite() {
        return Err(VisionError::InvalidParameter("filter delta must be finite".into()));
    }
    Ok(())
}

fn correlate<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel: &Kernel2D,
    delta: f64,
    border: BorderMode<T, CHANNELS>,
) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            let mut sums = [delta; CHANNELS];
            for ky in 0..kernel.height {
                for kx in 0..kernel.width {
                    let pixel = fetch(
                        input,
                        x as isize + kx as isize - kernel.anchor_x as isize,
                        y as isize + ky as isize - kernel.anchor_y as isize,
                        border,
                    );
                    let weight = kernel.coefficients[ky * kernel.width + kx];
                    for channel in 0..CHANNELS {
                        sums[channel] += pixel[channel].to_f64() * weight;
                    }
                }
            }
            output.extend_from_slice(&sums);
        }
    }
    output
}

fn separable_accumulators<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel_x: &Kernel1D,
    kernel_y: &Kernel1D,
    delta: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Vec<f64>> {
    validate_delta(delta)?;
    let mut horizontal = vec![0.0; input.width() * input.height() * CHANNELS];
    for y in 0..input.height() {
        for x in 0..input.width() {
            for (kx, &weight) in kernel_x.coefficients.iter().enumerate() {
                let pixel = fetch(
                    input,
                    x as isize + kx as isize - kernel_x.anchor as isize,
                    y as isize,
                    border,
                );
                for channel in 0..CHANNELS {
                    horizontal[(y * input.width() + x) * CHANNELS + channel] +=
                        pixel[channel].to_f64() * weight;
                }
            }
        }
    }

    let constant_horizontal = match border {
        BorderMode::Constant(pixel) => {
            let sum_x = kernel_x.coefficients.iter().sum::<f64>();
            std::array::from_fn(|channel| pixel[channel].to_f64() * sum_x)
        }
        _ => [0.0; CHANNELS],
    };
    let mut output = Vec::with_capacity(horizontal.len());
    for y in 0..input.height() {
        for x in 0..input.width() {
            let mut sums = [delta; CHANNELS];
            for (ky, &weight) in kernel_y.coefficients.iter().enumerate() {
                let source_y = y as isize + ky as isize - kernel_y.anchor as isize;
                if let Some(mapped_y) = map_index(source_y, input.height(), border) {
                    let offset = (mapped_y * input.width() + x) * CHANNELS;
                    for channel in 0..CHANNELS {
                        sums[channel] += horizontal[offset + channel] * weight;
                    }
                } else {
                    for channel in 0..CHANNELS {
                        sums[channel] += constant_horizontal[channel] * weight;
                    }
                }
            }
            output.extend_from_slice(&sums);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{
        box_blur, convolve2d, filter2d, gaussian_blur, gaussian_blur_u8, gaussian_blur_u8_into,
        separable_filter, GaussianBlurU8Workspace, Kernel1D, Kernel2D,
    };
    use crate::BorderMode;
    use proptest::prelude::*;
    use spatialrust_image::{Image, ImageRegion, ImageView};

    #[test]
    fn filter2d_is_correlation_and_convolution_reverses() {
        let image = Image::<u8, 1>::try_new(3, 1, vec![1, 2, 4]).unwrap();
        let kernel = Kernel2D::try_new_with_anchor(2, 1, 0, 0, vec![1.0, 10.0]).unwrap();
        let correlation = filter2d(image.view(), &kernel, 0.0, BorderMode::Replicate).unwrap();
        let convolution = convolve2d(image.view(), &kernel, 0.0, BorderMode::Replicate).unwrap();
        assert_eq!(correlation.as_slice(), &[21, 42, 44]);
        assert_eq!(convolution.as_slice(), &[11, 12, 24]);
    }

    #[test]
    fn separable_matches_outer_product_with_constant_border() {
        let image = Image::<u8, 1>::try_new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let x = Kernel1D::try_new(vec![0.25, 0.5, 0.25]).unwrap();
        let y = Kernel1D::try_new(vec![0.25, 0.5, 0.25]).unwrap();
        let kernel = Kernel2D::try_new(
            3,
            3,
            vec![0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625],
        )
        .unwrap();
        let expected = filter2d(image.view(), &kernel, 0.0, BorderMode::Constant([9])).unwrap();
        let actual =
            separable_filter(image.view(), &x, &y, 0.0, BorderMode::Constant([9])).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn roi_and_packed_filter_results_match() {
        let parent = Image::<u8, 1>::try_new(5, 3, (0..15).collect()).unwrap();
        let roi = parent.view().subview(ImageRegion::new(1, 1, 3, 2)).unwrap();
        let packed = Image::<u8, 1>::try_new(3, 2, vec![6, 7, 8, 11, 12, 13]).unwrap();
        let kernel = Kernel2D::try_new(3, 1, vec![0.25, 0.5, 0.25]).unwrap();
        assert_eq!(
            filter2d(roi, &kernel, 0.0, BorderMode::Reflect101).unwrap(),
            filter2d(packed.view(), &kernel, 0.0, BorderMode::Reflect101).unwrap()
        );
    }

    #[test]
    fn normalized_blurs_preserve_constant_images() {
        let image = Image::<f32, 3>::from_pixel(4, 3, [2.0, 4.0, 8.0]).unwrap();
        let box_output = box_blur(image.view(), 4, 2, BorderMode::Replicate).unwrap();
        let gaussian = gaussian_blur(image.view(), 5, 3, 1.2, 0.8, BorderMode::Reflect101).unwrap();
        assert_eq!(box_output, image);
        for (actual, expected) in gaussian.as_slice().iter().zip(image.as_slice()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn specialized_gaussian_matches_generic_for_strided_rgb_input() {
        let width = 17;
        let height = 11;
        let stride = width * 3 + 7;
        let mut storage = vec![199_u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                for channel in 0..3 {
                    storage[y * stride + x * 3 + channel] =
                        ((x * 31 + y * 17 + channel * 73) & 255) as u8;
                }
            }
        }
        let view = spatialrust_image::ImageView::new(width, height, stride, &storage).unwrap();
        for &(size, sigma) in &[(3, 0.8), (5, 1.2), (7, 2.0)] {
            for border in [
                BorderMode::Replicate,
                BorderMode::Reflect,
                BorderMode::Reflect101,
                BorderMode::Wrap,
                BorderMode::Constant([11, 23, 47]),
            ] {
                let expected = gaussian_blur(view, size, size, sigma, sigma, border).unwrap();
                let actual = gaussian_blur_u8(view, size, size, sigma, sigma, border).unwrap();
                assert!(expected
                    .as_slice()
                    .iter()
                    .zip(actual.as_slice())
                    .all(|(&left, &right)| left.abs_diff(right) <= 1));
            }
        }
    }

    #[test]
    fn specialized_gaussian_reuses_workspace_and_validates_output() {
        let image =
            Image::<u8, 3>::try_new(8, 6, (0..144).map(|value| value as u8).collect()).unwrap();
        let mut output = vec![0; 8 * 6 * 3];
        let mut workspace = GaussianBlurU8Workspace::new();
        gaussian_blur_u8_into(
            image.view(),
            5,
            5,
            1.2,
            1.2,
            BorderMode::Reflect101,
            &mut output,
            &mut workspace,
        )
        .unwrap();
        let capacity = workspace.capacity();
        gaussian_blur_u8_into(
            image.view(),
            5,
            5,
            1.2,
            1.2,
            BorderMode::Reflect101,
            &mut output,
            &mut workspace,
        )
        .unwrap();
        assert_eq!(workspace.capacity(), capacity);
        assert!(gaussian_blur_u8_into(
            image.view(),
            5,
            5,
            1.2,
            1.2,
            BorderMode::Reflect101,
            &mut output[..10],
            &mut workspace,
        )
        .is_err());
        assert!(gaussian_blur_u8(image.view(), 9, 5, 1.2, 1.2, BorderMode::Reflect101,).is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn high_precision_7x7_matches_generic_with_strided_input(
            width in 1_usize..33,
            height in 1_usize..33,
            padding in 0_usize..8,
            sigma_x in 0.3_f64..4.0,
            sigma_y in 0.3_f64..4.0,
            seed in any::<u64>(),
        ) {
            let stride = width * 3 + padding;
            let mut storage = vec![173_u8; stride * height];
            let mut state = seed;
            for y in 0..height {
                for value in &mut storage[y * stride..y * stride + width * 3] {
                    state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    *value = (state >> 32) as u8;
                }
            }
            let input = ImageView::<u8, 3>::new(width, height, stride, &storage).unwrap();
            let expected = gaussian_blur(
                input,
                7,
                7,
                sigma_x,
                sigma_y,
                BorderMode::Reflect101,
            )
            .unwrap();
            let mut output = vec![0_u8; width * height * 3];
            let mut workspace = GaussianBlurU8Workspace::new();
            gaussian_blur_u8_into(
                input,
                7,
                7,
                sigma_x,
                sigma_y,
                BorderMode::Reflect101,
                &mut output,
                &mut workspace,
            )
            .unwrap();
            prop_assert!(expected
                .as_slice()
                .iter()
                .zip(&output)
                .all(|(&left, &right)| left.abs_diff(right) <= 1));
            let capacity = workspace.capacity();
            gaussian_blur_u8_into(
                input,
                7,
                7,
                sigma_x,
                sigma_y,
                BorderMode::Reflect101,
                &mut output,
                &mut workspace,
            )
            .unwrap();
            prop_assert_eq!(workspace.capacity(), capacity);
        }
    }

    #[test]
    fn invalid_kernels_are_rejected() {
        assert!(Kernel2D::try_new(0, 1, Vec::new()).is_err());
        assert!(Kernel2D::try_new(2, 2, vec![1.0; 3]).is_err());
        assert!(Kernel1D::try_new(vec![f64::NAN]).is_err());
    }
}
