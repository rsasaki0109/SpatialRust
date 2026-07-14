//! Linear CPU image filters with explicit border and kernel contracts.

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
        box_blur, convolve2d, filter2d, gaussian_blur, separable_filter, Kernel1D, Kernel2D,
    };
    use crate::BorderMode;
    use spatialrust_image::{Image, ImageRegion};

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
    fn invalid_kernels_are_rejected() {
        assert!(Kernel2D::try_new(0, 1, Vec::new()).is_err());
        assert!(Kernel2D::try_new(2, 2, vec![1.0; 3]).is_err());
        assert!(Kernel1D::try_new(vec![f64::NAN]).is_err());
    }
}
