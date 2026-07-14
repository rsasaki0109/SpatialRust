//! Non-linear filters, image derivatives, and Gaussian pyramids.

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
