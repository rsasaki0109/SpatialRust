//! Canny edge detection with inspectable CPU intermediates.

use std::collections::VecDeque;

use spatialrust_image::{Image, ImageView};

use crate::{sobel, spatial_gradient_u8, BorderMode, VisionError, VisionResult};

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

/// Finds edges in a single-channel u8 image.
pub fn canny(input: ImageView<'_, u8, 1>, options: CannyOptions) -> VisionResult<Image<u8, 1>> {
    Ok(canny_with_intermediates(input, options)?.edges)
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

#[cfg(test)]
mod tests {
    use super::{canny, canny_with_intermediates, CannyOptions};
    use spatialrust_image::{Image, ImageRegion};

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
}
