//! Image remapping and geometric warp primitives.

use spatialrust_image::{Image, ImageView};

use crate::{Interpolation, PixelComponent, VisionError, VisionResult};

/// Out-of-bounds sampling behavior for geometric image operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BorderMode<T, const CHANNELS: usize> {
    /// Returns a fixed pixel outside the source image.
    Constant([T; CHANNELS]),
    /// Repeats the closest edge pixel.
    Replicate,
    /// Reflects including the edge pixel (`fedcba|abcdefgh|hgfedc`).
    Reflect,
    /// Reflects without repeating the edge (`gfedcb|abcdefgh|gfedcb`).
    Reflect101,
    /// Periodically wraps source coordinates.
    Wrap,
}

/// A source-to-destination 2D affine transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineTransform {
    /// First two rows of a homogeneous 3x3 transform.
    pub matrix: [[f64; 3]; 2],
}

impl AffineTransform {
    /// Identity affine transform.
    #[must_use]
    pub const fn identity() -> Self {
        Self { matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] }
    }

    /// Applies the source-to-destination transform.
    #[must_use]
    pub fn map_point(self, x: f64, y: f64) -> (f64, f64) {
        (
            self.matrix[0][0].mul_add(x, self.matrix[0][1].mul_add(y, self.matrix[0][2])),
            self.matrix[1][0].mul_add(x, self.matrix[1][1].mul_add(y, self.matrix[1][2])),
        )
    }

    /// Computes the inverse affine transform.
    pub fn inverse(self) -> VisionResult<Self> {
        let a = self.matrix[0][0];
        let b = self.matrix[0][1];
        let c = self.matrix[0][2];
        let d = self.matrix[1][0];
        let e = self.matrix[1][1];
        let f = self.matrix[1][2];
        let determinant = a.mul_add(e, -(b * d));
        if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
            return Err(VisionError::SingularTransform);
        }
        let inv = 1.0 / determinant;
        Ok(Self {
            matrix: [
                [e * inv, -b * inv, (b * f - e * c) * inv],
                [-d * inv, a * inv, (d * c - a * f) * inv],
            ],
        })
    }
}

/// A source-to-destination projective 2D transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerspectiveTransform {
    /// Homogeneous 3x3 transform matrix.
    pub matrix: [[f64; 3]; 3],
}

impl PerspectiveTransform {
    /// Identity projective transform.
    #[must_use]
    pub const fn identity() -> Self {
        Self { matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] }
    }

    /// Applies the source-to-destination homogeneous transform.
    #[must_use]
    pub fn map_point(self, x: f64, y: f64) -> Option<(f64, f64)> {
        let denominator =
            self.matrix[2][0].mul_add(x, self.matrix[2][1].mul_add(y, self.matrix[2][2]));
        if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
            return None;
        }
        Some((
            self.matrix[0][0].mul_add(x, self.matrix[0][1].mul_add(y, self.matrix[0][2]))
                / denominator,
            self.matrix[1][0].mul_add(x, self.matrix[1][1].mul_add(y, self.matrix[1][2]))
                / denominator,
        ))
    }

    /// Computes the inverse projective transform.
    pub fn inverse(self) -> VisionResult<Self> {
        let m = self.matrix;
        let c00 = m[1][1].mul_add(m[2][2], -(m[1][2] * m[2][1]));
        let c01 = -(m[1][0].mul_add(m[2][2], -(m[1][2] * m[2][0])));
        let c02 = m[1][0].mul_add(m[2][1], -(m[1][1] * m[2][0]));
        let c10 = -(m[0][1].mul_add(m[2][2], -(m[0][2] * m[2][1])));
        let c11 = m[0][0].mul_add(m[2][2], -(m[0][2] * m[2][0]));
        let c12 = -(m[0][0].mul_add(m[2][1], -(m[0][1] * m[2][0])));
        let c20 = m[0][1].mul_add(m[1][2], -(m[0][2] * m[1][1]));
        let c21 = -(m[0][0].mul_add(m[1][2], -(m[0][2] * m[1][0])));
        let c22 = m[0][0].mul_add(m[1][1], -(m[0][1] * m[1][0]));
        let determinant = m[0][0].mul_add(c00, m[0][1].mul_add(c01, m[0][2] * c02));
        if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
            return Err(VisionError::SingularTransform);
        }
        let inv = 1.0 / determinant;
        Ok(Self {
            matrix: [
                [c00 * inv, c10 * inv, c20 * inv],
                [c01 * inv, c11 * inv, c21 * inv],
                [c02 * inv, c12 * inv, c22 * inv],
            ],
        })
    }
}

/// Samples an image at absolute coordinates supplied by two single-channel maps.
pub fn remap<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    map_x: ImageView<'_, f32, 1>,
    map_y: ImageView<'_, f32, 1>,
    interpolation: Interpolation,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    if map_x.width() != map_y.width() || map_x.height() != map_y.height() {
        return Err(VisionError::ShapeMismatch("map_x and map_y dimensions must match".to_owned()));
    }
    if interpolation == Interpolation::Area {
        return Err(VisionError::InvalidParameter(
            "area interpolation is not defined for arbitrary remap".to_owned(),
        ));
    }
    let mut output = Vec::with_capacity(map_x.width() * map_x.height() * CHANNELS);
    for y in 0..map_x.height() {
        for x in 0..map_x.width() {
            let sx = f64::from(map_x.get(x, y).expect("map coordinate in bounds")[0]);
            let sy = f64::from(map_y.get(x, y).expect("map coordinate in bounds")[0]);
            let pixel = sample(input, sx, sy, interpolation, border);
            output.extend_from_slice(&pixel);
        }
    }
    Ok(Image::try_new_with_metadata(map_x.width(), map_x.height(), output, input.metadata())?)
}

/// Warps an image with a source-to-destination affine transform.
pub fn warp_affine<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    transform: AffineTransform,
    output_width: usize,
    output_height: usize,
    interpolation: Interpolation,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    let inverse = transform.inverse()?;
    warp_with_mapping(input, output_width, output_height, interpolation, border, |x, y| {
        Some(inverse.map_point(x, y))
    })
}

/// Warps an image with a source-to-destination projective transform.
pub fn warp_perspective<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    transform: PerspectiveTransform,
    output_width: usize,
    output_height: usize,
    interpolation: Interpolation,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    let inverse = transform.inverse()?;
    warp_with_mapping(input, output_width, output_height, interpolation, border, |x, y| {
        inverse.map_point(x, y)
    })
}

fn warp_with_mapping<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    output_width: usize,
    output_height: usize,
    interpolation: Interpolation,
    border: BorderMode<T, CHANNELS>,
    mut mapping: impl FnMut(f64, f64) -> Option<(f64, f64)>,
) -> VisionResult<Image<T, CHANNELS>> {
    if interpolation == Interpolation::Area {
        return Err(VisionError::InvalidParameter(
            "area interpolation is not defined for geometric warp".to_owned(),
        ));
    }
    let mut output = Vec::with_capacity(output_width * output_height * CHANNELS);
    for y in 0..output_height {
        for x in 0..output_width {
            let pixel = mapping(x as f64, y as f64).map_or_else(
                || constant_pixel(border),
                |(sx, sy)| sample(input, sx, sy, interpolation, border),
            );
            output.extend_from_slice(&pixel);
        }
    }
    Ok(Image::try_new_with_metadata(output_width, output_height, output, input.metadata())?)
}

fn constant_pixel<T: PixelComponent, const CHANNELS: usize>(
    border: BorderMode<T, CHANNELS>,
) -> [T; CHANNELS] {
    match border {
        BorderMode::Constant(pixel) => pixel,
        BorderMode::Replicate | BorderMode::Reflect | BorderMode::Reflect101 | BorderMode::Wrap => {
            std::array::from_fn(|_| T::from_f64(0.0))
        }
    }
}

fn sample<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: f64,
    y: f64,
    interpolation: Interpolation,
    border: BorderMode<T, CHANNELS>,
) -> [T; CHANNELS] {
    if !x.is_finite() || !y.is_finite() || input.width() == 0 || input.height() == 0 {
        return constant_pixel(border);
    }
    match interpolation {
        Interpolation::Nearest => fetch(input, x.round() as isize, y.round() as isize, border),
        Interpolation::Bilinear => sample_bilinear(input, x, y, border),
        Interpolation::Bicubic => sample_bicubic(input, x, y, border),
        Interpolation::Area => unreachable!("area rejected before sampling"),
    }
}

fn sample_bilinear<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: f64,
    y: f64,
    border: BorderMode<T, CHANNELS>,
) -> [T; CHANNELS] {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let wx = x - x.floor();
    let wy = y - y.floor();
    let p00 = fetch(input, x0, y0, border);
    let p10 = fetch(input, x0 + 1, y0, border);
    let p01 = fetch(input, x0, y0 + 1, border);
    let p11 = fetch(input, x0 + 1, y0 + 1, border);
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
    border: BorderMode<T, CHANNELS>,
) -> [T; CHANNELS] {
    let base_x = x.floor() as isize;
    let base_y = y.floor() as isize;
    let mut sums = [0.0_f64; CHANNELS];
    let mut total_weight = 0.0;
    for dy in -1..=2 {
        let wy = cubic_weight(y - (base_y + dy) as f64);
        for dx in -1..=2 {
            let weight = wy * cubic_weight(x - (base_x + dx) as f64);
            let pixel = fetch(input, base_x + dx, base_y + dy, border);
            for channel in 0..CHANNELS {
                sums[channel] += pixel[channel].to_f64() * weight;
            }
            total_weight += weight;
        }
    }
    std::array::from_fn(|channel| T::from_f64(sums[channel] / total_weight))
}

fn fetch<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: isize,
    y: isize,
    border: BorderMode<T, CHANNELS>,
) -> [T; CHANNELS] {
    if x >= 0 && y >= 0 && x < input.width() as isize && y < input.height() as isize {
        return *input.get(x as usize, y as usize).expect("coordinate checked");
    }
    match border {
        BorderMode::Constant(pixel) => pixel,
        BorderMode::Replicate => {
            let ix = x.clamp(0, input.width().saturating_sub(1) as isize) as usize;
            let iy = y.clamp(0, input.height().saturating_sub(1) as isize) as usize;
            *input.get(ix, iy).expect("replicated coordinate")
        }
        BorderMode::Reflect => {
            let ix = border_index(x, input.width(), false);
            let iy = border_index(y, input.height(), false);
            *input.get(ix, iy).expect("reflected coordinate")
        }
        BorderMode::Reflect101 => {
            let ix = border_index(x, input.width(), true);
            let iy = border_index(y, input.height(), true);
            *input.get(ix, iy).expect("reflected coordinate")
        }
        BorderMode::Wrap => {
            let ix = x.rem_euclid(input.width() as isize) as usize;
            let iy = y.rem_euclid(input.height() as isize) as usize;
            *input.get(ix, iy).expect("wrapped coordinate")
        }
    }
}

fn border_index(mut index: isize, length: usize, reflect101: bool) -> usize {
    if length <= 1 {
        return 0;
    }
    let length = length as isize;
    while index < 0 || index >= length {
        index = if index < 0 {
            if reflect101 {
                -index
            } else {
                -index - 1
            }
        } else if reflect101 {
            2 * length - index - 2
        } else {
            2 * length - index - 1
        };
    }
    index as usize
}

#[cfg(test)]
mod tests {
    use super::{
        remap, warp_affine, warp_perspective, AffineTransform, BorderMode, PerspectiveTransform,
    };
    use crate::Interpolation;
    use spatialrust_image::Image;

    #[test]
    fn identity_remap_is_exact() {
        let input = Image::<u8, 1>::try_new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mx = Image::<f32, 1>::try_new(2, 2, vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        let my = Image::<f32, 1>::try_new(2, 2, vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let output = remap(
            input.view(),
            mx.view(),
            my.view(),
            Interpolation::Bilinear,
            BorderMode::Constant([0]),
        )
        .unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn affine_translation_uses_constant_border() {
        let input = Image::<u8, 1>::try_new(3, 1, vec![1, 2, 3]).unwrap();
        let transform = AffineTransform { matrix: [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]] };
        let output = warp_affine(
            input.view(),
            transform,
            3,
            1,
            Interpolation::Nearest,
            BorderMode::Constant([9]),
        )
        .unwrap();
        assert_eq!(output.as_slice(), &[9, 1, 2]);
    }

    #[test]
    fn perspective_identity_is_exact() {
        let input = Image::<u8, 1>::try_new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let output = warp_perspective(
            input.view(),
            PerspectiveTransform::identity(),
            2,
            2,
            Interpolation::Nearest,
            BorderMode::Replicate,
        )
        .unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn border_modes_are_distinct() {
        let input = Image::<u8, 1>::try_new(3, 1, vec![10, 20, 30]).unwrap();
        let mx = Image::<f32, 1>::try_new(1, 1, vec![-1.0]).unwrap();
        let my = Image::<f32, 1>::try_new(1, 1, vec![0.0]).unwrap();
        let run = |border| {
            remap(input.view(), mx.view(), my.view(), Interpolation::Nearest, border)
                .unwrap()
                .as_slice()[0]
        };
        assert_eq!(run(BorderMode::Constant([5])), 5);
        assert_eq!(run(BorderMode::Replicate), 10);
        assert_eq!(run(BorderMode::Reflect), 10);
        assert_eq!(run(BorderMode::Reflect101), 20);
        assert_eq!(run(BorderMode::Wrap), 30);
    }
}
