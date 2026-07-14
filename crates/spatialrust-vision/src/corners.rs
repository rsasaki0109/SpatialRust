//! Classical corner and accelerated segment-test detectors.

use std::cmp::Ordering;

use spatialrust_image::{Image, ImageView};

use crate::border::fetch;
use crate::{sobel, BorderMode, Keypoint2, VisionError, VisionResult};

/// Shared response thresholding and spatial suppression options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CornerSelectionOptions {
    /// Maximum retained corners; zero keeps every accepted corner.
    pub max_corners: usize,
    /// Fraction of the strongest positive response in `(0, 1]`.
    pub quality_level: f32,
    /// Minimum Euclidean distance between retained corners in pixels.
    pub min_distance: f32,
    /// Odd structure-tensor summation window size.
    pub block_size: usize,
    /// Sobel aperture size: 3, 5, or 7.
    pub gradient_size: usize,
    /// Source-image border extrapolation used by derivatives.
    pub border: BorderMode<u8, 1>,
}

impl Default for CornerSelectionOptions {
    fn default() -> Self {
        Self {
            max_corners: 0,
            quality_level: 0.01,
            min_distance: 1.0,
            block_size: 3,
            gradient_size: 3,
            border: BorderMode::Reflect101,
        }
    }
}

impl CornerSelectionOptions {
    fn validate(self) -> VisionResult<Self> {
        if !self.quality_level.is_finite() || self.quality_level <= 0.0 || self.quality_level > 1.0
        {
            return Err(VisionError::InvalidParameter(
                "corner quality_level must be finite and in (0, 1]".into(),
            ));
        }
        if !self.min_distance.is_finite() || self.min_distance < 0.0 {
            return Err(VisionError::InvalidParameter(
                "corner min_distance must be finite and non-negative".into(),
            ));
        }
        if self.block_size == 0 || self.block_size % 2 == 0 {
            return Err(VisionError::InvalidParameter(
                "corner block_size must be positive and odd".into(),
            ));
        }
        if !matches!(self.gradient_size, 3 | 5 | 7) {
            return Err(VisionError::InvalidParameter(
                "corner gradient_size must be 3, 5, or 7".into(),
            ));
        }
        Ok(self)
    }
}

/// Harris detector configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarrisOptions {
    /// Shared candidate selection options.
    pub selection: CornerSelectionOptions,
    /// Harris trace penalty, normally near `0.04`.
    pub k: f32,
}

impl Default for HarrisOptions {
    fn default() -> Self {
        Self { selection: CornerSelectionOptions::default(), k: 0.04 }
    }
}

/// Shi–Tomasi minimum-eigenvalue detector configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShiTomasiOptions {
    /// Shared candidate selection options.
    pub selection: CornerSelectionOptions,
}

/// Computes an unnormalized Harris structure-tensor response image.
pub fn harris_response(
    input: ImageView<'_, u8, 1>,
    block_size: usize,
    gradient_size: usize,
    k: f32,
    border: BorderMode<u8, 1>,
) -> VisionResult<Image<f32, 1>> {
    if !k.is_finite() || k < 0.0 {
        return Err(VisionError::InvalidParameter(
            "Harris k must be finite and non-negative".into(),
        ));
    }
    corner_response(input, block_size, gradient_size, border, |a, b, c| {
        a.mul_add(c, -(b * b)) - k * (a + c) * (a + c)
    })
}

/// Computes an unnormalized Shi–Tomasi minimum-eigenvalue response image.
pub fn shi_tomasi_response(
    input: ImageView<'_, u8, 1>,
    block_size: usize,
    gradient_size: usize,
    border: BorderMode<u8, 1>,
) -> VisionResult<Image<f32, 1>> {
    corner_response(input, block_size, gradient_size, border, |a, b, c| {
        0.5 * (a + c - ((a - c) * (a - c) + 4.0 * b * b).sqrt())
    })
}

/// Detects Harris corners and returns strongest-first keypoints.
pub fn detect_harris(
    input: ImageView<'_, u8, 1>,
    options: HarrisOptions,
) -> VisionResult<Vec<Keypoint2>> {
    let selection = options.selection.validate()?;
    let response = harris_response(
        input,
        selection.block_size,
        selection.gradient_size,
        options.k,
        selection.border,
    )?;
    select_corner_responses(response.view(), selection)
}

/// Detects Shi–Tomasi corners and returns strongest-first keypoints.
pub fn detect_shi_tomasi(
    input: ImageView<'_, u8, 1>,
    options: ShiTomasiOptions,
) -> VisionResult<Vec<Keypoint2>> {
    let selection = options.selection.validate()?;
    let response = shi_tomasi_response(
        input,
        selection.block_size,
        selection.gradient_size,
        selection.border,
    )?;
    select_corner_responses(response.view(), selection)
}

fn corner_response(
    input: ImageView<'_, u8, 1>,
    block_size: usize,
    gradient_size: usize,
    border: BorderMode<u8, 1>,
    score: impl Fn(f32, f32, f32) -> f32,
) -> VisionResult<Image<f32, 1>> {
    let validation = CornerSelectionOptions {
        block_size,
        gradient_size,
        border,
        ..CornerSelectionOptions::default()
    }
    .validate()?;
    let gradient_x = sobel(input, 1, 0, validation.gradient_size, 1.0, 0.0, border)?;
    let gradient_y = sobel(input, 0, 1, validation.gradient_size, 1.0, 0.0, border)?;
    let products_xx = Image::try_new(
        input.width(),
        input.height(),
        gradient_x.as_slice().iter().map(|value| value * value).collect(),
    )?;
    let products_xy = Image::try_new(
        input.width(),
        input.height(),
        gradient_x.as_slice().iter().zip(gradient_y.as_slice()).map(|(x, y)| x * y).collect(),
    )?;
    let products_yy = Image::try_new(
        input.width(),
        input.height(),
        gradient_y.as_slice().iter().map(|value| value * value).collect(),
    )?;
    let product_border = product_border(border);
    let radius = (block_size / 2) as isize;
    let mut output = Vec::with_capacity(input.width() * input.height());
    for y in 0..input.height() {
        for x in 0..input.width() {
            let mut a = 0.0_f32;
            let mut b = 0.0_f32;
            let mut c = 0.0_f32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    a +=
                        fetch(products_xx.view(), x as isize + dx, y as isize + dy, product_border)
                            [0];
                    b +=
                        fetch(products_xy.view(), x as isize + dx, y as isize + dy, product_border)
                            [0];
                    c +=
                        fetch(products_yy.view(), x as isize + dx, y as isize + dy, product_border)
                            [0];
                }
            }
            output.push(score(a, b, c));
        }
    }
    Ok(Image::try_new(input.width(), input.height(), output)?)
}

fn product_border(border: BorderMode<u8, 1>) -> BorderMode<f32, 1> {
    match border {
        BorderMode::Constant(_) => BorderMode::Constant([0.0]),
        BorderMode::Replicate => BorderMode::Replicate,
        BorderMode::Reflect => BorderMode::Reflect,
        BorderMode::Reflect101 => BorderMode::Reflect101,
        BorderMode::Wrap => BorderMode::Wrap,
    }
}

fn select_corner_responses(
    response: ImageView<'_, f32, 1>,
    options: CornerSelectionOptions,
) -> VisionResult<Vec<Keypoint2>> {
    let maximum = (0..response.height())
        .flat_map(|y| response.row(y).expect("coordinate in bounds").iter().copied())
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if maximum <= 0.0 || !maximum.is_finite() {
        return Ok(Vec::new());
    }
    if response.width() < 3 || response.height() < 3 {
        return Ok(Vec::new());
    }
    let threshold = maximum * options.quality_level;
    let mut candidates = Vec::new();
    // OpenCV goodFeaturesToTrack applies 3x3 dilation/NMS only to pixels with
    // a complete immediate neighborhood, excluding the outermost image ring.
    for y in 1..response.height() - 1 {
        for x in 1..response.width() - 1 {
            let value = response.get(x, y).expect("coordinate in bounds")[0];
            if value <= threshold || !is_local_maximum(response, x, y, value) {
                continue;
            }
            candidates.push((x, y, value));
        }
    }
    candidates.sort_by(|left, right| {
        right
            .2
            .partial_cmp(&left.2)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.0.cmp(&right.0))
    });
    let minimum_squared = options.min_distance * options.min_distance;
    let mut selected = Vec::<Keypoint2>::new();
    for (x, y, value) in candidates {
        if options.min_distance > 0.0
            && selected.iter().any(|keypoint| {
                let dx = keypoint.x() - x as f32;
                let dy = keypoint.y() - y as f32;
                dx.mul_add(dx, dy * dy) < minimum_squared
            })
        {
            continue;
        }
        selected.push(
            Keypoint2::try_new(x as f32, y as f32, value)?.with_size(options.block_size as f32)?,
        );
        if options.max_corners != 0 && selected.len() == options.max_corners {
            break;
        }
    }
    Ok(selected)
}

fn is_local_maximum(response: ImageView<'_, f32, 1>, x: usize, y: usize, value: f32) -> bool {
    for dy in -1_isize..=1 {
        for dx in -1_isize..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if nx >= 0
                && ny >= 0
                && (nx as usize) < response.width()
                && (ny as usize) < response.height()
                && response.get(nx as usize, ny as usize).expect("checked coordinate")[0] > value
            {
                return false;
            }
        }
    }
    true
}

/// FAST detector configuration for the standard 9-of-16 radius-three circle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastOptions {
    /// Strict center-to-circle intensity threshold.
    pub threshold: u8,
    /// Retain only strict 3x3 maxima of the FAST score.
    pub nonmax_suppression: bool,
}

impl Default for FastOptions {
    fn default() -> Self {
        Self { threshold: 10, nonmax_suppression: true }
    }
}

const FAST_CIRCLE: [(isize, isize); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

/// Detects FAST-9/16 corners in scan order, matching OpenCV's radius-three ROI.
pub fn detect_fast(
    input: ImageView<'_, u8, 1>,
    options: FastOptions,
) -> VisionResult<Vec<Keypoint2>> {
    if input.width() < 7 || input.height() < 7 {
        return Ok(Vec::new());
    }
    let mut scores = vec![0_u8; input.width() * input.height()];
    let mut candidates = Vec::new();
    for y in 3..input.height() - 3 {
        for x in 3..input.width() - 3 {
            let score = fast_score(input, x, y);
            if score >= options.threshold {
                scores[y * input.width() + x] = score;
                candidates.push((x, y));
            }
        }
    }
    let mut keypoints = Vec::new();
    for (x, y) in candidates {
        let score = scores[y * input.width() + x];
        if options.nonmax_suppression {
            let strict_maximum = (-1_isize..=1).all(|dy| {
                (-1_isize..=1).all(|dx| {
                    (dx == 0 && dy == 0)
                        || score
                            > scores[(y as isize + dy) as usize * input.width()
                                + (x as isize + dx) as usize]
                })
            });
            if !strict_maximum {
                continue;
            }
        }
        keypoints.push(
            Keypoint2::try_new(
                x as f32,
                y as f32,
                if options.nonmax_suppression { f32::from(score) } else { 0.0 },
            )?
            .with_size(7.0)?,
        );
    }
    Ok(keypoints)
}

fn fast_score(input: ImageView<'_, u8, 1>, x: usize, y: usize) -> u8 {
    let center = i16::from(input.get(x, y).expect("coordinate in bounds")[0]);
    let differences = std::array::from_fn::<_, 16, _>(|index| {
        let (dx, dy) = FAST_CIRCLE[index];
        i16::from(
            input
                .get((x as isize + dx) as usize, (y as isize + dy) as usize)
                .expect("FAST radius checked")[0],
        ) - center
    });
    let mut best = 0_i16;
    for start in 0..16 {
        let mut bright = i16::MAX;
        let mut dark = i16::MAX;
        for offset in 0..9 {
            let difference = differences[(start + offset) % 16];
            bright = bright.min(difference);
            dark = dark.min(-difference);
        }
        best = best.max(bright).max(dark);
    }
    best.saturating_sub(1).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::{
        detect_fast, detect_harris, detect_shi_tomasi, FastOptions, HarrisOptions, ShiTomasiOptions,
    };
    use spatialrust_image::{Image, ImageRegion};

    fn square() -> Image<u8, 1> {
        let mut image = Image::try_new(17, 15, vec![0; 17 * 15]).unwrap();
        for y in 4..11 {
            for x in 5..13 {
                image.get_mut(x, y).unwrap()[0] = 255;
            }
        }
        image
    }

    #[test]
    fn harris_and_shi_tomasi_find_square_corners() {
        let image = square();
        let harris = detect_harris(image.view(), HarrisOptions::default()).unwrap();
        let shi = detect_shi_tomasi(image.view(), ShiTomasiOptions::default()).unwrap();
        for corners in [&harris, &shi] {
            assert!(corners.iter().any(|point| point.x() <= 6.0 && point.y() <= 5.0));
            assert!(corners.iter().any(|point| point.x() >= 11.0 && point.y() >= 9.0));
        }
    }

    #[test]
    fn corner_detectors_accept_strided_roi() {
        let image = square();
        let roi = image.view().subview(ImageRegion::new(2, 2, 13, 11)).unwrap();
        assert!(!detect_harris(roi, HarrisOptions::default()).unwrap().is_empty());
    }

    #[test]
    fn fast_detects_contrast_corner_and_handles_tiny_images() {
        let mut image = Image::try_new(9, 9, vec![0; 81]).unwrap();
        image.get_mut(4, 4).unwrap()[0] = 255;
        let corners =
            detect_fast(image.view(), FastOptions { threshold: 20, nonmax_suppression: true })
                .unwrap();
        assert_eq!(corners.len(), 1);
        assert_eq!((corners[0].x(), corners[0].y()), (4.0, 4.0));
        assert!(corners.iter().all(|point| point.size() == 7.0));
        let tiny = Image::try_new(6, 6, vec![0; 36]).unwrap();
        assert!(detect_fast(tiny.view(), FastOptions::default()).unwrap().is_empty());
    }
}
