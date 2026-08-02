//! Runtime-free computational photography and pairwise panorama composition.

use spatialrust_image::{Image, ImageView};

use crate::{
    estimate_homography_ransac, PerspectiveTransform, PointCorrespondence2,
    RobustEstimationOptions, VisionError, VisionResult,
};

/// Applies deterministic gray-world white balance to an RGB image.
pub fn gray_world_white_balance(input: ImageView<'_, u8, 3>) -> VisionResult<Image<u8, 3>> {
    if input.width() == 0 || input.height() == 0 {
        return Ok(Image::try_new_with_metadata(0, 0, Vec::new(), input.metadata())?);
    }
    let pixels = input.width().saturating_mul(input.height()) as f64;
    let mut sums = [0.0; 3];
    for y in 0..input.height() {
        let row = input.row(y).expect("validated RGB row");
        for pixel in row.chunks_exact(3) {
            for channel in 0..3 {
                sums[channel] += f64::from(pixel[channel]);
            }
        }
    }
    let means = sums.map(|sum| sum / pixels);
    let target = means.iter().sum::<f64>() / 3.0;
    let gains = means.map(|mean| if mean > 0.0 { target / mean } else { 1.0 });
    let mut data = Vec::with_capacity(input.width() * input.height() * 3);
    for y in 0..input.height() {
        for (index, &value) in input.row(y).expect("validated RGB row").iter().enumerate() {
            data.push((f64::from(value) * gains[index % 3]).round().clamp(0.0, 255.0) as u8);
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), data, input.metadata())?)
}

/// Well-exposedness fusion controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureFusionOptions {
    /// Standard deviation around middle gray in normalized intensity.
    pub well_exposed_sigma: f64,
    /// Positive floor preventing completely black weights.
    pub weight_floor: f64,
}

impl Default for ExposureFusionOptions {
    fn default() -> Self {
        Self { well_exposed_sigma: 0.2, weight_floor: 1e-6 }
    }
}

/// Fuses aligned RGB exposures with normalized per-pixel well-exposedness.
pub fn fuse_exposures(
    inputs: &[ImageView<'_, u8, 3>],
    options: ExposureFusionOptions,
) -> VisionResult<Image<u8, 3>> {
    let first = *inputs.first().ok_or_else(|| {
        VisionError::InvalidParameter("exposure fusion requires at least one image".into())
    })?;
    if !options.well_exposed_sigma.is_finite()
        || options.well_exposed_sigma <= 0.0
        || !options.weight_floor.is_finite()
        || options.weight_floor <= 0.0
    {
        return Err(VisionError::InvalidParameter("invalid exposure fusion weights".into()));
    }
    if inputs.iter().any(|image| {
        image.width() != first.width()
            || image.height() != first.height()
            || image.metadata() != first.metadata()
    }) {
        return Err(VisionError::ShapeMismatch(
            "exposure inputs must share dimensions and metadata".into(),
        ));
    }
    let mut output = Vec::with_capacity(first.width() * first.height() * 3);
    let sigma2 = 2.0 * options.well_exposed_sigma.powi(2);
    for y in 0..first.height() {
        for x in 0..first.width() {
            let mut weighted = [0.0; 3];
            let mut total = 0.0;
            for image in inputs {
                let pixel = image.get(x, y).expect("validated coordinates");
                let luminance = (0.2126 * f64::from(pixel[0])
                    + 0.7152 * f64::from(pixel[1])
                    + 0.0722 * f64::from(pixel[2]))
                    / 255.0;
                let weight = (-(luminance - 0.5).powi(2) / sigma2).exp() + options.weight_floor;
                for channel in 0..3 {
                    weighted[channel] += f64::from(pixel[channel]) * weight;
                }
                total += weight;
            }
            output.extend(weighted.map(|value| (value / total).round().clamp(0.0, 255.0) as u8));
        }
    }
    Ok(Image::try_new_with_metadata(first.width(), first.height(), output, first.metadata())?)
}

/// Bounded pairwise panorama settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanoramaOptions {
    /// Hard ceiling for output pixels before allocation.
    pub max_output_pixels: usize,
}

impl Default for PanoramaOptions {
    fn default() -> Self {
        Self { max_output_pixels: 64 * 1024 * 1024 }
    }
}

/// Pairwise panorama and its integer world-coordinate origin.
#[derive(Clone, Debug, PartialEq)]
pub struct Panorama {
    image: Image<u8, 3>,
    origin_x: i32,
    origin_y: i32,
}

impl Panorama {
    /// Returns the blended RGB canvas.
    pub const fn image(&self) -> &Image<u8, 3> {
        &self.image
    }
    /// Returns the right-image world x coordinate represented by canvas x=0.
    pub const fn origin_x(&self) -> i32 {
        self.origin_x
    }
    /// Returns the right-image world y coordinate represented by canvas y=0.
    pub const fn origin_y(&self) -> i32 {
        self.origin_y
    }
}

/// Estimates a source-to-target homography and stitches the pair.
pub fn estimate_and_stitch_panorama(
    source: ImageView<'_, u8, 3>,
    target: ImageView<'_, u8, 3>,
    correspondences: &[PointCorrespondence2],
    robust: RobustEstimationOptions,
    options: PanoramaOptions,
) -> VisionResult<Panorama> {
    let estimate = estimate_homography_ransac(correspondences, robust)?;
    stitch_panorama_pair(
        source,
        target,
        PerspectiveTransform { matrix: estimate.model().matrix().m },
        options,
    )
}

/// Warps `source` into `target` coordinates and feather-blends their overlap.
pub fn stitch_panorama_pair(
    source: ImageView<'_, u8, 3>,
    target: ImageView<'_, u8, 3>,
    source_to_target: PerspectiveTransform,
    options: PanoramaOptions,
) -> VisionResult<Panorama> {
    if source.metadata() != target.metadata() {
        return Err(VisionError::ShapeMismatch("panorama images must share color metadata".into()));
    }
    if source.width() == 0 || source.height() == 0 || target.width() == 0 || target.height() == 0 {
        return Err(VisionError::InvalidDimensions("panorama inputs must be non-empty".into()));
    }
    if options.max_output_pixels == 0 {
        return Err(VisionError::InvalidParameter("panorama pixel budget must be positive".into()));
    }
    let mut corners = vec![
        (0.0, 0.0),
        ((target.width() - 1) as f64, 0.0),
        (0.0, (target.height() - 1) as f64),
        ((target.width() - 1) as f64, (target.height() - 1) as f64),
    ];
    for &(x, y) in &[
        (0.0, 0.0),
        ((source.width() - 1) as f64, 0.0),
        (0.0, (source.height() - 1) as f64),
        ((source.width() - 1) as f64, (source.height() - 1) as f64),
    ] {
        corners.push(source_to_target.map_point(x, y).ok_or(VisionError::SingularTransform)?);
    }
    let min_x = corners.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).floor() as i32;
    let min_y = corners.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).floor() as i32;
    let max_x = corners.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max).ceil() as i32;
    let max_y = corners.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max).ceil() as i32;
    let width = usize::try_from(max_x - min_x + 1).map_err(|_| {
        VisionError::InvalidDimensions("panorama width is not representable".into())
    })?;
    let height = usize::try_from(max_y - min_y + 1).map_err(|_| {
        VisionError::InvalidDimensions("panorama height is not representable".into())
    })?;
    if width.checked_mul(height).filter(|&pixels| pixels <= options.max_output_pixels).is_none() {
        return Err(VisionError::InvalidDimensions("panorama exceeds output pixel budget".into()));
    }
    let inverse = source_to_target.inverse()?;
    let mut data = Vec::with_capacity(width * height * 3);
    for canvas_y in 0..height {
        for canvas_x in 0..width {
            let world_x = f64::from(min_x) + canvas_x as f64;
            let world_y = f64::from(min_y) + canvas_y as f64;
            let target_sample = sample_inside(target, world_x, world_y);
            let source_sample =
                inverse.map_point(world_x, world_y).and_then(|(x, y)| sample_inside(source, x, y));
            let pixel = match (source_sample, target_sample) {
                (Some((left, lw)), Some((right, rw))) => {
                    let total = lw + rw;
                    std::array::from_fn(|channel| {
                        ((f64::from(left[channel]) * lw + f64::from(right[channel]) * rw) / total)
                            .round()
                            .clamp(0.0, 255.0) as u8
                    })
                }
                (Some((pixel, _)), None) | (None, Some((pixel, _))) => pixel,
                (None, None) => [0; 3],
            };
            data.extend_from_slice(&pixel);
        }
    }
    Ok(Panorama {
        image: Image::try_new_with_metadata(width, height, data, target.metadata())?,
        origin_x: min_x,
        origin_y: min_y,
    })
}

fn sample_inside(image: ImageView<'_, u8, 3>, x: f64, y: f64) -> Option<([u8; 3], f64)> {
    if x < 0.0 || y < 0.0 || x > (image.width() - 1) as f64 || y > (image.height() - 1) as f64 {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let wx = x - x0 as f64;
    let wy = y - y0 as f64;
    let pixel = std::array::from_fn(|channel| {
        let p00 = f64::from(image.get(x0, y0).unwrap()[channel]);
        let p10 = f64::from(image.get(x1, y0).unwrap()[channel]);
        let p01 = f64::from(image.get(x0, y1).unwrap()[channel]);
        let p11 = f64::from(image.get(x1, y1).unwrap()[channel]);
        let top = p00 * (1.0 - wx) + p10 * wx;
        let bottom = p01 * (1.0 - wx) + p11 * wx;
        (top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8
    });
    let edge =
        x.min(y).min((image.width() - 1) as f64 - x).min((image.height() - 1) as f64 - y) + 1.0;
    Some((pixel, edge.max(1e-6)))
}

#[cfg(test)]
mod tests {
    use super::{
        fuse_exposures, gray_world_white_balance, stitch_panorama_pair, ExposureFusionOptions,
        PanoramaOptions,
    };
    use crate::PerspectiveTransform;
    use spatialrust_image::Image;

    #[test]
    fn white_balance_equalizes_channel_means() {
        let image = Image::try_new(2, 1, vec![40, 80, 120, 20, 40, 60]).unwrap();
        let balanced = gray_world_white_balance(image.view()).unwrap();
        let sums = [
            balanced.as_slice()[0] as u16 + balanced.as_slice()[3] as u16,
            balanced.as_slice()[1] as u16 + balanced.as_slice()[4] as u16,
            balanced.as_slice()[2] as u16 + balanced.as_slice()[5] as u16,
        ];
        assert!(sums.iter().max().unwrap() - sums.iter().min().unwrap() <= 1);
    }

    #[test]
    fn exposure_fusion_prefers_middle_gray() {
        let dark = Image::from_pixel(1, 1, [10, 10, 10]).unwrap();
        let middle = Image::from_pixel(1, 1, [128, 128, 128]).unwrap();
        let fused = fuse_exposures(&[dark.view(), middle.view()], ExposureFusionOptions::default())
            .unwrap();
        assert!(fused.as_slice()[0] >= 120);
    }

    #[test]
    fn translated_pair_expands_canvas_and_preserves_sides() {
        let source = Image::from_pixel(3, 2, [200, 0, 0]).unwrap();
        let target = Image::from_pixel(3, 2, [0, 0, 200]).unwrap();
        let panorama = stitch_panorama_pair(
            source.view(),
            target.view(),
            PerspectiveTransform { matrix: [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] },
            PanoramaOptions::default(),
        )
        .unwrap();
        assert_eq!((panorama.image().width(), panorama.image().height()), (5, 2));
        assert_eq!(panorama.image().get(0, 0).unwrap(), &[0, 0, 200]);
        assert_eq!(panorama.image().get(4, 0).unwrap(), &[200, 0, 0]);
        assert!(stitch_panorama_pair(
            source.view(),
            target.view(),
            PerspectiveTransform::identity(),
            PanoramaOptions { max_output_pixels: 5 },
        )
        .is_err());
    }
}
