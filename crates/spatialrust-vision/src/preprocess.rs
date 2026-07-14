use spatialrust_image::{ColorSpace, Image, ImageMetadata, ImageRegion, ImageView, PlanarImage};

use crate::{resize, Interpolation, PixelComponent, VisionError, VisionResult};

/// Padding applied around an image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Padding {
    /// Columns before the source image.
    pub left: usize,
    /// Columns after the source image.
    pub right: usize,
    /// Rows before the source image.
    pub top: usize,
    /// Rows after the source image.
    pub bottom: usize,
}

/// Mapping between a source image and its letterboxed output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LetterboxTransform {
    /// Uniform source-to-output scale.
    pub scale: f64,
    /// Left padding in output pixels.
    pub pad_left: usize,
    /// Top padding in output pixels.
    pub pad_top: usize,
    /// Resized content width.
    pub content_width: usize,
    /// Resized content height.
    pub content_height: usize,
    /// Final output width.
    pub output_width: usize,
    /// Final output height.
    pub output_height: usize,
}

impl LetterboxTransform {
    /// Maps a source pixel coordinate into letterboxed output coordinates.
    #[must_use]
    pub fn map_point(self, x: f64, y: f64) -> (f64, f64) {
        (x.mul_add(self.scale, self.pad_left as f64), y.mul_add(self.scale, self.pad_top as f64))
    }

    /// Maps an output coordinate back into the source image.
    #[must_use]
    pub fn unmap_point(self, x: f64, y: f64) -> (f64, f64) {
        ((x - self.pad_left as f64) / self.scale, (y - self.pad_top as f64) / self.scale)
    }
}

/// Copies a checked rectangular region into a packed image.
pub fn crop<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    region: ImageRegion,
) -> VisionResult<Image<T, CHANNELS>> {
    let view = input.subview(region)?;
    let mut data = Vec::with_capacity(region.width * region.height * CHANNELS);
    for y in 0..region.height {
        data.extend_from_slice(view.row(y).expect("validated crop row"));
    }
    Ok(Image::try_new_with_metadata(region.width, region.height, data, input.metadata())?)
}

/// Pads an image with a constant pixel value.
pub fn pad<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    padding: Padding,
    value: [T; CHANNELS],
) -> VisionResult<Image<T, CHANNELS>> {
    let width = input
        .width()
        .checked_add(padding.left)
        .and_then(|value| value.checked_add(padding.right))
        .ok_or_else(|| VisionError::InvalidDimensions("padding width overflow".to_owned()))?;
    let height = input
        .height()
        .checked_add(padding.top)
        .and_then(|value| value.checked_add(padding.bottom))
        .ok_or_else(|| VisionError::InvalidDimensions("padding height overflow".to_owned()))?;
    let mut output = Vec::with_capacity(width * height * CHANNELS);
    for y in 0..height {
        for x in 0..width {
            if x >= padding.left
                && x < padding.left + input.width()
                && y >= padding.top
                && y < padding.top + input.height()
            {
                output.extend_from_slice(
                    input.get(x - padding.left, y - padding.top).expect("validated pad coordinate"),
                );
            } else {
                output.extend_from_slice(&value);
            }
        }
    }
    Ok(Image::try_new_with_metadata(width, height, output, input.metadata())?)
}

/// Aspect-preserving resize followed by centered constant padding.
pub fn letterbox<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    output_width: usize,
    output_height: usize,
    interpolation: Interpolation,
    value: [T; CHANNELS],
) -> VisionResult<(Image<T, CHANNELS>, LetterboxTransform)> {
    if input.width() == 0 || input.height() == 0 || output_width == 0 || output_height == 0 {
        return Err(VisionError::InvalidDimensions(
            "letterbox requires non-empty input and output".to_owned(),
        ));
    }
    let scale = (output_width as f64 / input.width() as f64)
        .min(output_height as f64 / input.height() as f64);
    let content_width = ((input.width() as f64 * scale).round() as usize).clamp(1, output_width);
    let content_height = ((input.height() as f64 * scale).round() as usize).clamp(1, output_height);
    let resized = resize(input, content_width, content_height, interpolation)?;
    let remaining_x = output_width - content_width;
    let remaining_y = output_height - content_height;
    let padding = Padding {
        left: remaining_x / 2,
        right: remaining_x - remaining_x / 2,
        top: remaining_y / 2,
        bottom: remaining_y - remaining_y / 2,
    };
    let transform = LetterboxTransform {
        scale,
        pad_left: padding.left,
        pad_top: padding.top,
        content_width,
        content_height,
        output_width,
        output_height,
    };
    Ok((pad(resized.view(), padding, value)?, transform))
}

/// Converts packed scalar values to `f32`, then applies `value * scale`, mean
/// subtraction, and per-channel standard-deviation normalization.
pub fn normalize<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    scale: f32,
    mean: [f32; CHANNELS],
    std: [f32; CHANNELS],
) -> VisionResult<Image<f32, CHANNELS>> {
    if !scale.is_finite() || std.iter().any(|value| !value.is_finite() || *value == 0.0) {
        return Err(VisionError::InvalidParameter(
            "normalization scale/std must be finite and std non-zero".to_owned(),
        ));
    }
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            let pixel = input.get(x, y).expect("image coordinate in bounds");
            for channel in 0..CHANNELS {
                output
                    .push((pixel[channel].to_f64() as f32 * scale - mean[channel]) / std[channel]);
            }
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Packs and normalizes an interleaved image into planar CHW storage.
pub fn pack_chw<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    scale: f32,
    mean: [f32; CHANNELS],
    std: [f32; CHANNELS],
) -> VisionResult<PlanarImage<f32, CHANNELS>> {
    if !scale.is_finite() || std.iter().any(|value| !value.is_finite() || *value == 0.0) {
        return Err(VisionError::InvalidParameter(
            "normalization scale/std must be finite and std non-zero".to_owned(),
        ));
    }
    let plane_len = input.width() * input.height();
    let mut output = vec![0.0_f32; plane_len * CHANNELS];
    for y in 0..input.height() {
        for x in 0..input.width() {
            let pixel = input.get(x, y).expect("image coordinate in bounds");
            let index = y * input.width() + x;
            for channel in 0..CHANNELS {
                output[channel * plane_len + index] =
                    (pixel[channel].to_f64() as f32 * scale - mean[channel]) / std[channel];
            }
        }
    }
    Ok(PlanarImage::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Swaps the red and blue channels of a three-channel image.
pub fn swap_red_blue<T: PixelComponent>(input: ImageView<'_, T, 3>) -> VisionResult<Image<T, 3>> {
    let mut data = Vec::with_capacity(input.width() * input.height() * 3);
    for y in 0..input.height() {
        for x in 0..input.width() {
            let pixel = input.get(x, y).expect("image coordinate in bounds");
            data.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
        }
    }
    let mut metadata = input.metadata();
    metadata.color_space = match metadata.color_space {
        ColorSpace::Rgb => ColorSpace::Bgr,
        ColorSpace::Bgr => ColorSpace::Rgb,
        other => other,
    };
    Ok(Image::try_new_with_metadata(input.width(), input.height(), data, metadata)?)
}

/// Converts RGB `u8` pixels to BT.601 luma using fixed-point coefficients.
pub fn rgb_to_gray(input: ImageView<'_, u8, 3>) -> VisionResult<Image<u8, 1>> {
    let mut data = Vec::with_capacity(input.width() * input.height());
    for y in 0..input.height() {
        for x in 0..input.width() {
            let pixel = input.get(x, y).expect("image coordinate in bounds");
            let value = (77_u32 * u32::from(pixel[0])
                + 150_u32 * u32::from(pixel[1])
                + 29_u32 * u32::from(pixel[2])
                + 128)
                >> 8;
            data.push(value as u8);
        }
    }
    let metadata = ImageMetadata { color_space: ColorSpace::Gray, ..input.metadata() };
    Ok(Image::try_new_with_metadata(input.width(), input.height(), data, metadata)?)
}

/// Replicates a gray channel into RGB.
pub fn gray_to_rgb<T: PixelComponent>(input: ImageView<'_, T, 1>) -> VisionResult<Image<T, 3>> {
    let mut data = Vec::with_capacity(input.width() * input.height() * 3);
    for y in 0..input.height() {
        for x in 0..input.width() {
            let value = input.get(x, y).expect("image coordinate in bounds")[0];
            data.extend_from_slice(&[value, value, value]);
        }
    }
    let metadata = ImageMetadata { color_space: ColorSpace::Rgb, ..input.metadata() };
    Ok(Image::try_new_with_metadata(input.width(), input.height(), data, metadata)?)
}

/// Converts RGB `u8` to OpenCV-style HSV (`H` in `0..=179`, `S/V` in `0..=255`).
pub fn rgb_to_hsv(input: ImageView<'_, u8, 3>) -> VisionResult<Image<u8, 3>> {
    let mut data = Vec::with_capacity(input.width() * input.height() * 3);
    for y in 0..input.height() {
        for x in 0..input.width() {
            let [r, g, b] = *input.get(x, y).expect("image coordinate in bounds");
            let rf = f64::from(r) / 255.0;
            let gf = f64::from(g) / 255.0;
            let bf = f64::from(b) / 255.0;
            let max = rf.max(gf).max(bf);
            let min = rf.min(gf).min(bf);
            let delta = max - min;
            let mut hue = if delta == 0.0 {
                0.0
            } else if max == rf {
                60.0 * ((gf - bf) / delta).rem_euclid(6.0)
            } else if max == gf {
                60.0 * ((bf - rf) / delta + 2.0)
            } else {
                60.0 * ((rf - gf) / delta + 4.0)
            };
            if hue >= 360.0 {
                hue = 0.0;
            }
            let saturation = if max == 0.0 { 0.0 } else { delta / max };
            data.extend_from_slice(&[
                (hue / 2.0).round().clamp(0.0, 179.0) as u8,
                (saturation * 255.0).round() as u8,
                (max * 255.0).round() as u8,
            ]);
        }
    }
    let metadata = ImageMetadata { color_space: ColorSpace::Hsv, ..input.metadata() };
    Ok(Image::try_new_with_metadata(input.width(), input.height(), data, metadata)?)
}

#[cfg(test)]
mod tests {
    use super::{
        crop, gray_to_rgb, letterbox, normalize, pack_chw, pad, rgb_to_gray, rgb_to_hsv, Padding,
    };
    use crate::Interpolation;
    use spatialrust_image::{ColorSpace, Image, ImageMetadata, ImageRegion};

    #[test]
    fn crop_and_pad_roundtrip_center() {
        let image = Image::<u8, 1>::try_new(3, 2, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let cropped = crop(image.view(), ImageRegion::new(1, 0, 2, 2)).unwrap();
        assert_eq!(cropped.as_slice(), &[2, 3, 5, 6]);
        let padded =
            pad(cropped.view(), Padding { left: 1, right: 0, top: 1, bottom: 0 }, [0]).unwrap();
        assert_eq!(padded.as_slice(), &[0, 0, 0, 0, 2, 3, 0, 5, 6]);
    }

    #[test]
    fn letterbox_preserves_aspect_and_maps_points() {
        let image = Image::<u8, 1>::try_new(4, 2, vec![1; 8]).unwrap();
        let (output, transform) =
            letterbox(image.view(), 8, 8, Interpolation::Nearest, [0]).unwrap();
        assert_eq!((output.width(), output.height()), (8, 8));
        assert_eq!((transform.content_width, transform.content_height), (8, 4));
        assert_eq!(transform.pad_top, 2);
        let mapped = transform.map_point(1.0, 1.0);
        assert_eq!(transform.unmap_point(mapped.0, mapped.1), (1.0, 1.0));
    }

    #[test]
    fn normalize_and_chw_have_expected_layout() {
        let image = Image::<u8, 3>::try_new(2, 1, vec![0, 10, 20, 30, 40, 50]).unwrap();
        let normalized = normalize(image.view(), 0.1, [0.0; 3], [1.0; 3]).unwrap();
        assert_eq!(normalized.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let chw = pack_chw(image.view(), 0.1, [0.0; 3], [1.0; 3]).unwrap();
        assert_eq!(chw.as_slice(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn color_conversions_match_known_primaries() {
        let metadata = ImageMetadata { color_space: ColorSpace::Rgb, ..Default::default() };
        let image = Image::<u8, 3>::try_new_with_metadata(
            3,
            1,
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255],
            metadata,
        )
        .unwrap();
        assert_eq!(rgb_to_gray(image.view()).unwrap().as_slice(), &[77, 149, 29]);
        assert_eq!(
            rgb_to_hsv(image.view()).unwrap().as_slice(),
            &[0, 255, 255, 60, 255, 255, 120, 255, 255]
        );
        let gray = Image::<u8, 1>::try_new(1, 1, vec![12]).unwrap();
        assert_eq!(gray_to_rgb(gray.view()).unwrap().as_slice(), &[12, 12, 12]);
    }
}
