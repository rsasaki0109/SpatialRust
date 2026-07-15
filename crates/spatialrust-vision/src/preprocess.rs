use pulp::Arch;
use rayon::prelude::*;
use spatialrust_image::{
    ColorSpace, Image, ImageMetadata, ImageRegion, ImageView, ImageViewMut, PlanarImage,
};

use crate::{
    resize, BilinearResizeU8Plan, Interpolation, PixelComponent, VisionError, VisionResult,
};

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
    validate_normalization(scale, std)?;
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for pixel in input.row(y).expect("input row in bounds").chunks_exact(CHANNELS) {
            for channel in 0..CHANNELS {
                output
                    .push((pixel[channel].to_f64() as f32 * scale - mean[channel]) / std[channel]);
            }
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Normalizes interleaved pixels into caller-owned `f32` storage.
pub fn normalize_into<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    mut output: ImageViewMut<'_, f32, CHANNELS>,
    scale: f32,
    mean: [f32; CHANNELS],
    std: [f32; CHANNELS],
) -> VisionResult<()> {
    validate_normalization(scale, std)?;
    validate_output_dimensions(input, output.width(), output.height())?;
    output.set_metadata(input.metadata())?;
    for y in 0..input.height() {
        let source = input.row(y).expect("input row in bounds");
        let target = output.row_mut(y).expect("output row in bounds");
        for (source_pixel, target_pixel) in
            source.chunks_exact(CHANNELS).zip(target.chunks_exact_mut(CHANNELS))
        {
            for channel in 0..CHANNELS {
                target_pixel[channel] =
                    (source_pixel[channel].to_f64() as f32 * scale - mean[channel]) / std[channel];
            }
        }
    }
    Ok(())
}

/// Packs and normalizes an interleaved image into planar CHW storage.
pub fn pack_chw<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    scale: f32,
    mean: [f32; CHANNELS],
    std: [f32; CHANNELS],
) -> VisionResult<PlanarImage<f32, CHANNELS>> {
    let plane_len = input.width() * input.height();
    let mut output = vec![0.0_f32; plane_len * CHANNELS];
    pack_chw_into(input, scale, mean, std, &mut output)?;
    Ok(PlanarImage::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Packs and normalizes interleaved pixels into a reusable planar CHW slice.
///
/// Large multi-channel inputs dispatch across channels with safe scoped
/// threads. Smaller inputs remain scalar to avoid scheduling overhead.
pub fn pack_chw_into<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    scale: f32,
    mean: [f32; CHANNELS],
    std: [f32; CHANNELS],
    output: &mut [f32],
) -> VisionResult<()> {
    validate_normalization(scale, std)?;
    let plane_len = input.width().checked_mul(input.height()).ok_or_else(|| {
        VisionError::InvalidDimensions("CHW plane dimensions overflow".to_owned())
    })?;
    let required = plane_len.checked_mul(CHANNELS).ok_or_else(|| {
        VisionError::InvalidDimensions("CHW output dimensions overflow".to_owned())
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
    const PARALLEL_PLANE_THRESHOLD: usize = 256 * 1024;
    if CHANNELS > 1 && plane_len >= PARALLEL_PLANE_THRESHOLD {
        std::thread::scope(|scope| {
            for (channel, plane) in output.chunks_exact_mut(plane_len).enumerate() {
                scope.spawn(move || {
                    fill_chw_plane(input, channel, scale, mean[channel], std[channel], plane)
                });
            }
        });
    } else {
        for (channel, plane) in output.chunks_exact_mut(plane_len).enumerate() {
            fill_chw_plane(input, channel, scale, mean[channel], std[channel], plane);
        }
    }
    Ok(())
}

/// Fuses bilinear RGB resize, normalization, and CHW packing without an intermediate image.
pub fn resize_pack_chw(
    input: ImageView<'_, u8, 3>,
    output_width: usize,
    output_height: usize,
    scale: f32,
    mean: [f32; 3],
    std: [f32; 3],
) -> VisionResult<PlanarImage<f32, 3>> {
    BilinearResizeU8Plan::new(input.width(), input.height(), output_width, output_height)?
        .resize_rgb_to_chw(input, scale, mean, std)
}

/// Fuses bilinear RGB resize, normalization, and CHW packing into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn resize_pack_chw_into(
    input: ImageView<'_, u8, 3>,
    output_width: usize,
    output_height: usize,
    scale: f32,
    mean: [f32; 3],
    std: [f32; 3],
    output: &mut [f32],
) -> VisionResult<()> {
    BilinearResizeU8Plan::new(input.width(), input.height(), output_width, output_height)?
        .resize_rgb_to_chw_into(input, scale, mean, std, output)
}

fn fill_chw_plane<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    channel: usize,
    scale: f32,
    mean: f32,
    std: f32,
    output: &mut [f32],
) {
    for y in 0..input.height() {
        let row = input.row(y).expect("input row in bounds");
        for (x, pixel) in row.chunks_exact(CHANNELS).enumerate() {
            output[y * input.width() + x] = (pixel[channel].to_f64() as f32 * scale - mean) / std;
        }
    }
}

fn validate_normalization<const CHANNELS: usize>(
    scale: f32,
    std: [f32; CHANNELS],
) -> VisionResult<()> {
    if !scale.is_finite() || std.iter().any(|value| !value.is_finite() || *value == 0.0) {
        return Err(VisionError::InvalidParameter(
            "normalization scale/std must be finite and std non-zero".to_owned(),
        ));
    }
    Ok(())
}

fn validate_output_dimensions<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    output_width: usize,
    output_height: usize,
) -> VisionResult<()> {
    if (input.width(), input.height()) != (output_width, output_height) {
        return Err(VisionError::ShapeMismatch(format!(
            "output dimensions {output_width}x{output_height} do not match input {}x{}",
            input.width(),
            input.height()
        )));
    }
    Ok(())
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
    let metadata = ImageMetadata { color_space: ColorSpace::Gray, ..input.metadata() };
    let len = input
        .width()
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("gray output is too large".to_owned()))?;
    let mut output =
        Image::try_new_with_metadata(input.width(), input.height(), vec![0; len], metadata)?;
    rgb_to_gray_into(input, output.view_mut())?;
    Ok(output)
}

/// Converts RGB `u8` pixels into caller-owned gray storage without allocating.
pub fn rgb_to_gray_into(
    input: ImageView<'_, u8, 3>,
    mut output: ImageViewMut<'_, u8, 1>,
) -> VisionResult<()> {
    validate_output_dimensions(input, output.width(), output.height())?;
    output.set_metadata(ImageMetadata { color_space: ColorSpace::Gray, ..input.metadata() })?;
    let output_stride = output.row_stride();
    let pixels = input
        .width()
        .checked_mul(input.height())
        .ok_or_else(|| VisionError::InvalidDimensions("gray output is too large".to_owned()))?;
    let arch = Arch::new();
    if pixels >= 100_000 {
        if input.height() >= 2_000 {
            const ROWS_PER_TASK: usize = 8;
            let block_stride = output_stride.checked_mul(ROWS_PER_TASK).ok_or_else(|| {
                VisionError::InvalidDimensions("gray row block is too large".to_owned())
            })?;
            output.as_mut_slice().par_chunks_mut(block_stride).enumerate().for_each(
                |(block, rows)| {
                    arch.dispatch(|| {
                        for (row, target) in rows.chunks_mut(output_stride).enumerate() {
                            rgb_to_gray_row(input, block * ROWS_PER_TASK + row, target);
                        }
                    });
                },
            );
        } else {
            output
                .as_mut_slice()
                .par_chunks_mut(output_stride)
                .enumerate()
                .for_each(|(y, target)| arch.dispatch(|| rgb_to_gray_row(input, y, target)));
        }
    } else {
        output
            .as_mut_slice()
            .chunks_mut(output_stride)
            .enumerate()
            .for_each(|(y, target)| arch.dispatch(|| rgb_to_gray_row(input, y, target)));
    }
    Ok(())
}

/// Fuses bilinear RGB resize and BT.601 grayscale conversion without an intermediate RGB image.
pub fn resize_rgb_to_gray(
    input: ImageView<'_, u8, 3>,
    output_width: usize,
    output_height: usize,
) -> VisionResult<Image<u8, 1>> {
    BilinearResizeU8Plan::new(input.width(), input.height(), output_width, output_height)?
        .resize_rgb_to_gray(input)
}

/// Fuses bilinear RGB resize and BT.601 grayscale conversion into caller-owned storage.
pub fn resize_rgb_to_gray_into(
    input: ImageView<'_, u8, 3>,
    output: ImageViewMut<'_, u8, 1>,
) -> VisionResult<()> {
    BilinearResizeU8Plan::new(input.width(), input.height(), output.width(), output.height())?
        .resize_rgb_to_gray_into(input, output)
}

fn rgb_to_gray_row(input: ImageView<'_, u8, 3>, y: usize, target: &mut [u8]) {
    let source = input.row(y).expect("input row in bounds");
    for (pixel, target_value) in source.chunks_exact(3).zip(target.iter_mut()) {
        *target_value = rgb_luma(pixel);
    }
}

#[inline(always)]
fn rgb_luma(pixel: &[u8]) -> u8 {
    ((4_899_u32 * u32::from(pixel[0])
        + 9_617_u32 * u32::from(pixel[1])
        + 1_868_u32 * u32::from(pixel[2])
        + 8_192)
        >> 14) as u8
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
        crop, gray_to_rgb, letterbox, normalize, normalize_into, pack_chw, pack_chw_into, pad,
        resize_pack_chw, resize_pack_chw_into, resize_rgb_to_gray, resize_rgb_to_gray_into,
        rgb_to_gray, rgb_to_gray_into, rgb_to_hsv, Padding,
    };
    use crate::Interpolation;
    use spatialrust_image::{
        ColorSpace, Image, ImageMetadata, ImageRegion, ImageView, ImageViewMut,
    };

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
    fn reusable_normalize_and_chw_match_owned_outputs() {
        let image = Image::<u8, 3>::try_new(2, 1, vec![0, 10, 20, 30, 40, 50]).unwrap();
        let mut interleaved = vec![-1.0_f32; 9];
        let output = ImageViewMut::<f32, 3>::new(2, 1, 9, &mut interleaved).unwrap();
        normalize_into(image.view(), output, 0.1, [0.0; 3], [1.0; 3]).unwrap();
        assert_eq!(
            &interleaved[..6],
            normalize(image.view(), 0.1, [0.0; 3], [1.0; 3]).unwrap().as_slice()
        );
        assert_eq!(&interleaved[6..], &[-1.0; 3]);

        let mut chw = vec![0.0; 6];
        pack_chw_into(image.view(), 0.1, [0.0; 3], [1.0; 3], &mut chw).unwrap();
        assert_eq!(chw, pack_chw(image.view(), 0.1, [0.0; 3], [1.0; 3]).unwrap().as_slice());
        assert!(pack_chw_into(image.view(), 1.0, [0.0; 3], [1.0; 3], &mut chw[..5]).is_err());

        let empty = Image::<u8, 3>::try_new(0, 0, Vec::new()).unwrap();
        pack_chw_into(empty.view(), 1.0, [0.0; 3], [1.0; 3], &mut []).unwrap();
    }

    #[test]
    fn large_chw_parallel_dispatch_matches_channel_layout() {
        let image = Image::<u8, 3>::try_new(512, 512, vec![10; 512 * 512 * 3]).unwrap();
        let mut chw = vec![0.0; 512 * 512 * 3];
        pack_chw_into(image.view(), 0.1, [0.0, 0.5, 1.0], [1.0; 3], &mut chw).unwrap();
        let plane = 512 * 512;
        assert!(chw[..plane].iter().all(|&value| value == 1.0));
        assert!(chw[plane..2 * plane].iter().all(|&value| value == 0.5));
        assert!(chw[2 * plane..].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn fused_resize_chw_matches_unfused_general_and_preserves_metadata() {
        let metadata = ImageMetadata {
            color_space: ColorSpace::Rgb,
            color_range: spatialrust_image::ColorRange::Full,
            ..ImageMetadata::default()
        };
        let input = Image::<u8, 3>::try_new_with_metadata(
            7,
            5,
            (0..105).map(|value| (value * 37 % 256) as u8).collect(),
            metadata,
        )
        .unwrap();
        let plan = crate::BilinearResizeU8Plan::new(7, 5, 11, 8).unwrap();
        let resized = plan.resize(input.view()).unwrap();
        let expected =
            pack_chw(resized.view(), 1.0 / 255.0, [0.1, 0.2, 0.3], [0.5, 1.0, 2.0]).unwrap();
        let actual =
            resize_pack_chw(input.view(), 11, 8, 1.0 / 255.0, [0.1, 0.2, 0.3], [0.5, 1.0, 2.0])
                .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.metadata(), metadata);
    }

    #[test]
    fn fused_resize_chw_matches_unfused_half_scale_for_strided_input_and_reuse() {
        let mut storage = vec![231_u8; 169];
        for y in 0..6 {
            for x in 0..24 {
                storage[y * 29 + x] = (y * 41 + x * 13) as u8;
            }
        }
        let input = ImageView::<u8, 3>::new(8, 6, 29, &storage).unwrap();
        let plan = crate::BilinearResizeU8Plan::new(8, 6, 4, 3).unwrap();
        let resized = plan.resize(input).unwrap();
        let expected = pack_chw(resized.view(), 0.25, [1.0, 2.0, 3.0], [1.0; 3]).unwrap();
        let mut output = vec![-1.0_f32; 36];
        resize_pack_chw_into(input, 4, 3, 0.25, [1.0, 2.0, 3.0], [1.0; 3], &mut output).unwrap();
        assert_eq!(output, expected.as_slice());
        assert!(
            resize_pack_chw_into(input, 4, 3, 0.25, [0.0; 3], [1.0; 3], &mut output[..35]).is_err()
        );
    }

    #[test]
    fn rgb_to_gray_into_accepts_strided_output() {
        let image = Image::<u8, 3>::try_new(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap();
        let mut storage = vec![200_u8; 4];
        let output = ImageViewMut::<u8, 1>::new(2, 1, 4, &mut storage).unwrap();
        rgb_to_gray_into(image.view(), output).unwrap();
        assert_eq!(&storage[..2], rgb_to_gray(image.view()).unwrap().as_slice());
        assert_eq!(&storage[2..], &[200, 200]);
    }

    #[test]
    fn rgb_to_gray_accepts_strided_input_and_preserves_metadata() {
        let storage = [255, 0, 0, 0, 255, 0, 91, 92, 0, 0, 255, 255, 255, 255];
        let metadata = ImageMetadata { color_space: ColorSpace::Rgb, ..Default::default() };
        let input = ImageView::<u8, 3>::new_with_metadata(2, 2, 8, &storage, metadata).unwrap();
        let gray = rgb_to_gray(input).unwrap();
        assert_eq!(gray.as_slice(), &[76, 150, 29, 255]);
        assert_eq!(gray.metadata().color_space, ColorSpace::Gray);
    }

    #[test]
    fn fused_resize_rgb_to_gray_matches_unfused_half_scale_exactly() {
        let metadata = ImageMetadata {
            color_space: ColorSpace::Rgb,
            color_range: spatialrust_image::ColorRange::Full,
            ..ImageMetadata::default()
        };
        let input = Image::<u8, 3>::try_new_with_metadata(
            8,
            6,
            (0..144).map(|value| (value * 53 % 256) as u8).collect(),
            metadata,
        )
        .unwrap();
        let resized =
            crate::BilinearResizeU8Plan::new(8, 6, 4, 3).unwrap().resize(input.view()).unwrap();
        let expected = rgb_to_gray(resized.view()).unwrap();
        let actual = resize_rgb_to_gray(input.view(), 4, 3).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.metadata().color_space, ColorSpace::Gray);
        assert_eq!(actual.metadata().color_range, spatialrust_image::ColorRange::Full);
    }

    #[test]
    fn fused_resize_rgb_to_gray_matches_unfused_general_and_strided_output() {
        let mut input_storage = vec![231_u8; 83];
        for y in 0..5 {
            for x in 0..15 {
                input_storage[y * 17 + x] = (y * 41 + x * 13) as u8;
            }
        }
        let input = ImageView::<u8, 3>::new(5, 5, 17, &input_storage).unwrap();
        let resized = crate::BilinearResizeU8Plan::new(5, 5, 7, 3).unwrap().resize(input).unwrap();
        let expected = rgb_to_gray(resized.view()).unwrap();
        let mut output_storage = vec![199_u8; 29];
        let output = ImageViewMut::<u8, 1>::new(7, 3, 11, &mut output_storage).unwrap();
        resize_rgb_to_gray_into(input, output).unwrap();
        for y in 0..3 {
            assert_eq!(&output_storage[y * 11..y * 11 + 7], expected.view().row(y).unwrap());
            if y < 2 {
                assert_eq!(&output_storage[y * 11 + 7..(y + 1) * 11], &[199; 4]);
            }
        }
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
        assert_eq!(rgb_to_gray(image.view()).unwrap().as_slice(), &[76, 150, 29]);
        assert_eq!(
            rgb_to_hsv(image.view()).unwrap().as_slice(),
            &[0, 255, 255, 60, 255, 255, 120, 255, 255]
        );
        let gray = Image::<u8, 1>::try_new(1, 1, vec![12]).unwrap();
        assert_eq!(gray_to_rgb(gray.view()).unwrap().as_slice(), &[12, 12, 12]);
    }
}
