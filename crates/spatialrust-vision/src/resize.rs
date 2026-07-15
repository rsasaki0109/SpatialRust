use spatialrust_image::{Image, ImageView, ImageViewMut};

use crate::{PixelComponent, VisionError, VisionResult};

/// Sampling filter used by image resampling and geometric warps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Interpolation {
    /// Closest source pixel using half-pixel coordinate mapping.
    Nearest,
    /// Bilinear interpolation using half-pixel coordinate mapping.
    #[default]
    Bilinear,
    /// Bicubic interpolation with OpenCV-compatible cubic coefficient `-0.75`.
    Bicubic,
    /// Pixel-area integration when shrinking; bilinear when enlarging.
    Area,
}

/// Resizes an interleaved image while preserving semantic metadata.
pub fn resize<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    output_width: usize,
    output_height: usize,
    interpolation: Interpolation,
) -> VisionResult<Image<T, CHANNELS>> {
    if output_width == 0 || output_height == 0 {
        return Image::try_new_with_metadata(
            output_width,
            output_height,
            Vec::new(),
            input.metadata(),
        )
        .map_err(Into::into);
    }
    if input.width() == 0 || input.height() == 0 {
        return Err(VisionError::InvalidDimensions(
            "cannot resize an empty input to a non-empty output".to_owned(),
        ));
    }

    let mut output = Vec::with_capacity(output_width * output_height * CHANNELS);
    let area_downsample = interpolation == Interpolation::Area
        && (output_width < input.width() || output_height < input.height());
    for y in 0..output_height {
        for x in 0..output_width {
            let pixel = resized_pixel(
                input,
                x,
                y,
                output_width,
                output_height,
                interpolation,
                area_downsample,
            );
            output.extend_from_slice(&pixel);
        }
    }
    Ok(Image::try_new_with_metadata(output_width, output_height, output, input.metadata())?)
}

/// Resizes into caller-owned storage without allocating.
///
/// The output dimensions select the requested size. Packed and strided output
/// views are accepted, and semantic metadata is replaced with the input
/// metadata after channel validation.
pub fn resize_into<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    mut output: ImageViewMut<'_, T, CHANNELS>,
    interpolation: Interpolation,
) -> VisionResult<()> {
    let output_width = output.width();
    let output_height = output.height();
    output.set_metadata(input.metadata())?;
    if output_width == 0 || output_height == 0 {
        return Ok(());
    }
    if input.width() == 0 || input.height() == 0 {
        return Err(VisionError::InvalidDimensions(
            "cannot resize an empty input to a non-empty output".to_owned(),
        ));
    }
    let area_downsample = interpolation == Interpolation::Area
        && (output_width < input.width() || output_height < input.height());
    for y in 0..output_height {
        let row = output.row_mut(y).expect("output row in bounds");
        for x in 0..output_width {
            let pixel = resized_pixel(
                input,
                x,
                y,
                output_width,
                output_height,
                interpolation,
                area_downsample,
            );
            row[x * CHANNELS..(x + 1) * CHANNELS].copy_from_slice(&pixel);
        }
    }
    Ok(())
}

fn resized_pixel<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: usize,
    y: usize,
    output_width: usize,
    output_height: usize,
    interpolation: Interpolation,
    area_downsample: bool,
) -> [T; CHANNELS] {
    if area_downsample {
        sample_area(input, x, y, output_width, output_height)
    } else {
        let sx = half_pixel_coordinate(x, input.width(), output_width);
        let sy = half_pixel_coordinate(y, input.height(), output_height);
        match interpolation {
            Interpolation::Nearest => sample_nearest(input, sx, sy),
            Interpolation::Bilinear | Interpolation::Area => sample_bilinear(input, sx, sy),
            Interpolation::Bicubic => sample_bicubic(input, sx, sy),
        }
    }
}

fn half_pixel_coordinate(output: usize, input_len: usize, output_len: usize) -> f64 {
    (output as f64 + 0.5) * input_len as f64 / output_len as f64 - 0.5
}

fn clamped_index(value: isize, length: usize) -> usize {
    value.clamp(0, length.saturating_sub(1) as isize) as usize
}

pub(crate) fn sample_nearest<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: f64,
    y: f64,
) -> [T; CHANNELS] {
    let ix = clamped_index(x.round() as isize, input.width());
    let iy = clamped_index(y.round() as isize, input.height());
    *input.get(ix, iy).expect("clamped resize coordinate")
}

pub(crate) fn sample_bilinear<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: f64,
    y: f64,
) -> [T; CHANNELS] {
    let x0_raw = x.floor() as isize;
    let y0_raw = y.floor() as isize;
    let x1_raw = x0_raw + 1;
    let y1_raw = y0_raw + 1;
    let wx = x - x.floor();
    let wy = y - y.floor();
    let x0 = clamped_index(x0_raw, input.width());
    let x1 = clamped_index(x1_raw, input.width());
    let y0 = clamped_index(y0_raw, input.height());
    let y1 = clamped_index(y1_raw, input.height());
    let p00 = input.get(x0, y0).expect("clamped resize coordinate");
    let p10 = input.get(x1, y0).expect("clamped resize coordinate");
    let p01 = input.get(x0, y1).expect("clamped resize coordinate");
    let p11 = input.get(x1, y1).expect("clamped resize coordinate");
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
) -> [T; CHANNELS] {
    let base_x = x.floor() as isize;
    let base_y = y.floor() as isize;
    let mut sums = [0.0_f64; CHANNELS];
    let mut total_weight = 0.0;
    for dy in -1..=2 {
        let wy = cubic_weight(y - (base_y + dy) as f64);
        let iy = clamped_index(base_y + dy, input.height());
        for dx in -1..=2 {
            let weight = wy * cubic_weight(x - (base_x + dx) as f64);
            let ix = clamped_index(base_x + dx, input.width());
            let pixel = input.get(ix, iy).expect("clamped resize coordinate");
            for channel in 0..CHANNELS {
                sums[channel] += pixel[channel].to_f64() * weight;
            }
            total_weight += weight;
        }
    }
    std::array::from_fn(|channel| T::from_f64(sums[channel] / total_weight))
}

fn sample_area<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    output_x: usize,
    output_y: usize,
    output_width: usize,
    output_height: usize,
) -> [T; CHANNELS] {
    let scale_x = input.width() as f64 / output_width as f64;
    let scale_y = input.height() as f64 / output_height as f64;
    let start_x = output_x as f64 * scale_x;
    let end_x = (output_x + 1) as f64 * scale_x;
    let start_y = output_y as f64 * scale_y;
    let end_y = (output_y + 1) as f64 * scale_y;
    let mut sums = [0.0_f64; CHANNELS];
    let mut total_weight = 0.0;
    for iy in start_y.floor() as usize..end_y.ceil().min(input.height() as f64) as usize {
        let overlap_y = (end_y.min(iy as f64 + 1.0) - start_y.max(iy as f64)).max(0.0);
        for ix in start_x.floor() as usize..end_x.ceil().min(input.width() as f64) as usize {
            let overlap_x = (end_x.min(ix as f64 + 1.0) - start_x.max(ix as f64)).max(0.0);
            let weight = overlap_x * overlap_y;
            let pixel = input.get(ix, iy).expect("area sample within image");
            for channel in 0..CHANNELS {
                sums[channel] += pixel[channel].to_f64() * weight;
            }
            total_weight += weight;
        }
    }
    std::array::from_fn(|channel| T::from_f64(sums[channel] / total_weight))
}

#[cfg(test)]
mod tests {
    use super::{resize, resize_into, Interpolation};
    use spatialrust_image::{Image, ImageViewMut};

    #[test]
    fn nearest_repeats_pixels() {
        let input = Image::<u8, 1>::try_new(2, 1, vec![10, 20]).unwrap();
        let output = resize(input.view(), 4, 1, Interpolation::Nearest).unwrap();
        assert_eq!(output.as_slice(), &[10, 10, 20, 20]);
    }

    #[test]
    fn bilinear_interpolates_center() {
        let input = Image::<f32, 1>::try_new(2, 2, vec![0.0, 10.0, 20.0, 30.0]).unwrap();
        let output = resize(input.view(), 3, 3, Interpolation::Bilinear).unwrap();
        assert!((output[(1, 1)][0] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn area_computes_block_average() {
        let input = Image::<u8, 1>::try_new(2, 2, vec![0, 10, 20, 30]).unwrap();
        let output = resize(input.view(), 1, 1, Interpolation::Area).unwrap();
        assert_eq!(output.as_slice(), &[15]);
    }

    #[test]
    fn identity_resize_is_exact_for_all_filters() {
        let input = Image::<u8, 3>::try_new(2, 2, (0..12).collect()).unwrap();
        for filter in [
            Interpolation::Nearest,
            Interpolation::Bilinear,
            Interpolation::Bicubic,
            Interpolation::Area,
        ] {
            assert_eq!(resize(input.view(), 2, 2, filter).unwrap(), input);
        }
    }

    #[test]
    fn resize_into_reuses_strided_output_and_preserves_padding() {
        let input = Image::<u8, 1>::try_new(2, 2, vec![0, 10, 20, 30]).unwrap();
        let mut storage = vec![99_u8; 15];
        let output = ImageViewMut::<u8, 1>::new(3, 3, 6, &mut storage).unwrap();
        resize_into(input.view(), output, Interpolation::Bilinear).unwrap();
        let expected = resize(input.view(), 3, 3, Interpolation::Bilinear).unwrap();
        for y in 0..3 {
            assert_eq!(&storage[y * 6..y * 6 + 3], expected.view().row(y).unwrap());
            if y < 2 {
                assert_eq!(&storage[y * 6 + 3..(y + 1) * 6], &[99; 3]);
            }
        }
    }
}
