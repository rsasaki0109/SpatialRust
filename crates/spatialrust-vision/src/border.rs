//! Shared border extrapolation for CPU image kernels.

#[cfg(any(
    feature = "warp",
    feature = "imgproc-filter",
    feature = "imgproc-morphology",
    feature = "imgproc-analysis"
))]
use spatialrust_image::ImageView;

#[cfg(any(
    feature = "warp",
    feature = "imgproc-filter",
    feature = "imgproc-morphology",
    feature = "imgproc-analysis"
))]
use crate::PixelComponent;

/// Out-of-bounds sampling behavior for CPU image operations.
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

#[cfg(any(
    feature = "warp",
    feature = "imgproc-filter",
    feature = "imgproc-morphology",
    feature = "imgproc-analysis"
))]
pub(crate) fn constant_pixel<T: PixelComponent, const CHANNELS: usize>(
    border: BorderMode<T, CHANNELS>,
) -> [T; CHANNELS] {
    match border {
        BorderMode::Constant(pixel) => pixel,
        BorderMode::Replicate | BorderMode::Reflect | BorderMode::Reflect101 | BorderMode::Wrap => {
            std::array::from_fn(|_| T::from_f64(0.0))
        }
    }
}

#[cfg(any(
    feature = "warp",
    feature = "imgproc-filter",
    feature = "imgproc-morphology",
    feature = "imgproc-analysis"
))]
pub(crate) fn fetch<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    x: isize,
    y: isize,
    border: BorderMode<T, CHANNELS>,
) -> [T; CHANNELS] {
    if let (Some(ix), Some(iy)) =
        (map_index(x, input.width(), border), map_index(y, input.height(), border))
    {
        return *input.get(ix, iy).expect("mapped coordinate is in bounds");
    }
    constant_pixel(border)
}

#[cfg(any(
    feature = "warp",
    feature = "imgproc-filter",
    feature = "imgproc-morphology",
    feature = "imgproc-analysis"
))]
pub(crate) fn map_index<T, const CHANNELS: usize>(
    index: isize,
    length: usize,
    border: BorderMode<T, CHANNELS>,
) -> Option<usize> {
    if length == 0 {
        return None;
    }
    if index >= 0 && index < length as isize {
        return Some(index as usize);
    }
    match border {
        BorderMode::Constant(_) => None,
        BorderMode::Replicate => Some(index.clamp(0, length as isize - 1) as usize),
        BorderMode::Reflect => Some(reflect_index(index, length, false)),
        BorderMode::Reflect101 => Some(reflect_index(index, length, true)),
        BorderMode::Wrap => Some(index.rem_euclid(length as isize) as usize),
    }
}

#[cfg(any(
    feature = "warp",
    feature = "imgproc-filter",
    feature = "imgproc-morphology",
    feature = "imgproc-analysis"
))]
fn reflect_index(mut index: isize, length: usize, reflect101: bool) -> usize {
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

#[cfg(all(
    test,
    any(
        feature = "warp",
        feature = "imgproc-filter",
        feature = "imgproc-morphology",
        feature = "imgproc-analysis"
    )
))]
mod tests {
    use super::{map_index, BorderMode};

    #[test]
    fn maps_single_pixel_without_looping() {
        for border in [
            BorderMode::<u8, 1>::Replicate,
            BorderMode::Reflect,
            BorderMode::Reflect101,
            BorderMode::Wrap,
        ] {
            assert_eq!(map_index(-100, 1, border), Some(0));
            assert_eq!(map_index(100, 1, border), Some(0));
        }
    }

    #[test]
    fn empty_images_always_map_to_constant_space() {
        assert_eq!(map_index(0, 0, BorderMode::<u8, 1>::Wrap), None);
    }
}
