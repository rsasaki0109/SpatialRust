//! CPU mathematical morphology with explicit structuring elements and borders.

use spatialrust_image::{Image, ImageView};

use crate::border::fetch;
use crate::{BorderMode, PixelComponent, VisionError, VisionResult};

/// Built-in structuring-element geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MorphologyShape {
    /// Every element in the bounding rectangle is active.
    Rect,
    /// The anchor row and column are active.
    Cross,
    /// A filled ellipse inscribed in the bounding rectangle.
    Ellipse,
    /// A filled Manhattan-distance diamond.
    Diamond,
}

/// A validated binary neighborhood mask and anchor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuringElement {
    width: usize,
    height: usize,
    anchor_x: usize,
    anchor_y: usize,
    mask: Vec<bool>,
}

impl StructuringElement {
    /// Creates a built-in element anchored at its integer center.
    pub fn try_new(shape: MorphologyShape, width: usize, height: usize) -> VisionResult<Self> {
        Self::try_new_with_anchor(shape, width, height, width / 2, height / 2)
    }

    /// Creates a built-in element with an explicit anchor.
    pub fn try_new_with_anchor(
        shape: MorphologyShape,
        width: usize,
        height: usize,
        anchor_x: usize,
        anchor_y: usize,
    ) -> VisionResult<Self> {
        validate_dimensions(width, height, anchor_x, anchor_y)?;
        let mut mask = vec![false; width * height];
        match shape {
            MorphologyShape::Rect => mask.fill(true),
            MorphologyShape::Cross => {
                for y in 0..height {
                    for x in 0..width {
                        mask[y * width + x] = x == anchor_x || y == anchor_y;
                    }
                }
            }
            MorphologyShape::Ellipse => fill_ellipse(&mut mask, width, height),
            MorphologyShape::Diamond => {
                let rx = anchor_x.max(width - 1 - anchor_x).max(1);
                let ry = anchor_y.max(height - 1 - anchor_y).max(1);
                for y in 0..height {
                    for x in 0..width {
                        let dx = x.abs_diff(anchor_x) as f64 / rx as f64;
                        let dy = y.abs_diff(anchor_y) as f64 / ry as f64;
                        mask[y * width + x] = dx + dy <= 1.0 + f64::EPSILON;
                    }
                }
            }
        }
        Self::try_from_mask(width, height, anchor_x, anchor_y, mask)
    }

    /// Creates an element from a row-major binary mask.
    pub fn try_from_mask(
        width: usize,
        height: usize,
        anchor_x: usize,
        anchor_y: usize,
        mask: Vec<bool>,
    ) -> VisionResult<Self> {
        validate_dimensions(width, height, anchor_x, anchor_y)?;
        let expected = width
            .checked_mul(height)
            .ok_or_else(|| VisionError::InvalidParameter("structuring element overflows".into()))?;
        if mask.len() != expected {
            return Err(VisionError::ShapeMismatch(format!(
                "structuring element needs {expected} mask values, found {}",
                mask.len()
            )));
        }
        if !mask.iter().any(|&active| active) {
            return Err(VisionError::InvalidParameter(
                "structuring element must contain an active sample".into(),
            ));
        }
        Ok(Self { width, height, anchor_x, anchor_y, mask })
    }

    /// Element width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Element height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Anchor coordinate `(x, y)`.
    #[must_use]
    pub const fn anchor(&self) -> (usize, usize) {
        (self.anchor_x, self.anchor_y)
    }

    /// Row-major binary mask.
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }
}

/// Composite morphology operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MorphologyOperation {
    /// Erosion followed by dilation.
    Open,
    /// Dilation followed by erosion.
    Close,
    /// Dilation minus erosion.
    Gradient,
    /// Input minus opening.
    TopHat,
    /// Closing minus input.
    BlackHat,
}

/// Erodes each channel independently.
pub fn erode<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    repeat_extreme(input, element, iterations, border, Extreme::Minimum)
}

/// Dilates each channel independently.
pub fn dilate<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    repeat_extreme(input, element, iterations, border, Extreme::Maximum)
}

/// Applies a composite morphology operation.
pub fn morphology_ex<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    operation: MorphologyOperation,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    let original = pack(input)?;
    match operation {
        MorphologyOperation::Open => {
            let eroded = erode(input, element, iterations, border)?;
            dilate(eroded.view(), element, iterations, border)
        }
        MorphologyOperation::Close => {
            let dilated = dilate(input, element, iterations, border)?;
            erode(dilated.view(), element, iterations, border)
        }
        MorphologyOperation::Gradient => {
            let high = dilate(input, element, iterations, border)?;
            let low = erode(input, element, iterations, border)?;
            subtract(high.view(), low.view())
        }
        MorphologyOperation::TopHat => {
            let opened =
                morphology_ex(input, MorphologyOperation::Open, element, iterations, border)?;
            subtract(original.view(), opened.view())
        }
        MorphologyOperation::BlackHat => {
            let closed =
                morphology_ex(input, MorphologyOperation::Close, element, iterations, border)?;
            subtract(closed.view(), original.view())
        }
    }
}

#[derive(Clone, Copy)]
enum Extreme {
    Minimum,
    Maximum,
}

fn repeat_extreme<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<T, CHANNELS>,
    extreme: Extreme,
) -> VisionResult<Image<T, CHANNELS>> {
    let mut output = pack(input)?;
    for _ in 0..iterations {
        output = extreme_once(output.view(), element, border, extreme)?;
    }
    Ok(output)
}

fn extreme_once<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    element: &StructuringElement,
    border: BorderMode<T, CHANNELS>,
    extreme: Extreme,
) -> VisionResult<Image<T, CHANNELS>> {
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            let mut values = [0.0; CHANNELS];
            let mut initialized = false;
            for ey in 0..element.height {
                for ex in 0..element.width {
                    if !element.mask[ey * element.width + ex] {
                        continue;
                    }
                    let pixel = fetch(
                        input,
                        x as isize + ex as isize - element.anchor_x as isize,
                        y as isize + ey as isize - element.anchor_y as isize,
                        border,
                    );
                    if !initialized {
                        values = pixel.map(PixelComponent::to_f64);
                        initialized = true;
                    } else {
                        for channel in 0..CHANNELS {
                            let candidate = pixel[channel].to_f64();
                            let ordering = candidate.total_cmp(&values[channel]);
                            if matches!(extreme, Extreme::Minimum) && ordering.is_lt()
                                || matches!(extreme, Extreme::Maximum) && ordering.is_gt()
                            {
                                values[channel] = candidate;
                            }
                        }
                    }
                }
            }
            output.extend(values.map(T::from_f64));
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

fn subtract<T: PixelComponent, const CHANNELS: usize>(
    left: ImageView<'_, T, CHANNELS>,
    right: ImageView<'_, T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    if left.width() != right.width() || left.height() != right.height() {
        return Err(VisionError::ShapeMismatch(
            "morphology subtraction dimensions must match".into(),
        ));
    }
    let mut output = Vec::with_capacity(left.width() * left.height() * CHANNELS);
    for y in 0..left.height() {
        for x in 0..left.width() {
            let a = left.get(x, y).expect("left coordinate in bounds");
            let b = right.get(x, y).expect("right coordinate in bounds");
            output.extend(std::array::from_fn::<_, CHANNELS, _>(|channel| {
                T::from_f64(a[channel].to_f64() - b[channel].to_f64())
            }));
        }
    }
    Ok(Image::try_new_with_metadata(left.width(), left.height(), output, left.metadata())?)
}

fn pack<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            output.extend_from_slice(input.get(x, y).expect("input coordinate in bounds"));
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

fn validate_dimensions(
    width: usize,
    height: usize,
    anchor_x: usize,
    anchor_y: usize,
) -> VisionResult<()> {
    if width == 0 || height == 0 {
        return Err(VisionError::InvalidParameter(
            "structuring element dimensions must be non-zero".into(),
        ));
    }
    if anchor_x >= width || anchor_y >= height {
        return Err(VisionError::InvalidParameter(format!(
            "structuring element anchor ({anchor_x}, {anchor_y}) is outside {width}x{height}"
        )));
    }
    width
        .checked_mul(height)
        .ok_or_else(|| VisionError::InvalidParameter("structuring element overflows".into()))?;
    Ok(())
}

fn fill_ellipse(mask: &mut [bool], width: usize, height: usize) {
    let center_x = width / 2;
    let center_y = height / 2;
    let radius_y = center_y.max(1) as f64;
    for y in 0..height {
        let dy = y.abs_diff(center_y) as f64;
        if dy > radius_y {
            continue;
        }
        let extent =
            (center_x as f64 * (1.0 - dy * dy / (radius_y * radius_y)).sqrt()).round() as usize;
        let start = center_x.saturating_sub(extent);
        let end = (center_x + extent).min(width - 1);
        for x in start..=end {
            mask[y * width + x] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dilate, erode, morphology_ex, MorphologyOperation, MorphologyShape, StructuringElement,
    };
    use crate::BorderMode;
    use spatialrust_image::{Image, ImageRegion};

    #[test]
    fn built_in_masks_have_expected_3x3_layouts() {
        let rect = StructuringElement::try_new(MorphologyShape::Rect, 3, 3).unwrap();
        let cross = StructuringElement::try_new(MorphologyShape::Cross, 3, 3).unwrap();
        let ellipse = StructuringElement::try_new(MorphologyShape::Ellipse, 3, 3).unwrap();
        let diamond = StructuringElement::try_new(MorphologyShape::Diamond, 3, 3).unwrap();
        assert_eq!(rect.mask().iter().filter(|&&v| v).count(), 9);
        assert_eq!(cross.mask().iter().filter(|&&v| v).count(), 5);
        assert_eq!(ellipse.mask().iter().filter(|&&v| v).count(), 5);
        assert_eq!(diamond.mask().iter().filter(|&&v| v).count(), 5);
    }

    #[test]
    fn erode_and_dilate_match_known_extrema_on_roi() {
        let parent = Image::<u8, 1>::try_new(5, 3, (0..15).collect()).unwrap();
        let roi = parent.view().subview(ImageRegion::new(1, 0, 3, 3)).unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Rect, 3, 3).unwrap();
        let low = erode(roi, &element, 1, BorderMode::Replicate).unwrap();
        let high = dilate(roi, &element, 1, BorderMode::Replicate).unwrap();
        assert_eq!(low[(1, 1)][0], 1);
        assert_eq!(high[(1, 1)][0], 13);
    }

    #[test]
    fn opening_removes_isolated_impulse() {
        let image = Image::<u16, 1>::try_new(3, 3, vec![0, 0, 0, 0, 100, 0, 0, 0, 0]).unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Rect, 3, 3).unwrap();
        let output = morphology_ex(
            image.view(),
            MorphologyOperation::Open,
            &element,
            1,
            BorderMode::Constant([0]),
        )
        .unwrap();
        assert!(output.as_slice().iter().all(|&value| value == 0));
    }

    #[test]
    fn gradient_is_dilate_minus_erode_for_float() {
        let image = Image::<f32, 1>::try_new(3, 1, vec![1.0, 4.0, 9.0]).unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Rect, 3, 1).unwrap();
        let output = morphology_ex(
            image.view(),
            MorphologyOperation::Gradient,
            &element,
            1,
            BorderMode::Replicate,
        )
        .unwrap();
        assert_eq!(output.as_slice(), &[3.0, 8.0, 5.0]);
    }

    #[test]
    fn zero_iterations_return_packed_copy() {
        let image = Image::<u8, 3>::from_pixel(2, 2, [1, 2, 3]).unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Rect, 3, 3).unwrap();
        assert_eq!(erode(image.view(), &element, 0, BorderMode::Replicate).unwrap(), image);
    }

    #[test]
    fn invalid_masks_are_rejected() {
        assert!(StructuringElement::try_from_mask(2, 2, 0, 0, vec![false; 4]).is_err());
        assert!(StructuringElement::try_from_mask(2, 2, 0, 0, vec![true; 3]).is_err());
    }
}
