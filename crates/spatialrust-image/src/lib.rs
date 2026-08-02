//! Typed, CPU-resident image buffers and zero-copy strided views.
//!
//! Channel count is part of the type. Packed interleaved ownership is the
//! default; planar ownership and explicitly-strided views are also available.
//! Device-backed images belong in dedicated GPU crates so transfers remain
//! explicit.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use std::ops::{Index, IndexMut};

/// Errors raised while constructing or indexing images.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ImageError {
    /// A zero channel image type was requested.
    #[error("image channel count must be greater than zero")]
    ZeroChannels,
    /// Image dimensions overflowed `usize` arithmetic.
    #[error("image dimensions overflow addressable memory")]
    DimensionOverflow,
    /// The provided storage does not match the image layout.
    #[error("image storage is too short: need at least {required} elements, found {found}")]
    StorageTooShort {
        /// Minimum required element count.
        required: usize,
        /// Provided element count.
        found: usize,
    },
    /// A row stride cannot hold one row of pixels.
    #[error("row stride {stride} is smaller than packed row width {minimum}")]
    InvalidStride {
        /// Provided stride in scalar elements.
        stride: usize,
        /// Packed row size in scalar elements.
        minimum: usize,
    },
    /// A planar channel stride overlaps the preceding channel plane.
    #[error("plane stride {stride} is smaller than one plane span {minimum}")]
    InvalidPlaneStride {
        /// Provided plane stride in scalar elements.
        stride: usize,
        /// Minimum non-overlapping plane span.
        minimum: usize,
    },
    /// Color metadata is incompatible with the compile-time channel count.
    #[error("color space {color_space:?} requires {expected} channels, image has {found}")]
    MetadataChannelMismatch {
        /// Declared color space.
        color_space: ColorSpace,
        /// Required channel count.
        expected: usize,
        /// Image channel count.
        found: usize,
    },
    /// A requested image region lies outside its parent image.
    #[error("region ({x}, {y}, {region_width}, {region_height}) exceeds image bounds {image_width}x{image_height}")]
    InvalidRegion {
        /// Region x origin.
        x: usize,
        /// Region y origin.
        y: usize,
        /// Region width.
        region_width: usize,
        /// Region height.
        region_height: usize,
        /// Parent image width.
        image_width: usize,
        /// Parent image height.
        image_height: usize,
    },
    /// Pixel coordinates were outside the image.
    #[error("pixel ({x}, {y}) is outside image bounds {width}x{height}")]
    OutOfBounds {
        /// Pixel x coordinate.
        x: usize,
        /// Pixel y coordinate.
        y: usize,
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
    },
}

/// Physical channel arrangement in CPU storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageLayout {
    /// Pixel channels are adjacent (`RGBRGB...`).
    #[default]
    Interleaved,
    /// Each channel occupies a separate plane (`RR...GG...BB...`).
    Planar,
}

/// Semantic interpretation of image channels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// Channel semantics are not specified.
    #[default]
    Unknown,
    /// One-channel luminance.
    Gray,
    /// Nonlinear red, green, blue.
    Rgb,
    /// Nonlinear blue, green, red.
    Bgr,
    /// Nonlinear red, green, blue, alpha.
    Rgba,
    /// Nonlinear blue, green, red, alpha.
    Bgra,
    /// Linear-light red, green, blue.
    LinearRgb,
    /// Hue, saturation, value.
    Hsv,
    /// Metric or sensor depth values.
    Depth,
    /// Integer semantic or instance labels.
    Label,
}

impl ColorSpace {
    /// Returns the required channel count when the color space fixes one.
    #[must_use]
    pub const fn required_channels(self) -> Option<usize> {
        match self {
            Self::Unknown => None,
            Self::Gray | Self::Depth | Self::Label => Some(1),
            Self::Rgb | Self::Bgr | Self::LinearRgb | Self::Hsv => Some(3),
            Self::Rgba | Self::Bgra => Some(4),
        }
    }
}

/// Numeric range convention associated with image channels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ColorRange {
    /// Range is not specified or is naturally unbounded (for example depth).
    #[default]
    Unspecified,
    /// Full dtype or normalized range.
    Full,
    /// Video-range encoding such as limited-range YUV.
    Limited,
}

/// Alpha-channel interpretation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    /// No alpha channel is declared.
    #[default]
    None,
    /// Straight (unassociated) alpha.
    Straight,
    /// RGB values are premultiplied by alpha.
    Premultiplied,
}

/// Lightweight semantic metadata carried by images and borrowed views.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ImageMetadata {
    /// Channel color/depth/label interpretation.
    pub color_space: ColorSpace,
    /// Numeric range convention.
    pub color_range: ColorRange,
    /// Alpha convention.
    pub alpha_mode: AlphaMode,
}

impl ImageMetadata {
    /// Validates metadata against a compile-time channel count.
    pub fn validate<const CHANNELS: usize>(self) -> Result<(), ImageError> {
        if CHANNELS == 0 {
            return Err(ImageError::ZeroChannels);
        }
        if let Some(expected) = self.color_space.required_channels() {
            if expected != CHANNELS {
                return Err(ImageError::MetadataChannelMismatch {
                    color_space: self.color_space,
                    expected,
                    found: CHANNELS,
                });
            }
        }
        Ok(())
    }
}

/// A checked rectangular image region.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ImageRegion {
    /// Horizontal origin.
    pub x: usize,
    /// Vertical origin.
    pub y: usize,
    /// Region width.
    pub width: usize,
    /// Region height.
    pub height: usize,
}

impl ImageRegion {
    /// Creates a rectangular region.
    #[must_use]
    pub const fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self { x, y, width, height }
    }

    fn validate(self, image_width: usize, image_height: usize) -> Result<(), ImageError> {
        let end_x = self.x.checked_add(self.width).ok_or(ImageError::DimensionOverflow)?;
        let end_y = self.y.checked_add(self.height).ok_or(ImageError::DimensionOverflow)?;
        if end_x > image_width || end_y > image_height {
            return Err(ImageError::InvalidRegion {
                x: self.x,
                y: self.y,
                region_width: self.width,
                region_height: self.height,
                image_width,
                image_height,
            });
        }
        Ok(())
    }
}

fn packed_len<const CHANNELS: usize>(width: usize, height: usize) -> Result<usize, ImageError> {
    if CHANNELS == 0 {
        return Err(ImageError::ZeroChannels);
    }
    width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(CHANNELS))
        .ok_or(ImageError::DimensionOverflow)
}

fn strided_span(height: usize, row_stride: usize, packed_row: usize) -> Result<usize, ImageError> {
    if height == 0 {
        return Ok(0);
    }
    row_stride
        .checked_mul(height - 1)
        .and_then(|offset| offset.checked_add(packed_row))
        .ok_or(ImageError::DimensionOverflow)
}

/// An owning, densely packed, interleaved image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image<T, const CHANNELS: usize> {
    width: usize,
    height: usize,
    data: Vec<T>,
    metadata: ImageMetadata,
}

impl<T, const CHANNELS: usize> Image<T, CHANNELS> {
    /// Creates an image from densely packed, interleaved scalar elements.
    pub fn try_new(width: usize, height: usize, data: Vec<T>) -> Result<Self, ImageError> {
        let required = packed_len::<CHANNELS>(width, height)?;
        if data.len() != required {
            return Err(ImageError::StorageTooShort { required, found: data.len() });
        }
        Ok(Self { width, height, data, metadata: ImageMetadata::default() })
    }

    /// Creates a packed image and validates its semantic metadata.
    pub fn try_new_with_metadata(
        width: usize,
        height: usize,
        data: Vec<T>,
        metadata: ImageMetadata,
    ) -> Result<Self, ImageError> {
        metadata.validate::<CHANNELS>()?;
        let mut image = Self::try_new(width, height, data)?;
        image.metadata = metadata;
        Ok(image)
    }

    /// Returns image width in pixels.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns image height in pixels.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns the packed row stride in scalar elements.
    #[must_use]
    pub const fn row_stride(&self) -> usize {
        self.width * CHANNELS
    }

    /// Returns physical channel layout.
    #[must_use]
    pub const fn layout(&self) -> ImageLayout {
        ImageLayout::Interleaved
    }

    /// Returns semantic image metadata.
    #[must_use]
    pub const fn metadata(&self) -> ImageMetadata {
        self.metadata
    }

    /// Replaces semantic metadata after validating the channel count.
    pub fn set_metadata(&mut self, metadata: ImageMetadata) -> Result<(), ImageError> {
        metadata.validate::<CHANNELS>()?;
        self.metadata = metadata;
        Ok(())
    }

    /// Returns the packed scalar storage.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns mutable packed scalar storage.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrows this image as a zero-copy view.
    #[must_use]
    pub fn view(&self) -> ImageView<'_, T, CHANNELS> {
        ImageView {
            width: self.width,
            height: self.height,
            row_stride: self.row_stride(),
            data: &self.data,
            metadata: self.metadata,
        }
    }

    /// Borrows this image as a mutable zero-copy view.
    #[must_use]
    pub fn view_mut(&mut self) -> ImageViewMut<'_, T, CHANNELS> {
        ImageViewMut {
            width: self.width,
            height: self.height,
            row_stride: self.width * CHANNELS,
            data: &mut self.data,
            metadata: self.metadata,
        }
    }

    /// Returns one pixel, or `None` outside image bounds.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<&[T; CHANNELS]> {
        self.view().get(x, y)
    }

    /// Returns one mutable pixel, or `None` outside image bounds.
    #[must_use]
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut [T; CHANNELS]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let offset = (y * self.width + x) * CHANNELS;
        self.data.get_mut(offset..offset + CHANNELS)?.try_into().ok()
    }

    /// Consumes the image and returns its packed scalar storage.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Clone, const CHANNELS: usize> Image<T, CHANNELS> {
    /// Creates a densely packed image filled with one pixel value.
    pub fn from_pixel(
        width: usize,
        height: usize,
        pixel: [T; CHANNELS],
    ) -> Result<Self, ImageError> {
        let required = packed_len::<CHANNELS>(width, height)?;
        let mut data = Vec::with_capacity(required);
        for _ in 0..width.saturating_mul(height) {
            data.extend_from_slice(&pixel);
        }
        Self::try_new(width, height, data)
    }
}

impl<T, const CHANNELS: usize> Index<(usize, usize)> for Image<T, CHANNELS> {
    type Output = [T; CHANNELS];

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        self.get(x, y).expect("image index out of bounds")
    }
}

impl<T, const CHANNELS: usize> IndexMut<(usize, usize)> for Image<T, CHANNELS> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        self.get_mut(x, y).expect("image index out of bounds")
    }
}

/// A read-only, zero-copy image view with an explicit row stride.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a, T, const CHANNELS: usize> {
    width: usize,
    height: usize,
    row_stride: usize,
    data: &'a [T],
    metadata: ImageMetadata,
}

impl<'a, T, const CHANNELS: usize> ImageView<'a, T, CHANNELS> {
    /// Creates a view over interleaved storage.
    ///
    /// `row_stride` is measured in scalar elements, not bytes.
    pub fn new(
        width: usize,
        height: usize,
        row_stride: usize,
        data: &'a [T],
    ) -> Result<Self, ImageError> {
        Self::new_with_metadata(width, height, row_stride, data, ImageMetadata::default())
    }

    /// Creates a view with validated semantic metadata.
    pub fn new_with_metadata(
        width: usize,
        height: usize,
        row_stride: usize,
        data: &'a [T],
        metadata: ImageMetadata,
    ) -> Result<Self, ImageError> {
        metadata.validate::<CHANNELS>()?;
        let packed_row = width.checked_mul(CHANNELS).ok_or(ImageError::DimensionOverflow)?;
        if row_stride < packed_row {
            return Err(ImageError::InvalidStride { stride: row_stride, minimum: packed_row });
        }
        let required = if height == 0 {
            0
        } else {
            row_stride
                .checked_mul(height - 1)
                .and_then(|offset| offset.checked_add(packed_row))
                .ok_or(ImageError::DimensionOverflow)?
        };
        if data.len() < required {
            return Err(ImageError::StorageTooShort { required, found: data.len() });
        }
        Ok(Self { width, height, row_stride, data, metadata })
    }

    /// Returns image width in pixels.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Returns image height in pixels.
    #[must_use]
    pub const fn height(self) -> usize {
        self.height
    }

    /// Returns row stride in scalar elements.
    #[must_use]
    pub const fn row_stride(self) -> usize {
        self.row_stride
    }

    /// Returns physical channel layout.
    #[must_use]
    pub const fn layout(self) -> ImageLayout {
        ImageLayout::Interleaved
    }

    /// Returns semantic image metadata.
    #[must_use]
    pub const fn metadata(self) -> ImageMetadata {
        self.metadata
    }

    /// Returns one pixel, or `None` outside image bounds.
    #[must_use]
    pub fn get(self, x: usize, y: usize) -> Option<&'a [T; CHANNELS]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let offset = y * self.row_stride + x * CHANNELS;
        self.data.get(offset..offset + CHANNELS)?.try_into().ok()
    }

    /// Returns a packed row without its trailing padding.
    #[must_use]
    pub fn row(self, y: usize) -> Option<&'a [T]> {
        if y >= self.height {
            return None;
        }
        let start = y * self.row_stride;
        self.data.get(start..start + self.width * CHANNELS)
    }

    /// Creates a checked zero-copy subview.
    pub fn subview(self, region: ImageRegion) -> Result<Self, ImageError> {
        region.validate(self.width, self.height)?;
        if region.width == 0 || region.height == 0 {
            return Ok(Self {
                width: region.width,
                height: region.height,
                row_stride: self.row_stride,
                data: &self.data[..0],
                metadata: self.metadata,
            });
        }
        let start = region.y * self.row_stride + region.x * CHANNELS;
        let span = (region.height - 1) * self.row_stride + region.width * CHANNELS;
        Ok(Self {
            width: region.width,
            height: region.height,
            row_stride: self.row_stride,
            data: &self.data[start..start + span],
            metadata: self.metadata,
        })
    }
}

/// A mutable, zero-copy interleaved image view with an explicit row stride.
#[derive(Debug)]
pub struct ImageViewMut<'a, T, const CHANNELS: usize> {
    width: usize,
    height: usize,
    row_stride: usize,
    data: &'a mut [T],
    metadata: ImageMetadata,
}

impl<'a, T, const CHANNELS: usize> ImageViewMut<'a, T, CHANNELS> {
    /// Creates a mutable view over interleaved storage.
    pub fn new(
        width: usize,
        height: usize,
        row_stride: usize,
        data: &'a mut [T],
    ) -> Result<Self, ImageError> {
        Self::new_with_metadata(width, height, row_stride, data, ImageMetadata::default())
    }

    /// Creates a mutable view with validated semantic metadata.
    pub fn new_with_metadata(
        width: usize,
        height: usize,
        row_stride: usize,
        data: &'a mut [T],
        metadata: ImageMetadata,
    ) -> Result<Self, ImageError> {
        metadata.validate::<CHANNELS>()?;
        let packed_row = width.checked_mul(CHANNELS).ok_or(ImageError::DimensionOverflow)?;
        if row_stride < packed_row {
            return Err(ImageError::InvalidStride { stride: row_stride, minimum: packed_row });
        }
        let required = strided_span(height, row_stride, packed_row)?;
        if data.len() < required {
            return Err(ImageError::StorageTooShort { required, found: data.len() });
        }
        Ok(Self { width, height, row_stride, data, metadata })
    }

    /// Returns image width in pixels.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns image height in pixels.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns row stride in scalar elements.
    #[must_use]
    pub const fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns semantic image metadata.
    #[must_use]
    pub const fn metadata(&self) -> ImageMetadata {
        self.metadata
    }

    /// Replaces semantic metadata after validating the channel count.
    pub fn set_metadata(&mut self, metadata: ImageMetadata) -> Result<(), ImageError> {
        metadata.validate::<CHANNELS>()?;
        self.metadata = metadata;
        Ok(())
    }

    /// Reborrows this mutable view as read-only.
    #[must_use]
    pub fn as_view(&self) -> ImageView<'_, T, CHANNELS> {
        ImageView {
            width: self.width,
            height: self.height,
            row_stride: self.row_stride,
            data: self.data,
            metadata: self.metadata,
        }
    }

    /// Returns one read-only pixel, or `None` outside image bounds.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<&[T; CHANNELS]> {
        self.as_view().get(x, y)
    }

    /// Returns one mutable pixel, or `None` outside image bounds.
    #[must_use]
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut [T; CHANNELS]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let offset = y * self.row_stride + x * CHANNELS;
        self.data.get_mut(offset..offset + CHANNELS)?.try_into().ok()
    }

    /// Returns a mutable packed row without trailing padding.
    #[must_use]
    pub fn row_mut(&mut self, y: usize) -> Option<&mut [T]> {
        if y >= self.height {
            return None;
        }
        let start = y * self.row_stride;
        self.data.get_mut(start..start + self.width * CHANNELS)
    }

    /// Returns the complete mutable backing span, including inter-row padding.
    ///
    /// The final row has no required trailing padding, so the returned length is
    /// the minimum checked span accepted by [`ImageViewMut::new`].
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// Creates a checked mutable zero-copy subview.
    pub fn subview(
        &mut self,
        region: ImageRegion,
    ) -> Result<ImageViewMut<'_, T, CHANNELS>, ImageError> {
        region.validate(self.width, self.height)?;
        if region.width == 0 || region.height == 0 {
            return Ok(ImageViewMut {
                width: region.width,
                height: region.height,
                row_stride: self.row_stride,
                data: &mut self.data[..0],
                metadata: self.metadata,
            });
        }
        let start = region.y * self.row_stride + region.x * CHANNELS;
        let span = (region.height - 1) * self.row_stride + region.width * CHANNELS;
        Ok(ImageViewMut {
            width: region.width,
            height: region.height,
            row_stride: self.row_stride,
            data: &mut self.data[start..start + span],
            metadata: self.metadata,
        })
    }
}

/// An owning, densely packed planar image.
///
/// Storage order is all values for channel 0, followed by channel 1, and so on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanarImage<T, const CHANNELS: usize> {
    width: usize,
    height: usize,
    data: Vec<T>,
    metadata: ImageMetadata,
}

impl<T, const CHANNELS: usize> PlanarImage<T, CHANNELS> {
    /// Creates an image from densely packed planar scalar elements.
    pub fn try_new(width: usize, height: usize, data: Vec<T>) -> Result<Self, ImageError> {
        let required = packed_len::<CHANNELS>(width, height)?;
        if data.len() != required {
            return Err(ImageError::StorageTooShort { required, found: data.len() });
        }
        Ok(Self { width, height, data, metadata: ImageMetadata::default() })
    }

    /// Creates a planar image with validated semantic metadata.
    pub fn try_new_with_metadata(
        width: usize,
        height: usize,
        data: Vec<T>,
        metadata: ImageMetadata,
    ) -> Result<Self, ImageError> {
        metadata.validate::<CHANNELS>()?;
        let mut image = Self::try_new(width, height, data)?;
        image.metadata = metadata;
        Ok(image)
    }

    /// Returns image width in pixels.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns image height in pixels.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns physical channel layout.
    #[must_use]
    pub const fn layout(&self) -> ImageLayout {
        ImageLayout::Planar
    }

    /// Returns packed row stride within each plane.
    #[must_use]
    pub const fn row_stride(&self) -> usize {
        self.width
    }

    /// Returns the scalar distance between channel-plane origins.
    #[must_use]
    pub const fn plane_stride(&self) -> usize {
        self.width * self.height
    }

    /// Returns semantic image metadata.
    #[must_use]
    pub const fn metadata(&self) -> ImageMetadata {
        self.metadata
    }

    /// Replaces semantic metadata after validating the channel count.
    pub fn set_metadata(&mut self, metadata: ImageMetadata) -> Result<(), ImageError> {
        metadata.validate::<CHANNELS>()?;
        self.metadata = metadata;
        Ok(())
    }

    /// Returns planar scalar storage.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns mutable planar scalar storage.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrows this image as a planar view.
    #[must_use]
    pub fn view(&self) -> PlanarImageView<'_, T, CHANNELS> {
        PlanarImageView {
            width: self.width,
            height: self.height,
            row_stride: self.width,
            plane_stride: self.width * self.height,
            data: &self.data,
            metadata: self.metadata,
        }
    }

    /// Returns one channel value, or `None` outside image bounds.
    #[must_use]
    pub fn get(&self, channel: usize, x: usize, y: usize) -> Option<&T> {
        self.view().get(channel, x, y)
    }

    /// Returns one mutable channel value, or `None` outside image bounds.
    #[must_use]
    pub fn get_mut(&mut self, channel: usize, x: usize, y: usize) -> Option<&mut T> {
        if channel >= CHANNELS || x >= self.width || y >= self.height {
            return None;
        }
        let offset = channel * self.width * self.height + y * self.width + x;
        self.data.get_mut(offset)
    }

    /// Consumes the image and returns planar scalar storage.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

/// A read-only, zero-copy planar image view with explicit strides.
#[derive(Clone, Copy, Debug)]
pub struct PlanarImageView<'a, T, const CHANNELS: usize> {
    width: usize,
    height: usize,
    row_stride: usize,
    plane_stride: usize,
    data: &'a [T],
    metadata: ImageMetadata,
}

impl<'a, T, const CHANNELS: usize> PlanarImageView<'a, T, CHANNELS> {
    /// Creates a planar view. Strides are measured in scalar elements.
    pub fn new(
        width: usize,
        height: usize,
        row_stride: usize,
        plane_stride: usize,
        data: &'a [T],
    ) -> Result<Self, ImageError> {
        Self::new_with_metadata(
            width,
            height,
            row_stride,
            plane_stride,
            data,
            ImageMetadata::default(),
        )
    }

    /// Creates a planar view with validated semantic metadata.
    pub fn new_with_metadata(
        width: usize,
        height: usize,
        row_stride: usize,
        plane_stride: usize,
        data: &'a [T],
        metadata: ImageMetadata,
    ) -> Result<Self, ImageError> {
        metadata.validate::<CHANNELS>()?;
        if row_stride < width {
            return Err(ImageError::InvalidStride { stride: row_stride, minimum: width });
        }
        let plane_span = strided_span(height, row_stride, width)?;
        if plane_stride < plane_span {
            return Err(ImageError::InvalidPlaneStride {
                stride: plane_stride,
                minimum: plane_span,
            });
        }
        let required = if width == 0 || height == 0 {
            0
        } else {
            plane_stride
                .checked_mul(CHANNELS - 1)
                .and_then(|offset| offset.checked_add(plane_span))
                .ok_or(ImageError::DimensionOverflow)?
        };
        if data.len() < required {
            return Err(ImageError::StorageTooShort { required, found: data.len() });
        }
        Ok(Self { width, height, row_stride, plane_stride, data, metadata })
    }

    /// Returns image width in pixels.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Returns image height in pixels.
    #[must_use]
    pub const fn height(self) -> usize {
        self.height
    }

    /// Returns row stride within a plane.
    #[must_use]
    pub const fn row_stride(self) -> usize {
        self.row_stride
    }

    /// Returns scalar distance between channel-plane origins.
    #[must_use]
    pub const fn plane_stride(self) -> usize {
        self.plane_stride
    }

    /// Returns physical channel layout.
    #[must_use]
    pub const fn layout(self) -> ImageLayout {
        ImageLayout::Planar
    }

    /// Returns semantic image metadata.
    #[must_use]
    pub const fn metadata(self) -> ImageMetadata {
        self.metadata
    }

    /// Returns one channel value, or `None` outside image bounds.
    #[must_use]
    pub fn get(self, channel: usize, x: usize, y: usize) -> Option<&'a T> {
        if channel >= CHANNELS || x >= self.width || y >= self.height {
            return None;
        }
        self.data.get(channel * self.plane_stride + y * self.row_stride + x)
    }

    /// Copies one pixel from its channel planes.
    #[must_use]
    pub fn pixel(self, x: usize, y: usize) -> Option<[T; CHANNELS]>
    where
        T: Copy,
    {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(std::array::from_fn(|channel| {
            *self.get(channel, x, y).expect("validated planar coordinate")
        }))
    }

    /// Creates a checked zero-copy planar subview.
    pub fn subview(self, region: ImageRegion) -> Result<Self, ImageError> {
        region.validate(self.width, self.height)?;
        if region.width == 0 || region.height == 0 {
            return Ok(Self {
                width: region.width,
                height: region.height,
                row_stride: self.row_stride,
                plane_stride: self.plane_stride,
                data: &self.data[..0],
                metadata: self.metadata,
            });
        }
        let start = region.y * self.row_stride + region.x;
        let span = (CHANNELS - 1) * self.plane_stride
            + (region.height - 1) * self.row_stride
            + region.width;
        Ok(Self {
            width: region.width,
            height: region.height,
            row_stride: self.row_stride,
            plane_stride: self.plane_stride,
            data: &self.data[start..start + span],
            metadata: self.metadata,
        })
    }
}

/// A one-channel image.
pub type GrayImage<T> = Image<T, 1>;
/// A three-channel RGB image.
pub type RgbImage<T> = Image<T, 3>;

#[cfg(test)]
mod tests {
    use super::{
        AlphaMode, ColorRange, ColorSpace, Image, ImageError, ImageMetadata, ImageRegion,
        ImageView, ImageViewMut, PlanarImage, PlanarImageView,
    };

    #[test]
    fn packed_image_indexes_pixels() {
        let mut image = Image::<u8, 3>::try_new(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(image[(1, 0)], [4, 5, 6]);
        image[(0, 0)] = [7, 8, 9];
        assert_eq!(image.as_slice(), &[7, 8, 9, 4, 5, 6]);
    }

    #[test]
    fn strided_view_skips_padding() {
        let data = [1_u16, 2, 99, 3, 4];
        let view = ImageView::<u16, 1>::new(2, 2, 3, &data).unwrap();
        assert_eq!(view.get(0, 1), Some(&[3]));
        assert_eq!(view.row(0), Some(&[1, 2][..]));
    }

    #[test]
    fn rejects_short_storage() {
        assert_eq!(
            Image::<u8, 1>::try_new(2, 2, vec![0; 3]).unwrap_err(),
            ImageError::StorageTooShort { required: 4, found: 3 }
        );
    }

    #[test]
    fn mutable_roi_updates_only_selected_pixels() {
        let mut data = [0_u8, 1, 2, 99, 3, 4, 5];
        let mut view = ImageViewMut::<u8, 1>::new(3, 2, 4, &mut data).unwrap();
        {
            let mut roi = view.subview(ImageRegion::new(1, 0, 2, 2)).unwrap();
            *roi.get_mut(0, 0).unwrap() = [10];
            *roi.get_mut(1, 1).unwrap() = [20];
        }
        assert_eq!(data, [0, 10, 2, 99, 3, 4, 20]);
    }

    #[test]
    fn immutable_roi_preserves_parent_stride() {
        let data = [0_u8, 1, 2, 99, 3, 4, 5];
        let view = ImageView::<u8, 1>::new(3, 2, 4, &data).unwrap();
        let roi = view.subview(ImageRegion::new(1, 0, 2, 2)).unwrap();
        assert_eq!(roi.row_stride(), 4);
        assert_eq!(roi.get(0, 0), Some(&[1]));
        assert_eq!(roi.get(1, 1), Some(&[5]));
        assert!(view.subview(ImageRegion::new(2, 1, 2, 1)).is_err());
    }

    #[test]
    fn planar_image_reads_channels_and_subviews() {
        // R plane, G plane, B plane.
        let mut image = PlanarImage::<u8, 3>::try_new(
            2,
            2,
            vec![1, 2, 3, 4, 10, 20, 30, 40, 100, 110, 120, 130],
        )
        .unwrap();
        assert_eq!(image.view().pixel(1, 1), Some([4, 40, 130]));
        *image.get_mut(1, 0, 1).unwrap() = 31;
        let roi = image.view().subview(ImageRegion::new(0, 1, 2, 1)).unwrap();
        assert_eq!(roi.pixel(0, 0), Some([3, 31, 120]));
        assert_eq!(roi.pixel(1, 0), Some([4, 40, 130]));
    }

    #[test]
    fn planar_view_honors_row_and_plane_padding() {
        let data = [1_u8, 2, 99, 3, 4, 88, 77, 10, 20, 99, 30, 40];
        let view = PlanarImageView::<u8, 2>::new(2, 2, 3, 7, &data).unwrap();
        assert_eq!(view.pixel(0, 1), Some([3, 30]));
        assert_eq!(view.pixel(1, 1), Some([4, 40]));
    }

    #[test]
    fn validates_color_metadata_channel_count() {
        let metadata = ImageMetadata {
            color_space: ColorSpace::Rgb,
            color_range: ColorRange::Full,
            alpha_mode: AlphaMode::None,
        };
        let image = Image::<u8, 3>::try_new_with_metadata(1, 1, vec![1, 2, 3], metadata).unwrap();
        assert_eq!(image.metadata(), metadata);
        assert!(matches!(
            Image::<u8, 1>::try_new_with_metadata(1, 1, vec![1], metadata),
            Err(ImageError::MetadataChannelMismatch { .. })
        ));
    }
}
