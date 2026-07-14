//! Bounded, feature-gated image decoding and encoding for SpatialRust.
//!
//! All input is staged in CPU memory with an explicit compressed-input limit.
//! This crate never uploads to or reads from a device. Codec dependencies are
//! selected independently from the storage-only `spatialrust-image` crate.
//!
//! ```no_run
//! use spatialrust_image_io::{decode_path, DecodeOptions, DecodedPixels};
//!
//! let decoded = decode_path("frame.png", DecodeOptions::default())?;
//! match decoded.pixels() {
//!     DecodedPixels::Rgb8(image) => assert!(image.width() > 0),
//!     other => println!("decoded {other:?}"),
//! }
//! # Ok::<(), spatialrust_image_io::ImageIoError>(())
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]

use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, Write};
use std::path::Path;

use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, ImageOutputFormat};
use spatialrust_image::{AlphaMode, ColorRange, ColorSpace, Image, ImageError, ImageMetadata};

/// Errors produced by bounded image IO.
#[derive(Debug, thiserror::Error)]
pub enum ImageIoError {
    /// Reading or writing the stream failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A codec rejected the image or encoded output.
    #[error(transparent)]
    Codec(#[from] image::ImageError),
    /// Constructing a typed SpatialRust image failed.
    #[error(transparent)]
    Image(#[from] ImageError),
    /// The stream exceeded the configured compressed-input limit.
    #[error("encoded input exceeds the {maximum} byte limit")]
    InputTooLarge {
        /// Maximum accepted encoded byte count.
        maximum: usize,
    },
    /// Decoded dimensions or pixel count exceeded the configured limits.
    #[error("decoded image dimensions {width}x{height} exceed configured limits")]
    DimensionsTooLarge {
        /// Encoded image width.
        width: u32,
        /// Encoded image height.
        height: u32,
    },
    /// The detected/requested format is not enabled in this build.
    #[error("image format `{0}` is not enabled in this build")]
    FormatDisabled(ImageFileFormat),
    /// The input format is not one of SpatialRust's image-IO formats.
    #[error("unsupported image format: {0}")]
    UnsupportedFormat(String),
    /// Encoding options were invalid.
    #[error("invalid encode option: {0}")]
    InvalidEncodeOption(String),
    /// A decoded dimension cannot be represented by this platform.
    #[error("decoded dimensions cannot be represented by usize")]
    DimensionOverflow,
}

/// File formats recognized by the stable SpatialRust image-IO boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageFileFormat {
    /// Portable Network Graphics.
    Png,
    /// JPEG/JFIF image.
    Jpeg,
    /// Portable anymap family (PBM/PGM/PPM/PAM).
    Pnm,
    /// Tagged Image File Format.
    Tiff,
    /// OpenEXR high-dynamic-range image.
    OpenExr,
}

impl std::fmt::Display for ImageFileFormat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Pnm => "pnm",
            Self::Tiff => "tiff",
            Self::OpenExr => "openexr",
        })
    }
}

impl ImageFileFormat {
    /// Returns whether this codec is compiled into the current crate.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        match self {
            Self::Png => cfg!(feature = "png"),
            Self::Jpeg => cfg!(feature = "jpeg"),
            Self::Pnm => cfg!(feature = "pnm"),
            Self::Tiff => cfg!(feature = "tiff"),
            Self::OpenExr => cfg!(feature = "openexr"),
        }
    }

    fn from_backend(format: ImageFormat) -> Result<Self, ImageIoError> {
        match format {
            ImageFormat::Png => Ok(Self::Png),
            ImageFormat::Jpeg => Ok(Self::Jpeg),
            ImageFormat::Pnm => Ok(Self::Pnm),
            ImageFormat::Tiff => Ok(Self::Tiff),
            ImageFormat::OpenExr => Ok(Self::OpenExr),
            other => Err(ImageIoError::UnsupportedFormat(format!("{other:?}"))),
        }
    }

    fn require_enabled(self) -> Result<(), ImageIoError> {
        if self.is_enabled() {
            Ok(())
        } else {
            Err(ImageIoError::FormatDisabled(self))
        }
    }
}

/// Strict application-level resource limits for one decode operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Maximum compressed input size in bytes.
    pub max_input_bytes: usize,
    /// Maximum decoded width.
    pub max_width: u32,
    /// Maximum decoded height.
    pub max_height: u32,
    /// Maximum decoded pixel count.
    pub max_pixels: u64,
    /// Best-effort maximum simultaneous allocation performed by the backend.
    pub max_alloc_bytes: u64,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 256 * 1024 * 1024,
            max_width: 32_768,
            max_height: 32_768,
            max_pixels: 100_000_000,
            max_alloc_bytes: 512 * 1024 * 1024,
        }
    }
}

impl DecodeLimits {
    fn validate_dimensions(self, width: u32, height: u32) -> Result<(), ImageIoError> {
        let pixels = u64::from(width).saturating_mul(u64::from(height));
        if width > self.max_width || height > self.max_height || pixels > self.max_pixels {
            return Err(ImageIoError::DimensionsTooLarge { width, height });
        }
        Ok(())
    }

    fn backend(self) -> image::io::Limits {
        let mut limits = image::io::Limits::default();
        limits.max_image_width = Some(self.max_width);
        limits.max_image_height = Some(self.max_height);
        limits.max_alloc = Some(self.max_alloc_bytes);
        limits
    }
}

/// Options controlling one decode operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Resource limits applied before and during decoding.
    pub limits: DecodeLimits,
    /// Apply Exif orientation to produce canonical top-left pixels.
    pub apply_orientation: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self { limits: DecodeLimits::default(), apply_orientation: true }
    }
}

/// Exif image orientation in encoded-pixel coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Orientation {
    /// No orientation tag was found.
    #[default]
    Unspecified = 0,
    /// Top-left origin; no transform.
    Normal = 1,
    /// Mirror left-to-right.
    FlipHorizontal = 2,
    /// Rotate 180 degrees.
    Rotate180 = 3,
    /// Mirror top-to-bottom.
    FlipVertical = 4,
    /// Reflect across the top-left to bottom-right diagonal.
    Transpose = 5,
    /// Rotate 90 degrees clockwise.
    Rotate90 = 6,
    /// Reflect across the top-right to bottom-left diagonal.
    Transverse = 7,
    /// Rotate 270 degrees clockwise.
    Rotate270 = 8,
}

impl Orientation {
    #[cfg(feature = "exif")]
    fn from_exif(value: u32) -> Self {
        match value {
            1 => Self::Normal,
            2 => Self::FlipHorizontal,
            3 => Self::Rotate180,
            4 => Self::FlipVertical,
            5 => Self::Transpose,
            6 => Self::Rotate90,
            7 => Self::Transverse,
            8 => Self::Rotate270,
            _ => Self::Unspecified,
        }
    }

    fn changes_pixels(self) -> bool {
        !matches!(self, Self::Unspecified | Self::Normal)
    }
}

/// Encoded sample/channel type before conversion into SpatialRust storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceColorType {
    /// 8-bit grayscale.
    Gray8,
    /// 8-bit grayscale plus alpha.
    GrayAlpha8,
    /// 8-bit RGB.
    Rgb8,
    /// 8-bit RGBA.
    Rgba8,
    /// 16-bit grayscale.
    Gray16,
    /// 16-bit grayscale plus alpha.
    GrayAlpha16,
    /// 16-bit RGB.
    Rgb16,
    /// 16-bit RGBA.
    Rgba16,
    /// 32-bit floating-point RGB.
    Rgb32Float,
    /// 32-bit floating-point RGBA.
    Rgba32Float,
}

impl SourceColorType {
    fn from_backend(color: image::ColorType) -> Result<Self, ImageIoError> {
        match color {
            image::ColorType::L8 => Ok(Self::Gray8),
            image::ColorType::La8 => Ok(Self::GrayAlpha8),
            image::ColorType::Rgb8 => Ok(Self::Rgb8),
            image::ColorType::Rgba8 => Ok(Self::Rgba8),
            image::ColorType::L16 => Ok(Self::Gray16),
            image::ColorType::La16 => Ok(Self::GrayAlpha16),
            image::ColorType::Rgb16 => Ok(Self::Rgb16),
            image::ColorType::Rgba16 => Ok(Self::Rgba16),
            image::ColorType::Rgb32F => Ok(Self::Rgb32Float),
            image::ColorType::Rgba32F => Ok(Self::Rgba32Float),
            other => Err(ImageIoError::UnsupportedFormat(format!(
                "unsupported decoded color type {other:?}"
            ))),
        }
    }
}

/// Owned typed pixels produced by a decoder or accepted by an encoder.
#[derive(Clone, Debug, PartialEq)]
pub enum DecodedPixels {
    /// One-channel 8-bit grayscale.
    Gray8(Image<u8, 1>),
    /// Two-channel 8-bit grayscale and alpha.
    GrayAlpha8(Image<u8, 2>),
    /// Three-channel 8-bit RGB.
    Rgb8(Image<u8, 3>),
    /// Four-channel 8-bit RGBA.
    Rgba8(Image<u8, 4>),
    /// One-channel 16-bit grayscale.
    Gray16(Image<u16, 1>),
    /// Two-channel 16-bit grayscale and alpha.
    GrayAlpha16(Image<u16, 2>),
    /// Three-channel 16-bit RGB.
    Rgb16(Image<u16, 3>),
    /// Four-channel 16-bit RGBA.
    Rgba16(Image<u16, 4>),
    /// Three-channel linear-light floating-point RGB.
    Rgb32Float(Image<f32, 3>),
    /// Four-channel linear-light floating-point RGBA.
    Rgba32Float(Image<f32, 4>),
}

impl DecodedPixels {
    /// Returns pixel width.
    #[must_use]
    pub fn width(&self) -> usize {
        match self {
            Self::Gray8(image) => image.width(),
            Self::GrayAlpha8(image) => image.width(),
            Self::Rgb8(image) => image.width(),
            Self::Rgba8(image) => image.width(),
            Self::Gray16(image) => image.width(),
            Self::GrayAlpha16(image) => image.width(),
            Self::Rgb16(image) => image.width(),
            Self::Rgba16(image) => image.width(),
            Self::Rgb32Float(image) => image.width(),
            Self::Rgba32Float(image) => image.width(),
        }
    }

    /// Returns pixel height.
    #[must_use]
    pub fn height(&self) -> usize {
        match self {
            Self::Gray8(image) => image.height(),
            Self::GrayAlpha8(image) => image.height(),
            Self::Rgb8(image) => image.height(),
            Self::Rgba8(image) => image.height(),
            Self::Gray16(image) => image.height(),
            Self::GrayAlpha16(image) => image.height(),
            Self::Rgb16(image) => image.height(),
            Self::Rgba16(image) => image.height(),
            Self::Rgb32Float(image) => image.height(),
            Self::Rgba32Float(image) => image.height(),
        }
    }

    fn to_dynamic(&self) -> Result<DynamicImage, ImageIoError> {
        let width = u32::try_from(self.width()).map_err(|_| ImageIoError::DimensionOverflow)?;
        let height = u32::try_from(self.height()).map_err(|_| ImageIoError::DimensionOverflow)?;
        macro_rules! buffer {
            ($image:expr, $pixel:ty, $variant:path) => {{
                let typed = ImageBuffer::<$pixel, Vec<_>>::from_raw(
                    width,
                    height,
                    $image.as_slice().to_vec(),
                )
                .ok_or(ImageIoError::DimensionOverflow)?;
                $variant(typed)
            }};
        }
        Ok(match self {
            Self::Gray8(image) => buffer!(image, image::Luma<u8>, DynamicImage::ImageLuma8),
            Self::GrayAlpha8(image) => {
                buffer!(image, image::LumaA<u8>, DynamicImage::ImageLumaA8)
            }
            Self::Rgb8(image) => buffer!(image, image::Rgb<u8>, DynamicImage::ImageRgb8),
            Self::Rgba8(image) => buffer!(image, image::Rgba<u8>, DynamicImage::ImageRgba8),
            Self::Gray16(image) => buffer!(image, image::Luma<u16>, DynamicImage::ImageLuma16),
            Self::GrayAlpha16(image) => {
                buffer!(image, image::LumaA<u16>, DynamicImage::ImageLumaA16)
            }
            Self::Rgb16(image) => buffer!(image, image::Rgb<u16>, DynamicImage::ImageRgb16),
            Self::Rgba16(image) => buffer!(image, image::Rgba<u16>, DynamicImage::ImageRgba16),
            Self::Rgb32Float(image) => {
                buffer!(image, image::Rgb<f32>, DynamicImage::ImageRgb32F)
            }
            Self::Rgba32Float(image) => {
                buffer!(image, image::Rgba<f32>, DynamicImage::ImageRgba32F)
            }
        })
    }
}

/// Metadata describing encoded pixels and orientation handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodedMetadata {
    /// Detected container/codec format.
    pub format: ImageFileFormat,
    /// Sample and channel representation reported by the decoder.
    pub source_color_type: SourceColorType,
    /// Orientation tag found in the encoded container.
    pub orientation: Orientation,
    /// Whether a non-identity orientation was applied to the returned pixels.
    pub orientation_applied: bool,
}

/// One decoded image with owned typed pixels and source metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedImage {
    pixels: DecodedPixels,
    metadata: DecodedMetadata,
}

impl DecodedImage {
    /// Borrows decoded pixels.
    #[must_use]
    pub fn pixels(&self) -> &DecodedPixels {
        &self.pixels
    }

    /// Returns source metadata.
    #[must_use]
    pub const fn metadata(&self) -> DecodedMetadata {
        self.metadata
    }

    /// Consumes the wrapper and returns pixels.
    #[must_use]
    pub fn into_pixels(self) -> DecodedPixels {
        self.pixels
    }

    /// Returns decoded width after orientation handling.
    #[must_use]
    pub fn width(&self) -> usize {
        self.pixels.width()
    }

    /// Returns decoded height after orientation handling.
    #[must_use]
    pub fn height(&self) -> usize {
        self.pixels.height()
    }
}

/// Options controlling encoded output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Output codec/container.
    pub format: ImageFileFormat,
    /// JPEG quality in the inclusive range 1–100.
    pub jpeg_quality: u8,
}

impl EncodeOptions {
    /// Creates options for one format with JPEG quality 90.
    #[must_use]
    pub const fn new(format: ImageFileFormat) -> Self {
        Self { format, jpeg_quality: 90 }
    }

    fn output_format(self) -> Result<ImageOutputFormat, ImageIoError> {
        self.format.require_enabled()?;
        if self.jpeg_quality == 0 || self.jpeg_quality > 100 {
            return Err(ImageIoError::InvalidEncodeOption(
                "jpeg_quality must be in 1..=100".to_owned(),
            ));
        }
        match self.format {
            #[cfg(feature = "png")]
            ImageFileFormat::Png => Ok(ImageOutputFormat::Png),
            #[cfg(not(feature = "png"))]
            ImageFileFormat::Png => Err(ImageIoError::FormatDisabled(self.format)),
            #[cfg(feature = "jpeg")]
            ImageFileFormat::Jpeg => Ok(ImageOutputFormat::Jpeg(self.jpeg_quality)),
            #[cfg(not(feature = "jpeg"))]
            ImageFileFormat::Jpeg => Err(ImageIoError::FormatDisabled(self.format)),
            #[cfg(feature = "pnm")]
            ImageFileFormat::Pnm => {
                Ok(ImageOutputFormat::Pnm(image::codecs::pnm::PnmSubtype::ArbitraryMap))
            }
            #[cfg(not(feature = "pnm"))]
            ImageFileFormat::Pnm => Err(ImageIoError::FormatDisabled(self.format)),
            #[cfg(feature = "tiff")]
            ImageFileFormat::Tiff => Ok(ImageOutputFormat::Tiff),
            #[cfg(not(feature = "tiff"))]
            ImageFileFormat::Tiff => Err(ImageIoError::FormatDisabled(self.format)),
            #[cfg(feature = "openexr")]
            ImageFileFormat::OpenExr => Ok(ImageOutputFormat::OpenExr),
            #[cfg(not(feature = "openexr"))]
            ImageFileFormat::OpenExr => Err(ImageIoError::FormatDisabled(self.format)),
        }
    }
}

/// Decodes an image from an encoded byte slice.
pub fn decode_bytes(bytes: &[u8], options: DecodeOptions) -> Result<DecodedImage, ImageIoError> {
    if bytes.len() > options.limits.max_input_bytes {
        return Err(ImageIoError::InputTooLarge { maximum: options.limits.max_input_bytes });
    }
    let backend_format = image::guess_format(bytes)?;
    let format = ImageFileFormat::from_backend(backend_format)?;
    format.require_enabled()?;

    let dimensions =
        image::io::Reader::with_format(Cursor::new(bytes), backend_format).into_dimensions()?;
    options.limits.validate_dimensions(dimensions.0, dimensions.1)?;

    let orientation = read_orientation(bytes);
    let mut reader = image::io::Reader::with_format(Cursor::new(bytes), backend_format);
    reader.limits(options.limits.backend());
    let dynamic = reader.decode()?;
    let source_color_type = SourceColorType::from_backend(dynamic.color())?;
    let orientation_applied = options.apply_orientation && orientation.changes_pixels();
    let dynamic =
        if orientation_applied { apply_orientation(dynamic, orientation) } else { dynamic };
    let pixels = dynamic_to_pixels(dynamic)?;
    Ok(DecodedImage {
        pixels,
        metadata: DecodedMetadata { format, source_color_type, orientation, orientation_applied },
    })
}

/// Reads a bounded encoded stream and decodes it.
pub fn decode_reader<R: Read>(
    reader: R,
    options: DecodeOptions,
) -> Result<DecodedImage, ImageIoError> {
    let maximum = options.limits.max_input_bytes;
    let take_limit = u64::try_from(maximum).unwrap_or(u64::MAX).saturating_add(1);
    let mut bytes = Vec::new();
    reader.take(take_limit).read_to_end(&mut bytes)?;
    if bytes.len() > maximum {
        return Err(ImageIoError::InputTooLarge { maximum });
    }
    decode_bytes(&bytes, options)
}

/// Opens and decodes one image path using content-based format detection.
pub fn decode_path(
    path: impl AsRef<Path>,
    options: DecodeOptions,
) -> Result<DecodedImage, ImageIoError> {
    decode_reader(BufReader::new(File::open(path)?), options)
}

/// Encodes typed pixels to a seekable writer.
///
/// The codec backend receives a temporary owned CPU buffer. No device transfer
/// occurs, and the source image remains unchanged.
pub fn encode_writer<W: Write + Seek>(
    writer: &mut W,
    pixels: &DecodedPixels,
    options: EncodeOptions,
) -> Result<(), ImageIoError> {
    let dynamic = pixels.to_dynamic()?;
    dynamic.write_to(writer, options.output_format()?)?;
    Ok(())
}

/// Encodes typed pixels into an owned byte vector.
pub fn encode_bytes(
    pixels: &DecodedPixels,
    options: EncodeOptions,
) -> Result<Vec<u8>, ImageIoError> {
    let mut cursor = Cursor::new(Vec::new());
    encode_writer(&mut cursor, pixels, options)?;
    Ok(cursor.into_inner())
}

/// Creates or truncates a path and encodes typed pixels into it.
pub fn encode_path(
    path: impl AsRef<Path>,
    pixels: &DecodedPixels,
    options: EncodeOptions,
) -> Result<(), ImageIoError> {
    let mut writer = BufWriter::new(File::create(path)?);
    encode_writer(&mut writer, pixels, options)?;
    writer.flush()?;
    Ok(())
}

fn metadata(color_space: ColorSpace, alpha_mode: AlphaMode, floating: bool) -> ImageMetadata {
    ImageMetadata {
        color_space,
        color_range: if floating { ColorRange::Unspecified } else { ColorRange::Full },
        alpha_mode,
    }
}

fn dimensions(dynamic: &DynamicImage) -> Result<(usize, usize), ImageIoError> {
    let (width, height) = dynamic.dimensions();
    Ok((
        usize::try_from(width).map_err(|_| ImageIoError::DimensionOverflow)?,
        usize::try_from(height).map_err(|_| ImageIoError::DimensionOverflow)?,
    ))
}

fn dynamic_to_pixels(dynamic: DynamicImage) -> Result<DecodedPixels, ImageIoError> {
    let (width, height) = dimensions(&dynamic)?;
    Ok(match dynamic {
        DynamicImage::ImageLuma8(image) => DecodedPixels::Gray8(Image::try_new_with_metadata(
            width,
            height,
            image.into_raw(),
            metadata(ColorSpace::Gray, AlphaMode::None, false),
        )?),
        DynamicImage::ImageLumaA8(image) => {
            DecodedPixels::GrayAlpha8(Image::try_new_with_metadata(
                width,
                height,
                image.into_raw(),
                metadata(ColorSpace::Unknown, AlphaMode::Straight, false),
            )?)
        }
        DynamicImage::ImageRgb8(image) => DecodedPixels::Rgb8(Image::try_new_with_metadata(
            width,
            height,
            image.into_raw(),
            metadata(ColorSpace::Rgb, AlphaMode::None, false),
        )?),
        DynamicImage::ImageRgba8(image) => DecodedPixels::Rgba8(Image::try_new_with_metadata(
            width,
            height,
            image.into_raw(),
            metadata(ColorSpace::Rgba, AlphaMode::Straight, false),
        )?),
        DynamicImage::ImageLuma16(image) => DecodedPixels::Gray16(Image::try_new_with_metadata(
            width,
            height,
            image.into_raw(),
            metadata(ColorSpace::Gray, AlphaMode::None, false),
        )?),
        DynamicImage::ImageLumaA16(image) => {
            DecodedPixels::GrayAlpha16(Image::try_new_with_metadata(
                width,
                height,
                image.into_raw(),
                metadata(ColorSpace::Unknown, AlphaMode::Straight, false),
            )?)
        }
        DynamicImage::ImageRgb16(image) => DecodedPixels::Rgb16(Image::try_new_with_metadata(
            width,
            height,
            image.into_raw(),
            metadata(ColorSpace::Rgb, AlphaMode::None, false),
        )?),
        DynamicImage::ImageRgba16(image) => DecodedPixels::Rgba16(Image::try_new_with_metadata(
            width,
            height,
            image.into_raw(),
            metadata(ColorSpace::Rgba, AlphaMode::Straight, false),
        )?),
        DynamicImage::ImageRgb32F(image) => {
            DecodedPixels::Rgb32Float(Image::try_new_with_metadata(
                width,
                height,
                image.into_raw(),
                metadata(ColorSpace::LinearRgb, AlphaMode::None, true),
            )?)
        }
        DynamicImage::ImageRgba32F(image) => {
            DecodedPixels::Rgba32Float(Image::try_new_with_metadata(
                width,
                height,
                image.into_raw(),
                metadata(ColorSpace::Unknown, AlphaMode::Straight, true),
            )?)
        }
        _ => {
            return Err(ImageIoError::UnsupportedFormat(
                "decoder returned an unsupported dynamic image variant".to_owned(),
            ))
        }
    })
}

fn apply_orientation(image: DynamicImage, orientation: Orientation) -> DynamicImage {
    match orientation {
        Orientation::Unspecified | Orientation::Normal => image,
        Orientation::FlipHorizontal => image.fliph(),
        Orientation::Rotate180 => image.rotate180(),
        Orientation::FlipVertical => image.flipv(),
        Orientation::Transpose => image.rotate90().fliph(),
        Orientation::Rotate90 => image.rotate90(),
        Orientation::Transverse => image.rotate90().flipv(),
        Orientation::Rotate270 => image.rotate270(),
    }
}

#[cfg(feature = "exif")]
fn read_orientation(bytes: &[u8]) -> Orientation {
    use exif::{In, Reader, Tag};

    let mut cursor = Cursor::new(bytes);
    Reader::new()
        .read_from_container(&mut cursor)
        .ok()
        .and_then(|exif| {
            exif.get_field(Tag::Orientation, In::PRIMARY).and_then(|field| field.value.get_uint(0))
        })
        .map(Orientation::from_exif)
        .unwrap_or(Orientation::Unspecified)
}

#[cfg(not(feature = "exif"))]
fn read_orientation(_bytes: &[u8]) -> Orientation {
    Orientation::Unspecified
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[cfg(any(feature = "png", feature = "jpeg", feature = "pnm"))]
    fn rgb_fixture() -> DecodedPixels {
        DecodedPixels::Rgb8(
            Image::try_new_with_metadata(
                3,
                2,
                vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                metadata(ColorSpace::Rgb, AlphaMode::None, false),
            )
            .unwrap(),
        )
    }

    #[cfg(feature = "png")]
    #[test]
    fn png_memory_roundtrip_is_exact() {
        let source = rgb_fixture();
        let bytes = encode_bytes(&source, EncodeOptions::new(ImageFileFormat::Png)).unwrap();
        let decoded = decode_bytes(&bytes, DecodeOptions::default()).unwrap();
        assert_eq!(decoded.pixels(), &source);
        assert_eq!(decoded.metadata().format, ImageFileFormat::Png);
        assert_eq!(decoded.metadata().source_color_type, SourceColorType::Rgb8);
    }

    #[cfg(feature = "pnm")]
    #[test]
    fn pnm_reader_roundtrip_is_exact() {
        let source = rgb_fixture();
        let bytes = encode_bytes(&source, EncodeOptions::new(ImageFileFormat::Pnm)).unwrap();
        let decoded = decode_reader(Cursor::new(bytes), DecodeOptions::default()).unwrap();
        assert_eq!(decoded.pixels(), &source);
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn jpeg_roundtrip_preserves_shape_and_type() {
        let bytes =
            encode_bytes(&rgb_fixture(), EncodeOptions::new(ImageFileFormat::Jpeg)).unwrap();
        let decoded = decode_bytes(&bytes, DecodeOptions::default()).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (3, 2));
        assert!(matches!(decoded.pixels(), DecodedPixels::Rgb8(_)));
    }

    #[cfg(feature = "png")]
    #[test]
    fn dimensions_and_input_are_bounded_before_decode() {
        let bytes = encode_bytes(&rgb_fixture(), EncodeOptions::new(ImageFileFormat::Png)).unwrap();
        let mut options = DecodeOptions::default();
        options.limits.max_width = 2;
        assert!(matches!(
            decode_bytes(&bytes, options),
            Err(ImageIoError::DimensionsTooLarge { .. }) | Err(ImageIoError::Codec(_))
        ));

        let mut options = DecodeOptions::default();
        options.limits.max_input_bytes = bytes.len() - 1;
        assert!(matches!(
            decode_reader(Cursor::new(bytes), options),
            Err(ImageIoError::InputTooLarge { .. })
        ));
    }

    #[test]
    fn rotation_and_diagonal_reflection_have_expected_layout() {
        let image = image::GrayImage::from_raw(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let rotated =
            apply_orientation(DynamicImage::ImageLuma8(image.clone()), Orientation::Rotate90);
        assert_eq!(rotated.dimensions(), (3, 2));
        assert_eq!(rotated.to_luma8().into_raw(), vec![5, 3, 1, 6, 4, 2]);

        let transposed = apply_orientation(DynamicImage::ImageLuma8(image), Orientation::Transpose);
        assert_eq!(transposed.dimensions(), (3, 2));
        assert_eq!(transposed.to_luma8().into_raw(), vec![1, 3, 5, 2, 4, 6]);
    }

    #[cfg(feature = "exif")]
    #[test]
    fn reads_orientation_from_minimal_tiff_exif() {
        let bytes = [
            b'I', b'I', 42, 0, 8, 0, 0, 0, // TIFF header and first IFD offset
            1, 0, // one entry
            0x12, 0x01, // Orientation tag
            3, 0, // SHORT
            1, 0, 0, 0, // count
            6, 0, 0, 0, // Rotate 90 clockwise
            0, 0, 0, 0, // next IFD
        ];
        assert_eq!(read_orientation(&bytes), Orientation::Rotate90);
    }

    #[test]
    fn rejects_malformed_input() {
        assert!(decode_bytes(b"not an image", DecodeOptions::default()).is_err());
    }

    #[test]
    fn disabled_format_reports_feature_boundary() {
        if !ImageFileFormat::Tiff.is_enabled() {
            assert!(matches!(
                EncodeOptions::new(ImageFileFormat::Tiff).output_format(),
                Err(ImageIoError::FormatDisabled(ImageFileFormat::Tiff))
            ));
        }
    }

    proptest! {
        #[test]
        fn arbitrary_small_input_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..4096)) {
            let mut options = DecodeOptions::default();
            options.limits.max_input_bytes = 4096;
            options.limits.max_width = 256;
            options.limits.max_height = 256;
            options.limits.max_pixels = 65_536;
            options.limits.max_alloc_bytes = 4 * 1024 * 1024;
            let _ = decode_bytes(&bytes, options);
        }
    }

    #[cfg(feature = "png")]
    #[test]
    fn path_roundtrip_uses_content_detection() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("image-without-extension");
        let source = rgb_fixture();
        encode_path(&path, &source, EncodeOptions::new(ImageFileFormat::Png)).unwrap();
        let decoded = decode_path(path, DecodeOptions::default()).unwrap();
        assert_eq!(decoded.pixels(), &source);
    }

    #[cfg(feature = "tiff")]
    #[test]
    fn tiff_preserves_sixteen_bit_grayscale() {
        let source = DecodedPixels::Gray16(
            Image::try_new_with_metadata(
                3,
                1,
                vec![0, 1024, u16::MAX],
                metadata(ColorSpace::Gray, AlphaMode::None, false),
            )
            .unwrap(),
        );
        let bytes = encode_bytes(&source, EncodeOptions::new(ImageFileFormat::Tiff)).unwrap();
        let decoded = decode_bytes(&bytes, DecodeOptions::default()).unwrap();
        assert_eq!(decoded.pixels(), &source);
    }

    #[cfg(feature = "openexr")]
    #[test]
    fn openexr_preserves_float_rgb() {
        let source = DecodedPixels::Rgb32Float(
            Image::try_new_with_metadata(
                2,
                1,
                vec![0.0, 0.25, 1.0, 4.0, -0.5, 2.0],
                metadata(ColorSpace::LinearRgb, AlphaMode::None, true),
            )
            .unwrap(),
        );
        let bytes = encode_bytes(&source, EncodeOptions::new(ImageFileFormat::OpenExr)).unwrap();
        let decoded = decode_bytes(&bytes, DecodeOptions::default()).unwrap();
        assert_eq!(decoded.pixels(), &source);
    }
}
