//! Explicit bridges between typed CPU images and generic tensors.

use bytemuck::{cast_slice, Pod};
use spatialrust_image::{Image, ImageView, PlanarImage, PlanarImageView};

use crate::{DataType, Device, TensorBuffer, TensorDescriptor, TensorError, TensorView};

/// A native-endian scalar that has a stable tensor dtype and no invalid bit patterns.
pub trait TensorElement: Pod {
    /// Tensor dtype corresponding to this Rust scalar.
    const DTYPE: DataType;
}

macro_rules! tensor_elements {
    ($($type:ty => $dtype:expr),+ $(,)?) => {
        $(impl TensorElement for $type {
            const DTYPE: DataType = $dtype;
        })+
    };
}

tensor_elements! {
    u8 => DataType::U8,
    u16 => DataType::U16,
    i8 => DataType::I8,
    i16 => DataType::I16,
    i32 => DataType::I32,
    i64 => DataType::I64,
    f32 => DataType::F32,
    f64 => DataType::F64,
}

/// Borrows a packed interleaved image as a zero-copy `[height, width, channels]` tensor.
pub fn interleaved_image_view<T: TensorElement, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
) -> Result<TensorView<'_>, TensorError> {
    let descriptor = TensorDescriptor::contiguous(
        T::DTYPE,
        vec![image.height(), image.width(), CHANNELS],
        Device::CPU,
    );
    TensorView::try_new(cast_slice(image.as_slice()), descriptor)
}

/// Borrows a packed planar image as a zero-copy `[channels, height, width]` tensor.
pub fn planar_image_view<T: TensorElement, const CHANNELS: usize>(
    image: &PlanarImage<T, CHANNELS>,
) -> Result<TensorView<'_>, TensorError> {
    let descriptor = TensorDescriptor::contiguous(
        T::DTYPE,
        vec![CHANNELS, image.height(), image.width()],
        Device::CPU,
    );
    TensorView::try_new(cast_slice(image.as_slice()), descriptor)
}

/// Explicitly packs a possibly strided interleaved image view into an owned HWC tensor.
pub fn pack_interleaved_image<T: TensorElement, const CHANNELS: usize>(
    image: ImageView<'_, T, CHANNELS>,
) -> Result<TensorBuffer, TensorError> {
    let scalar_count = image
        .width()
        .checked_mul(image.height())
        .and_then(|value| value.checked_mul(CHANNELS))
        .ok_or(TensorError::LayoutOverflow)?;
    let byte_count =
        scalar_count.checked_mul(T::DTYPE.element_size()).ok_or(TensorError::LayoutOverflow)?;
    let mut bytes = Vec::with_capacity(byte_count);
    for y in 0..image.height() {
        let row = image.row(y).expect("row index is within the image height");
        bytes.extend_from_slice(cast_slice(row));
    }
    let descriptor = TensorDescriptor::contiguous(
        T::DTYPE,
        vec![image.height(), image.width(), CHANNELS],
        Device::CPU,
    );
    TensorBuffer::try_new(bytes, descriptor)
}

/// Explicitly packs a possibly strided planar image view into an owned CHW tensor.
pub fn pack_planar_image<T: TensorElement, const CHANNELS: usize>(
    image: PlanarImageView<'_, T, CHANNELS>,
) -> Result<TensorBuffer, TensorError> {
    let scalar_count = image
        .width()
        .checked_mul(image.height())
        .and_then(|value| value.checked_mul(CHANNELS))
        .ok_or(TensorError::LayoutOverflow)?;
    let byte_count =
        scalar_count.checked_mul(T::DTYPE.element_size()).ok_or(TensorError::LayoutOverflow)?;
    let mut bytes = Vec::with_capacity(byte_count);
    for channel in 0..CHANNELS {
        for y in 0..image.height() {
            for x in 0..image.width() {
                let value = image
                    .get(channel, x, y)
                    .expect("channel and pixel coordinates are within the image");
                bytes.extend_from_slice(bytemuck::bytes_of(value));
            }
        }
    }
    let descriptor = TensorDescriptor::contiguous(
        T::DTYPE,
        vec![CHANNELS, image.height(), image.width()],
        Device::CPU,
    );
    TensorBuffer::try_new(bytes, descriptor)
}

#[cfg(test)]
mod tests {
    use super::{
        interleaved_image_view, pack_interleaved_image, pack_planar_image, planar_image_view,
    };
    use spatialrust_image::{Image, ImageRegion, ImageView, PlanarImage, PlanarImageView};

    #[test]
    fn packed_images_are_zero_copy_hwc_and_chw() {
        let interleaved = Image::<u16, 3>::try_new(2, 2, (0..12).collect()).unwrap();
        let tensor = interleaved_image_view(&interleaved).unwrap();
        assert_eq!(tensor.descriptor().shape(), &[2, 2, 3]);
        assert_eq!(tensor.allocation_bytes().as_ptr(), interleaved.as_slice().as_ptr().cast());

        let planar =
            PlanarImage::<f32, 3>::try_new(2, 2, (0..12).map(|v| v as f32).collect()).unwrap();
        let tensor = planar_image_view(&planar).unwrap();
        assert_eq!(tensor.descriptor().shape(), &[3, 2, 2]);
        assert_eq!(tensor.allocation_bytes().as_ptr(), planar.as_slice().as_ptr().cast());
    }

    #[test]
    fn strided_interleaved_roi_is_explicitly_packed() {
        let storage = (0_u8..30).collect::<Vec<_>>();
        let parent = ImageView::<u8, 3>::new(3, 3, 10, &storage).unwrap();
        let roi = parent.subview(ImageRegion::new(1, 1, 2, 2)).unwrap();
        let tensor = pack_interleaved_image(roi).unwrap();
        assert_eq!(tensor.descriptor().shape(), &[2, 2, 3]);
        assert_eq!(tensor.allocation_bytes(), &[13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28]);
    }

    #[test]
    fn padded_planar_view_is_explicitly_packed() {
        let storage = (0_u16..24).collect::<Vec<_>>();
        let view = PlanarImageView::<u16, 2>::new(2, 2, 3, 12, &storage).unwrap();
        let tensor = pack_planar_image(view).unwrap();
        assert_eq!(tensor.descriptor().shape(), &[2, 2, 2]);
        assert_eq!(
            bytemuck::cast_slice::<u8, u16>(tensor.allocation_bytes()),
            &[0, 1, 3, 4, 12, 13, 15, 16]
        );
    }
}
