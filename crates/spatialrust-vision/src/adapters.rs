//! Explicit image ↔ tensor adapters for AI pipelines.
//!
//! These helpers materialize contiguous host tensors and vision maps. They never
//! call an inference backend and never imply device transfers.

use spatialrust_image::{ImageView, PlanarImage};
use spatialrust_tensor::{DataType, Device, TensorBuffer, TensorDescriptor};

use crate::{
    letterbox, pack_chw, BoundingBox2, DepthMap, Detection, Interpolation, LetterboxTransform,
    VisionError, VisionResult,
};

/// Letterboxes an RGB `u8` image and packs it as contiguous NCHW `f32` `[1,3,H,W]`.
pub fn rgb_u8_to_nchw_f32(
    image: ImageView<'_, u8, 3>,
    width: usize,
    height: usize,
    interpolation: Interpolation,
    pad_value: [u8; 3],
    scale: f32,
    mean: [f32; 3],
    std: [f32; 3],
) -> VisionResult<(TensorBuffer, LetterboxTransform)> {
    let (letterboxed, mapping) = letterbox(image, width, height, interpolation, pad_value)?;
    let planar = pack_chw(letterboxed.view(), scale, mean, std)?;
    Ok((planar_f32_to_nchw(&planar)?, mapping))
}

/// Packs a planar CHW `f32` image into contiguous NCHW `[1,C,H,W]`.
pub fn planar_f32_to_nchw<const CHANNELS: usize>(
    planar: &PlanarImage<f32, CHANNELS>,
) -> VisionResult<TensorBuffer> {
    let values = planar.as_slice().to_vec();
    TensorBuffer::try_from_f32(
        values,
        TensorDescriptor::contiguous(
            DataType::F32,
            vec![1, CHANNELS, planar.height(), planar.width()],
            Device::CPU,
        ),
    )
    .map_err(|error| VisionError::InvalidParameter(error.to_string()))
}

/// Converts a host depth tensor into a [`DepthMap`].
///
/// Accepted shapes: `[H,W]`, `[1,H,W]`, or `[1,1,H,W]`.
pub fn depth_tensor_to_depth_map(tensor: &TensorBuffer) -> VisionResult<DepthMap> {
    let (height, width, values) = flatten_single_channel_f32(tensor, "depth")?;
    DepthMap::try_new(width, height, values)
}

/// Converts a host score tensor into a binary mask via threshold.
///
/// Accepted shapes: `[H,W]`, `[1,H,W]`, or `[1,1,H,W]`.
pub fn score_tensor_to_binary_mask(
    tensor: &TensorBuffer,
    threshold: f32,
) -> VisionResult<crate::BinaryMask> {
    if !threshold.is_finite() {
        return Err(VisionError::InvalidParameter("mask threshold must be finite".into()));
    }
    let (height, width, values) = flatten_single_channel_f32(tensor, "scores")?;
    crate::BinaryMask::try_new(
        width,
        height,
        values.into_iter().map(|value| u8::from(value.is_finite() && value >= threshold)).collect(),
    )
}

/// Decodes axis-aligned detections from an `[N,6]` tensor of
/// `x0,y0,x1,y1,score,class`.
pub fn detection_tensor_to_detections(tensor: &TensorBuffer) -> VisionResult<Vec<Detection>> {
    let descriptor = tensor.descriptor();
    if descriptor.dtype() != DataType::F32 {
        return Err(VisionError::InvalidParameter(format!(
            "detections require f32, found {:?}",
            descriptor.dtype()
        )));
    }
    let shape = descriptor.shape();
    if shape.len() != 2 || shape[1] != 6 {
        return Err(VisionError::ShapeMismatch(format!(
            "detections expect [N,6], found {shape:?}"
        )));
    }
    if !descriptor.is_c_contiguous() || descriptor.byte_offset() != 0 {
        return Err(VisionError::InvalidParameter(
            "detections tensor must be contiguous with byte_offset=0".into(),
        ));
    }
    let values = f32_slice(tensor, "detections")?;
    let mut detections = Vec::with_capacity(shape[0]);
    for row in values.chunks_exact(6) {
        let bbox = BoundingBox2::try_new(row[0], row[1], row[2], row[3])
            .map_err(|error| VisionError::InvalidParameter(error.to_string()))?;
        detections.push(Detection {
            bbox,
            score: row[4],
            class_id: row[5] as i64,
        });
    }
    Ok(detections)
}

fn flatten_single_channel_f32(
    tensor: &TensorBuffer,
    name: &str,
) -> VisionResult<(usize, usize, Vec<f32>)> {
    let descriptor = tensor.descriptor();
    if descriptor.dtype() != DataType::F32 {
        return Err(VisionError::InvalidParameter(format!(
            "{name} requires f32, found {:?}",
            descriptor.dtype()
        )));
    }
    if !descriptor.is_c_contiguous() || descriptor.byte_offset() != 0 {
        return Err(VisionError::InvalidParameter(format!(
            "{name} tensor must be contiguous with byte_offset=0"
        )));
    }
    let shape = descriptor.shape();
    let (height, width) = match shape {
        [h, w] => (*h, *w),
        [1, h, w] => (*h, *w),
        [1, 1, h, w] => (*h, *w),
        other => {
            return Err(VisionError::ShapeMismatch(format!(
                "{name} expects [H,W], [1,H,W], or [1,1,H,W]; found {other:?}"
            )));
        }
    };
    let values = f32_slice(tensor, name)?.to_vec();
    if values.len() != height.saturating_mul(width) {
        return Err(VisionError::ShapeMismatch(format!(
            "{name} length {} does not match {width}x{height}",
            values.len()
        )));
    }
    Ok((height, width, values))
}

fn f32_slice<'a>(tensor: &'a TensorBuffer, name: &str) -> VisionResult<&'a [f32]> {
    let bytes = tensor.allocation_bytes();
    if bytes.len() % 4 != 0 {
        return Err(VisionError::InvalidParameter(format!(
            "{name} f32 allocation is not a multiple of 4 bytes"
        )));
    }
    Ok(bytemuck::cast_slice(bytes))
}

#[cfg(test)]
mod tests {
    use super::{
        depth_tensor_to_depth_map, detection_tensor_to_detections, planar_f32_to_nchw,
        rgb_u8_to_nchw_f32, score_tensor_to_binary_mask,
    };
    use crate::Interpolation;
    use spatialrust_image::{Image, PlanarImage};
    use spatialrust_tensor::{DataType, Device, TensorBuffer, TensorDescriptor};

    #[test]
    fn letterbox_pack_produces_batch_nchw() {
        let image = Image::<u8, 3>::try_new(4, 2, vec![255; 4 * 2 * 3]).unwrap();
        let (tensor, mapping) = rgb_u8_to_nchw_f32(
            image.view(),
            4,
            4,
            Interpolation::Nearest,
            [0, 0, 0],
            1.0 / 255.0,
            [0.0; 3],
            [1.0; 3],
        )
        .unwrap();
        assert_eq!(tensor.descriptor().shape(), &[1, 3, 4, 4]);
        assert!(mapping.scale > 0.0);
    }

    #[test]
    fn planar_to_nchw_keeps_values() {
        let planar =
            PlanarImage::<f32, 3>::try_new(2, 2, (0..12).map(|v| v as f32).collect()).unwrap();
        let tensor = planar_f32_to_nchw(&planar).unwrap();
        assert_eq!(tensor.descriptor().shape(), &[1, 3, 2, 2]);
        assert_eq!(
            bytemuck::cast_slice::<u8, f32>(tensor.allocation_bytes()),
            planar.as_slice()
        );
    }

    #[test]
    fn depth_and_mask_decode_single_channel_shapes() {
        let depth = TensorBuffer::try_from_f32(
            vec![1.0, 2.0, 3.0, 4.0],
            TensorDescriptor::contiguous(DataType::F32, vec![1, 1, 2, 2], Device::CPU),
        )
        .unwrap();
        let map = depth_tensor_to_depth_map(&depth).unwrap();
        assert_eq!(map.image().width(), 2);
        assert_eq!(map.image().as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        let scores = TensorBuffer::try_from_f32(
            vec![0.1, 0.9, 0.4, 0.8],
            TensorDescriptor::contiguous(DataType::F32, vec![2, 2], Device::CPU),
        )
        .unwrap();
        let mask = score_tensor_to_binary_mask(&scores, 0.5).unwrap();
        assert_eq!(mask.image().as_slice(), &[0, 1, 0, 1]);
    }

    #[test]
    fn detections_decode_rows() {
        let tensor = TensorBuffer::try_from_f32(
            vec![0.0, 0.0, 2.0, 2.0, 0.9, 1.0],
            TensorDescriptor::contiguous(DataType::F32, vec![1, 6], Device::CPU),
        )
        .unwrap();
        let detections = detection_tensor_to_detections(&tensor).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class_id, 1);
        assert!((detections[0].score - 0.9).abs() < 1e-6);
    }
}
