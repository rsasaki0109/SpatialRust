use crate::{CameraError, PinholeCamera};
use spatialrust_core::{
    PointBuffer, PointBufferSet, PointCloud, SpatialError, SpatialMetadata, StandardSchemas,
};
use spatialrust_image::ImageView;
use spatialrust_math::Vec2;

/// Errors from RGB-D conversion.
#[derive(Debug, thiserror::Error)]
pub enum RgbdError {
    /// Camera dimensions and depth image dimensions differ.
    #[error("depth dimensions {image_width}x{image_height} do not match camera dimensions {camera_width}x{camera_height}")]
    DepthDimensionMismatch {
        /// Depth image width.
        image_width: usize,
        /// Depth image height.
        image_height: usize,
        /// Calibrated camera width.
        camera_width: usize,
        /// Calibrated camera height.
        camera_height: usize,
    },
    /// RGB and depth image dimensions differ.
    #[error("color dimensions {color_width}x{color_height} do not match depth dimensions {depth_width}x{depth_height}")]
    ColorDimensionMismatch {
        /// Color image width.
        color_width: usize,
        /// Color image height.
        color_height: usize,
        /// Depth image width.
        depth_width: usize,
        /// Depth image height.
        depth_height: usize,
    },
    /// Conversion settings were invalid.
    #[error("{0}")]
    InvalidOptions(String),
    /// Camera geometry failed.
    #[error(transparent)]
    Camera(#[from] CameraError),
    /// Point cloud construction failed.
    #[error(transparent)]
    Spatial(#[from] SpatialError),
}

/// Controls conversion from stored depth values to metric camera coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthConversionOptions {
    /// Multiplier converting each stored depth value to meters.
    pub depth_scale: f32,
    /// Inclusive minimum accepted metric depth.
    pub min_depth: f32,
    /// Inclusive maximum accepted metric depth.
    pub max_depth: f32,
}

impl Default for DepthConversionOptions {
    fn default() -> Self {
        Self { depth_scale: 1.0, min_depth: f32::EPSILON, max_depth: f32::INFINITY }
    }
}

impl DepthConversionOptions {
    fn validate(self) -> Result<(), RgbdError> {
        if !self.depth_scale.is_finite() || self.depth_scale <= 0.0 {
            return Err(RgbdError::InvalidOptions(
                "depth_scale must be finite and positive".to_owned(),
            ));
        }
        if !self.min_depth.is_finite()
            || self.min_depth < 0.0
            || self.max_depth.is_nan()
            || self.max_depth < self.min_depth
        {
            return Err(RgbdError::InvalidOptions(
                "depth range must be ordered, non-negative, and not NaN".to_owned(),
            ));
        }
        Ok(())
    }
}

fn validate_depth(depth: ImageView<'_, f32, 1>, camera: &PinholeCamera) -> Result<(), RgbdError> {
    let intrinsics = camera.intrinsics;
    if depth.width() != intrinsics.width || depth.height() != intrinsics.height {
        return Err(RgbdError::DepthDimensionMismatch {
            image_width: depth.width(),
            image_height: depth.height(),
            camera_width: intrinsics.width,
            camera_height: intrinsics.height,
        });
    }
    Ok(())
}

/// Converts an aligned depth image into an XYZ point cloud.
///
/// Zero, non-finite, and out-of-range depths are omitted.
pub fn depth_to_point_cloud(
    depth: ImageView<'_, f32, 1>,
    camera: &PinholeCamera,
    options: DepthConversionOptions,
) -> Result<PointCloud, RgbdError> {
    validate_depth(depth, camera)?;
    options.validate()?;
    let capacity = depth.width().saturating_mul(depth.height());
    let mut xs = Vec::with_capacity(capacity);
    let mut ys = Vec::with_capacity(capacity);
    let mut zs = Vec::with_capacity(capacity);
    for y in 0..depth.height() {
        for x in 0..depth.width() {
            let meters =
                depth.get(x, y).expect("validated image coordinates")[0] * options.depth_scale;
            if !meters.is_finite() || meters < options.min_depth || meters > options.max_depth {
                continue;
            }
            let point = camera.unproject(Vec2 { x: x as f64, y: y as f64 }, meters as f64)?;
            xs.push(point.x as f32);
            ys.push(point.y as f32);
            zs.push(point.z as f32);
        }
    }
    let mut buffers = PointBufferSet::new();
    buffers.insert("x", PointBuffer::from_f32(xs));
    buffers.insert("y", PointBuffer::from_f32(ys));
    buffers.insert("z", PointBuffer::from_f32(zs));
    Ok(PointCloud::try_from_parts(
        StandardSchemas::point_xyz(),
        buffers,
        SpatialMetadata::default(),
    )?)
}

/// Converts aligned RGB and depth images into an XYZRGB point cloud.
///
/// RGB channel order is preserved as `r`, `g`, and `b` `u8` fields.
pub fn rgbd_to_point_cloud(
    depth: ImageView<'_, f32, 1>,
    color: ImageView<'_, u8, 3>,
    camera: &PinholeCamera,
    options: DepthConversionOptions,
) -> Result<PointCloud, RgbdError> {
    validate_depth(depth, camera)?;
    options.validate()?;
    if color.width() != depth.width() || color.height() != depth.height() {
        return Err(RgbdError::ColorDimensionMismatch {
            color_width: color.width(),
            color_height: color.height(),
            depth_width: depth.width(),
            depth_height: depth.height(),
        });
    }

    let capacity = depth.width().saturating_mul(depth.height());
    let mut xs = Vec::with_capacity(capacity);
    let mut ys = Vec::with_capacity(capacity);
    let mut zs = Vec::with_capacity(capacity);
    let mut rs = Vec::with_capacity(capacity);
    let mut gs = Vec::with_capacity(capacity);
    let mut bs = Vec::with_capacity(capacity);
    for y in 0..depth.height() {
        for x in 0..depth.width() {
            let meters =
                depth.get(x, y).expect("validated image coordinates")[0] * options.depth_scale;
            if !meters.is_finite() || meters < options.min_depth || meters > options.max_depth {
                continue;
            }
            let point = camera.unproject(Vec2 { x: x as f64, y: y as f64 }, meters as f64)?;
            let rgb = color.get(x, y).expect("validated image coordinates");
            xs.push(point.x as f32);
            ys.push(point.y as f32);
            zs.push(point.z as f32);
            rs.push(rgb[0]);
            gs.push(rgb[1]);
            bs.push(rgb[2]);
        }
    }
    let mut buffers = PointBufferSet::new();
    buffers.insert("x", PointBuffer::from_f32(xs));
    buffers.insert("y", PointBuffer::from_f32(ys));
    buffers.insert("z", PointBuffer::from_f32(zs));
    buffers.insert("r", PointBuffer::U8(rs));
    buffers.insert("g", PointBuffer::U8(gs));
    buffers.insert("b", PointBuffer::U8(bs));
    Ok(PointCloud::try_from_parts(
        StandardSchemas::point_xyzrgb(),
        buffers,
        SpatialMetadata::default(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::{depth_to_point_cloud, rgbd_to_point_cloud, DepthConversionOptions};
    use crate::{CameraIntrinsics, PinholeCamera};
    use spatialrust_core::PointBuffer;
    use spatialrust_image::Image;

    fn camera() -> PinholeCamera {
        PinholeCamera::new(CameraIntrinsics::try_new(2.0, 2.0, 0.0, 0.0, 2, 2).unwrap())
    }

    #[test]
    fn skips_invalid_depth_and_unprojects() {
        let depth = Image::<f32, 1>::try_new(2, 2, vec![1.0, 0.0, f32::NAN, 2.0]).unwrap();
        let cloud = depth_to_point_cloud(depth.view(), &camera(), Default::default()).unwrap();
        assert_eq!(cloud.len(), 2);
        assert_eq!(cloud.field("x").unwrap().as_f32().unwrap(), &[0.0, 1.0]);
        assert_eq!(cloud.field("y").unwrap().as_f32().unwrap(), &[0.0, 1.0]);
        assert_eq!(cloud.field("z").unwrap().as_f32().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn rgb_fields_follow_valid_depths() {
        let depth = Image::<f32, 1>::try_new(2, 2, vec![1.0, 0.0, 3.0, 2.0]).unwrap();
        let color =
            Image::<u8, 3>::try_new(2, 2, vec![10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42])
                .unwrap();
        let options = DepthConversionOptions { max_depth: 2.0, ..Default::default() };
        let cloud = rgbd_to_point_cloud(depth.view(), color.view(), &camera(), options).unwrap();
        assert_eq!(cloud.len(), 2);
        assert_eq!(cloud.field("r").unwrap(), &PointBuffer::U8(vec![10, 40]));
        assert_eq!(cloud.field("g").unwrap(), &PointBuffer::U8(vec![11, 41]));
        assert_eq!(cloud.field("b").unwrap(), &PointBuffer::U8(vec![12, 42]));
    }
}
