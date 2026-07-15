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
        Self {
            depth_scale: 1.0,
            min_depth: f32::EPSILON,
            max_depth: f32::INFINITY,
        }
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

#[derive(Clone, Copy)]
struct FastPinholeF32 {
    inv_fx: f32,
    inv_fy: f32,
    cx: f32,
    cy: f32,
}

impl FastPinholeF32 {
    fn from_camera(camera: &PinholeCamera) -> Self {
        Self {
            inv_fx: (1.0 / camera.intrinsics.fx) as f32,
            inv_fy: (1.0 / camera.intrinsics.fy) as f32,
            cx: camera.intrinsics.cx as f32,
            cy: camera.intrinsics.cy as f32,
        }
    }

    #[inline]
    fn unproject(self, px: f32, py: f32, z: f32) -> (f32, f32, f32) {
        (
            (px - self.cx) * self.inv_fx * z,
            (py - self.cy) * self.inv_fy * z,
            z,
        )
    }
}

/// Converts depth into a dense `H×W×3` XYZ buffer (row-major interleaved).
///
/// Invalid / out-of-range depths become `NaN` triples, matching OpenCV
/// `rgbd.depthTo3d` dense semantics for undistorted pinhole cameras.
pub fn depth_to_xyz_dense(
    depth: ImageView<'_, f32, 1>,
    camera: &PinholeCamera,
    options: DepthConversionOptions,
) -> Result<Vec<f32>, RgbdError> {
    let len = depth
        .width()
        .saturating_mul(depth.height())
        .saturating_mul(3);
    let mut out = vec![0.0f32; len];
    depth_to_xyz_dense_into(depth, camera, options, &mut out)?;
    Ok(out)
}

/// Fills a caller-provided dense `H×W×3` XYZ buffer (length must be `H*W*3`).
pub fn depth_to_xyz_dense_into(
    depth: ImageView<'_, f32, 1>,
    camera: &PinholeCamera,
    options: DepthConversionOptions,
    out: &mut [f32],
) -> Result<(), RgbdError> {
    validate_depth(depth, camera)?;
    options.validate()?;
    let expected = depth
        .width()
        .saturating_mul(depth.height())
        .saturating_mul(3);
    if out.len() != expected {
        return Err(RgbdError::InvalidOptions(format!(
            "dense XYZ buffer length must be {expected}, found {}",
            out.len()
        )));
    }
    if camera.distortion.is_identity() {
        fill_xyz_dense_identity(depth, camera, options, out);
    } else {
        fill_xyz_dense_distorted(depth, camera, options, out)?;
    }
    Ok(())
}

fn fill_xyz_dense_identity(
    depth: ImageView<'_, f32, 1>,
    camera: &PinholeCamera,
    options: DepthConversionOptions,
    out: &mut [f32],
) {
    let pin = FastPinholeF32::from_camera(camera);
    let width = depth.width();
    let height = depth.height();
    let scale = options.depth_scale;
    let min_d = options.min_depth;
    let max_d = options.max_depth;
    let scale_is_one = scale == 1.0;
    let max_is_inf = !max_d.is_finite();
    // Per-column `(x - cx) / fx` factors so the inner loop is multiply-add only.
    let x_mul: Vec<f32> = (0..width)
        .map(|x| (x as f32 - pin.cx) * pin.inv_fx)
        .collect();
    let pixels = width.saturating_mul(height);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(1);
    // Thread spawn overhead dominates below ~2M pixels on typical hosts.
    if threads == 1 || pixels < 2_000_000 {
        fill_xyz_dense_identity_rows(
            depth,
            pin,
            &x_mul,
            0,
            height,
            width,
            scale,
            min_d,
            max_d,
            scale_is_one,
            max_is_inf,
            out,
        );
        return;
    }

    let row_stride = width * 3;
    let chunk = (height + threads - 1) / threads;
    std::thread::scope(|scope| {
        let mut rest = out;
        let mut y0 = 0usize;
        while y0 < height {
            let y1 = (y0 + chunk).min(height);
            let take = (y1 - y0) * row_stride;
            let (chunk_out, next) = rest.split_at_mut(take);
            rest = next;
            let x_mul = &x_mul;
            scope.spawn(move || {
                fill_xyz_dense_identity_rows(
                    depth,
                    pin,
                    x_mul,
                    y0,
                    y1,
                    width,
                    scale,
                    min_d,
                    max_d,
                    scale_is_one,
                    max_is_inf,
                    chunk_out,
                );
            });
            y0 = y1;
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn fill_xyz_dense_identity_rows(
    depth: ImageView<'_, f32, 1>,
    pin: FastPinholeF32,
    x_mul: &[f32],
    y0: usize,
    y1: usize,
    width: usize,
    scale: f32,
    min_d: f32,
    max_d: f32,
    scale_is_one: bool,
    max_is_inf: bool,
    out: &mut [f32],
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if scale_is_one && max_is_inf && is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection above; slices are row-validated.
            unsafe {
                fill_xyz_dense_identity_rows_avx2(
                    depth, pin, x_mul, y0, y1, width, min_d, out,
                );
            }
            return;
        }
    }

    let mut o = 0usize;
    for y in y0..y1 {
        let row = depth.row(y).expect("validated row");
        let y_mul = (y as f32 - pin.cy) * pin.inv_fy;
        for x in 0..width {
            let raw = row[x];
            let meters = if scale_is_one { raw } else { raw * scale };
            // Ordered compares treat NaN as invalid (NaN >= x is false).
            let z = if max_is_inf {
                if meters >= min_d {
                    meters
                } else {
                    f32::NAN
                }
            } else if meters >= min_d && meters <= max_d {
                meters
            } else {
                f32::NAN
            };
            out[o] = x_mul[x] * z;
            out[o + 1] = y_mul * z;
            out[o + 2] = z;
            o += 3;
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn fill_xyz_dense_identity_rows_avx2(
    depth: ImageView<'_, f32, 1>,
    pin: FastPinholeF32,
    x_mul: &[f32],
    y0: usize,
    y1: usize,
    width: usize,
    min_d: f32,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;

    let min_v = _mm256_set1_ps(min_d);
    let nan_v = _mm256_set1_ps(f32::NAN);
    let mut o = 0usize;
    for y in y0..y1 {
        let row = depth.row(y).expect("validated row");
        let y_mul = _mm256_set1_ps((y as f32 - pin.cy) * pin.inv_fy);
        let mut x = 0usize;
        while x + 8 <= width {
            let z_raw = _mm256_loadu_ps(row.as_ptr().add(x));
            // _CMP_GE_OQ: ordered → NaN lanes become false.
            let valid = _mm256_cmp_ps(z_raw, min_v, _CMP_GE_OQ);
            let z = _mm256_blendv_ps(nan_v, z_raw, valid);
            let xm = _mm256_loadu_ps(x_mul.as_ptr().add(x));
            let xs = _mm256_mul_ps(xm, z);
            let ys = _mm256_mul_ps(y_mul, z);

            let mut xa = [0.0f32; 8];
            let mut ya = [0.0f32; 8];
            let mut za = [0.0f32; 8];
            _mm256_storeu_ps(xa.as_mut_ptr(), xs);
            _mm256_storeu_ps(ya.as_mut_ptr(), ys);
            _mm256_storeu_ps(za.as_mut_ptr(), z);
            for i in 0..8 {
                *out.get_unchecked_mut(o) = xa[i];
                *out.get_unchecked_mut(o + 1) = ya[i];
                *out.get_unchecked_mut(o + 2) = za[i];
                o += 3;
            }
            x += 8;
        }
        let y_mul_s = (y as f32 - pin.cy) * pin.inv_fy;
        while x < width {
            let meters = *row.get_unchecked(x);
            let z = if meters >= min_d {
                meters
            } else {
                f32::NAN
            };
            *out.get_unchecked_mut(o) = *x_mul.get_unchecked(x) * z;
            *out.get_unchecked_mut(o + 1) = y_mul_s * z;
            *out.get_unchecked_mut(o + 2) = z;
            o += 3;
            x += 1;
        }
    }
}

fn fill_xyz_dense_distorted(
    depth: ImageView<'_, f32, 1>,
    camera: &PinholeCamera,
    options: DepthConversionOptions,
    out: &mut [f32],
) -> Result<(), RgbdError> {
    let width = depth.width();
    let mut o = 0usize;
    for y in 0..depth.height() {
        for x in 0..width {
            let meters =
                depth.get(x, y).expect("validated image coordinates")[0] * options.depth_scale;
            if meters.is_finite() && meters >= options.min_depth && meters <= options.max_depth {
                let point = camera.unproject(Vec2 { x: x as f64, y: y as f64 }, meters as f64)?;
                out[o] = point.x as f32;
                out[o + 1] = point.y as f32;
                out[o + 2] = point.z as f32;
            } else {
                out[o] = f32::NAN;
                out[o + 1] = f32::NAN;
                out[o + 2] = f32::NAN;
            }
            o += 3;
        }
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
    if camera.distortion.is_identity() {
        let pin = FastPinholeF32::from_camera(camera);
        let scale = options.depth_scale;
        let min_d = options.min_depth;
        let max_d = options.max_depth;
        for y in 0..depth.height() {
            let row = depth.row(y).expect("validated row");
            let py = y as f32;
            for x in 0..depth.width() {
                let meters = row[x] * scale;
                if !(meters.is_finite() && meters >= min_d && meters <= max_d) {
                    continue;
                }
                let (xx, yy, zz) = pin.unproject(x as f32, py, meters);
                xs.push(xx);
                ys.push(yy);
                zs.push(zz);
            }
        }
    } else {
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
    if camera.distortion.is_identity() {
        let pin = FastPinholeF32::from_camera(camera);
        let scale = options.depth_scale;
        let min_d = options.min_depth;
        let max_d = options.max_depth;
        for y in 0..depth.height() {
            let depth_row = depth.row(y).expect("validated row");
            let color_row = color.row(y).expect("validated row");
            let py = y as f32;
            for x in 0..depth.width() {
                let meters = depth_row[x] * scale;
                if !(meters.is_finite() && meters >= min_d && meters <= max_d) {
                    continue;
                }
                let (xx, yy, zz) = pin.unproject(x as f32, py, meters);
                let c = x * 3;
                xs.push(xx);
                ys.push(yy);
                zs.push(zz);
                rs.push(color_row[c]);
                gs.push(color_row[c + 1]);
                bs.push(color_row[c + 2]);
            }
        }
    } else {
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
    use super::{
        depth_to_point_cloud, depth_to_xyz_dense, rgbd_to_point_cloud, DepthConversionOptions,
    };
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
    fn dense_xyz_uses_nan_for_invalid() {
        let depth = Image::<f32, 1>::try_new(2, 2, vec![1.0, 0.0, f32::NAN, 2.0]).unwrap();
        let xyz = depth_to_xyz_dense(depth.view(), &camera(), Default::default()).unwrap();
        assert_eq!(xyz.len(), 12);
        assert_eq!(&xyz[0..3], &[0.0, 0.0, 1.0]);
        assert!(xyz[3].is_nan() && xyz[4].is_nan() && xyz[5].is_nan());
        assert!(xyz[6].is_nan());
        assert_eq!(&xyz[9..12], &[1.0, 1.0, 2.0]);
    }

    #[test]
    fn rgb_fields_follow_valid_depths() {
        let depth = Image::<f32, 1>::try_new(2, 2, vec![1.0, 0.0, 3.0, 2.0]).unwrap();
        let color =
            Image::<u8, 3>::try_new(2, 2, vec![10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42])
                .unwrap();
        let options = DepthConversionOptions {
            max_depth: 2.0,
            ..Default::default()
        };
        let cloud = rgbd_to_point_cloud(depth.view(), color.view(), &camera(), options).unwrap();
        assert_eq!(cloud.len(), 2);
        assert_eq!(cloud.field("r").unwrap(), &PointBuffer::U8(vec![10, 40]));
        assert_eq!(cloud.field("g").unwrap(), &PointBuffer::U8(vec![11, 41]));
        assert_eq!(cloud.field("b").unwrap(), &PointBuffer::U8(vec![12, 42]));
    }
}
