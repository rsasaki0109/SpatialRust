//! Stereo rig contracts, rectification maps, and block-matching disparity.

use spatialrust_image::{Image, ImageView};
use spatialrust_math::{Mat3, Vec3};

use crate::{
    CameraMatrix3, PixelComponent, RelativePose, VisionError, VisionResult,
};

/// Invalid disparity sentinel written by [`stereo_block_match`].
pub const INVALID_DISPARITY: f32 = -1.0;

/// Calibrated two-camera stereo geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoRig {
    left: CameraMatrix3,
    right: CameraMatrix3,
    pose: RelativePose,
}

impl StereoRig {
    /// Creates a rig from left/right intrinsics and the right-camera pose in the
    /// left-camera frame (`X_right = R X_left + t`).
    pub fn try_new(
        left: CameraMatrix3,
        right: CameraMatrix3,
        pose: RelativePose,
    ) -> VisionResult<Self> {
        Ok(Self { left, right, pose })
    }

    /// Returns the left intrinsic matrix.
    pub const fn left(self) -> CameraMatrix3 {
        self.left
    }

    /// Returns the right intrinsic matrix.
    pub const fn right(self) -> CameraMatrix3 {
        self.right
    }

    /// Returns the right camera pose expressed in the left camera frame.
    pub const fn pose(self) -> RelativePose {
        self.pose
    }

    /// Returns the absolute baseline length `|t|`.
    #[must_use]
    pub fn baseline(self) -> f64 {
        self.pose.translation().length()
    }
}

/// Remap grids produced by stereo rectification for explicit `warp::remap`.
#[derive(Clone, Debug, PartialEq)]
pub struct StereoRectifyMaps {
    left_map_x: Image<f32, 1>,
    left_map_y: Image<f32, 1>,
    right_map_x: Image<f32, 1>,
    right_map_y: Image<f32, 1>,
    rectified_left: CameraMatrix3,
    rectified_right: CameraMatrix3,
    baseline: f64,
}

impl StereoRectifyMaps {
    /// Returns the left absolute-x remap image.
    pub const fn left_map_x(&self) -> &Image<f32, 1> {
        &self.left_map_x
    }

    /// Returns the left absolute-y remap image.
    pub const fn left_map_y(&self) -> &Image<f32, 1> {
        &self.left_map_y
    }

    /// Returns the right absolute-x remap image.
    pub const fn right_map_x(&self) -> &Image<f32, 1> {
        &self.right_map_x
    }

    /// Returns the right absolute-y remap image.
    pub const fn right_map_y(&self) -> &Image<f32, 1> {
        &self.right_map_y
    }

    /// Returns the shared rectified left intrinsics.
    pub const fn rectified_left(&self) -> CameraMatrix3 {
        self.rectified_left
    }

    /// Returns the shared rectified right intrinsics.
    pub const fn rectified_right(&self) -> CameraMatrix3 {
        self.rectified_right
    }

    /// Returns the positive rectified baseline along +X.
    pub const fn baseline(&self) -> f64 {
        self.baseline
    }
}

/// Block-matching stereo options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoBmOptions {
    /// Odd SAD window size in pixels.
    pub window_size: usize,
    /// Inclusive minimum positive disparity in pixels.
    pub min_disparity: i32,
    /// Number of disparities searched (must be positive and even-friendly).
    pub num_disparities: i32,
    /// Uniqueness ratio in percent; candidates failing it are invalid.
    pub uniqueness_ratio: f32,
}

impl Default for StereoBmOptions {
    fn default() -> Self {
        Self {
            window_size: 15,
            min_disparity: 0,
            num_disparities: 64,
            uniqueness_ratio: 15.0,
        }
    }
}

impl StereoBmOptions {
    /// Validates window and disparity search settings.
    pub fn validate(self) -> VisionResult<Self> {
        if self.window_size < 3 || self.window_size % 2 == 0 {
            return Err(VisionError::InvalidParameter(
                "stereo BM window_size must be odd and at least 3".into(),
            ));
        }
        if self.num_disparities <= 0 {
            return Err(VisionError::InvalidParameter(
                "stereo BM num_disparities must be positive".into(),
            ));
        }
        if !self.uniqueness_ratio.is_finite() || self.uniqueness_ratio < 0.0 {
            return Err(VisionError::InvalidParameter(
                "stereo BM uniqueness_ratio must be finite and non-negative".into(),
            ));
        }
        Ok(self)
    }
}

/// Builds left/right absolute remap grids that make epipolar lines horizontal.
///
/// Callers feed these maps into `warp::remap`. Identical fronto-parallel
/// cameras with a pure +X baseline produce identity remaps.
pub fn stereo_rectify(
    rig: StereoRig,
    width: usize,
    height: usize,
) -> VisionResult<StereoRectifyMaps> {
    if width == 0 || height == 0 {
        return Err(VisionError::InvalidDimensions(
            "stereo rectify requires positive width and height".into(),
        ));
    }
    let translation = rig.pose().translation();
    let baseline = translation.length();
    if baseline <= f64::EPSILON {
        return Err(VisionError::InvalidParameter(
            "stereo baseline must be non-zero".into(),
        ));
    }
    let e1 = translation.normalize();
    let helper = if e1.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let e2 = e1.cross(helper).normalize();
    let e3 = e1.cross(e2).normalize();
    let r_rect = Mat3::from_rows(
        [e1.x, e1.y, e1.z],
        [e2.x, e2.y, e2.z],
        [e3.x, e3.y, e3.z],
    );
    let left_rotation = r_rect;
    let right_rotation = r_rect.mul_mat3(rig.pose().rotation());
    let fx = 0.5 * (rig.left().matrix().m[0][0] + rig.right().matrix().m[0][0]);
    let fy = 0.5 * (rig.left().matrix().m[1][1] + rig.right().matrix().m[1][1]);
    let cx = (width as f64 - 1.0) * 0.5;
    let cy = (height as f64 - 1.0) * 0.5;
    let new_k = Mat3::from_rows([fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]);
    let new_camera = CameraMatrix3::try_from_pinhole(fx, fy, cx, cy)?;
    let left_maps = build_rectify_maps(rig.left(), left_rotation, new_k, width, height)?;
    let right_maps = build_rectify_maps(rig.right(), right_rotation, new_k, width, height)?;
    Ok(StereoRectifyMaps {
        left_map_x: left_maps.0,
        left_map_y: left_maps.1,
        right_map_x: right_maps.0,
        right_map_y: right_maps.1,
        rectified_left: new_camera,
        rectified_right: new_camera,
        baseline,
    })
}

/// Dense SAD block matching on already-rectified grayscale stereo images.
///
/// Invalid disparities are set to [`INVALID_DISPARITY`]. Search looks for
/// matches of the left pixel in the right image at `x - d`.
pub fn stereo_block_match<T: PixelComponent>(
    left: ImageView<'_, T, 1>,
    right: ImageView<'_, T, 1>,
    options: StereoBmOptions,
) -> VisionResult<Image<f32, 1>> {
    let options = options.validate()?;
    if left.width() != right.width() || left.height() != right.height() {
        return Err(VisionError::ShapeMismatch(
            "stereo BM frames must share width and height".into(),
        ));
    }
    let width = left.width();
    let height = left.height();
    let radius = options.window_size / 2;
    let mut data = vec![INVALID_DISPARITY; width * height];
    for y in radius..(height.saturating_sub(radius)) {
        for x in radius..(width.saturating_sub(radius)) {
            let mut best_cost = f64::INFINITY;
            let mut second_cost = f64::INFINITY;
            let mut best_d = 0i32;
            let d0 = options.min_disparity;
            let d1 = options.min_disparity + options.num_disparities;
            for disparity in d0..d1 {
                let xr = x as i32 - disparity;
                if xr < radius as i32 || xr >= (width - radius) as i32 {
                    continue;
                }
                let mut cost = 0.0;
                for dy in -(radius as isize)..=(radius as isize) {
                    for dx in -(radius as isize)..=(radius as isize) {
                        let ly = (y as isize + dy) as usize;
                        let lx = (x as isize + dx) as usize;
                        let ry = ly;
                        let rx = (xr as isize + dx) as usize;
                        let left_value = left.get(lx, ly).expect("in-bounds")[0].to_f64();
                        let right_value = right.get(rx, ry).expect("in-bounds")[0].to_f64();
                        cost += (left_value - right_value).abs();
                    }
                }
                if cost < best_cost {
                    second_cost = best_cost;
                    best_cost = cost;
                    best_d = disparity;
                } else if cost < second_cost {
                    second_cost = cost;
                }
            }
            let unique = if !best_cost.is_finite() {
                false
            } else if !second_cost.is_finite() || second_cost <= best_cost {
                true
            } else {
                second_cost >= best_cost * (1.0 + f64::from(options.uniqueness_ratio) / 100.0)
            };
            if unique && best_d > options.min_disparity {
                data[y * width + x] = best_d as f32;
            }
        }
    }
    Ok(Image::try_new(width, height, data)?)
}

/// Converts horizontal disparity to metric depth with `Z = f * B / d`.
pub fn disparity_to_depth(
    disparity: ImageView<'_, f32, 1>,
    focal_length: f64,
    baseline: f64,
) -> VisionResult<Image<f32, 1>> {
    if !focal_length.is_finite() || focal_length <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "disparity_to_depth focal_length must be finite and positive".into(),
        ));
    }
    if !baseline.is_finite() || baseline <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "disparity_to_depth baseline must be finite and positive".into(),
        ));
    }
    let mut data = vec![0.0_f32; disparity.width() * disparity.height()];
    for y in 0..disparity.height() {
        for x in 0..disparity.width() {
            let d = f64::from(disparity.get(x, y).expect("in-bounds")[0]);
            data[y * disparity.width() + x] = if d > 0.0 && d.is_finite() {
                (focal_length * baseline / d) as f32
            } else {
                0.0
            };
        }
    }
    Ok(Image::try_new(disparity.width(), disparity.height(), data)?)
}

/// Reprojects disparity into left-camera XYZ using rectified intrinsics.
pub fn disparity_to_xyz(
    disparity: ImageView<'_, f32, 1>,
    camera: CameraMatrix3,
    baseline: f64,
) -> VisionResult<Image<f32, 3>> {
    if !baseline.is_finite() || baseline <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "disparity_to_xyz baseline must be finite and positive".into(),
        ));
    }
    let fx = camera.matrix().m[0][0];
    let fy = camera.matrix().m[1][1];
    let cx = camera.matrix().m[0][2];
    let cy = camera.matrix().m[1][2];
    let mut data = vec![0.0_f32; disparity.width() * disparity.height() * 3];
    for y in 0..disparity.height() {
        for x in 0..disparity.width() {
            let d = f64::from(disparity.get(x, y).expect("in-bounds")[0]);
            let index = (y * disparity.width() + x) * 3;
            if d > 0.0 && d.is_finite() {
                let z = fx * baseline / d;
                let xx = (x as f64 - cx) * z / fx;
                let yy = (y as f64 - cy) * z / fy;
                data[index] = xx as f32;
                data[index + 1] = yy as f32;
                data[index + 2] = z as f32;
            }
        }
    }
    Ok(Image::try_new(disparity.width(), disparity.height(), data)?)
}

fn build_rectify_maps(
    camera: CameraMatrix3,
    rotation: Mat3<f64>,
    new_k: Mat3<f64>,
    width: usize,
    height: usize,
) -> VisionResult<(Image<f32, 1>, Image<f32, 1>)> {
    let mut map_x = vec![0.0_f32; width * height];
    let mut map_y = vec![0.0_f32; width * height];
    let new_inverse = invert_intrinsic(new_k)?;
    let map_matrix = camera.matrix().mul_mat3(rotation.transpose()).mul_mat3(new_inverse);
    for y in 0..height {
        for x in 0..width {
            let destination = Vec3::new(x as f64, y as f64, 1.0);
            let source = map_matrix.mul_vec3(destination);
            if source.z.abs() <= 1e-12 {
                map_x[y * width + x] = -1.0;
                map_y[y * width + x] = -1.0;
            } else {
                map_x[y * width + x] = (source.x / source.z) as f32;
                map_y[y * width + x] = (source.y / source.z) as f32;
            }
        }
    }
    Ok((Image::try_new(width, height, map_x)?, Image::try_new(width, height, map_y)?))
}

fn invert_intrinsic(matrix: Mat3<f64>) -> VisionResult<Mat3<f64>> {
    let fx = matrix.m[0][0];
    let fy = matrix.m[1][1];
    let cx = matrix.m[0][2];
    let cy = matrix.m[1][2];
    if fx.abs() <= f64::EPSILON || fy.abs() <= f64::EPSILON {
        return Err(VisionError::InvalidParameter(
            "rectified intrinsics must have non-zero focal lengths".into(),
        ));
    }
    Ok(Mat3::from_rows(
        [1.0 / fx, 0.0, -cx / fx],
        [0.0, 1.0 / fy, -cy / fy],
        [0.0, 0.0, 1.0],
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        disparity_to_depth, stereo_block_match, stereo_rectify, StereoBmOptions, StereoRig,
        INVALID_DISPARITY,
    };
    use crate::{CameraMatrix3, RelativePose};
    use spatialrust_camera::CameraIntrinsics;
    use spatialrust_image::Image;
    use spatialrust_math::{Mat3, Vec3};

    fn camera() -> CameraMatrix3 {
        let intrinsics = CameraIntrinsics::try_new(400.0, 400.0, 80.0, 60.0, 160, 120).unwrap();
        CameraMatrix3::from_intrinsics(intrinsics)
    }

    #[test]
    fn fronto_parallel_stereo_recovers_plane_depth() {
        let camera = camera();
        let baseline = 0.1;
        let pose = RelativePose::try_new(
            Mat3::<f64>::identity(),
            Vec3::new(baseline, 0.0, 0.0),
        )
        .unwrap();
        let rig = StereoRig::try_new(camera, camera, pose).unwrap();
        let maps = stereo_rectify(rig, 160, 120).unwrap();
        assert!((maps.baseline() - baseline).abs() < 1e-12);

        // Synthetic textured fronto-parallel plane at Z=2 with disparity = f*B/Z = 20.
        let depth = 2.0;
        let disparity = (400.0 * baseline / depth) as i32;
        let width = 160;
        let height = 120;
        let mut left = vec![0u8; width * height];
        let mut right = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let value = (((x * 17 + y * 29) % 200) + 20) as u8;
                left[y * width + x] = value;
                let xr = x as i32 - disparity;
                if (0..width as i32).contains(&xr) {
                    right[y * width + xr as usize] = value;
                }
            }
        }
        let left = Image::<u8, 1>::try_new(width, height, left).unwrap();
        let right = Image::<u8, 1>::try_new(width, height, right).unwrap();
        let disparity_map = stereo_block_match(
            left.view(),
            right.view(),
            StereoBmOptions {
                window_size: 11,
                min_disparity: 1,
                num_disparities: 64,
                uniqueness_ratio: 5.0,
            },
        )
        .unwrap();
        let center = disparity_map.get(80, 60).unwrap()[0];
        assert!(
            (center - disparity as f32).abs() <= 1.0,
            "center disparity {center}, expected {disparity}"
        );
        assert_ne!(center, INVALID_DISPARITY);
        let depth_map = disparity_to_depth(disparity_map.view(), 400.0, baseline).unwrap();
        let recovered = depth_map.get(80, 60).unwrap()[0];
        assert!((f64::from(recovered) - depth).abs() < 0.15);
    }
}