//! Robust feature tracking and calibrated visual odometry building blocks.

use spatialrust_image::ImageView;
use spatialrust_math::{Vec2, Vec3};

use crate::{
    estimate_essential_ransac, recover_relative_pose, solve_pnp_ransac, track_points_lucas_kanade,
    AbsolutePose, CameraMatrix3, Keypoint2, LucasKanadeOptions, ObjectImageCorrespondence,
    PointCorrespondence2, RelativePose, RobustEstimationOptions, VisionError, VisionResult,
};

/// Controls deterministic spatial distribution of candidate keypoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridSelectionOptions {
    /// Width of one grid cell in pixels.
    pub cell_width: usize,
    /// Height of one grid cell in pixels.
    pub cell_height: usize,
    /// Maximum retained candidates per cell.
    pub max_per_cell: usize,
}

impl Default for GridSelectionOptions {
    fn default() -> Self {
        Self { cell_width: 32, cell_height: 32, max_per_cell: 4 }
    }
}

/// Retains the strongest keypoints in each image cell.
///
/// Output is ordered by cell scan order, then decreasing response, with the
/// input index as the deterministic tie breaker.
pub fn select_keypoints_grid(
    keypoints: &[Keypoint2],
    width: usize,
    height: usize,
    options: GridSelectionOptions,
) -> VisionResult<Vec<Keypoint2>> {
    if width == 0
        || height == 0
        || options.cell_width == 0
        || options.cell_height == 0
        || options.max_per_cell == 0
    {
        return Err(VisionError::InvalidParameter(
            "grid selection dimensions and limits must be positive".into(),
        ));
    }
    let columns = width.div_ceil(options.cell_width);
    let rows = height.div_ceil(options.cell_height);
    let mut cells = vec![Vec::<(usize, Keypoint2)>::new(); columns * rows];
    for (index, &keypoint) in keypoints.iter().enumerate() {
        if keypoint.x() < 0.0
            || keypoint.y() < 0.0
            || keypoint.x() >= width as f32
            || keypoint.y() >= height as f32
        {
            continue;
        }
        let column = keypoint.x() as usize / options.cell_width;
        let row = keypoint.y() as usize / options.cell_height;
        cells[row * columns + column].push((index, keypoint));
    }
    let mut selected = Vec::new();
    for cell in &mut cells {
        cell.sort_by(|(left_index, left), (right_index, right)| {
            right.response().total_cmp(&left.response()).then(left_index.cmp(right_index))
        });
        selected.extend(cell.iter().take(options.max_per_cell).map(|(_, point)| *point));
    }
    Ok(selected)
}

/// Forward/backward consistency settings layered over pyramidal LK.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustTrackOptions {
    /// Underlying LK settings in both directions.
    pub lucas_kanade: LucasKanadeOptions,
    /// Maximum accepted round-trip distance in pixels.
    pub max_forward_backward_error: f64,
}

impl Default for RobustTrackOptions {
    fn default() -> Self {
        Self { lucas_kanade: LucasKanadeOptions::default(), max_forward_backward_error: 1.0 }
    }
}

/// One target coordinate, validity decision, and round-trip error per source point.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustTracks {
    next_points: Vec<Vec2<f64>>,
    status: Vec<bool>,
    forward_backward_errors: Vec<f64>,
}

impl RobustTracks {
    /// Returns one next-frame coordinate per source point.
    pub fn next_points(&self) -> &[Vec2<f64>] {
        &self.next_points
    }
    /// Returns the combined forward, backward, and threshold decision.
    pub fn status(&self) -> &[bool] {
        &self.status
    }
    /// Returns round-trip pixel error, or infinity when either direction failed.
    pub fn forward_backward_errors(&self) -> &[f64] {
        &self.forward_backward_errors
    }
    /// Returns the accepted track count.
    pub fn accepted_count(&self) -> usize {
        self.status.iter().filter(|&&value| value).count()
    }
}

/// Tracks points in both directions and rejects inconsistent round trips.
pub fn track_points_forward_backward<T: crate::PixelComponent>(
    previous: ImageView<'_, T, 1>,
    next: ImageView<'_, T, 1>,
    points: &[Vec2<f64>],
    options: RobustTrackOptions,
) -> VisionResult<RobustTracks> {
    if !options.max_forward_backward_error.is_finite() || options.max_forward_backward_error < 0.0 {
        return Err(VisionError::InvalidParameter(
            "forward/backward threshold must be finite and non-negative".into(),
        ));
    }
    let forward = track_points_lucas_kanade(previous, next, points, options.lucas_kanade)?;
    let backward =
        track_points_lucas_kanade(next, previous, forward.next_points(), options.lucas_kanade)?;
    let mut status = Vec::with_capacity(points.len());
    let mut errors = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let valid = forward.status()[index] && backward.status()[index];
        let error = if valid {
            let dx = backward.next_points()[index].x - point.x;
            let dy = backward.next_points()[index].y - point.y;
            dx.hypot(dy)
        } else {
            f64::INFINITY
        };
        status.push(valid && error <= options.max_forward_backward_error);
        errors.push(error);
    }
    Ok(RobustTracks {
        next_points: forward.next_points().to_vec(),
        status,
        forward_backward_errors: errors,
    })
}

/// Scale-ambiguous monocular odometry result.
#[derive(Clone, Debug, PartialEq)]
pub struct MonocularOdometryEstimate {
    /// Source-to-target pose; translation has unit, arbitrary scale.
    pub pose: RelativePose,
    /// Essential-matrix RANSAC inliers.
    pub inliers: Vec<bool>,
    /// Correspondences triangulated in front of both cameras.
    pub positive_depth_count: usize,
}

/// Estimates calibrated monocular motion with essential RANSAC and cheirality.
pub fn estimate_monocular_odometry(
    correspondences: &[PointCorrespondence2],
    camera: CameraMatrix3,
    options: RobustEstimationOptions,
) -> VisionResult<MonocularOdometryEstimate> {
    let estimate = estimate_essential_ransac(correspondences, camera, camera, options)?;
    let inlier_pairs = correspondences
        .iter()
        .zip(estimate.inliers())
        .filter_map(|(&pair, &inlier)| inlier.then_some(pair))
        .collect::<Vec<_>>();
    if inlier_pairs.len() < 8 {
        return Err(VisionError::InvalidParameter(
            "monocular odometry requires at least eight essential inliers".into(),
        ));
    }
    let recovered = recover_relative_pose(*estimate.model(), &inlier_pairs, camera, camera)?;
    Ok(MonocularOdometryEstimate {
        pose: recovered.pose(),
        inliers: estimate.inliers().to_vec(),
        positive_depth_count: recovered.positive_depth_count(),
    })
}

/// Depth filtering and PnP RANSAC controls for metric RGB-D odometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbdOdometryOptions {
    /// Multiplier converting stored depth to metres.
    pub depth_scale: f64,
    /// Inclusive minimum accepted metric depth.
    pub min_depth: f64,
    /// Inclusive maximum accepted metric depth.
    pub max_depth: f64,
    /// PnP robust-estimation settings (threshold in target pixels).
    pub robust: RobustEstimationOptions,
}

impl Default for RgbdOdometryOptions {
    fn default() -> Self {
        Self {
            depth_scale: 1.0,
            min_depth: 0.1,
            max_depth: 100.0,
            robust: RobustEstimationOptions { threshold: 1.0, ..Default::default() },
        }
    }
}

/// Metric previous-camera to current-camera odometry result.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbdOdometryEstimate {
    /// Full metric source-to-target pose.
    pub pose: AbsolutePose,
    /// RANSAC decisions over correspondences that had valid source depth.
    pub inliers: Vec<bool>,
    /// Number of input rows discarded for missing/out-of-range source depth.
    pub rejected_depth_count: usize,
}

/// Estimates metric motion from source depth and source-to-target pixel tracks.
pub fn estimate_rgbd_odometry<T: crate::PixelComponent>(
    previous_depth: ImageView<'_, T, 1>,
    correspondences: &[PointCorrespondence2],
    camera: CameraMatrix3,
    options: RgbdOdometryOptions,
) -> VisionResult<RgbdOdometryEstimate> {
    if !options.depth_scale.is_finite()
        || options.depth_scale <= 0.0
        || !options.min_depth.is_finite()
        || options.min_depth <= 0.0
        || options.max_depth.is_nan()
        || options.max_depth < options.min_depth
    {
        return Err(VisionError::InvalidParameter("invalid RGB-D odometry depth range".into()));
    }
    let mut pairs = Vec::new();
    let mut rejected = 0;
    for &pair in correspondences {
        let pixel = pair.source();
        let x = pixel.x.round() as isize;
        let y = pixel.y.round() as isize;
        let Some(sample) =
            (x >= 0 && y >= 0).then(|| previous_depth.get(x as usize, y as usize)).flatten()
        else {
            rejected += 1;
            continue;
        };
        let depth = sample[0].to_f64() * options.depth_scale;
        if !depth.is_finite() || depth < options.min_depth || depth > options.max_depth {
            rejected += 1;
            continue;
        }
        let ray = camera.normalize_pixel(pixel);
        pairs.push(ObjectImageCorrespondence::try_new(
            Vec3::new(ray.x * depth, ray.y * depth, depth),
            pair.target(),
        )?);
    }
    if pairs.len() < 6 {
        return Err(VisionError::InvalidParameter(
            "RGB-D odometry requires at least six tracks with valid depth".into(),
        ));
    }
    let estimate = solve_pnp_ransac(&pairs, camera, options.robust)?;
    Ok(RgbdOdometryEstimate {
        pose: *estimate.model(),
        inliers: estimate.inliers().to_vec(),
        rejected_depth_count: rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        estimate_rgbd_odometry, select_keypoints_grid, GridSelectionOptions, RgbdOdometryOptions,
    };
    use crate::{CameraMatrix3, Keypoint2, PointCorrespondence2, RobustEstimationOptions};
    use spatialrust_camera::CameraIntrinsics;
    use spatialrust_image::Image;
    use spatialrust_math::Vec2;

    #[test]
    fn grid_selection_keeps_strongest_per_cell() {
        let points = vec![
            Keypoint2::try_new(2.0, 2.0, 1.0).unwrap(),
            Keypoint2::try_new(3.0, 3.0, 4.0).unwrap(),
            Keypoint2::try_new(12.0, 2.0, 2.0).unwrap(),
        ];
        let selected = select_keypoints_grid(
            &points,
            20,
            10,
            GridSelectionOptions { cell_width: 10, cell_height: 10, max_per_cell: 1 },
        )
        .unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].response(), 4.0);
        assert_eq!(selected[1].response(), 2.0);
    }

    #[test]
    fn rgbd_odometry_recovers_metric_translation() {
        let camera = CameraMatrix3::from_intrinsics(
            CameraIntrinsics::try_new(100.0, 100.0, 50.0, 40.0, 100, 80).unwrap(),
        );
        let mut depth = Image::<f32, 1>::from_pixel(100, 80, [f32::NAN]).unwrap();
        let mut pairs = Vec::new();
        for (index, (x, y)) in [
            (20, 20),
            (35, 20),
            (50, 20),
            (65, 20),
            (80, 20),
            (20, 35),
            (35, 35),
            (50, 35),
            (65, 35),
            (80, 35),
            (25, 55),
            (45, 55),
            (65, 55),
            (80, 55),
        ]
        .into_iter()
        .enumerate()
        {
            let z = 1.5 + index as f64 * 0.07;
            depth.get_mut(x, y).unwrap()[0] = z as f32;
            let source = Vec2 { x: x as f64, y: y as f64 };
            let target = Vec2 { x: source.x + 100.0 * 0.1 / z, y: source.y };
            pairs.push(PointCorrespondence2::try_new(source, target).unwrap());
        }
        let estimate = estimate_rgbd_odometry(
            depth.view(),
            &pairs,
            camera,
            RgbdOdometryOptions {
                robust: RobustEstimationOptions {
                    threshold: 0.05,
                    max_iterations: 500,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        assert!((estimate.pose.translation().x - 0.1).abs() < 1e-4);
        assert!(estimate.pose.translation().y.abs() < 1e-4);
        assert!(estimate.pose.translation().z.abs() < 1e-4);
        assert_eq!(estimate.inliers.iter().filter(|&&value| value).count(), pairs.len());
    }
}
