//! Spherical range-image projection for rotating-LiDAR scans.
//!
//! Projects a 3D scan into the dense 2D range image that learned LiDAR models
//! consume (RangeNet++, SqueezeSeg): rows span the sensor's vertical field of
//! view, columns span azimuth, and each pixel holds the nearest point's range.

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};

/// Configuration for [`range_image`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RangeImageConfig {
    /// Image width in pixels (azimuth resolution).
    pub width: usize,
    /// Image height in pixels (one row per laser beam).
    pub height: usize,
    /// Upper bound of the vertical field of view, in degrees.
    pub fov_up_deg: f32,
    /// Lower bound of the vertical field of view, in degrees (usually negative).
    pub fov_down_deg: f32,
}

impl Default for RangeImageConfig {
    fn default() -> Self {
        // Velodyne HDL-64E-like vertical FOV.
        Self { width: 1024, height: 64, fov_up_deg: 3.0, fov_down_deg: -25.0 }
    }
}

impl RangeImageConfig {
    /// Creates a config from the image dimensions (default vertical FOV).
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, ..Self::default() }
    }
}

/// A dense range image in row-major order (`row * width + col`).
#[derive(Clone, Debug, PartialEq)]
pub struct RangeImage {
    /// Image width (columns).
    pub width: usize,
    /// Image height (rows).
    pub height: usize,
    /// Range per pixel in meters; `0.0` marks an empty pixel.
    pub data: Vec<f32>,
}

impl RangeImage {
    /// Total number of pixels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the image has no pixels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of pixels that received a point.
    #[must_use]
    pub fn filled_count(&self) -> usize {
        self.data.iter().filter(|&&r| r > 0.0).count()
    }

    /// Range at `(row, col)`, or `None` if out of range.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row >= self.height || col >= self.width {
            return None;
        }
        Some(self.data[row * self.width + col])
    }
}

/// Projects a cloud into a spherical range image, keeping the nearest range per
/// pixel. Points outside the vertical field of view are dropped.
pub fn range_image(cloud: &PointCloud, config: RangeImageConfig) -> SpatialResult<RangeImage> {
    if config.width == 0 || config.height == 0 {
        return Err(SpatialError::InvalidArgument(
            "range image dimensions must be non-zero".to_owned(),
        ));
    }
    if config.fov_up_deg <= config.fov_down_deg {
        return Err(SpatialError::InvalidArgument(
            "fov_up must be greater than fov_down".to_owned(),
        ));
    }

    let (x, y, z) = cloud.positions3()?;
    let fov_up = config.fov_up_deg.to_radians();
    let fov_down = config.fov_down_deg.to_radians();
    let fov = fov_up - fov_down;

    let mut data = vec![0.0_f32; config.width * config.height];
    for i in 0..cloud.len() {
        let (px, py, pz) = (x[i], y[i], z[i]);
        let range = (px * px + py * py + pz * pz).sqrt();
        if range < 1e-6 {
            continue;
        }
        let azimuth = py.atan2(px); // [-pi, pi]
        let elevation = (pz / range).asin(); // [-pi/2, pi/2]

        // Azimuth -> column, elevation -> row (top row = fov_up).
        let u = 0.5 * (azimuth / std::f32::consts::PI + 1.0);
        let v = 1.0 - (elevation - fov_down) / fov;
        let col = (u * config.width as f32).floor();
        let row = (v * config.height as f32).floor();
        if row < 0.0 || col < 0.0 {
            continue;
        }
        let (row, col) = (row as usize, col as usize);
        if row >= config.height || col >= config.width {
            continue;
        }
        let idx = row * config.width + col;
        // Keep the nearest return (or fill an empty pixel).
        if data[idx] == 0.0 || range < data[idx] {
            data[idx] = range;
        }
    }

    Ok(RangeImage { width: config.width, height: config.height, data })
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    fn cloud(points: &[[f32; 3]]) -> PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for p in points {
            builder.push_point(*p).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn projects_points_into_pixels() {
        // A point straight ahead (+x) lands near the middle column; one to the
        // left (+y) lands a quarter of the way across.
        let c = cloud(&[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]);
        let img = range_image(&c, RangeImageConfig::new(360, 64)).unwrap();
        assert_eq!(img.filled_count(), 2);
        // Both points are at range 5.
        let total: f32 = img.data.iter().filter(|&&r| r > 0.0).sum();
        assert!((total - 10.0).abs() < 1e-3);
    }

    #[test]
    fn keeps_nearest_range_per_pixel() {
        // Two points in the same direction at different ranges share a pixel.
        let c = cloud(&[[10.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        let img = range_image(&c, RangeImageConfig::new(16, 8)).unwrap();
        assert_eq!(img.filled_count(), 1);
        let nearest = img.data.iter().copied().find(|&r| r > 0.0).unwrap();
        assert!((nearest - 3.0).abs() < 1e-3, "did not keep nearest range: {nearest}");
    }

    #[test]
    fn drops_points_outside_vertical_fov() {
        // A point straight up is well above a narrow forward FOV.
        let c = cloud(&[[0.0, 0.0, 5.0]]);
        let config = RangeImageConfig { fov_up_deg: 2.0, fov_down_deg: -2.0, width: 64, height: 8 };
        let img = range_image(&c, config).unwrap();
        assert_eq!(img.filled_count(), 0);
    }

    #[test]
    fn rejects_bad_config() {
        let c = cloud(&[[1.0, 0.0, 0.0]]);
        assert!(range_image(&c, RangeImageConfig::new(0, 8)).is_err());
        let bad_fov = RangeImageConfig { fov_up_deg: -5.0, fov_down_deg: 5.0, width: 8, height: 8 };
        assert!(range_image(&c, bad_fov).is_err());
    }
}
