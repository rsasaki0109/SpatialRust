//! Sparse pyramidal Lucas–Kanade point tracking.

use spatialrust_image::{Image, ImageView};
use spatialrust_math::Vec2;

use crate::{PixelComponent, VisionError, VisionResult};

/// Configuration for sparse Lucas–Kanade tracking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LucasKanadeOptions {
    /// Half-window radius in pixels at each pyramid level.
    pub window_radius: usize,
    /// Number of pyramid levels including the full-resolution level.
    pub pyramid_levels: usize,
    /// Maximum Gauss–Newton iterations per pyramid level.
    pub max_iterations: usize,
    /// Convergence threshold on the update length in pixels.
    pub epsilon: f64,
    /// Minimum structure-tensor determinant accepted as trackable.
    pub min_eigenvalue: f64,
}

impl Default for LucasKanadeOptions {
    fn default() -> Self {
        Self {
            window_radius: 5,
            pyramid_levels: 3,
            max_iterations: 30,
            epsilon: 1e-3,
            min_eigenvalue: 1e-4,
        }
    }
}

impl LucasKanadeOptions {
    /// Validates window, pyramid, and numerical thresholds.
    pub fn validate(self) -> VisionResult<Self> {
        if self.window_radius == 0 {
            return Err(VisionError::InvalidParameter(
                "Lucas–Kanade window_radius must be positive".into(),
            ));
        }
        if self.pyramid_levels == 0 {
            return Err(VisionError::InvalidParameter(
                "Lucas–Kanade pyramid_levels must be positive".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(VisionError::InvalidParameter(
                "Lucas–Kanade max_iterations must be positive".into(),
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "Lucas–Kanade epsilon must be finite and positive".into(),
            ));
        }
        if !self.min_eigenvalue.is_finite() || self.min_eigenvalue <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "Lucas–Kanade min_eigenvalue must be finite and positive".into(),
            ));
        }
        Ok(self)
    }
}

/// Result of sparse point tracking between two frames.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackedPoints {
    next_points: Vec<Vec2<f64>>,
    status: Vec<bool>,
}

impl TrackedPoints {
    /// Returns one next-frame coordinate per input point.
    pub fn next_points(&self) -> &[Vec2<f64>] {
        &self.next_points
    }

    /// Returns whether each point was successfully tracked.
    pub fn status(&self) -> &[bool] {
        &self.status
    }
}

/// Tracks sparse points from `previous` to `next` with pyramidal Lucas–Kanade.
pub fn track_points_lucas_kanade<T: PixelComponent>(
    previous: ImageView<'_, T, 1>,
    next: ImageView<'_, T, 1>,
    points: &[Vec2<f64>],
    options: LucasKanadeOptions,
) -> VisionResult<TrackedPoints> {
    let options = options.validate()?;
    if previous.width() != next.width() || previous.height() != next.height() {
        return Err(VisionError::ShapeMismatch(
            "Lucas–Kanade frames must share width and height".into(),
        ));
    }
    if previous.width() < 3 || previous.height() < 3 {
        return Err(VisionError::InvalidDimensions(
            "Lucas–Kanade requires at least 3x3 images".into(),
        ));
    }
    let previous_pyramid = build_pyramid(previous, options.pyramid_levels)?;
    let next_pyramid = build_pyramid(next, options.pyramid_levels)?;
    let mut next_points = points.to_vec();
    let mut status = vec![true; points.len()];
    let scale = 1.0 / f64::from(1u32 << (options.pyramid_levels.saturating_sub(1) as u32));
    for point in &mut next_points {
        point.x *= scale;
        point.y *= scale;
    }
    let mut previous_guess = next_points.clone();
    for level in (0..options.pyramid_levels).rev() {
        let level_scale = 1.0 / f64::from(1u32 << (level as u32));
        for (index, previous_point) in points.iter().enumerate() {
            if !status[index] {
                continue;
            }
            let mut guess = previous_guess[index];
            let target =
                Vec2 { x: previous_point.x * level_scale, y: previous_point.y * level_scale };
            match track_one_level(
                previous_pyramid[level].view(),
                next_pyramid[level].view(),
                target,
                guess,
                options,
            ) {
                Ok(updated) => {
                    guess = updated;
                    status[index] = true;
                }
                Err(_) => status[index] = false,
            }
            previous_guess[index] = guess;
            if level > 0 {
                previous_guess[index].x *= 2.0;
                previous_guess[index].y *= 2.0;
            }
        }
    }
    Ok(TrackedPoints { next_points: previous_guess, status })
}

fn track_one_level(
    previous: ImageView<'_, f32, 1>,
    next: ImageView<'_, f32, 1>,
    previous_point: Vec2<f64>,
    mut next_point: Vec2<f64>,
    options: LucasKanadeOptions,
) -> VisionResult<Vec2<f64>> {
    let radius = options.window_radius as f64;
    for _ in 0..options.max_iterations {
        let mut gxx = 0.0;
        let mut gxy = 0.0;
        let mut gyy = 0.0;
        let mut bx = 0.0;
        let mut by = 0.0;
        let mut samples = 0usize;
        let start = -(options.window_radius as isize);
        let end = options.window_radius as isize;
        for dy in start..=end {
            for dx in start..=end {
                let px = previous_point.x + dx as f64;
                let py = previous_point.y + dy as f64;
                let qx = next_point.x + dx as f64;
                let qy = next_point.y + dy as f64;
                if !(in_bounds(previous, px, py, radius) && in_bounds(next, qx, qy, radius)) {
                    continue;
                }
                let ix = 0.5 * (sample(previous, px + 1.0, py)? - sample(previous, px - 1.0, py)?);
                let iy = 0.5 * (sample(previous, px, py + 1.0)? - sample(previous, px, py - 1.0)?);
                let it = sample(next, qx, qy)? - sample(previous, px, py)?;
                gxx += ix * ix;
                gxy += ix * iy;
                gyy += iy * iy;
                bx += ix * it;
                by += iy * it;
                samples += 1;
            }
        }
        if samples < 4 {
            return Err(VisionError::InvalidParameter("Lucas–Kanade patch left the image".into()));
        }
        let det = gxx * gyy - gxy * gxy;
        if det.abs() < options.min_eigenvalue {
            return Err(VisionError::InvalidParameter(
                "Lucas–Kanade structure tensor is singular".into(),
            ));
        }
        let dx = (-gyy * bx + gxy * by) / det;
        let dy = (gxy * bx - gxx * by) / det;
        next_point.x += dx;
        next_point.y += dy;
        if dx.hypot(dy) < options.epsilon {
            break;
        }
    }
    if !in_bounds(next, next_point.x, next_point.y, 0.0) {
        return Err(VisionError::InvalidParameter(
            "Lucas–Kanade tracked point left the image".into(),
        ));
    }
    Ok(next_point)
}

fn build_pyramid<T: PixelComponent>(
    image: ImageView<'_, T, 1>,
    levels: usize,
) -> VisionResult<Vec<Image<f32, 1>>> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(to_f32_image(image)?);
    for _ in 1..levels {
        let previous = pyramid.last().expect("pyramid starts non-empty");
        pyramid.push(downsample(previous.view())?);
    }
    Ok(pyramid)
}

fn to_f32_image<T: PixelComponent>(image: ImageView<'_, T, 1>) -> VisionResult<Image<f32, 1>> {
    let mut data = Vec::with_capacity(image.width() * image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            data.push(image.get(x, y).expect("in-bounds")[0].to_f64() as f32);
        }
    }
    Ok(Image::try_new(image.width(), image.height(), data)?)
}

fn downsample(image: ImageView<'_, f32, 1>) -> VisionResult<Image<f32, 1>> {
    let width = (image.width() / 2).max(1);
    let height = (image.height() / 2).max(1);
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let sx = x * 2;
            let sy = y * 2;
            let mut sum = 0.0_f64;
            let mut count = 0.0_f64;
            for dy in 0..2 {
                for dx in 0..2 {
                    if let Some(pixel) = image.get(sx + dx, sy + dy) {
                        sum += f64::from(pixel[0]);
                        count += 1.0;
                    }
                }
            }
            data.push((sum / count.max(1.0)) as f32);
        }
    }
    Ok(Image::try_new(width, height, data)?)
}

fn sample(image: ImageView<'_, f32, 1>, x: f64, y: f64) -> VisionResult<f64> {
    if !x.is_finite() || !y.is_finite() {
        return Err(VisionError::InvalidParameter(
            "Lucas–Kanade sample coordinates must be finite".into(),
        ));
    }
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    if x0 < 0 || y0 < 0 || x1 >= image.width() as isize || y1 >= image.height() as isize {
        return Err(VisionError::InvalidParameter("Lucas–Kanade sample is out of bounds".into()));
    }
    let ax = x - x0 as f64;
    let ay = y - y0 as f64;
    let i00 = f64::from(image.get(x0 as usize, y0 as usize).expect("in-bounds")[0]);
    let i10 = f64::from(image.get(x1 as usize, y0 as usize).expect("in-bounds")[0]);
    let i01 = f64::from(image.get(x0 as usize, y1 as usize).expect("in-bounds")[0]);
    let i11 = f64::from(image.get(x1 as usize, y1 as usize).expect("in-bounds")[0]);
    Ok((1.0 - ax) * (1.0 - ay) * i00
        + ax * (1.0 - ay) * i10
        + (1.0 - ax) * ay * i01
        + ax * ay * i11)
}

fn in_bounds(image: ImageView<'_, f32, 1>, x: f64, y: f64, margin: f64) -> bool {
    x >= margin
        && y >= margin
        && x < image.width() as f64 - 1.0 - margin
        && y < image.height() as f64 - 1.0 - margin
}

#[cfg(test)]
mod tests {
    use super::{track_points_lucas_kanade, LucasKanadeOptions};
    use spatialrust_image::Image;
    use spatialrust_math::Vec2;

    #[test]
    fn tracks_integer_translation() {
        let width = 64;
        let height = 48;
        let mut previous = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                previous[y * width + x] = ((x * 13 + y * 7) % 200 + 20) as u8;
            }
        }
        let shift = 3isize;
        let mut next = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let sx = x as isize - shift;
                let sy = y as isize;
                if (0..width as isize).contains(&sx) && (0..height as isize).contains(&sy) {
                    next[y * width + x] = previous[sy as usize * width + sx as usize];
                }
            }
        }
        let previous = Image::<u8, 1>::try_new(width, height, previous).unwrap();
        let next = Image::<u8, 1>::try_new(width, height, next).unwrap();
        let points = [Vec2 { x: 24.0, y: 20.0 }];
        let tracked = track_points_lucas_kanade(
            previous.view(),
            next.view(),
            &points,
            LucasKanadeOptions {
                window_radius: 4,
                pyramid_levels: 1,
                max_iterations: 50,
                epsilon: 1e-4,
                min_eigenvalue: 1e-8,
            },
        )
        .unwrap();
        assert!(tracked.status().iter().all(|&ok| ok), "status={:?}", tracked.status());
        let actual = tracked.next_points()[0];
        assert!(
            (actual.x - (points[0].x + shift as f64)).abs() < 1.5,
            "x actual={} expected={}",
            actual.x,
            points[0].x + shift as f64
        );
        assert!(
            (actual.y - points[0].y).abs() < 1.5,
            "y actual={} expected={}",
            actual.y,
            points[0].y
        );
    }
}
