//! Oriented FAST and rotated BRIEF binary features.

use spatialrust_image::{Image, ImageView};

use crate::{
    detect_fast, gaussian_blur, resize, BorderMode, DescriptorBuffer, FastOptions, FeatureSet2,
    Interpolation, Keypoint2, VisionError, VisionResult,
};

const ORB_DESCRIPTOR_BYTES: usize = 32;

/// Response used to rank ORB keypoints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OrbScoreType {
    /// Rank FAST candidates by a local Harris response.
    #[default]
    Harris,
    /// Retain the FAST segment-test score.
    Fast,
}

/// Multi-scale ORB detector and descriptor configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbOptions {
    /// Maximum number of descriptor rows across every pyramid level.
    pub max_features: usize,
    /// Scale multiplier between adjacent levels; must be greater than one.
    pub scale_factor: f32,
    /// Number of pyramid levels.
    pub levels: usize,
    /// Minimum distance from a level-image edge.
    pub edge_threshold: usize,
    /// FAST intensity threshold.
    pub fast_threshold: u8,
    /// Odd intensity-centroid and descriptor patch diameter.
    pub patch_size: usize,
    /// Candidate ranking response.
    pub score_type: OrbScoreType,
}

impl Default for OrbOptions {
    fn default() -> Self {
        Self {
            max_features: 500,
            scale_factor: 1.2,
            levels: 8,
            edge_threshold: 31,
            fast_threshold: 20,
            patch_size: 31,
            score_type: OrbScoreType::Harris,
        }
    }
}

impl OrbOptions {
    fn validate(self) -> VisionResult<Self> {
        if !self.scale_factor.is_finite() || self.scale_factor <= 1.0 {
            return Err(VisionError::InvalidParameter(
                "ORB scale_factor must be finite and greater than one".into(),
            ));
        }
        if self.levels == 0 {
            return Err(VisionError::InvalidParameter("ORB levels must be positive".into()));
        }
        if self.patch_size < 7 || self.patch_size % 2 == 0 {
            return Err(VisionError::InvalidParameter(
                "ORB patch_size must be odd and at least seven".into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    level: usize,
    level_x: usize,
    level_y: usize,
    scale: f32,
    response: f32,
}

/// Detects oriented multi-scale keypoints and computes 256-bit rotated BRIEF descriptors.
///
/// The fixed-seed BRIEF sampling pattern is stable across platforms and SpatialRust
/// releases. It intentionally does not promise bit identity with OpenCV's private
/// learned sampling table.
pub fn detect_and_describe_orb(
    input: ImageView<'_, u8, 1>,
    options: OrbOptions,
) -> VisionResult<FeatureSet2> {
    let options = options.validate()?;
    if input.width() == 0 || input.height() == 0 || options.max_features == 0 {
        return FeatureSet2::try_new(
            Vec::new(),
            DescriptorBuffer::try_binary(0, ORB_DESCRIPTOR_BYTES, Vec::new())?,
        );
    }

    let mut pyramid = Vec::with_capacity(options.levels);
    let mut candidates = Vec::new();
    for level in 0..options.levels {
        let scale = options.scale_factor.powi(level as i32);
        let width = ((input.width() as f32 / scale).round() as usize).max(1);
        let height = ((input.height() as f32 / scale).round() as usize).max(1);
        let image = if level == 0 {
            Image::try_new_with_metadata(
                input.width(),
                input.height(),
                (0..input.height())
                    .flat_map(|y| {
                        (0..input.width())
                            .map(move |x| input.get(x, y).expect("coordinate in bounds")[0])
                    })
                    .collect(),
                input.metadata(),
            )?
        } else {
            resize(input, width, height, Interpolation::Bilinear)?
        };
        let margin = options.edge_threshold.max(options.patch_size / 2 + 1);
        let fast = detect_fast(
            image.view(),
            FastOptions { threshold: options.fast_threshold, nonmax_suppression: true },
        )?;
        for point in fast {
            let x = point.x() as usize;
            let y = point.y() as usize;
            if x < margin
                || y < margin
                || x.saturating_add(margin) >= image.width()
                || y.saturating_add(margin) >= image.height()
            {
                continue;
            }
            let response = match options.score_type {
                OrbScoreType::Harris => harris_score(image.view(), x, y),
                OrbScoreType::Fast => point.response(),
            };
            candidates.push(Candidate { level, level_x: x, level_y: y, scale, response });
        }
        pyramid.push(image);
    }

    candidates.sort_by(|left, right| {
        right
            .response
            .total_cmp(&left.response)
            .then_with(|| left.level.cmp(&right.level))
            .then_with(|| left.level_y.cmp(&right.level_y))
            .then_with(|| left.level_x.cmp(&right.level_x))
    });
    candidates.truncate(options.max_features);

    let pattern = brief_pattern(options.patch_size / 2);
    let mut blurred = Vec::with_capacity(pyramid.len());
    for image in &pyramid {
        blurred.push(gaussian_blur(image.view(), 7, 7, 2.0, 2.0, BorderMode::Reflect101)?);
    }
    let mut keypoints = Vec::with_capacity(candidates.len());
    let mut descriptors = Vec::with_capacity(candidates.len() * ORB_DESCRIPTOR_BYTES);
    for candidate in candidates {
        let image = blurred[candidate.level].view();
        let angle = intensity_centroid_angle(
            image,
            candidate.level_x,
            candidate.level_y,
            options.patch_size / 2,
        );
        keypoints.push(
            Keypoint2::try_new(
                candidate.level_x as f32 * candidate.scale,
                candidate.level_y as f32 * candidate.scale,
                candidate.response,
            )?
            .with_size(options.patch_size as f32 * candidate.scale)?
            .with_angle_degrees(angle.to_degrees())?
            .with_octave(candidate.level as i32),
        );
        descriptors.extend(describe(image, candidate.level_x, candidate.level_y, angle, &pattern));
    }
    FeatureSet2::try_new(
        keypoints,
        DescriptorBuffer::try_binary(
            descriptors.len() / ORB_DESCRIPTOR_BYTES,
            ORB_DESCRIPTOR_BYTES,
            descriptors,
        )?,
    )
}

fn harris_score(image: ImageView<'_, u8, 1>, x: usize, y: usize) -> f32 {
    let mut xx = 0.0_f32;
    let mut xy = 0.0_f32;
    let mut yy = 0.0_f32;
    for dy in -3_isize..=3 {
        for dx in -3_isize..=3 {
            let px = (x as isize + dx) as usize;
            let py = (y as isize + dy) as usize;
            let gx = f32::from(image.get(px + 1, py).unwrap()[0])
                - f32::from(image.get(px - 1, py).unwrap()[0]);
            let gy = f32::from(image.get(px, py + 1).unwrap()[0])
                - f32::from(image.get(px, py - 1).unwrap()[0]);
            xx += gx * gx;
            xy += gx * gy;
            yy += gy * gy;
        }
    }
    xx.mul_add(yy, -(xy * xy)) - 0.04 * (xx + yy) * (xx + yy)
}

fn intensity_centroid_angle(image: ImageView<'_, u8, 1>, x: usize, y: usize, radius: usize) -> f32 {
    let mut m10 = 0_i64;
    let mut m01 = 0_i64;
    let radius_squared = (radius * radius) as isize;
    for dy in -(radius as isize)..=radius as isize {
        let extent = ((radius_squared - dy * dy) as f64).sqrt().floor() as isize;
        for dx in -extent..=extent {
            let intensity = i64::from(
                image.get((x as isize + dx) as usize, (y as isize + dy) as usize).unwrap()[0],
            );
            m10 += dx as i64 * intensity;
            m01 += dy as i64 * intensity;
        }
    }
    (m01 as f32).atan2(m10 as f32)
}

fn describe(
    image: ImageView<'_, u8, 1>,
    x: usize,
    y: usize,
    angle: f32,
    pattern: &[((i8, i8), (i8, i8))],
) -> [u8; ORB_DESCRIPTOR_BYTES] {
    let (sin, cos) = angle.sin_cos();
    let mut descriptor = [0_u8; ORB_DESCRIPTOR_BYTES];
    for (bit, &(first, second)) in pattern.iter().enumerate() {
        let sample = |point: (i8, i8)| {
            let rx = (f32::from(point.0) * cos - f32::from(point.1) * sin).round() as isize;
            let ry = (f32::from(point.0) * sin + f32::from(point.1) * cos).round() as isize;
            image.get((x as isize + rx) as usize, (y as isize + ry) as usize).unwrap()[0]
        };
        if sample(first) < sample(second) {
            descriptor[bit / 8] |= 1 << (bit % 8);
        }
    }
    descriptor
}

fn brief_pattern(radius: usize) -> Vec<((i8, i8), (i8, i8))> {
    let radius = radius.min(i8::MAX as usize) as i32;
    let mut state = 0x6d2b_79f5_u32;
    let mut next_point = || loop {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let span = (radius * 2 + 1) as u32;
        let x = (state % span) as i32 - radius;
        state = state.rotate_left(11).wrapping_mul(0x9e37_79b1);
        let y = (state % span) as i32 - radius;
        if x * x + y * y <= radius * radius {
            return (x as i8, y as i8);
        }
    };
    (0..ORB_DESCRIPTOR_BYTES * 8)
        .map(|_| {
            let first = next_point();
            let mut second = next_point();
            while second == first {
                second = next_point();
            }
            (first, second)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{detect_and_describe_orb, OrbOptions};
    use crate::{match_descriptors, MatchOptions};
    use spatialrust_image::Image;

    fn textured_image() -> Image<u8, 1> {
        Image::try_new(
            96,
            80,
            (0..80)
                .flat_map(|y| {
                    (0..96).map(move |x| {
                        (((x * 37 + y * 19) ^ (x * y * 3) ^ ((x / 8 + y / 8) * 127)) & 255) as u8
                    })
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn orb_is_deterministic_bounded_and_well_formed() {
        let image = textured_image();
        let options = OrbOptions { max_features: 80, edge_threshold: 16, ..OrbOptions::default() };
        let first = detect_and_describe_orb(image.view(), options).unwrap();
        let second = detect_and_describe_orb(image.view(), options).unwrap();
        assert_eq!(first, second);
        assert!(!first.keypoints().is_empty());
        assert!(first.keypoints().len() <= 80);
        assert_eq!(first.descriptors().width(), 32);
        assert!(first.keypoints().iter().all(|point| point.angle_degrees().is_some()));
    }

    #[test]
    fn orb_descriptors_self_match_at_zero_distance() {
        let image = textured_image();
        let features = detect_and_describe_orb(
            image.view(),
            OrbOptions { max_features: 40, edge_threshold: 16, ..OrbOptions::default() },
        )
        .unwrap();
        let matches = match_descriptors(
            features.descriptors(),
            features.descriptors(),
            MatchOptions { cross_check: true, ..MatchOptions::default() },
        )
        .unwrap();
        assert_eq!(matches.len(), features.keypoints().len());
        assert!(matches.iter().all(|feature_match| feature_match.distance() == 0.0));
    }

    #[test]
    fn invalid_or_empty_orb_inputs_are_handled() {
        let image = textured_image();
        assert!(detect_and_describe_orb(
            image.view(),
            OrbOptions { levels: 0, ..OrbOptions::default() }
        )
        .is_err());
        let empty = detect_and_describe_orb(
            image.view(),
            OrbOptions { max_features: 0, ..OrbOptions::default() },
        )
        .unwrap();
        assert!(empty.keypoints().is_empty());
        assert_eq!(empty.descriptors().width(), 32);
    }
}
