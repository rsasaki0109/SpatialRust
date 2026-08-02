//! Feature2D keypoint, descriptor, and correspondence data contracts.

use crate::{VisionError, VisionResult};

/// One scale-space image keypoint in pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keypoint2 {
    x: f32,
    y: f32,
    size: f32,
    angle_degrees: Option<f32>,
    response: f32,
    octave: i32,
    class_id: Option<i32>,
}

impl Keypoint2 {
    /// Creates an unoriented one-pixel keypoint with a detector response.
    pub fn try_new(x: f32, y: f32, response: f32) -> VisionResult<Self> {
        if !x.is_finite() || !y.is_finite() || !response.is_finite() {
            return Err(VisionError::InvalidParameter(
                "keypoint coordinates and response must be finite".into(),
            ));
        }
        Ok(Self { x, y, size: 1.0, angle_degrees: None, response, octave: 0, class_id: None })
    }

    /// Sets a positive finite feature diameter in pixels.
    pub fn with_size(mut self, size: f32) -> VisionResult<Self> {
        if !size.is_finite() || size <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "keypoint size must be positive and finite".into(),
            ));
        }
        self.size = size;
        Ok(self)
    }

    /// Sets a finite orientation, normalized into `[0, 360)` degrees.
    pub fn with_angle_degrees(mut self, angle: f32) -> VisionResult<Self> {
        if !angle.is_finite() {
            return Err(VisionError::InvalidParameter("keypoint angle must be finite".into()));
        }
        self.angle_degrees = Some(angle.rem_euclid(360.0));
        Ok(self)
    }

    /// Sets a scale-pyramid octave.
    #[must_use]
    pub const fn with_octave(mut self, octave: i32) -> Self {
        self.octave = octave;
        self
    }

    /// Sets an optional detector- or application-defined class identifier.
    #[must_use]
    pub const fn with_class_id(mut self, class_id: i32) -> Self {
        self.class_id = Some(class_id);
        self
    }

    /// Returns the horizontal pixel coordinate.
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Returns the vertical pixel coordinate.
    pub const fn y(self) -> f32 {
        self.y
    }

    /// Returns the feature diameter in pixels.
    pub const fn size(self) -> f32 {
        self.size
    }

    /// Returns the normalized orientation, or `None` when not estimated.
    pub const fn angle_degrees(self) -> Option<f32> {
        self.angle_degrees
    }

    /// Returns the detector response; larger values are more salient.
    pub const fn response(self) -> f32 {
        self.response
    }

    /// Returns the scale-pyramid octave.
    pub const fn octave(self) -> i32 {
        self.octave
    }

    /// Returns the optional application-defined class identifier.
    pub const fn class_id(self) -> Option<i32> {
        self.class_id
    }
}

/// Scalar representation used by a descriptor matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DescriptorKind {
    /// Packed binary descriptor; matching uses Hamming distance.
    Binary,
    /// Float32 descriptor; matching normally uses L2 distance.
    Float32,
}

#[derive(Clone, Debug, PartialEq)]
enum DescriptorStorage {
    Binary(Vec<u8>),
    Float32(Vec<f32>),
}

/// Row-major descriptor matrix with a fixed width and explicit scalar kind.
#[derive(Clone, Debug, PartialEq)]
pub struct DescriptorBuffer {
    rows: usize,
    width: usize,
    storage: DescriptorStorage,
}

impl DescriptorBuffer {
    /// Creates a checked row-major binary descriptor matrix.
    pub fn try_binary(rows: usize, bytes_per_row: usize, data: Vec<u8>) -> VisionResult<Self> {
        Self::validate_layout(rows, bytes_per_row, data.len())?;
        Ok(Self { rows, width: bytes_per_row, storage: DescriptorStorage::Binary(data) })
    }

    /// Creates a checked row-major float32 descriptor matrix.
    pub fn try_float32(rows: usize, values_per_row: usize, data: Vec<f32>) -> VisionResult<Self> {
        if data.iter().any(|value| !value.is_finite()) {
            return Err(VisionError::InvalidParameter(
                "float descriptors must contain only finite values".into(),
            ));
        }
        Self::validate_layout(rows, values_per_row, data.len())?;
        Ok(Self { rows, width: values_per_row, storage: DescriptorStorage::Float32(data) })
    }

    fn validate_layout(rows: usize, width: usize, actual: usize) -> VisionResult<()> {
        if width == 0 {
            return Err(VisionError::InvalidParameter("descriptor width must be positive".into()));
        }
        let expected = rows
            .checked_mul(width)
            .ok_or_else(|| VisionError::InvalidParameter("descriptor layout overflows".into()))?;
        if actual != expected {
            return Err(VisionError::DescriptorLayout { rows, width, expected, actual });
        }
        Ok(())
    }

    /// Returns the descriptor scalar representation.
    pub const fn kind(&self) -> DescriptorKind {
        match self.storage {
            DescriptorStorage::Binary(_) => DescriptorKind::Binary,
            DescriptorStorage::Float32(_) => DescriptorKind::Float32,
        }
    }

    /// Returns the descriptor row count.
    pub const fn len(&self) -> usize {
        self.rows
    }

    /// Returns whether there are no descriptor rows.
    pub const fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Returns bytes or float values per descriptor.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns one binary descriptor, or `None` for a wrong kind or row index.
    pub fn binary_row(&self, index: usize) -> Option<&[u8]> {
        let DescriptorStorage::Binary(data) = &self.storage else {
            return None;
        };
        let start = index.checked_mul(self.width)?;
        data.get(start..start + self.width)
    }

    /// Returns one float descriptor, or `None` for a wrong kind or row index.
    pub fn float32_row(&self, index: usize) -> Option<&[f32]> {
        let DescriptorStorage::Float32(data) = &self.storage else {
            return None;
        };
        let start = index.checked_mul(self.width)?;
        data.get(start..start + self.width)
    }

    /// Returns packed binary storage when the descriptor kind is binary.
    pub fn binary_data(&self) -> Option<&[u8]> {
        match &self.storage {
            DescriptorStorage::Binary(data) => Some(data),
            DescriptorStorage::Float32(_) => None,
        }
    }

    /// Returns packed float storage when the descriptor kind is float32.
    pub fn float32_data(&self) -> Option<&[f32]> {
        match &self.storage {
            DescriptorStorage::Float32(data) => Some(data),
            DescriptorStorage::Binary(_) => None,
        }
    }
}

/// Keypoints paired one-to-one with descriptor rows.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureSet2 {
    keypoints: Vec<Keypoint2>,
    descriptors: DescriptorBuffer,
}

impl FeatureSet2 {
    /// Validates and owns a keypoint/descriptor pair.
    pub fn try_new(keypoints: Vec<Keypoint2>, descriptors: DescriptorBuffer) -> VisionResult<Self> {
        if keypoints.len() != descriptors.len() {
            return Err(VisionError::FeatureCountMismatch {
                keypoints: keypoints.len(),
                descriptors: descriptors.len(),
            });
        }
        Ok(Self { keypoints, descriptors })
    }

    /// Returns keypoints in descriptor-row order.
    pub fn keypoints(&self) -> &[Keypoint2] {
        &self.keypoints
    }

    /// Returns the associated descriptor matrix.
    pub const fn descriptors(&self) -> &DescriptorBuffer {
        &self.descriptors
    }
}

/// One descriptor correspondence from a query feature to a train feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureMatch {
    query_index: usize,
    train_index: usize,
    distance: f32,
}

impl FeatureMatch {
    /// Creates a correspondence with a finite non-negative distance.
    pub fn try_new(query_index: usize, train_index: usize, distance: f32) -> VisionResult<Self> {
        if !distance.is_finite() || distance < 0.0 {
            return Err(VisionError::InvalidParameter(
                "feature-match distance must be finite and non-negative".into(),
            ));
        }
        Ok(Self { query_index, train_index, distance })
    }

    /// Validates both indices against their feature collections.
    pub fn validate(self, query_count: usize, train_count: usize) -> VisionResult<Self> {
        if self.query_index >= query_count || self.train_index >= train_count {
            return Err(VisionError::MatchIndexOutOfBounds {
                query: self.query_index,
                queries: query_count,
                train: self.train_index,
                trains: train_count,
            });
        }
        Ok(self)
    }

    /// Returns the query descriptor row index.
    pub const fn query_index(self) -> usize {
        self.query_index
    }

    /// Returns the train descriptor row index.
    pub const fn train_index(self) -> usize {
        self.train_index
    }

    /// Returns the descriptor distance; smaller is a better match.
    pub const fn distance(self) -> f32 {
        self.distance
    }
}

#[cfg(test)]
mod tests {
    use super::{DescriptorBuffer, DescriptorKind, FeatureMatch, FeatureSet2, Keypoint2};
    use crate::VisionError;

    #[test]
    fn keypoint_normalizes_orientation_and_rejects_non_finite_values() {
        let keypoint = Keypoint2::try_new(2.5, 3.5, -0.2)
            .unwrap()
            .with_size(7.0)
            .unwrap()
            .with_angle_degrees(-45.0)
            .unwrap()
            .with_octave(2)
            .with_class_id(9);
        assert_eq!(keypoint.angle_degrees(), Some(315.0));
        assert_eq!((keypoint.x(), keypoint.y(), keypoint.size()), (2.5, 3.5, 7.0));
        assert_eq!(
            (keypoint.response(), keypoint.octave(), keypoint.class_id()),
            (-0.2, 2, Some(9))
        );
        assert!(Keypoint2::try_new(f32::NAN, 0.0, 1.0).is_err());
    }

    #[test]
    fn descriptor_layout_and_kind_are_checked() {
        let binary = DescriptorBuffer::try_binary(2, 4, (0..8).collect()).unwrap();
        assert_eq!(binary.kind(), DescriptorKind::Binary);
        assert_eq!(binary.binary_row(1), Some(&[4, 5, 6, 7][..]));
        assert!(binary.float32_row(0).is_none());
        assert!(matches!(
            DescriptorBuffer::try_float32(2, 3, vec![0.0; 5]),
            Err(VisionError::DescriptorLayout { expected: 6, actual: 5, .. })
        ));
        assert!(DescriptorBuffer::try_float32(1, 1, vec![f32::NAN]).is_err());
    }

    #[test]
    fn feature_rows_and_match_indices_are_checked() {
        let keypoint = Keypoint2::try_new(1.0, 2.0, 3.0).unwrap();
        let descriptors = DescriptorBuffer::try_binary(1, 2, vec![0xaa, 0x55]).unwrap();
        let features = FeatureSet2::try_new(vec![keypoint], descriptors).unwrap();
        assert_eq!(features.keypoints(), &[keypoint]);
        assert_eq!(features.descriptors().len(), 1);
        let correspondence = FeatureMatch::try_new(0, 0, 4.0).unwrap().validate(1, 1).unwrap();
        assert_eq!((correspondence.query_index(), correspondence.train_index()), (0, 0));
        assert_eq!(correspondence.distance(), 4.0);
        assert!(FeatureMatch::try_new(0, 0, f32::INFINITY).is_err());
        assert!(matches!(
            FeatureMatch::try_new(1, 0, 0.0).unwrap().validate(1, 1),
            Err(VisionError::MatchIndexOutOfBounds { .. })
        ));
    }
}
