//! Bridges dense vision outputs into SpatialRust camera and point-cloud APIs.

use spatialrust_camera::{depth_to_point_cloud, DepthConversionOptions, PinholeCamera};
use spatialrust_core::{PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas};

use crate::{ConfidenceMap, DepthMap, PointMap, VisionError, VisionResult};

/// Unprojects a depth map with a calibrated camera into an XYZ point cloud.
pub fn depth_map_to_point_cloud(
    depth: &DepthMap,
    camera: &PinholeCamera,
    options: DepthConversionOptions,
) -> VisionResult<PointCloud> {
    depth_to_point_cloud(depth.image().view(), camera, options)
        .map_err(|error| VisionError::InvalidParameter(error.to_string()))
}

/// Flattens valid point-map pixels into an XYZ cloud, optionally filtering by confidence.
pub fn point_map_to_point_cloud(
    point_map: &PointMap,
    confidence: Option<&ConfidenceMap>,
    min_confidence: f32,
) -> VisionResult<PointCloud> {
    if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
        return Err(VisionError::InvalidParameter(
            "minimum confidence must be finite and in [0, 1]".to_owned(),
        ));
    }
    if let Some(confidence) = confidence {
        if confidence.width() != point_map.width() || confidence.height() != point_map.height() {
            return Err(VisionError::ShapeMismatch(
                "point map and confidence map dimensions must match".to_owned(),
            ));
        }
    }

    let capacity = point_map.width().saturating_mul(point_map.height());
    let mut xs = Vec::with_capacity(capacity);
    let mut ys = Vec::with_capacity(capacity);
    let mut zs = Vec::with_capacity(capacity);
    for y in 0..point_map.height() {
        for x in 0..point_map.width() {
            let point = point_map.image().get(x, y).expect("point-map coordinate in bounds");
            if !point.iter().all(|value| value.is_finite()) {
                continue;
            }
            if let Some(confidence) = confidence {
                let score =
                    confidence.image().get(x, y).expect("confidence coordinate in bounds")[0];
                if score < min_confidence {
                    continue;
                }
            }
            xs.push(point[0]);
            ys.push(point[1]);
            zs.push(point[2]);
        }
    }

    let mut buffers = PointBufferSet::new();
    buffers.insert("x", PointBuffer::from_f32(xs));
    buffers.insert("y", PointBuffer::from_f32(ys));
    buffers.insert("z", PointBuffer::from_f32(zs));
    PointCloud::try_from_parts(StandardSchemas::point_xyz(), buffers, SpatialMetadata::default())
        .map_err(|error| VisionError::InvalidParameter(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{depth_map_to_point_cloud, point_map_to_point_cloud};
    use crate::{ConfidenceMap, DepthMap, PointMap};
    use spatialrust_camera::{CameraIntrinsics, PinholeCamera};

    #[test]
    fn depth_map_uses_camera_unprojection() {
        let depth = DepthMap::try_new(2, 1, vec![1.0, 2.0]).unwrap();
        let camera =
            PinholeCamera::new(CameraIntrinsics::try_new(2.0, 2.0, 0.0, 0.0, 2, 1).unwrap());
        let cloud = depth_map_to_point_cloud(&depth, &camera, Default::default()).unwrap();
        assert_eq!(cloud.field("x").unwrap().as_f32().unwrap(), &[0.0, 1.0]);
        assert_eq!(cloud.field("z").unwrap().as_f32().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn point_map_filters_invalid_and_low_confidence_points() {
        let points =
            PointMap::try_new(3, 1, vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, f32::NAN, 0.0, 1.0])
                .unwrap();
        let confidence = ConfidenceMap::try_new(3, 1, vec![0.9, 0.1, 1.0]).unwrap();
        let cloud = point_map_to_point_cloud(&points, Some(&confidence), 0.5).unwrap();
        assert_eq!(cloud.len(), 1);
        assert_eq!(cloud.field("x").unwrap().as_f32().unwrap(), &[0.0]);
    }
}
