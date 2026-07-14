//! Detection post-processing primitives.

use std::cmp::Ordering;

use crate::{VisionError, VisionResult};

/// Axis-aligned bounding box using half-open `(min, max)` coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoundingBox2 {
    /// Minimum x coordinate.
    pub x_min: f32,
    /// Minimum y coordinate.
    pub y_min: f32,
    /// Maximum x coordinate.
    pub x_max: f32,
    /// Maximum y coordinate.
    pub y_max: f32,
}

impl BoundingBox2 {
    /// Creates a validated axis-aligned box.
    pub fn try_new(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> VisionResult<Self> {
        let bbox = Self { x_min, y_min, x_max, y_max };
        if !bbox.is_valid() {
            return Err(VisionError::InvalidParameter(
                "box coordinates must be finite and max >= min".to_owned(),
            ));
        }
        Ok(bbox)
    }

    /// Returns whether coordinates are finite and ordered.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.x_min.is_finite()
            && self.y_min.is_finite()
            && self.x_max.is_finite()
            && self.y_max.is_finite()
            && self.x_max >= self.x_min
            && self.y_max >= self.y_min
    }

    /// Box width.
    #[must_use]
    pub fn width(self) -> f32 {
        (self.x_max - self.x_min).max(0.0)
    }

    /// Box height.
    #[must_use]
    pub fn height(self) -> f32 {
        (self.y_max - self.y_min).max(0.0)
    }

    /// Box area.
    #[must_use]
    pub fn area(self) -> f32 {
        self.width() * self.height()
    }

    /// Intersection box, including zero-area edge contact.
    #[must_use]
    pub fn intersection(self, other: Self) -> Option<Self> {
        let result = Self {
            x_min: self.x_min.max(other.x_min),
            y_min: self.y_min.max(other.y_min),
            x_max: self.x_max.min(other.x_max),
            y_max: self.y_max.min(other.y_max),
        };
        result.is_valid().then_some(result)
    }

    /// Intersection over union, returning zero for two zero-area boxes.
    #[must_use]
    pub fn iou(self, other: Self) -> f32 {
        let intersection = self.intersection(other).map_or(0.0, Self::area);
        let union = self.area() + other.area() - intersection;
        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    /// Generalized intersection over union (GIoU).
    #[must_use]
    pub fn generalized_iou(self, other: Self) -> f32 {
        let iou = self.iou(other);
        let enclosing = Self {
            x_min: self.x_min.min(other.x_min),
            y_min: self.y_min.min(other.y_min),
            x_max: self.x_max.max(other.x_max),
            y_max: self.y_max.max(other.y_max),
        };
        let enclosing_area = enclosing.area();
        if enclosing_area == 0.0 {
            return iou;
        }
        let intersection = self.intersection(other).map_or(0.0, Self::area);
        let union = self.area() + other.area() - intersection;
        iou - (enclosing_area - union) / enclosing_area
    }

    /// Clips coordinates into an image rectangle.
    #[must_use]
    pub fn clip(self, width: f32, height: f32) -> Self {
        Self {
            x_min: self.x_min.clamp(0.0, width),
            y_min: self.y_min.clamp(0.0, height),
            x_max: self.x_max.clamp(0.0, width),
            y_max: self.y_max.clamp(0.0, height),
        }
    }

    /// Applies uniform scaling and translation, useful for letterbox mapping.
    #[must_use]
    pub fn scale_translate(self, scale: f32, tx: f32, ty: f32) -> Self {
        Self {
            x_min: self.x_min.mul_add(scale, tx),
            y_min: self.y_min.mul_add(scale, ty),
            x_max: self.x_max.mul_add(scale, tx),
            y_max: self.y_max.mul_add(scale, ty),
        }
    }
}

/// One scored detection used by class-aware post-processing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    /// Detected box.
    pub bbox: BoundingBox2,
    /// Confidence score.
    pub score: f32,
    /// Model-defined class identifier.
    pub class_id: i64,
}

/// Soft-NMS score-decay strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SoftNmsMethod {
    /// Hard suppression above the IoU threshold.
    Hard,
    /// Linear score decay above the IoU threshold.
    Linear,
    /// Gaussian score decay at every overlap.
    Gaussian {
        /// Positive Gaussian variance parameter.
        sigma: f32,
    },
}

/// Index and possibly updated score returned by Soft-NMS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredIndex {
    /// Index into the original input arrays.
    pub index: usize,
    /// Score after overlap decay.
    pub score: f32,
}

/// Greedy non-maximum suppression. Returned indices are score-descending.
pub fn nms(
    boxes: &[BoundingBox2],
    scores: &[f32],
    score_threshold: f32,
    iou_threshold: f32,
) -> VisionResult<Vec<usize>> {
    validate_nms_inputs(boxes, scores, score_threshold, iou_threshold)?;
    let mut order: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter_map(|(index, &score)| (score >= score_threshold).then_some(index))
        .collect();
    sort_indices_by_score(&mut order, scores);
    let mut keep: Vec<usize> = Vec::with_capacity(order.len());
    'candidate: for index in order {
        for &selected in &keep {
            if boxes[index].iou(boxes[selected]) > iou_threshold {
                continue 'candidate;
            }
        }
        keep.push(index);
    }
    Ok(keep)
}

/// Class-aware greedy NMS over detection records.
pub fn batched_nms(
    detections: &[Detection],
    score_threshold: f32,
    iou_threshold: f32,
) -> VisionResult<Vec<usize>> {
    if !score_threshold.is_finite()
        || !iou_threshold.is_finite()
        || !(0.0..=1.0).contains(&iou_threshold)
    {
        return Err(VisionError::InvalidParameter(
            "NMS thresholds must be finite and IoU in [0, 1]".to_owned(),
        ));
    }
    if detections.iter().any(|detection| !detection.bbox.is_valid() || !detection.score.is_finite())
    {
        return Err(VisionError::InvalidParameter(
            "detections must contain valid boxes and finite scores".to_owned(),
        ));
    }
    let scores: Vec<f32> = detections.iter().map(|detection| detection.score).collect();
    let mut order: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter_map(|(index, &score)| (score >= score_threshold).then_some(index))
        .collect();
    sort_indices_by_score(&mut order, &scores);
    let mut keep: Vec<usize> = Vec::with_capacity(order.len());
    'candidate: for index in order {
        for &selected in &keep {
            if detections[index].class_id == detections[selected].class_id
                && detections[index].bbox.iou(detections[selected].bbox) > iou_threshold
            {
                continue 'candidate;
            }
        }
        keep.push(index);
    }
    Ok(keep)
}

/// Soft non-maximum suppression with deterministic score ordering.
pub fn soft_nms(
    boxes: &[BoundingBox2],
    scores: &[f32],
    score_threshold: f32,
    iou_threshold: f32,
    method: SoftNmsMethod,
) -> VisionResult<Vec<ScoredIndex>> {
    validate_nms_inputs(boxes, scores, score_threshold, iou_threshold)?;
    if matches!(method, SoftNmsMethod::Gaussian { sigma } if !sigma.is_finite() || sigma <= 0.0) {
        return Err(VisionError::InvalidParameter(
            "Soft-NMS Gaussian sigma must be finite and positive".to_owned(),
        ));
    }
    let mut candidates: Vec<ScoredIndex> = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(index, score)| ScoredIndex { index, score })
        .collect();
    let mut output = Vec::new();
    while !candidates.is_empty() {
        candidates.sort_by(score_order);
        let selected = candidates.remove(0);
        if selected.score < score_threshold {
            break;
        }
        output.push(selected);
        for candidate in &mut candidates {
            let overlap = boxes[selected.index].iou(boxes[candidate.index]);
            let weight = match method {
                SoftNmsMethod::Hard => {
                    if overlap <= iou_threshold {
                        1.0
                    } else {
                        0.0
                    }
                }
                SoftNmsMethod::Linear => {
                    if overlap > iou_threshold {
                        1.0 - overlap
                    } else {
                        1.0
                    }
                }
                SoftNmsMethod::Gaussian { sigma } => (-(overlap * overlap) / sigma).exp(),
            };
            candidate.score *= weight;
        }
        candidates.retain(|candidate| candidate.score >= score_threshold);
    }
    Ok(output)
}

fn validate_nms_inputs(
    boxes: &[BoundingBox2],
    scores: &[f32],
    score_threshold: f32,
    iou_threshold: f32,
) -> VisionResult<()> {
    if boxes.len() != scores.len() {
        return Err(VisionError::ShapeMismatch(
            "boxes and scores must have equal lengths".to_owned(),
        ));
    }
    if !score_threshold.is_finite()
        || !iou_threshold.is_finite()
        || !(0.0..=1.0).contains(&iou_threshold)
    {
        return Err(VisionError::InvalidParameter(
            "NMS thresholds must be finite and IoU in [0, 1]".to_owned(),
        ));
    }
    if boxes.iter().any(|bbox| !bbox.is_valid()) || scores.iter().any(|score| !score.is_finite()) {
        return Err(VisionError::InvalidParameter(
            "boxes must be valid and scores finite".to_owned(),
        ));
    }
    Ok(())
}

fn sort_indices_by_score(indices: &mut [usize], scores: &[f32]) {
    indices.sort_by(|&left, &right| {
        scores[right]
            .partial_cmp(&scores[left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
}

fn score_order(left: &ScoredIndex, right: &ScoredIndex) -> Ordering {
    right
        .score
        .partial_cmp(&left.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.index.cmp(&right.index))
}

#[cfg(test)]
mod tests {
    use super::{batched_nms, nms, soft_nms, BoundingBox2, Detection, SoftNmsMethod};

    fn bbox(x0: f32, y0: f32, x1: f32, y1: f32) -> BoundingBox2 {
        BoundingBox2::try_new(x0, y0, x1, y1).unwrap()
    }

    #[test]
    fn iou_matches_known_overlap() {
        let a = bbox(0.0, 0.0, 2.0, 2.0);
        let b = bbox(1.0, 1.0, 3.0, 3.0);
        assert!((a.iou(b) - 1.0 / 7.0).abs() < 1e-6);
        assert!(a.generalized_iou(b) < a.iou(b));
    }

    #[test]
    fn nms_suppresses_lower_scored_overlap() {
        let boxes = [bbox(0.0, 0.0, 2.0, 2.0), bbox(0.1, 0.1, 2.1, 2.1), bbox(5.0, 5.0, 6.0, 6.0)];
        assert_eq!(nms(&boxes, &[0.9, 0.8, 0.7], 0.0, 0.5).unwrap(), vec![0, 2]);
    }

    #[test]
    fn batched_nms_keeps_overlapping_different_classes() {
        let detections = [
            Detection { bbox: bbox(0.0, 0.0, 2.0, 2.0), score: 0.9, class_id: 1 },
            Detection { bbox: bbox(0.0, 0.0, 2.0, 2.0), score: 0.8, class_id: 2 },
        ];
        assert_eq!(batched_nms(&detections, 0.0, 0.5).unwrap(), vec![0, 1]);
    }

    #[test]
    fn soft_nms_decays_overlapping_score() {
        let boxes = [bbox(0.0, 0.0, 2.0, 2.0), bbox(0.1, 0.1, 2.1, 2.1)];
        let result = soft_nms(&boxes, &[0.9, 0.8], 0.01, 0.5, SoftNmsMethod::Linear).unwrap();
        assert_eq!(result[0].index, 0);
        assert!(result[1].score < 0.8);
    }
}
