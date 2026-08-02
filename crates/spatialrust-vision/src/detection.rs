//! Detection post-processing primitives.

use std::cmp::Ordering;
use std::collections::HashMap;

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
    let areas: Vec<f32> = boxes.iter().copied().map(BoundingBox2::area).collect();
    let mut keep: Vec<usize> = Vec::with_capacity(order.len());
    'candidate: for index in order {
        for &selected in &keep {
            if iou_with_areas(boxes[index], areas[index], boxes[selected], areas[selected])
                > iou_threshold
            {
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
    let areas: Vec<f32> = detections.iter().map(|detection| detection.bbox.area()).collect();
    let mut keep: Vec<usize> = Vec::with_capacity(order.len());
    let mut keep_by_class: HashMap<i64, Vec<usize>> = HashMap::new();
    'candidate: for index in order {
        let class_id = detections[index].class_id;
        let selected_for_class = keep_by_class.entry(class_id).or_default();
        for &selected in selected_for_class.iter() {
            if iou_with_areas(
                detections[index].bbox,
                areas[index],
                detections[selected].bbox,
                areas[selected],
            ) > iou_threshold
            {
                continue 'candidate;
            }
        }
        keep.push(index);
        selected_for_class.push(index);
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
    let areas: Vec<f32> = boxes.iter().copied().map(BoundingBox2::area).collect();
    let mut output = Vec::with_capacity(candidates.len());
    let mut best = best_scored_candidate(&candidates);
    while let Some(best_index) = best {
        if candidates[best_index].score < score_threshold {
            break;
        }
        let selected = candidates.swap_remove(best_index);
        output.push(selected);
        let selected_box = boxes[selected.index];
        let selected_area = areas[selected.index];
        let mut active_count = 0;
        let mut next_best = None;
        for read_index in 0..candidates.len() {
            let mut candidate = candidates[read_index];
            let overlap = iou_with_areas(
                selected_box,
                selected_area,
                boxes[candidate.index],
                areas[candidate.index],
            );
            if overlap != 0.0 {
                match method {
                    SoftNmsMethod::Hard => {
                        if overlap > iou_threshold {
                            candidate.score *= 0.0;
                        }
                    }
                    SoftNmsMethod::Linear => {
                        if overlap > iou_threshold {
                            candidate.score *= 1.0 - overlap;
                        }
                    }
                    SoftNmsMethod::Gaussian { sigma } => {
                        candidate.score *= (-(overlap * overlap) / sigma).exp();
                    }
                }
            }
            if candidate.score >= score_threshold {
                candidates[active_count] = candidate;
                let is_next_best = match next_best {
                    Some(index) => scored_candidate_precedes(candidate, candidates[index]),
                    None => true,
                };
                if is_next_best {
                    next_best = Some(active_count);
                }
                active_count += 1;
            }
        }
        candidates.truncate(active_count);
        best = next_best;
    }
    Ok(output)
}

fn best_scored_candidate(candidates: &[ScoredIndex]) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let mut best = 0;
    for index in 1..candidates.len() {
        if scored_candidate_precedes(candidates[index], candidates[best]) {
            best = index;
        }
    }
    Some(best)
}

fn scored_candidate_precedes(left: ScoredIndex, right: ScoredIndex) -> bool {
    left.score > right.score || left.score == right.score && left.index < right.index
}

fn iou_with_areas(left: BoundingBox2, left_area: f32, right: BoundingBox2, right_area: f32) -> f32 {
    let intersection_width = left.x_max.min(right.x_max) - left.x_min.max(right.x_min);
    if intersection_width <= 0.0 {
        return 0.0;
    }
    let intersection_height = left.y_max.min(right.y_max) - left.y_min.max(right.y_min);
    if intersection_height <= 0.0 {
        return 0.0;
    }
    let intersection = intersection_width * intersection_height;
    let union = left_area + right_area - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
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

#[cfg(test)]
mod tests {
    use super::{
        batched_nms, iou_with_areas, nms, soft_nms, BoundingBox2, Detection, SoftNmsMethod,
    };

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
    fn cached_area_iou_matches_public_iou() {
        let boxes = [
            BoundingBox2::try_new(0.0, 0.0, 20.0, 20.0).unwrap(),
            BoundingBox2::try_new(2.0, 2.0, 18.0, 18.0).unwrap(),
            BoundingBox2::try_new(30.0, 30.0, 45.0, 45.0).unwrap(),
            BoundingBox2::try_new(5.0, 5.0, 5.0, 8.0).unwrap(),
        ];
        for &left in &boxes {
            for &right in &boxes {
                assert_eq!(iou_with_areas(left, left.area(), right, right.area()), left.iou(right),);
            }
        }
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
    fn batched_nms_suppresses_per_class_in_global_score_order() {
        let detections = [
            Detection { bbox: bbox(0.0, 0.0, 2.0, 2.0), score: 0.7, class_id: 4 },
            Detection { bbox: bbox(0.1, 0.1, 2.1, 2.1), score: 0.9, class_id: 4 },
            Detection { bbox: bbox(0.1, 0.1, 2.1, 2.1), score: 0.8, class_id: 9 },
            Detection { bbox: bbox(5.0, 5.0, 6.0, 6.0), score: 0.6, class_id: 4 },
        ];
        assert_eq!(batched_nms(&detections, 0.0, 0.5).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn class_buckets_match_global_scan_reference() {
        let mut state = 119_u64;
        let detections = (0..257)
            .map(|index| {
                let x = sample(&mut state) * 64.0;
                let y = sample(&mut state) * 64.0;
                let width = 1.0 + sample(&mut state) * 20.0;
                let height = 1.0 + sample(&mut state) * 20.0;
                Detection {
                    bbox: bbox(x, y, x + width, y + height),
                    score: sample(&mut state),
                    class_id: (index % 11) as i64 - 5,
                }
            })
            .collect::<Vec<_>>();
        let actual = batched_nms(&detections, 0.2, 0.45).unwrap();

        let mut order = (0..detections.len())
            .filter(|&index| detections[index].score >= 0.2)
            .collect::<Vec<_>>();
        order.sort_by(|&left, &right| {
            detections[right]
                .score
                .partial_cmp(&detections[left].score)
                .unwrap()
                .then_with(|| left.cmp(&right))
        });
        let mut expected: Vec<usize> = Vec::new();
        'candidate: for index in order {
            for &selected in &expected {
                if detections[index].class_id == detections[selected].class_id
                    && detections[index].bbox.iou(detections[selected].bbox) > 0.45
                {
                    continue 'candidate;
                }
            }
            expected.push(index);
        }
        assert_eq!(actual, expected);
    }

    fn sample(state: &mut u64) -> f32 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        (value >> 40) as f32 / (1_u32 << 24) as f32
    }

    #[test]
    fn soft_nms_decays_overlapping_score() {
        let boxes = [bbox(0.0, 0.0, 2.0, 2.0), bbox(0.1, 0.1, 2.1, 2.1)];
        let result = soft_nms(&boxes, &[0.9, 0.8], 0.01, 0.5, SoftNmsMethod::Linear).unwrap();
        assert_eq!(result[0].index, 0);
        assert!(result[1].score < 0.8);
        assert!(soft_nms(&boxes, &[0.9, 0.8], 0.01, 0.5, SoftNmsMethod::Gaussian { sigma: 0.0 },)
            .is_err());
    }

    #[test]
    fn soft_nms_selection_scan_preserves_sorting_semantics() {
        let mut state = 127_u64;
        let boxes = (0..73)
            .map(|_| {
                let x = sample(&mut state) * 64.0;
                let y = sample(&mut state) * 64.0;
                let width = 1.0 + sample(&mut state) * 20.0;
                let height = 1.0 + sample(&mut state) * 20.0;
                bbox(x, y, x + width, y + height)
            })
            .collect::<Vec<_>>();
        let mut scores = (0..boxes.len()).map(|_| sample(&mut state)).collect::<Vec<_>>();
        scores[5] = scores[2];
        for method in
            [SoftNmsMethod::Hard, SoftNmsMethod::Linear, SoftNmsMethod::Gaussian { sigma: 0.5 }]
        {
            let actual = soft_nms(&boxes, &scores, 0.2, 0.45, method).unwrap();
            let expected = sorting_soft_nms_reference(&boxes, &scores, 0.2, 0.45, method);
            assert_eq!(actual, expected);
        }

        let overlapping = [bbox(0.0, 0.0, 2.0, 2.0), bbox(0.0, 0.0, 2.0, 2.0)];
        let negative_scores = [1.0, -0.6];
        assert_eq!(
            soft_nms(&overlapping, &negative_scores, -0.5, 0.45, SoftNmsMethod::Hard).unwrap(),
            sorting_soft_nms_reference(
                &overlapping,
                &negative_scores,
                -0.5,
                0.45,
                SoftNmsMethod::Hard,
            ),
        );
    }

    fn sorting_soft_nms_reference(
        boxes: &[BoundingBox2],
        scores: &[f32],
        score_threshold: f32,
        iou_threshold: f32,
        method: SoftNmsMethod,
    ) -> Vec<super::ScoredIndex> {
        let mut candidates = scores
            .iter()
            .copied()
            .enumerate()
            .map(|(index, score)| super::ScoredIndex { index, score })
            .collect::<Vec<_>>();
        let mut output = Vec::new();
        while !candidates.is_empty() {
            candidates.sort_by(|left, right| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap()
                    .then_with(|| left.index.cmp(&right.index))
            });
            let selected = candidates.remove(0);
            if selected.score < score_threshold {
                break;
            }
            output.push(selected);
            for candidate in &mut candidates {
                let overlap = boxes[selected.index].iou(boxes[candidate.index]);
                let weight = match method {
                    SoftNmsMethod::Hard => (overlap <= iou_threshold) as u8 as f32,
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
        output
    }
}
