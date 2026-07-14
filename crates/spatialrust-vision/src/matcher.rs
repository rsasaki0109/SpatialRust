//! Brute-force descriptor matching with explicit distance semantics.

use crate::{DescriptorBuffer, DescriptorKind, FeatureMatch, VisionError, VisionResult};

/// Filtering applied to brute-force descriptor correspondences.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MatchOptions {
    /// Keep only pairs whose reverse nearest neighbour is the query row.
    pub cross_check: bool,
    /// Lowe ratio threshold in `(0, 1)`; requires at least two train rows.
    pub ratio: Option<f32>,
    /// Optional inclusive maximum distance.
    pub max_distance: Option<f32>,
}

impl MatchOptions {
    fn validate(self) -> VisionResult<Self> {
        if self.ratio.is_some_and(|ratio| !ratio.is_finite() || ratio <= 0.0 || ratio >= 1.0) {
            return Err(VisionError::InvalidParameter(
                "descriptor match ratio must be finite and in (0, 1)".into(),
            ));
        }
        if self.max_distance.is_some_and(|distance| !distance.is_finite() || distance < 0.0) {
            return Err(VisionError::InvalidParameter(
                "descriptor maximum distance must be finite and non-negative".into(),
            ));
        }
        Ok(self)
    }
}

/// Matches each query descriptor to its nearest train descriptor.
///
/// Binary rows use Hamming distance and float rows use Euclidean L2 distance.
/// Equal distances are resolved by the lowest row index. Returned matches remain
/// in ascending query-row order.
pub fn match_descriptors(
    query: &DescriptorBuffer,
    train: &DescriptorBuffer,
    options: MatchOptions,
) -> VisionResult<Vec<FeatureMatch>> {
    let options = options.validate()?;
    validate_compatibility(query, train)?;
    if query.is_empty() || train.is_empty() {
        return Ok(Vec::new());
    }
    if options.ratio.is_some() && train.len() < 2 {
        return Err(VisionError::InvalidParameter(
            "descriptor ratio matching requires at least two train rows".into(),
        ));
    }

    let reverse_best = options
        .cross_check
        .then(|| (0..train.len()).map(|index| nearest(train, index, query).0).collect::<Vec<_>>());
    let mut matches = Vec::with_capacity(query.len());
    for query_index in 0..query.len() {
        let (train_index, best, second) = nearest(query, query_index, train);
        if options.ratio.is_some_and(|ratio| best >= ratio * second.unwrap_or(f32::INFINITY)) {
            continue;
        }
        if options.max_distance.is_some_and(|maximum| best > maximum) {
            continue;
        }
        if reverse_best.as_ref().is_some_and(|indices| indices[train_index] != query_index) {
            continue;
        }
        matches.push(FeatureMatch::try_new(query_index, train_index, best)?);
    }
    Ok(matches)
}

fn validate_compatibility(query: &DescriptorBuffer, train: &DescriptorBuffer) -> VisionResult<()> {
    if query.kind() != train.kind() || query.width() != train.width() {
        return Err(VisionError::ShapeMismatch(format!(
            "descriptor matrices must have equal kind and width (query {:?}/{}, train {:?}/{})",
            query.kind(),
            query.width(),
            train.kind(),
            train.width()
        )));
    }
    Ok(())
}

fn nearest(
    source: &DescriptorBuffer,
    source_index: usize,
    target: &DescriptorBuffer,
) -> (usize, f32, Option<f32>) {
    let mut candidates = (0..target.len())
        .map(|target_index| (target_index, distance(source, source_index, target, target_index)))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.1.total_cmp(&right.1).then_with(|| left.0.cmp(&right.0)));
    let (best_index, best_distance) = candidates[0];
    (best_index, best_distance, candidates.get(1).map(|candidate| candidate.1))
}

fn distance(
    left: &DescriptorBuffer,
    left_index: usize,
    right: &DescriptorBuffer,
    right_index: usize,
) -> f32 {
    match left.kind() {
        DescriptorKind::Binary => left
            .binary_row(left_index)
            .expect("validated binary row")
            .iter()
            .zip(right.binary_row(right_index).expect("validated binary row"))
            .map(|(a, b)| (a ^ b).count_ones())
            .sum::<u32>() as f32,
        DescriptorKind::Float32 => left
            .float32_row(left_index)
            .expect("validated float row")
            .iter()
            .zip(right.float32_row(right_index).expect("validated float row"))
            .map(|(a, b)| {
                let delta = a - b;
                delta * delta
            })
            .sum::<f32>()
            .sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::{match_descriptors, MatchOptions};
    use crate::DescriptorBuffer;

    #[test]
    fn hamming_matching_is_deterministic_and_filters_distance() {
        let query = DescriptorBuffer::try_binary(2, 1, vec![0b0000_0000, 0b1111_0000]).unwrap();
        let train = DescriptorBuffer::try_binary(3, 1, vec![0b0000_0011, 0b0000_1100, 0b1111_1111])
            .unwrap();
        let matches = match_descriptors(&query, &train, MatchOptions::default()).unwrap();
        assert_eq!((matches[0].train_index(), matches[0].distance()), (0, 2.0));
        assert_eq!((matches[1].train_index(), matches[1].distance()), (2, 4.0));

        let filtered = match_descriptors(
            &query,
            &train,
            MatchOptions { max_distance: Some(2.0), ..MatchOptions::default() },
        )
        .unwrap();
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn l2_ratio_and_cross_check_match_expected_rows() {
        let query = DescriptorBuffer::try_float32(2, 2, vec![0.0, 0.0, 10.0, 10.0]).unwrap();
        let train =
            DescriptorBuffer::try_float32(3, 2, vec![1.0, 0.0, 3.0, 0.0, 10.0, 9.0]).unwrap();
        let matches = match_descriptors(
            &query,
            &train,
            MatchOptions { cross_check: true, ratio: Some(0.8), max_distance: None },
        )
        .unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!((matches[0].query_index(), matches[0].train_index()), (0, 0));
        assert_eq!((matches[1].query_index(), matches[1].train_index()), (1, 2));
        assert_eq!(matches[0].distance(), 1.0);
        assert_eq!(matches[1].distance(), 1.0);
    }

    #[test]
    fn incompatible_and_invalid_match_options_are_rejected() {
        let binary = DescriptorBuffer::try_binary(1, 1, vec![0]).unwrap();
        let float = DescriptorBuffer::try_float32(1, 1, vec![0.0]).unwrap();
        assert!(match_descriptors(&binary, &float, MatchOptions::default()).is_err());
        assert!(match_descriptors(
            &binary,
            &binary,
            MatchOptions { ratio: Some(1.0), ..MatchOptions::default() }
        )
        .is_err());
    }
}
