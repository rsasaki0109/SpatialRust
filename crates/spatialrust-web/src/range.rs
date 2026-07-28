use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{WebError, WebResult};

/// Half-open remote byte range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct ByteRange {
    start: u64,
    end_exclusive: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ByteRangeInput {
    start: u64,
    end_exclusive: u64,
}

impl<'de> Deserialize<'de> for ByteRange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let input = ByteRangeInput::deserialize(deserializer)?;
        Self::try_new(input.start, input.end_exclusive).map_err(serde::de::Error::custom)
    }
}

impl ByteRange {
    /// Creates a non-empty range.
    pub fn try_new(start: u64, end_exclusive: u64) -> WebResult<Self> {
        if end_exclusive <= start {
            return Err(WebError::Range("byte range must satisfy start < end_exclusive".into()));
        }
        Ok(Self { start, end_exclusive })
    }

    /// Inclusive start offset.
    #[must_use]
    pub const fn start(self) -> u64 {
        self.start
    }

    /// Exclusive end offset.
    #[must_use]
    pub const fn end_exclusive(self) -> u64 {
        self.end_exclusive
    }

    /// Range length.
    #[must_use]
    pub const fn len(self) -> u64 {
        self.end_exclusive - self.start
    }

    /// A validated range is never empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end_exclusive
    }

    /// HTTP `Range` header value.
    #[must_use]
    pub fn http_header(self) -> String {
        format!("bytes={}-{}", self.start, self.end_exclusive - 1)
    }
}

/// Hard limits for one remote planning/cache session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeBudget {
    /// Maximum cache misses emitted by one plan.
    pub max_requests_per_plan: usize,
    /// Maximum aggregate bytes emitted by one plan.
    pub max_requested_bytes_per_plan: u64,
    /// Maximum one range length.
    pub max_single_range_bytes: u64,
    /// Maximum cached response bytes.
    pub max_cache_bytes: u64,
}

impl RangeBudget {
    /// Validates positive limits and compatible range/cache sizes.
    pub fn validate(self) -> WebResult<()> {
        if self.max_requests_per_plan == 0
            || self.max_requested_bytes_per_plan == 0
            || self.max_single_range_bytes == 0
            || self.max_cache_bytes == 0
            || self.max_single_range_bytes > self.max_requested_bytes_per_plan
        {
            return Err(WebError::Range(
                "range limits must be positive and single-range <= per-plan bytes".into(),
            ));
        }
        Ok(())
    }
}

/// Deterministic cache-hit/fetch/denial decision.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangePlan {
    /// Exact cached ranges.
    pub cached: Vec<ByteRange>,
    /// Misses admitted for fetch.
    pub fetch: Vec<ByteRange>,
    /// Requests denied by cancellation or hard limits.
    pub denied: Vec<ByteRange>,
    /// Exact admitted fetch bytes.
    pub requested_bytes: u64,
    /// Monotonic plan generation.
    pub generation: u64,
}

/// Result of admitting one fetched response to the cache.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeAdmissionReceipt {
    /// Admitted range.
    pub range: ByteRange,
    /// Exact response bytes copied into the cache.
    pub response_bytes: u64,
    /// LRU ranges evicted before admission.
    pub evicted: Vec<ByteRange>,
    /// Cache bytes after admission.
    pub cached_bytes: u64,
}

#[derive(Clone, Debug)]
struct CacheEntry {
    bytes: Vec<u8>,
    last_used: u64,
}

/// Exact-range response cache with deterministic LRU eviction.
#[derive(Clone, Debug)]
pub struct RangeCache {
    budget: RangeBudget,
    entries: BTreeMap<ByteRange, CacheEntry>,
    cached_bytes: u64,
    tick: u64,
}

impl RangeCache {
    /// Creates an empty cache.
    pub fn try_new(budget: RangeBudget) -> WebResult<Self> {
        budget.validate()?;
        Ok(Self { budget, entries: BTreeMap::new(), cached_bytes: 0, tick: 0 })
    }

    /// Returns exact cached bytes and updates recency.
    pub fn get(&mut self, range: ByteRange) -> Option<&[u8]> {
        self.tick = self.tick.saturating_add(1);
        let entry = self.entries.get_mut(&range)?;
        entry.last_used = self.tick;
        Some(&entry.bytes)
    }

    /// Whether an exact range is cached, without changing recency.
    #[must_use]
    pub fn contains(&self, range: ByteRange) -> bool {
        self.entries.contains_key(&range)
    }

    /// Current cached bytes.
    #[must_use]
    pub const fn cached_bytes(&self) -> u64 {
        self.cached_bytes
    }

    /// Admits an exact-length response, evicting deterministic LRU entries.
    ///
    /// Failure leaves the cache unchanged.
    pub fn admit(
        &mut self,
        range: ByteRange,
        response: Vec<u8>,
    ) -> WebResult<RangeAdmissionReceipt> {
        let expected = range.len();
        let actual = u64::try_from(response.len())
            .map_err(|_| WebError::Range("response length exceeds u64".into()))?;
        if actual != expected {
            return Err(WebError::Range(format!(
                "range response length {actual} does not match requested {expected}"
            )));
        }
        if actual > self.budget.max_single_range_bytes || actual > self.budget.max_cache_bytes {
            return Err(WebError::Range(
                "range response cannot fit single-range/cache budget".into(),
            ));
        }
        let existing = self.entries.get(&range).map_or(0, |entry| entry.bytes.len() as u64);
        let mut next_bytes = self
            .cached_bytes
            .checked_sub(existing)
            .and_then(|value| value.checked_add(actual))
            .ok_or_else(|| WebError::Range("cache byte accounting overflow".into()))?;
        let mut candidates: Vec<_> = self
            .entries
            .iter()
            .filter(|(candidate, _)| **candidate != range)
            .map(|(candidate, entry)| (*candidate, entry.last_used, entry.bytes.len() as u64))
            .collect();
        candidates.sort_by_key(|(candidate, last_used, _)| (*last_used, *candidate));
        let mut evicted = Vec::new();
        for (candidate, _, bytes) in candidates {
            if next_bytes <= self.budget.max_cache_bytes {
                break;
            }
            next_bytes -= bytes;
            evicted.push(candidate);
        }
        if next_bytes > self.budget.max_cache_bytes {
            return Err(WebError::Range("cache capacity unavailable".into()));
        }
        let next_tick = self
            .tick
            .checked_add(1)
            .ok_or_else(|| WebError::Range("cache recency overflow".into()))?;
        for candidate in &evicted {
            self.entries.remove(candidate);
        }
        self.tick = next_tick;
        self.entries.insert(range, CacheEntry { bytes: response, last_used: self.tick });
        self.cached_bytes = next_bytes;
        Ok(RangeAdmissionReceipt {
            range,
            response_bytes: actual,
            evicted,
            cached_bytes: next_bytes,
        })
    }
}

/// Stateful deterministic range planner.
#[derive(Clone, Debug)]
pub struct RangePlanner {
    budget: RangeBudget,
    generation: u64,
}

impl RangePlanner {
    /// Creates a planner.
    pub fn try_new(budget: RangeBudget) -> WebResult<Self> {
        budget.validate()?;
        Ok(Self { budget, generation: 0 })
    }

    /// Deduplicates/sorts ranges and emits bounded cache misses.
    pub fn plan(
        &mut self,
        ranges: impl IntoIterator<Item = ByteRange>,
        cache: &RangeCache,
        cancelled: bool,
    ) -> WebResult<RangePlan> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| WebError::Range("range plan generation overflow".into()))?;
        let ranges: BTreeSet<_> = ranges.into_iter().collect();
        let mut plan = RangePlan { generation: self.generation, ..RangePlan::default() };
        for range in ranges {
            if range.is_empty() || range.len() > self.budget.max_single_range_bytes {
                plan.denied.push(range);
                continue;
            }
            if cache.contains(range) {
                plan.cached.push(range);
                continue;
            }
            let next_bytes = plan.requested_bytes.checked_add(range.len());
            if cancelled
                || plan.fetch.len() >= self.budget.max_requests_per_plan
                || next_bytes.map_or(true, |bytes| bytes > self.budget.max_requested_bytes_per_plan)
            {
                plan.denied.push(range);
                continue;
            }
            plan.requested_bytes = next_bytes.expect("checked above");
            plan.fetch.push(range);
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::{ByteRange, RangeBudget, RangeCache, RangePlanner};

    fn budget() -> RangeBudget {
        RangeBudget {
            max_requests_per_plan: 2,
            max_requested_bytes_per_plan: 8,
            max_single_range_bytes: 4,
            max_cache_bytes: 8,
        }
    }

    #[test]
    fn plan_deduplicates_orders_hits_and_denies_before_fetch() {
        let a = ByteRange::try_new(0, 4).unwrap();
        let b = ByteRange::try_new(4, 8).unwrap();
        let c = ByteRange::try_new(8, 12).unwrap();
        let mut cache = RangeCache::try_new(budget()).unwrap();
        cache.admit(a, vec![1; 4]).unwrap();
        let mut planner = RangePlanner::try_new(budget()).unwrap();
        let plan = planner.plan([c, b, a, b], &cache, false).unwrap();
        assert_eq!(plan.cached, vec![a]);
        assert_eq!(plan.fetch, vec![b, c]);
        assert_eq!(plan.requested_bytes, 8);

        let cancelled = planner.plan([b], &cache, true).unwrap();
        assert!(cancelled.fetch.is_empty());
        assert_eq!(cancelled.denied, vec![b]);
    }

    #[test]
    fn range_json_rejects_empty_reversed_and_unknown_fields() {
        assert!(serde_json::from_str::<ByteRange>(r#"{"start":4,"end_exclusive":4}"#).is_err());
        assert!(serde_json::from_str::<ByteRange>(r#"{"start":8,"end_exclusive":4}"#).is_err());
        assert!(serde_json::from_str::<ByteRange>(
            r#"{"start":0,"end_exclusive":4,"unexpected":true}"#
        )
        .is_err());
    }

    #[test]
    fn cache_checks_exact_length_and_evicts_lru_deterministically() {
        let a = ByteRange::try_new(0, 4).unwrap();
        let b = ByteRange::try_new(4, 8).unwrap();
        let c = ByteRange::try_new(8, 12).unwrap();
        let mut cache = RangeCache::try_new(budget()).unwrap();
        cache.admit(a, vec![1; 4]).unwrap();
        cache.admit(b, vec![2; 4]).unwrap();
        cache.get(a).unwrap();
        let receipt = cache.admit(c, vec![3; 4]).unwrap();
        assert_eq!(receipt.evicted, vec![b]);
        assert!(cache.contains(a));
        assert!(cache.contains(c));
        assert_eq!(cache.cached_bytes(), 8);

        let before = cache.cached_bytes();
        assert!(cache.admit(ByteRange::try_new(12, 16).unwrap(), vec![0; 3]).is_err());
        assert_eq!(cache.cached_bytes(), before);
    }
}
