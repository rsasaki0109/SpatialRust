//! Byte-bounded retention shared by storage-buffer and image-texture pools.

use std::collections::HashMap;
use std::hash::Hash;

/// Default retained-byte budget for each GPU resource pool (64 MiB).
pub const DEFAULT_GPU_POOL_BUDGET_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PER_KEY: usize = 8;

/// Snapshot of one pool's retained resources and lifetime acquisition counters.
///
/// Bytes describe requested buffer sizes or RGBA8 texel payloads, excluding
/// driver overhead, live caller-owned resources, uniforms, and readback staging.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuResourcePoolStats {
    /// Maximum bytes retained after resources are recycled.
    pub budget_bytes: u64,
    /// Bytes currently retained for reuse.
    pub cached_bytes: u64,
    /// Resources currently retained for reuse.
    pub cached_resources: usize,
    /// Device allocations requested through this pool since its creation.
    pub allocation_count: u64,
    /// Acquisitions satisfied by an already retained resource.
    pub reuse_count: u64,
    /// Recycled or cached resources dropped instead of retained.
    pub discarded_count: u64,
}

#[derive(Debug)]
pub(crate) struct ResourceCache<K, V> {
    free: HashMap<K, Vec<(u64, V)>>,
    stats: GpuResourcePoolStats,
}

impl<K: Eq + Hash, V> ResourceCache<K, V> {
    pub(crate) fn new(budget_bytes: u64) -> Self {
        Self {
            free: HashMap::new(),
            stats: GpuResourcePoolStats { budget_bytes, ..Default::default() },
        }
    }

    pub(crate) fn take(&mut self, key: &K) -> Option<V> {
        let values = self.free.get_mut(key)?;
        let (bytes, value) = values.pop()?;
        if values.is_empty() {
            self.free.remove(key);
        }
        self.stats.cached_bytes -= bytes;
        self.stats.cached_resources -= 1;
        self.stats.reuse_count = self.stats.reuse_count.saturating_add(1);
        Some(value)
    }

    pub(crate) fn record_allocation(&mut self) {
        self.stats.allocation_count = self.stats.allocation_count.saturating_add(1);
    }

    pub(crate) fn record_discard(&mut self) {
        self.stats.discarded_count = self.stats.discarded_count.saturating_add(1);
    }

    pub(crate) fn put(&mut self, key: K, bytes: u64, value: V) {
        if bytes == 0
            || bytes > self.stats.budget_bytes.saturating_sub(self.stats.cached_bytes)
            || self.free.get(&key).is_some_and(|values| values.len() >= MAX_PER_KEY)
        {
            self.record_discard();
            return;
        }
        self.free.entry(key).or_default().push((bytes, value));
        self.stats.cached_bytes += bytes;
        self.stats.cached_resources += 1;
    }

    pub(crate) fn stats(&self) -> GpuResourcePoolStats {
        self.stats
    }

    pub(crate) fn clear(&mut self) {
        self.stats.discarded_count =
            self.stats.discarded_count.saturating_add(self.stats.cached_resources as u64);
        self.free.clear();
        self.stats.cached_bytes = 0;
        self.stats.cached_resources = 0;
    }

    pub(crate) fn set_budget(&mut self, budget_bytes: u64) {
        self.stats.budget_bytes = budget_bytes;
        if self.stats.cached_bytes > budget_bytes {
            self.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_boundary_and_key_churn_bound_retained_memory() {
        let mut cache = ResourceCache::new(16);
        cache.put(1, 8, 10);
        cache.put(2, 8, 20);
        for key in 3..10_000 {
            cache.put(key, 4, 30);
        }
        assert_eq!(cache.stats().cached_bytes, 16);
        assert_eq!(cache.free.len(), 2);
        assert_eq!(cache.take(&1), Some(10));
        assert!(!cache.free.contains_key(&1));
        assert_eq!(cache.take(&1), None);
        assert_eq!(cache.stats().reuse_count, 1);
        cache.put(3, 8, 30);
        assert_eq!(cache.stats().cached_bytes, 16);
        assert_eq!(cache.stats().cached_resources, 2);
    }

    #[test]
    fn per_key_limit_shrinking_and_zero_budget_keep_lifetime_counters() {
        let mut cache = ResourceCache::new(u64::MAX);
        for _ in 0..10 {
            cache.record_allocation();
            cache.put(1, 4, ());
        }
        assert_eq!(cache.stats().cached_resources, MAX_PER_KEY);
        assert_eq!(cache.stats().discarded_count, 2);
        cache.set_budget(31);
        assert_eq!(cache.stats().cached_bytes, 0);
        assert!(cache.free.is_empty());
        assert_eq!(cache.stats().allocation_count, 10);
        assert_eq!(cache.stats().discarded_count, 10);
        cache.set_budget(0);
        cache.put(2, 4, ());
        cache.put(3, 0, ());
        assert_eq!(cache.stats().cached_resources, 0);
        assert_eq!(cache.stats().discarded_count, 12);
        cache.set_budget(4);
        cache.put(4, 4, ());
        cache.clear();
        assert_eq!(cache.stats().budget_bytes, 4);
        assert_eq!(cache.stats().cached_resources, 0);
    }
}
