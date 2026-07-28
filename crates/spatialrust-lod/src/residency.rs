use std::collections::{BTreeMap, BTreeSet};

#[cfg(feature = "records")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "records")]
use spatialrust_records::{CancellationToken, MemoryBudget, MemoryReservation, MemoryTracker};

use crate::{LodError, LodResult, NodeId};

/// Hard limits shared by selection, decode/upload admission, and GPU residency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodBudgets {
    /// Maximum selected/resident points.
    pub max_points: u64,
    /// Maximum simultaneously leased decoded host bytes.
    pub max_host_bytes: u64,
    /// Maximum resident GPU bytes.
    pub max_gpu_bytes: u64,
    /// Maximum upload bytes admitted in one frame generation.
    pub max_upload_bytes_per_frame: u64,
    /// Maximum concurrently leased/in-flight chunks.
    pub max_in_flight: usize,
}

impl LodBudgets {
    /// Requires every hard limit to be positive.
    pub fn validate(self) -> LodResult<()> {
        if self.max_points == 0
            || self.max_host_bytes == 0
            || self.max_gpu_bytes == 0
            || self.max_upload_bytes_per_frame == 0
            || self.max_in_flight == 0
        {
            return Err(LodError::BudgetExceeded(
                "all LOD resource limits must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// One device-resident node tracked without owning backend-specific handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentNode {
    /// Node identity.
    pub id: NodeId,
    /// Resident points.
    pub point_count: u64,
    /// Resident allocation bytes.
    pub gpu_bytes: u64,
    /// Last frame generation that displayed/touched this node.
    pub last_used_generation: u64,
}

/// Exact result of one GPU cache admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuCacheReceipt {
    /// Newly admitted node.
    pub admitted: NodeId,
    /// Nodes explicitly evicted before admission.
    pub evicted: Vec<NodeId>,
    /// Resident points after admission.
    pub resident_points: u64,
    /// Resident GPU bytes after admission.
    pub resident_gpu_bytes: u64,
}

/// Backend-neutral GPU residency ledger with deterministic LRU eviction.
#[derive(Clone, Debug)]
pub struct LodGpuCache {
    budgets: LodBudgets,
    residents: BTreeMap<NodeId, ResidentNode>,
    resident_points: u64,
    resident_gpu_bytes: u64,
}

impl LodGpuCache {
    /// Creates an empty bounded cache.
    pub fn try_new(budgets: LodBudgets) -> LodResult<Self> {
        budgets.validate()?;
        Ok(Self { budgets, residents: BTreeMap::new(), resident_points: 0, resident_gpu_bytes: 0 })
    }

    /// Current resident IDs.
    #[must_use]
    pub fn resident_ids(&self) -> BTreeSet<NodeId> {
        self.residents.keys().copied().collect()
    }

    /// Finds a resident.
    #[must_use]
    pub fn resident(&self, id: NodeId) -> Option<&ResidentNode> {
        self.residents.get(&id)
    }

    /// Marks a resident as used by `generation`.
    pub fn touch(&mut self, id: NodeId, generation: u64) -> LodResult<()> {
        let node = self.residents.get_mut(&id).ok_or(LodError::UnknownNode(id.0))?;
        node.last_used_generation = generation;
        Ok(())
    }

    /// Admits a completed explicit upload, evicting only unprotected LRU nodes.
    ///
    /// Failure leaves the cache unchanged.
    pub fn admit(
        &mut self,
        id: NodeId,
        point_count: u64,
        gpu_bytes: u64,
        generation: u64,
        protected: &BTreeSet<NodeId>,
    ) -> LodResult<GpuCacheReceipt> {
        if point_count == 0
            || gpu_bytes == 0
            || point_count > self.budgets.max_points
            || gpu_bytes > self.budgets.max_gpu_bytes
        {
            return Err(LodError::BudgetExceeded(format!(
                "node {} cannot fit GPU point/byte budget",
                id.0
            )));
        }
        if self.residents.contains_key(&id) {
            self.touch(id, generation)?;
            return Ok(GpuCacheReceipt {
                admitted: id,
                evicted: Vec::new(),
                resident_points: self.resident_points,
                resident_gpu_bytes: self.resident_gpu_bytes,
            });
        }

        let mut next_points = self
            .resident_points
            .checked_add(point_count)
            .ok_or_else(|| LodError::BudgetExceeded("resident point count overflow".into()))?;
        let mut next_bytes = self
            .resident_gpu_bytes
            .checked_add(gpu_bytes)
            .ok_or_else(|| LodError::BudgetExceeded("resident GPU byte count overflow".into()))?;
        let mut candidates: Vec<_> = self
            .residents
            .values()
            .filter(|resident| !protected.contains(&resident.id))
            .copied()
            .collect();
        candidates.sort_by_key(|resident| (resident.last_used_generation, resident.id));
        let mut evicted = Vec::new();
        for candidate in candidates {
            if next_points <= self.budgets.max_points && next_bytes <= self.budgets.max_gpu_bytes {
                break;
            }
            next_points -= candidate.point_count;
            next_bytes -= candidate.gpu_bytes;
            evicted.push(candidate.id);
        }
        if next_points > self.budgets.max_points || next_bytes > self.budgets.max_gpu_bytes {
            return Err(LodError::BudgetExceeded(
                "protected GPU residents leave insufficient capacity".into(),
            ));
        }
        for evicted_id in &evicted {
            self.residents.remove(evicted_id);
        }
        self.resident_points = next_points;
        self.resident_gpu_bytes = next_bytes;
        self.residents.insert(
            id,
            ResidentNode { id, point_count, gpu_bytes, last_used_generation: generation },
        );
        Ok(GpuCacheReceipt {
            admitted: id,
            evicted,
            resident_points: next_points,
            resident_gpu_bytes: next_bytes,
        })
    }

    /// Explicitly removes a node.
    pub fn remove(&mut self, id: NodeId) -> Option<ResidentNode> {
        let resident = self.residents.remove(&id)?;
        self.resident_points -= resident.point_count;
        self.resident_gpu_bytes -= resident.gpu_bytes;
        Some(resident)
    }
}

#[cfg(feature = "records")]
#[derive(Debug, Default)]
struct UploadState {
    generation: u64,
    admitted_upload_bytes: u64,
    in_flight: BTreeSet<NodeId>,
}

/// Explicit host-memory lease for one decoded LOD chunk.
#[cfg(feature = "records")]
#[derive(Debug)]
pub struct HostChunkLease {
    node: NodeId,
    point_count: u64,
    upload_bytes: u64,
    generation: u64,
    cancellation: CancellationToken,
    reservation: MemoryReservation,
    state: Arc<Mutex<UploadState>>,
    released: bool,
}

#[cfg(feature = "records")]
impl HostChunkLease {
    /// Node identity.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// Exact reserved host bytes.
    #[must_use]
    pub const fn host_bytes(&self) -> u64 {
        self.reservation.bytes()
    }

    /// Cooperative cancellation token checked before upload completion.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    fn release(&mut self) {
        if !self.released {
            self.state.lock().expect("LOD upload state poisoned").in_flight.remove(&self.node);
            self.released = true;
        }
    }
}

#[cfg(feature = "records")]
impl Drop for HostChunkLease {
    fn drop(&mut self) {
        self.release();
    }
}

/// Exact receipt for a host lease followed by caller-performed GPU upload.
#[cfg(feature = "records")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LodUploadReceipt {
    /// Uploaded node.
    pub node: NodeId,
    /// Plan/frame generation.
    pub generation: u64,
    /// Decoded host bytes held until upload completion.
    pub host_bytes: u64,
    /// Caller-declared bytes transferred to GPU.
    pub upload_bytes: u64,
    /// Caller-declared resident GPU bytes.
    pub gpu_bytes: u64,
    /// Nodes evicted to fit.
    pub evicted: Vec<NodeId>,
    /// Peak host lease bytes observed by the shared tracker.
    pub peak_host_bytes: u64,
}

/// Bounded lease/upload admission state.
#[cfg(feature = "records")]
#[derive(Clone, Debug)]
pub struct LodUploadSession {
    budgets: LodBudgets,
    memory: MemoryTracker,
    state: Arc<Mutex<UploadState>>,
}

#[cfg(feature = "records")]
impl LodUploadSession {
    /// Creates an empty upload session using the exact records memory tracker.
    pub fn try_new(budgets: LodBudgets) -> LodResult<Self> {
        budgets.validate()?;
        let memory_budget = MemoryBudget::new(budgets.max_host_bytes)
            .map_err(|error| LodError::Records(error.to_string()))?;
        Ok(Self {
            budgets,
            memory: MemoryTracker::new(memory_budget),
            state: Arc::new(Mutex::new(UploadState::default())),
        })
    }

    /// Starts a new frame/generation and resets only the per-frame upload ledger.
    pub fn begin_frame(&self, generation: u64) -> LodResult<()> {
        let mut state = self.state.lock().expect("LOD upload state poisoned");
        if generation <= state.generation {
            return Err(LodError::InvalidPlanner(
                "upload generations must increase monotonically".into(),
            ));
        }
        state.generation = generation;
        state.admitted_upload_bytes = 0;
        Ok(())
    }

    /// Reserves host memory, in-flight capacity, and this frame's upload bytes.
    pub fn try_lease(
        &self,
        node: NodeId,
        point_count: u64,
        host_bytes: u64,
        upload_bytes: u64,
        cancellation: CancellationToken,
    ) -> LodResult<HostChunkLease> {
        if point_count == 0 || host_bytes == 0 || upload_bytes == 0 {
            return Err(LodError::BudgetExceeded(
                "LOD lease counts and bytes must be positive".into(),
            ));
        }
        let mut state = self.state.lock().expect("LOD upload state poisoned");
        if state.in_flight.contains(&node) {
            return Err(LodError::BudgetExceeded(format!("node {} is already in flight", node.0)));
        }
        if state.in_flight.len() >= self.budgets.max_in_flight {
            return Err(LodError::BudgetExceeded("in-flight chunk budget exhausted".into()));
        }
        let next_upload = state
            .admitted_upload_bytes
            .checked_add(upload_bytes)
            .ok_or_else(|| LodError::BudgetExceeded("frame upload byte overflow".into()))?;
        if next_upload > self.budgets.max_upload_bytes_per_frame {
            return Err(LodError::BudgetExceeded("per-frame upload byte budget exhausted".into()));
        }
        let reservation = self
            .memory
            .try_reserve(host_bytes)
            .map_err(|error| LodError::Records(error.to_string()))?;
        state.in_flight.insert(node);
        state.admitted_upload_bytes = next_upload;
        Ok(HostChunkLease {
            node,
            point_count,
            upload_bytes,
            generation: state.generation,
            cancellation,
            reservation,
            state: Arc::clone(&self.state),
            released: false,
        })
    }

    /// Records a caller-completed upload and admits it to the GPU cache.
    pub fn complete_upload(
        &self,
        mut lease: HostChunkLease,
        gpu_bytes: u64,
        cache: &mut LodGpuCache,
        protected: &BTreeSet<NodeId>,
    ) -> LodResult<LodUploadReceipt> {
        lease.cancellation.check().map_err(|error| LodError::Records(error.to_string()))?;
        if gpu_bytes == 0 {
            return Err(LodError::BudgetExceeded("resident GPU bytes must be positive".into()));
        }
        let cache_receipt =
            cache.admit(lease.node, lease.point_count, gpu_bytes, lease.generation, protected)?;
        let receipt = LodUploadReceipt {
            node: lease.node,
            generation: lease.generation,
            host_bytes: lease.reservation.bytes(),
            upload_bytes: lease.upload_bytes,
            gpu_bytes,
            evicted: cache_receipt.evicted,
            peak_host_bytes: self.memory.snapshot().peak_bytes,
        };
        lease.release();
        Ok(receipt)
    }

    /// Current in-flight IDs.
    #[must_use]
    pub fn in_flight(&self) -> BTreeSet<NodeId> {
        self.state.lock().expect("LOD upload state poisoned").in_flight.clone()
    }

    /// Current and peak host lease bytes.
    #[must_use]
    pub fn host_memory(&self) -> spatialrust_records::MemorySnapshot {
        self.memory.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::{LodBudgets, LodGpuCache, NodeId};

    fn budgets() -> LodBudgets {
        LodBudgets {
            max_points: 100,
            max_host_bytes: 100,
            max_gpu_bytes: 100,
            max_upload_bytes_per_frame: 100,
            max_in_flight: 2,
        }
    }

    #[test]
    fn gpu_cache_evicts_deterministic_lru_and_preserves_protected_nodes() {
        let mut cache = LodGpuCache::try_new(budgets()).unwrap();
        cache.admit(NodeId(2), 40, 40, 1, &BTreeSet::new()).unwrap();
        cache.admit(NodeId(1), 40, 40, 1, &BTreeSet::new()).unwrap();
        cache.touch(NodeId(2), 2).unwrap();
        let receipt = cache.admit(NodeId(3), 50, 50, 3, &BTreeSet::new()).unwrap();
        assert_eq!(receipt.evicted, vec![NodeId(1)]);
        assert_eq!(cache.resident_ids(), BTreeSet::from([NodeId(2), NodeId(3)]));

        let before = cache.resident_ids();
        assert!(cache
            .admit(NodeId(4), 90, 90, 4, &BTreeSet::from([NodeId(2), NodeId(3)]))
            .is_err());
        assert_eq!(cache.resident_ids(), before);
    }

    #[cfg(feature = "records")]
    #[test]
    fn leases_enforce_memory_upload_inflight_cancellation_and_cleanup() {
        let session = super::LodUploadSession::try_new(budgets()).unwrap();
        session.begin_frame(1).unwrap();
        let first = session
            .try_lease(NodeId(1), 20, 60, 60, spatialrust_records::CancellationToken::default())
            .unwrap();
        assert!(session
            .try_lease(NodeId(2), 20, 50, 20, spatialrust_records::CancellationToken::default())
            .is_err());
        assert_eq!(session.host_memory().current_bytes, 60);
        drop(first);
        assert_eq!(session.host_memory().current_bytes, 0);
        assert!(session.in_flight().is_empty());

        session.begin_frame(2).unwrap();
        let cancellation = spatialrust_records::CancellationToken::default();
        let lease = session.try_lease(NodeId(3), 20, 40, 40, cancellation.clone()).unwrap();
        cancellation.cancel();
        let mut cache = LodGpuCache::try_new(budgets()).unwrap();
        assert!(session.complete_upload(lease, 40, &mut cache, &BTreeSet::new()).is_err());
        assert!(cache.resident_ids().is_empty());
        assert!(session.in_flight().is_empty());

        session.begin_frame(3).unwrap();
        let lease = session
            .try_lease(NodeId(4), 20, 40, 40, spatialrust_records::CancellationToken::default())
            .unwrap();
        let receipt = session.complete_upload(lease, 40, &mut cache, &BTreeSet::new()).unwrap();
        assert_eq!(receipt.host_bytes, 40);
        assert_eq!(receipt.upload_bytes, 40);
        assert_eq!(session.host_memory().current_bytes, 0);
        assert_eq!(cache.resident_ids(), BTreeSet::from([NodeId(4)]));
    }
}
