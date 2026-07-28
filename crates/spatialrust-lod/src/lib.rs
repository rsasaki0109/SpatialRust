//! Bounded camera-driven level-of-detail planning and explicit residency.
//!
//! Planning is independent of file formats and GPU backends. Optional adapters
//! connect plans to COPC queries and record-memory leases. Upload completion is
//! caller-reported; this crate performs no hidden host/device transfer.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod index;
mod planner;
mod residency;

#[cfg(feature = "copc")]
mod copc;

#[cfg(feature = "copc")]
pub use copc::copc_query_for_nodes;
pub use error::{LodError, LodResult};
pub use index::{LodBounds, LodIndex, LodNode, NodeId};
pub use planner::{LodPlan, LodPlanner, LodPlannerOptions};
pub use residency::{GpuCacheReceipt, LodBudgets, LodGpuCache, ResidentNode};
#[cfg(feature = "records")]
pub use residency::{HostChunkLease, LodUploadReceipt, LodUploadSession};
