//! Explicit edge/distributed execution contracts.
//!
//! Partition graphs, watermark backpressure, and named measurable transfers —
//! never implicit host/device copies.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod backpressure;
mod error;
mod graph;
mod transfer;

pub use backpressure::{BackpressurePolicy, BackpressureSignal, BoundedTransferQueue};
pub use error::{DistributeError, DistributeResult};
pub use graph::{ExecutionNode, ExecutionPartition, PartitionGraph};
pub use transfer::{NamedTransfer, TransferDirection, TransferKind, TransferLedger, TransferPlan};
