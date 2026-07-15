//! Explicit edge/distributed execution contracts.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod backpressure;
mod error;
mod graph;
mod transfer;

pub use backpressure::{BackpressurePolicy, BackpressureSignal};
pub use error::{DistributeError, DistributeResult};
pub use graph::{ExecutionNode, ExecutionPartition, PartitionGraph};
pub use transfer::{NamedTransfer, TransferDirection, TransferKind};
