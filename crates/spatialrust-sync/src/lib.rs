//! Sensor time domains, frame graphs, and deterministic multimodal replay.
//!
//! Enable the `mcap` feature for file-backed episode codecs (`sync-mcap` on the
//! facade). Default builds keep in-memory episode contracts only.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod clock;
mod error;
mod frame_graph;
mod replay;
mod stamped;

#[cfg(feature = "mcap")]
mod mcap_io;

pub use clock::{ClockDomain, ClockId, StampedTime, SyncQuality};
pub use error::{SyncError, SyncResult};
pub use frame_graph::{FrameEdge, FrameGraph};
pub use replay::{DeterministicReplayer, EpisodeIndex, MemoryEpisode, SyncWindow, TopicId};
pub use stamped::StampedRecord;

#[cfg(feature = "mcap")]
pub use mcap_io::{read_memory_episode_mcap, write_memory_episode_mcap};
