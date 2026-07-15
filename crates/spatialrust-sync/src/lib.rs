//! Sensor time domains, frame graphs, and deterministic multimodal replay.
//!
//! MCAP file codecs remain behind the `mcap` feature; the default build is
//! in-memory episode contracts only.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod clock;
mod error;
mod frame_graph;
mod replay;
mod stamped;

pub use clock::{ClockDomain, ClockId, StampedTime, SyncQuality};
pub use error::{SyncError, SyncResult};
pub use frame_graph::{FrameEdge, FrameGraph};
pub use replay::{
    DeterministicReplayer, EpisodeIndex, MemoryEpisode, SyncWindow, TopicId,
};
pub use stamped::StampedRecord;
