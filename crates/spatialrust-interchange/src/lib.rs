//! glTF and OpenUSD scene interchange adapters.
//!
//! `openusd` provides in-memory stages plus USDA ASCII mesh export/import.
//! Native libusd bindings remain optional and outside the default tree.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;

#[cfg(feature = "gltf")]
mod gltf;
#[cfg(feature = "openusd")]
mod usd;

pub use error::{InterchangeError, InterchangeResult};

#[cfg(feature = "gltf")]
pub use gltf::{export_triangle_mesh_gltf_json, import_triangle_mesh_gltf_json};
#[cfg(feature = "openusd")]
pub use usd::{
    export_stage_usda, import_mesh_from_usda, MemoryUsdStageAdapter, UsdPrimPath, UsdStageAdapter,
    UsdStageDescription,
};
