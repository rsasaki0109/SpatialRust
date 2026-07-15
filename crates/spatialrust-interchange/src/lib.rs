//! glTF and OpenUSD scene interchange adapters.

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
pub use usd::{MemoryUsdStageAdapter, UsdPrimPath, UsdStageAdapter, UsdStageDescription};
