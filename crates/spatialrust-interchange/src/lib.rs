//! glTF, OpenUSD, and OGC 3D Tiles scene interchange adapters.
//!
//! `openusd` provides in-memory stages plus USDA ASCII mesh export/import.
//! `tiles3d` provides deterministic OGC 3D Tiles 1.1 point-cloud tilesets
//! (`tileset.json` + `pnts` payloads). Native libusd bindings remain optional
//! and outside the default tree.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
#[cfg(feature = "tiles3d")]
mod json;

#[cfg(feature = "gltf")]
mod gltf;
#[cfg(feature = "tiles3d")]
mod tiles3d;
#[cfg(feature = "openusd")]
mod usd;

pub use error::{InterchangeError, InterchangeResult};

#[cfg(feature = "gltf")]
pub use gltf::{
    decode_triangle_mesh_gltf_json, export_triangle_mesh_gltf_json, import_triangle_mesh_gltf_json,
};
#[cfg(feature = "tiles3d")]
pub use tiles3d::{
    build_point_tileset, decode_pnts, encode_pnts, parse_tileset_json, serialize_tileset_json,
    write_point_tileset, BoundingVolume, BuiltTile, BuiltTileset, PntsFeatureTable, Refinement,
    Tile, TileContent, Tileset, TilesetBuilderOptions, TilesetWriteReceipt,
};
#[cfg(feature = "tiles3d-copc")]
pub use tiles3d::{export_copc_tileset, CopcTilesetOptions};
#[cfg(feature = "openusd")]
pub use usd::{
    export_stage_usda, import_mesh_from_usda, MemoryUsdStageAdapter, UsdPrimPath, UsdStageAdapter,
    UsdStageDescription,
};
