//! OGC 3D Tiles 1.1 point-cloud tileset export (`tiles3d`).
//!
//! This feature provides a dependency-light `pnts` codec, a validated
//! `tileset.json` model, and a deterministic octree tileset builder. It never
//! depends on `spatialrust-core`, a GPU backend, or serde; data is always
//! caller-owned host memory.

mod builder;
mod pnts;
mod tileset;

pub use builder::{
    build_point_tileset, write_point_tileset, BuiltTile, BuiltTileset, TilesetBuilderOptions,
    TilesetWriteReceipt,
};
pub use pnts::{decode_pnts, encode_pnts, PntsFeatureTable};
pub use tileset::{
    parse_tileset_json, serialize_tileset_json, BoundingVolume, Refinement, Tile, TileContent,
    Tileset,
};
