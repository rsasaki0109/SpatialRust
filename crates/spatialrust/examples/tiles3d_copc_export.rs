//! Bounded COPC → 3D Tiles 1.1 export example.
//!
//! Opens a COPC file, walks its octree hierarchy in deterministic order, and
//! writes one `.pnts` tile per node plus a `tileset.json` without materializing
//! the whole cloud. Any 3D Tiles 1.1 runtime can then stream the result.
//!
//! Run with:
//! `cargo run -p spatialrust --example tiles3d_copc_export --features "io-copc interchange-tiles3d-copc" -- <input.copc.laz> <output-dir> [--max-level N]`

use std::path::PathBuf;

use spatialrust::interchange::{export_copc_tileset, CopcTilesetOptions};

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| {
        panic!("usage: tiles3d_copc_export <input.copc.laz> <output-dir> [--max-level N]")
    });
    let output = args.next().unwrap_or_else(|| {
        panic!("usage: tiles3d_copc_export <input.copc.laz> <output-dir> [--max-level N]")
    });

    let mut max_level = None;
    let mut rest = args;
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--max-level" => {
                max_level = Some(
                    rest.next()
                        .expect("--max-level requires a value")
                        .parse()
                        .expect("max-level must be an integer"),
                );
            }
            other => panic!("unknown argument {other}"),
        }
    }

    let input = PathBuf::from(input);
    let output = PathBuf::from(output);
    let receipt = export_copc_tileset(
        &input,
        &output,
        &CopcTilesetOptions { max_level, ..Default::default() },
    )
    .expect("COPC tileset export failed");

    println!(
        "wrote {} tiles / {} points to {} (tileset.json {} B, pnts {} B)",
        receipt.tile_count,
        receipt.point_count,
        output.display(),
        receipt.tileset_json_bytes,
        receipt.pnts_bytes,
    );
}
