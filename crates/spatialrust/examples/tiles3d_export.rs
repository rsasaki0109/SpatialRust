//! 3D Tiles 1.1 point-cloud tileset export example.
//!
//! Reads a PCD file, downsamples with the voxel filter, and writes a
//! `tileset.json` plus one `.pnts` tile per octree node to the output
//! directory. The resulting tileset can be streamed by any 3D Tiles 1.1
//! runtime.
//!
//! Run with:
//! `cargo run -p spatialrust --example tiles3d_export --features "io-pcd filter-voxel interchange-tiles3d" -- <input.pcd> <output-dir>`

use std::path::PathBuf;

use spatialrust::filtering::{PointCloudFilter, VoxelGridDownsample, VoxelGridDownsampleConfig};
use spatialrust::interchange::{build_point_tileset, write_point_tileset, TilesetBuilderOptions};
use spatialrust::{FieldSemantic, HasPositions3};

fn main() {
    let mut args = std::env::args().skip(1);
    let input =
        args.next().unwrap_or_else(|| panic!("usage: tiles3d_export <input.pcd> <output-dir>"));
    let output =
        args.next().unwrap_or_else(|| panic!("usage: tiles3d_export <input.pcd> <output-dir>"));
    let input = PathBuf::from(input);
    let output = PathBuf::from(output);

    let mut cloud =
        spatialrust::io::read_point_cloud_file(&input).expect("failed to read input point cloud");
    cloud.validate().expect("input cloud is invalid");

    let config = VoxelGridDownsampleConfig::centroid(0.05);
    let voxel = VoxelGridDownsample::new(config);
    cloud = voxel.filter(&cloud).expect("voxel downsample failed");

    let (x, y, z) = cloud.positions3().expect("cloud must have positions");
    let mut positions = Vec::with_capacity(cloud.len() * 3);
    for index in 0..cloud.len() {
        positions.push(x[index]);
        positions.push(y[index]);
        positions.push(z[index]);
    }

    let rgb = cloud
        .schema()
        .fields()
        .iter()
        .position(|field| field.semantic == FieldSemantic::ColorR)
        .and_then(|_| {
            let (r, g, b) =
                (cloud.field("r").ok()?, cloud.field("g").ok()?, cloud.field("b").ok()?);
            let r = extract_u8(r)?;
            let g = extract_u8(g)?;
            let b = extract_u8(b)?;
            let mut out = Vec::with_capacity(r.len() * 3);
            for index in 0..r.len() {
                out.push(r[index]);
                out.push(g[index]);
                out.push(b[index]);
            }
            Some(out)
        });

    let built = build_point_tileset(
        &positions,
        rgb.as_deref(),
        &TilesetBuilderOptions { max_points_per_tile: 100_000, ..Default::default() },
    )
    .expect("tileset build failed");

    let receipt = write_point_tileset(&output, &built).expect("tileset write failed");
    let point_total: usize = built.tiles.iter().map(|tile| tile.point_count).sum();
    println!(
        "wrote {} tiles / {} points to {} (tileset.json {} B, pnts {} B)",
        receipt.tile_count,
        point_total,
        output.display(),
        receipt.tileset_json_bytes,
        receipt.pnts_bytes,
    );
}

fn extract_u8(buffer: &spatialrust::core::PointBuffer) -> Option<&[u8]> {
    match buffer {
        spatialrust::core::PointBuffer::U8(values) => Some(values),
        _ => None,
    }
}
