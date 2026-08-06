//! Bounded COPC → 3D Tiles 1.1 tileset export.
//!
//! Opens a COPC file once, walks its octree hierarchy in deterministic order,
//! and writes one `pnts` tile per COPC node plus a `tileset.json` that mirrors
//! the octree parent/child structure. The cloud is never materialized as a
//! whole: each node is decoded, re-centered to its own `RTC_CENTER`, encoded,
//! and dropped before the next node is processed.

use std::path::Path;

use spatialrust_core::HasPositions3;
use spatialrust_io::CopcNode;

use crate::tiles3d::builder::{BuiltTile, BuiltTileset, TilesetWriteReceipt};
use crate::tiles3d::pnts::{encode_pnts, PntsFeatureTable};
use crate::tiles3d::tileset::{
    serialize_tileset_json, BoundingVolume, Refinement, Tile, TileContent, Tileset,
};
use crate::{InterchangeError, InterchangeResult};

/// Options controlling COPC → 3D Tiles export.
#[derive(Clone, Debug)]
pub struct CopcTilesetOptions {
    /// Maximum octree depth to export; `None` exports the whole hierarchy.
    pub max_level: Option<i32>,
    /// Refinement policy recorded on the root tile.
    pub refine: Refinement,
    /// Root geometric error; defaults to the root bounds diagonal when `None`.
    pub root_geometric_error: Option<f64>,
    /// Multiplier applied to the parent geometric error for each child level.
    pub geometric_error_scale: f64,
}

impl Default for CopcTilesetOptions {
    fn default() -> Self {
        Self {
            max_level: None,
            refine: Refinement::Replace,
            root_geometric_error: None,
            geometric_error_scale: 0.5,
        }
    }
}

impl CopcTilesetOptions {
    fn validate(&self) -> InterchangeResult<()> {
        if let Some(level) = self.max_level {
            if level < 0 {
                return Err(InterchangeError::InvalidConfiguration(
                    "max_level must be non-negative".into(),
                ));
            }
        }
        if let Some(error) = self.root_geometric_error {
            if !error.is_finite() || error < 0.0 {
                return Err(InterchangeError::InvalidConfiguration(
                    "root_geometric_error must be finite and non-negative".into(),
                ));
            }
        }
        if !self.geometric_error_scale.is_finite()
            || !(0.0..=1.0).contains(&self.geometric_error_scale)
        {
            return Err(InterchangeError::InvalidConfiguration(
                "geometric_error_scale must be within [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

/// Exports a COPC file into a 3D Tiles 1.1 tileset without materializing the
/// whole cloud. Returns the byte/point/tile receipt.
pub fn export_copc_tileset(
    copc_path: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &CopcTilesetOptions,
) -> InterchangeResult<TilesetWriteReceipt> {
    options.validate()?;
    let mut reader = spatialrust_io::CopcNodeReader::open(copc_path.as_ref())
        .map_err(|error| InterchangeError::InvalidConfiguration(format!("COPC open: {error}")))?;

    let nodes: Vec<CopcNode> = reader
        .nodes()
        .iter()
        .filter(|node| options.max_level.map_or(true, |level| node.level <= level))
        .copied()
        .collect();
    if nodes.is_empty() {
        return Err(InterchangeError::InvalidConfiguration(
            "COPC file contains no nodes at or above max_level".into(),
        ));
    }

    let root_error = match options.root_geometric_error {
        Some(error) => error,
        None => bounds_diagonal(nodes[0].bounds.min, nodes[0].bounds.max),
    };

    let index_of = |level: i32, x: i32, y: i32, z: i32| -> Option<usize> {
        nodes
            .iter()
            .position(|node| node.level == level && node.x == x && node.y == y && node.z == z)
    };

    let mut tiles: Vec<BuiltTile> = Vec::with_capacity(nodes.len());
    let mut children_by_parent: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        if node.level == 0 {
            continue;
        }
        if let Some(parent) = index_of(node.level - 1, node.x >> 1, node.y >> 1, node.z >> 1) {
            children_by_parent[parent].push(index);
        }
    }
    for children in &mut children_by_parent {
        children.sort_unstable();
    }

    // Recursive materialization in deterministic parent-before-child order.
    let mut root_tile_index = None;
    for (index, node) in nodes.iter().enumerate() {
        if node.level == 0 {
            root_tile_index = Some(index);
            break;
        }
    }
    let root_index = root_tile_index.ok_or_else(|| {
        InterchangeError::InvalidConfiguration("COPC file has no level-0 root node".into())
    })?;

    fn materialize(
        reader: &mut spatialrust_io::CopcNodeReader,
        nodes: &[CopcNode],
        children: &[Vec<usize>],
        tiles: &mut Vec<BuiltTile>,
        index: usize,
        parent_error: f64,
        scale: f64,
    ) -> InterchangeResult<Tile> {
        let node = &nodes[index];
        let bounds = node.bounds;
        let center = [
            (bounds.min[0] + bounds.max[0]) * 0.5,
            (bounds.min[1] + bounds.max[1]) * 0.5,
            (bounds.min[2] + bounds.max[2]) * 0.5,
        ];
        let geometric_error = if node.level == 0 { parent_error } else { parent_error * scale };

        let cloud = reader.read_node(index).map_err(|error| {
            InterchangeError::InvalidConfiguration(format!("COPC node: {error}"))
        })?;
        let (x, y, z) = cloud.positions3().map_err(|error| {
            InterchangeError::InvalidConfiguration(format!("COPC schema: {error}"))
        })?;
        let rgb = extract_rgb(&cloud);
        let mut local_positions = Vec::with_capacity(cloud.len() * 3);
        for point_index in 0..cloud.len() {
            local_positions.push(x[point_index] - center[0] as f32);
            local_positions.push(y[point_index] - center[1] as f32);
            local_positions.push(z[point_index] - center[2] as f32);
        }
        let table = PntsFeatureTable { positions: local_positions, rgb, rtc_center: Some(center) };
        let pnts = encode_pnts(&table)?;
        let uri = format!("{}.pnts", tiles.len());
        let point_count = table.point_count();
        tiles.push(BuiltTile { uri: uri.clone(), pnts, point_count });

        let mut child_tiles = Vec::new();
        for &child in &children[index] {
            child_tiles.push(materialize(
                reader,
                nodes,
                children,
                tiles,
                child,
                geometric_error,
                scale,
            )?);
        }

        Ok(Tile {
            bounding_volume: BoundingVolume::box_from_bounds(bounds.min, bounds.max)?,
            geometric_error,
            refine: None,
            content: Some(TileContent { uri }),
            children: child_tiles,
        })
    }

    let root_tile = materialize(
        &mut reader,
        &nodes,
        &children_by_parent,
        &mut tiles,
        root_index,
        root_error,
        options.geometric_error_scale,
    )?;
    let mut root_tile = root_tile;
    root_tile.refine = Some(options.refine);

    let built =
        BuiltTileset { tileset: Tileset { geometric_error: root_error, root: root_tile }, tiles };
    write_built(out_dir.as_ref(), &built)
}

/// Writes a built tileset to `dir` and returns the receipt.
fn write_built(dir: &Path, built: &BuiltTileset) -> InterchangeResult<TilesetWriteReceipt> {
    std::fs::create_dir_all(dir).map_err(io_error)?;
    let document = serialize_tileset_json(&built.tileset)?;
    std::fs::write(dir.join("tileset.json"), document.as_bytes()).map_err(io_error)?;

    let mut tile_count = 0u64;
    let mut point_count = 0u64;
    let mut pnts_bytes = 0u64;
    for tile in &built.tiles {
        std::fs::write(dir.join(&tile.uri), &tile.pnts).map_err(io_error)?;
        tile_count += 1;
        point_count = point_count
            .checked_add(tile.point_count as u64)
            .ok_or_else(|| InterchangeError::InvalidConfiguration("point count overflow".into()))?;
        pnts_bytes = pnts_bytes.checked_add(tile.pnts.len() as u64).ok_or_else(|| {
            InterchangeError::InvalidConfiguration("pnts byte count overflow".into())
        })?;
    }
    Ok(TilesetWriteReceipt {
        tileset_json_bytes: document.len() as u64,
        tile_count,
        point_count,
        pnts_bytes,
    })
}

fn bounds_diagonal(min: [f64; 3], max: [f64; 3]) -> f64 {
    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Extracts interleaved 8-bit RGB from a cloud's color fields when present.
///
/// LAS/COPC color is stored as `u16` (0–65535); 3D Tiles `pnts` RGB uses
/// `u8` (0–255), so each channel is shifted right by eight bits. Returns
/// `None` when the cloud has no color fields.
fn extract_rgb(cloud: &spatialrust_core::PointCloud) -> Option<Vec<u8>> {
    use spatialrust_core::{FieldSemantic, PointBuffer};

    let field = |semantic: FieldSemantic| {
        let name = match semantic {
            FieldSemantic::ColorR => "red",
            FieldSemantic::ColorG => "green",
            FieldSemantic::ColorB => "blue",
            _ => return None,
        };
        let buffer = cloud.field(name).ok()?;
        match buffer {
            PointBuffer::U16(values) => Some(values.as_slice()),
            _ => None,
        }
    };

    let (r, g, b) = match (
        field(FieldSemantic::ColorR),
        field(FieldSemantic::ColorG),
        field(FieldSemantic::ColorB),
    ) {
        (Some(r), Some(g), Some(b)) => (r, g, b),
        _ => return None,
    };
    if r.len() != cloud.len() || g.len() != cloud.len() || b.len() != cloud.len() {
        return None;
    }
    let mut out = Vec::with_capacity(cloud.len() * 3);
    for index in 0..cloud.len() {
        out.push((r[index] >> 8) as u8);
        out.push((g[index] >> 8) as u8);
        out.push((b[index] >> 8) as u8);
    }
    Some(out)
}

fn io_error(error: std::io::Error) -> InterchangeError {
    InterchangeError::InvalidConfiguration(format!("tileset IO failure: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{export_copc_tileset, CopcTilesetOptions};
    use crate::tiles3d::pnts::decode_pnts;
    use crate::tiles3d::tileset::parse_tileset_json;
    use spatialrust_core::PointCloudBuilder;
    use spatialrust_io::{write_copc_file, write_copc_file_with_params, CopcWriterParams};

    fn dense_grid_cloud(count: usize) -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::xyz();
        for index in 0..count {
            let x = (index % 31) as f32 - 15.0;
            let y = ((index / 31) % 29) as f32 - 14.0;
            let z = ((index / (31 * 29)) % 23) as f32 - 11.0;
            builder.push_point([x, y, z]).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn exports_copc_without_full_materialization() {
        let cloud = dense_grid_cloud(7_000);
        let copc_path = std::env::temp_dir()
            .join(format!("spatialrust_tiles3d_copc_{}.copc.laz", std::process::id()));
        write_copc_file_with_params(
            &copc_path,
            &cloud,
            &CopcWriterParams { max_points_per_node: 96, max_depth: 8 },
        )
        .unwrap();

        let out_dir = std::env::temp_dir()
            .join(format!("spatialrust_tiles3d_copc_out_{}", std::process::id()));
        let receipt =
            export_copc_tileset(&copc_path, &out_dir, &CopcTilesetOptions::default()).unwrap();
        assert_eq!(receipt.point_count, cloud.len() as u64);
        assert!(receipt.tile_count > 1);

        let document = std::fs::read_to_string(out_dir.join("tileset.json")).unwrap();
        let tileset = parse_tileset_json(&document).unwrap();
        assert_eq!(tileset.root.content.as_ref().unwrap().uri, "0.pnts");

        for tile in 0..receipt.tile_count {
            let pnts = std::fs::read(out_dir.join(format!("{tile}.pnts"))).unwrap();
            assert!(decode_pnts(&pnts).is_ok());
        }

        let _ = std::fs::remove_dir_all(&out_dir);
        let _ = std::fs::remove_file(&copc_path);
    }

    #[test]
    fn max_level_bounds_export() {
        let cloud = dense_grid_cloud(7_000);
        let copc_path = std::env::temp_dir()
            .join(format!("spatialrust_tiles3d_copc_lvl_{}.copc.laz", std::process::id()));
        write_copc_file_with_params(
            &copc_path,
            &cloud,
            &CopcWriterParams { max_points_per_node: 96, max_depth: 8 },
        )
        .unwrap();

        let out_dir = std::env::temp_dir()
            .join(format!("spatialrust_tiles3d_copc_lvl_out_{}", std::process::id()));
        let receipt = export_copc_tileset(
            &copc_path,
            &out_dir,
            &CopcTilesetOptions { max_level: Some(0), ..Default::default() },
        )
        .unwrap();
        assert_eq!(receipt.tile_count, 1);
        assert!(
            receipt.point_count < cloud.len() as u64,
            "level-0 export must expose only the root node chunk"
        );

        let _ = std::fs::remove_dir_all(&out_dir);
        let _ = std::fs::remove_file(&copc_path);
    }

    #[test]
    fn preserves_rgb_from_las_color() {
        use spatialrust_core::{PointCloudBuilder, StandardSchemas};

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzrgb());
        for index in 0..2_000usize {
            let x = (index % 31) as f32 - 15.0;
            let y = ((index / 31) % 29) as f32 - 14.0;
            let z = ((index / (31 * 29)) % 23) as f32 - 11.0;
            let r = ((index % 256) << 8) as f32;
            let g = (((index * 7) % 256) << 8) as f32;
            let b = (((index * 13) % 256) << 8) as f32;
            builder.push_point([x, y, z, r, g, b]).unwrap();
        }
        let cloud = builder.build().unwrap();

        let copc_path = std::env::temp_dir()
            .join(format!("spatialrust_tiles3d_copc_rgb_{}.copc.laz", std::process::id()));
        write_copc_file(&copc_path, &cloud).unwrap();

        let out_dir = std::env::temp_dir()
            .join(format!("spatialrust_tiles3d_copc_rgb_out_{}", std::process::id()));
        let receipt =
            export_copc_tileset(&copc_path, &out_dir, &CopcTilesetOptions::default()).unwrap();
        assert_eq!(receipt.point_count, cloud.len() as u64);

        let mut rgb_points = 0usize;
        for tile in 0..receipt.tile_count {
            let pnts = std::fs::read(out_dir.join(format!("{tile}.pnts"))).unwrap();
            let decoded = decode_pnts(&pnts).unwrap();
            let rgb = decoded.rgb.as_ref().expect("color-bearing COPC must write RGB");
            assert_eq!(rgb.len(), decoded.point_count() * 3);
            rgb_points += decoded.point_count();
        }
        assert_eq!(rgb_points, cloud.len());

        let _ = std::fs::remove_dir_all(&out_dir);
        let _ = std::fs::remove_file(&copc_path);
    }
}
