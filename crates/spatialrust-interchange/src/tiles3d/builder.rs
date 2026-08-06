//! Deterministic octree 3D Tiles 1.1 tileset builder for point data.
//!
//! The builder splits an input point set into an octree whose nodes become
//! tiles. Each tile carries a `pnts` payload with a per-tile `RTC_CENTER` so
//! `f32` precision is retained far from the coordinate origin. Split order is
//! deterministic (fixed octant bit order) and internal geometric error halves
//! each level, so the same input and options always produce the same tree.

use std::path::Path;

use crate::tiles3d::pnts::{encode_pnts, PntsFeatureTable};
use crate::tiles3d::tileset::{
    serialize_tileset_json, BoundingVolume, Refinement, Tile, TileContent, Tileset,
};
use crate::{InterchangeError, InterchangeResult};

const OCTANTS: usize = 8;
const BITS: [u8; 8] = [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111];

/// Options controlling octree construction.
#[derive(Clone, Debug)]
pub struct TilesetBuilderOptions {
    /// Maximum points materialized per tile before the node splits.
    pub max_points_per_tile: usize,
    /// Maximum octree depth; nodes at this depth never split.
    pub max_depth: u32,
    /// Root geometric error; defaults to the root bounds diagonal when `None`.
    pub root_geometric_error: Option<f64>,
    /// Multiplier applied to the parent geometric error for each child level.
    pub geometric_error_scale: f64,
    /// Refinement policy recorded on the root tile.
    pub refine: Refinement,
}

impl Default for TilesetBuilderOptions {
    fn default() -> Self {
        Self {
            max_points_per_tile: 100_000,
            max_depth: 12,
            root_geometric_error: None,
            geometric_error_scale: 0.5,
            refine: Refinement::Replace,
        }
    }
}

impl TilesetBuilderOptions {
    fn validate(&self) -> InterchangeResult<()> {
        if self.max_points_per_tile == 0 {
            return Err(InterchangeError::InvalidConfiguration(
                "max_points_per_tile must be positive".into(),
            ));
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

/// One generated tile payload paired with its content URI.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltTile {
    /// Content URI relative to `tileset.json`.
    pub uri: String,
    /// Encoded `pnts` payload.
    pub pnts: Vec<u8>,
    /// Points materialized by this tile.
    pub point_count: usize,
}

/// A validated tileset plus its `pnts` payloads.
#[derive(Clone, Debug)]
pub struct BuiltTileset {
    /// Serialized tileset document.
    pub tileset: Tileset,
    /// Tile payloads in deterministic content order.
    pub tiles: Vec<BuiltTile>,
}

/// Builds a deterministic octree tileset from interleaved positions.
///
/// `positions` holds `N*3` interleaved `x,y,z` values; `rgb` optionally holds
/// `N*3` interleaved bytes. Positions must all be finite.
pub fn build_point_tileset(
    positions: &[f32],
    rgb: Option<&[u8]>,
    options: &TilesetBuilderOptions,
) -> InterchangeResult<BuiltTileset> {
    options.validate()?;
    if positions.is_empty() || positions.len() % 3 != 0 {
        return Err(InterchangeError::InvalidConfiguration(
            "point positions must be a non-empty multiple of 3".into(),
        ));
    }
    let point_count = positions.len() / 3;
    if let Some(rgb) = rgb {
        if rgb.len() != point_count * 3 {
            return Err(InterchangeError::InvalidConfiguration(
                "RGB length must equal three times the point count".into(),
            ));
        }
    }
    if positions.iter().any(|value| !value.is_finite()) {
        return Err(InterchangeError::InvalidConfiguration(
            "point positions must contain finite values".into(),
        ));
    }

    let (min, max) = compute_bounds(positions);
    let mut indices: Vec<u32> = (0..point_count).map(|index| index as u32).collect();
    let root_error = options.root_geometric_error.unwrap_or_else(|| bounds_diagonal(min, max));
    let node = build_node(&mut indices, min, max, root_error, options, positions)?;

    let mut tiles = Vec::new();
    let mut tileset_root = materialize(node, positions, rgb, &mut tiles)?;
    tileset_root.geometric_error = root_error;
    tileset_root.refine = Some(options.refine);

    Ok(BuiltTileset { tileset: Tileset { geometric_error: root_error, root: tileset_root }, tiles })
}

/// Writes a built tileset to `dir` as `tileset.json` plus one `.pnts` file per tile.
pub fn write_point_tileset(
    dir: impl AsRef<Path>,
    built: &BuiltTileset,
) -> InterchangeResult<TilesetWriteReceipt> {
    let dir = dir.as_ref();
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

/// Receipt for a written 3D Tiles tileset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilesetWriteReceipt {
    /// Bytes written for `tileset.json`.
    pub tileset_json_bytes: u64,
    /// Number of `pnts` tile files written.
    pub tile_count: u64,
    /// Total points across all tiles.
    pub point_count: u64,
    /// Total bytes across all `pnts` payloads.
    pub pnts_bytes: u64,
}

struct Node {
    min: [f64; 3],
    max: [f64; 3],
    geometric_error: f64,
    points: Vec<u32>,
    children: Vec<Node>,
}

fn build_node(
    indices: &mut Vec<u32>,
    min: [f64; 3],
    max: [f64; 3],
    geometric_error: f64,
    options: &TilesetBuilderOptions,
    positions: &[f32],
) -> InterchangeResult<Node> {
    let leaf = indices.len() <= options.max_points_per_tile;
    if leaf || indices.is_empty() {
        return Ok(Node {
            min,
            max,
            geometric_error: 0.0,
            points: std::mem::take(indices),
            children: Vec::new(),
        });
    }

    let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];

    let mut buckets: [Vec<u32>; OCTANTS] = std::array::from_fn(|_| Vec::new());
    for &point in indices.iter() {
        let index = point as usize;
        let x = f64::from(positions[index * 3]);
        let y = f64::from(positions[index * 3 + 1]);
        let z = f64::from(positions[index * 3 + 2]);
        let mut octant = 0u8;
        if x >= center[0] {
            octant |= 0b100;
        }
        if y >= center[1] {
            octant |= 0b010;
        }
        if z >= center[2] {
            octant |= 0b001;
        }
        buckets[octant as usize].push(point);
    }
    indices.clear();

    let child_error = geometric_error * options.geometric_error_scale;
    let mut children = Vec::new();
    for (octant_index, bucket) in buckets.iter_mut().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let bits = BITS[octant_index];
        let child_min = [
            if bits & 0b100 != 0 { center[0] } else { min[0] },
            if bits & 0b010 != 0 { center[1] } else { min[1] },
            if bits & 0b001 != 0 { center[2] } else { min[2] },
        ];
        let child_max = [
            if bits & 0b100 != 0 { max[0] } else { center[0] },
            if bits & 0b010 != 0 { max[1] } else { center[1] },
            if bits & 0b001 != 0 { max[2] } else { center[2] },
        ];
        let child = build_node(bucket, child_min, child_max, child_error, options, positions)?;
        if !child.points.is_empty() || !child.children.is_empty() {
            children.push(child);
        }
    }

    Ok(Node { min, max, geometric_error, points: Vec::new(), children })
}

fn materialize(
    node: Node,
    positions: &[f32],
    rgb: Option<&[u8]>,
    tiles: &mut Vec<BuiltTile>,
) -> InterchangeResult<Tile> {
    let center = [
        (node.min[0] + node.max[0]) * 0.5,
        (node.min[1] + node.max[1]) * 0.5,
        (node.min[2] + node.max[2]) * 0.5,
    ];

    let mut content = None;
    if !node.points.is_empty() {
        let mut local_positions = Vec::with_capacity(node.points.len() * 3);
        for &point in &node.points {
            let index = point as usize;
            local_positions.push(positions[index * 3] - center[0] as f32);
            local_positions.push(positions[index * 3 + 1] - center[1] as f32);
            local_positions.push(positions[index * 3 + 2] - center[2] as f32);
        }
        let tile_rgb = rgb.map(|rgb| {
            node.points
                .iter()
                .flat_map(|&point| {
                    let index = point as usize * 3;
                    rgb[index..index + 3].to_vec()
                })
                .collect::<Vec<u8>>()
        });
        let table = PntsFeatureTable {
            positions: local_positions,
            rgb: tile_rgb,
            rtc_center: Some(center),
        };
        let pnts = encode_pnts(&table)?;
        let uri = format!("{}.pnts", tiles.len());
        let point_count = table.point_count();
        tiles.push(BuiltTile { uri: uri.clone(), pnts, point_count });
        content = Some(TileContent { uri });
    }

    let mut children = Vec::new();
    for child in node.children {
        children.push(materialize(child, positions, rgb, tiles)?);
    }

    let bounding_volume = BoundingVolume::box_from_bounds(node.min, node.max)?;
    Ok(Tile {
        bounding_volume,
        geometric_error: node.geometric_error,
        refine: None,
        content,
        children,
    })
}

fn compute_bounds(positions: &[f32]) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for (index, value) in positions.iter().enumerate() {
        let axis = index % 3;
        min[axis] = min[axis].min(f64::from(*value));
        max[axis] = max[axis].max(f64::from(*value));
    }
    for axis in 0..3 {
        if !(max[axis] - min[axis]).is_normal() {
            min[axis] -= 1.0e-6;
            max[axis] += 1.0e-6;
        }
    }
    (min, max)
}

fn bounds_diagonal(min: [f64; 3], max: [f64; 3]) -> f64 {
    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn io_error(error: std::io::Error) -> InterchangeError {
    InterchangeError::InvalidConfiguration(format!("tileset IO failure: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{build_point_tileset, write_point_tileset, TilesetBuilderOptions};
    use crate::tiles3d::pnts::decode_pnts;

    fn grid_points(size: usize, step: f32) -> Vec<f32> {
        let mut out = Vec::new();
        for x in 0..size {
            for y in 0..size {
                for z in 0..size {
                    out.push(x as f32 * step);
                    out.push(y as f32 * step);
                    out.push(z as f32 * step);
                }
            }
        }
        out
    }

    #[test]
    fn splits_into_multiple_tiles() {
        let positions = grid_points(16, 1.0);
        let built = build_point_tileset(
            &positions,
            None,
            &TilesetBuilderOptions { max_points_per_tile: 64, ..Default::default() },
        )
        .unwrap();
        assert!(built.tiles.len() > 1);
        let total: usize = built.tiles.iter().map(|tile| tile.point_count).sum();
        assert_eq!(total, positions.len() / 3);
        for tile in &built.tiles {
            let decoded = decode_pnts(&tile.pnts).unwrap();
            assert_eq!(decoded.point_count(), tile.point_count);
            assert!(decoded.rtc_center.is_some());
        }
    }

    #[test]
    fn single_tile_within_budget() {
        let positions = grid_points(2, 1.0);
        let built =
            build_point_tileset(&positions, None, &TilesetBuilderOptions::default()).unwrap();
        assert_eq!(built.tiles.len(), 1);
        assert_eq!(built.tiles[0].point_count, 8);
    }

    #[test]
    fn preserves_rgb_per_tile() {
        let positions = grid_points(8, 1.0);
        let rgb: Vec<u8> = (0..positions.len()).map(|index| (index % 251) as u8).collect();
        let built = build_point_tileset(
            &positions,
            Some(&rgb),
            &TilesetBuilderOptions { max_points_per_tile: 16, ..Default::default() },
        )
        .unwrap();
        let mut rgb_points = 0usize;
        for tile in &built.tiles {
            let decoded = decode_pnts(&tile.pnts).unwrap();
            assert_eq!(decoded.rgb.as_ref().unwrap().len(), decoded.point_count() * 3);
            rgb_points += decoded.point_count();
        }
        assert_eq!(rgb_points, positions.len() / 3);
    }

    #[test]
    fn deterministic_output() {
        let positions = grid_points(12, 0.5);
        let options = TilesetBuilderOptions { max_points_per_tile: 32, ..Default::default() };
        let first = build_point_tileset(&positions, None, &options).unwrap();
        let second = build_point_tileset(&positions, None, &options).unwrap();
        assert_eq!(first.tiles, second.tiles);
        assert_eq!(first.tileset, second.tileset);
    }

    #[test]
    fn rejects_invalid_input() {
        assert!(build_point_tileset(&[0.0, 0.0], None, &TilesetBuilderOptions::default()).is_err());
        assert!(build_point_tileset(
            &[f32::NAN, 0.0, 0.0],
            None,
            &TilesetBuilderOptions::default()
        )
        .is_err());
        let options = TilesetBuilderOptions { max_points_per_tile: 0, ..Default::default() };
        assert!(build_point_tileset(&[0.0, 0.0, 0.0], None, &options).is_err());
    }

    #[test]
    fn writes_files_and_receipt() {
        let positions = grid_points(6, 1.0);
        let built =
            build_point_tileset(&positions, None, &TilesetBuilderOptions::default()).unwrap();
        let dir = std::env::temp_dir().join(format!("spatialrust-tiles3d-{}", std::process::id()));
        let receipt = write_point_tileset(&dir, &built).unwrap();
        assert_eq!(receipt.tile_count as usize, built.tiles.len());
        assert_eq!(receipt.point_count, (positions.len() / 3) as u64);
        assert!(dir.join("tileset.json").exists());
        assert!(dir.join(&built.tiles[0].uri).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
