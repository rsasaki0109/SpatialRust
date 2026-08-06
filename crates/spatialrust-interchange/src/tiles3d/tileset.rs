//! OGC 3D Tiles 1.1 `tileset.json` model, validation, and codec.

use crate::json::{parse_json, serialize_json, Json};
use crate::{InterchangeError, InterchangeResult};

/// Supported refinement policy for a tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refinement {
    /// Child tiles are rendered in addition to the parent.
    Add,
    /// Child tiles replace the parent when refined.
    Replace,
}

impl Refinement {
    fn as_str(self) -> &'static str {
        match self {
            Refinement::Add => "ADD",
            Refinement::Replace => "REPLACE",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "ADD" => Some(Refinement::Add),
            "REPLACE" => Some(Refinement::Replace),
            _ => None,
        }
    }
}

/// Bounding volume for a tile.
#[derive(Clone, Debug, PartialEq)]
pub enum BoundingVolume {
    /// Right-handed axis-aligned box: `[center(3), x-half-axis(3), y-half-axis(3), z-half-axis(3)]`.
    Box([f64; 12]),
}

impl BoundingVolume {
    /// Creates an axis-aligned box from ordered min/max corners.
    pub fn box_from_bounds(min: [f64; 3], max: [f64; 3]) -> InterchangeResult<Self> {
        for axis in 0..3 {
            if !min[axis].is_finite() || !max[axis].is_finite() || min[axis] > max[axis] {
                return Err(InterchangeError::InvalidConfiguration(
                    "tile box bounds must be finite and ordered".into(),
                ));
            }
        }
        let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
        let half = [(max[0] - min[0]) * 0.5, (max[1] - min[1]) * 0.5, (max[2] - min[2]) * 0.5];
        let mut box_value = [0.0f64; 12];
        box_value[0..3].copy_from_slice(&center);
        box_value[3] = half[0];
        box_value[7] = half[1];
        box_value[11] = half[2];
        Ok(BoundingVolume::Box(box_value))
    }

    fn to_json(&self) -> Json {
        match self {
            BoundingVolume::Box(value) => Json::object(vec![(
                "box",
                Json::Array(value.iter().map(|v| Json::Number(v.to_string())).collect()),
            )]),
        }
    }

    fn from_json(json: &Json) -> InterchangeResult<Self> {
        let box_values = json
            .get("box")
            .and_then(Json::as_array)
            .ok_or_else(|| InterchangeError::InvalidConfiguration("missing tile box".into()))?;
        if box_values.len() != 12 {
            return Err(InterchangeError::InvalidConfiguration(
                "tile box must contain twelve values".into(),
            ));
        }
        let mut value = [0.0f64; 12];
        for (index, item) in box_values.iter().enumerate() {
            value[index] = item.as_f64().ok_or_else(|| {
                InterchangeError::InvalidConfiguration("tile box value is not numeric".into())
            })?;
            if !value[index].is_finite() {
                return Err(InterchangeError::InvalidConfiguration(
                    "tile box must contain finite values".into(),
                ));
            }
        }
        Ok(BoundingVolume::Box(value))
    }
}

/// Content reference for a tile.
#[derive(Clone, Debug, PartialEq)]
pub struct TileContent {
    /// Content URI relative to the tileset document.
    pub uri: String,
}

/// One tile in a 3D Tiles hierarchy.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    /// Bounding volume for this tile.
    pub bounding_volume: BoundingVolume,
    /// Geometric error in world units; leaves should be `0.0`.
    pub geometric_error: f64,
    /// Optional refinement policy; absent inherits the parent policy.
    pub refine: Option<Refinement>,
    /// Optional content reference.
    pub content: Option<TileContent>,
    /// Child tiles in deterministic order.
    pub children: Vec<Tile>,
}

impl Tile {
    fn validate(&self) -> InterchangeResult<()> {
        if !self.geometric_error.is_finite() || self.geometric_error < 0.0 {
            return Err(InterchangeError::InvalidConfiguration(
                "tile geometric error must be finite and non-negative".into(),
            ));
        }
        let child_ids: Vec<_> = self.children.iter().map(|child| child.hash_key()).collect();
        for index in 0..child_ids.len() {
            if child_ids.iter().skip(index + 1).any(|id| *id == child_ids[index]) {
                return Err(InterchangeError::InvalidConfiguration(
                    "tileset contains duplicate child tiles".into(),
                ));
            }
        }
        for child in &self.children {
            child.validate()?;
        }
        Ok(())
    }

    fn hash_key(&self) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        serialize_json(&self.bounding_volume.to_json()).hash(&mut hasher);
        hasher.finish().to_string()
    }

    fn to_json(&self) -> Json {
        let mut members = vec![
            ("boundingVolume", self.bounding_volume.to_json()),
            ("geometricError", Json::Number(self.geometric_error.to_string())),
        ];
        if let Some(refine) = self.refine {
            members.push(("refine", Json::String(refine.as_str().to_owned())));
        }
        if let Some(content) = &self.content {
            members
                .push(("content", Json::object(vec![("uri", Json::String(content.uri.clone()))])));
        }
        if !self.children.is_empty() {
            members
                .push(("children", Json::Array(self.children.iter().map(Tile::to_json).collect())));
        }
        Json::object(members)
    }

    fn from_json(json: &Json, depth: usize) -> InterchangeResult<Self> {
        if depth > 64 {
            return Err(InterchangeError::InvalidConfiguration(
                "tileset nesting exceeds 64 levels".into(),
            ));
        }
        let bounding_volume =
            json.get("boundingVolume").map(BoundingVolume::from_json).transpose()?.ok_or_else(
                || InterchangeError::InvalidConfiguration("tile missing boundingVolume".into()),
            )?;
        let geometric_error =
            json.get("geometricError").and_then(Json::as_f64).ok_or_else(|| {
                InterchangeError::InvalidConfiguration("missing geometricError".into())
            })?;
        let refine = json.get("refine").and_then(Json::as_str).and_then(Refinement::from_str);
        let content = match json.get("content") {
            None => None,
            Some(content) => Some(TileContent {
                uri: content
                    .get("uri")
                    .and_then(Json::as_str)
                    .ok_or_else(|| {
                        InterchangeError::InvalidConfiguration("content missing uri".into())
                    })?
                    .to_owned(),
            }),
        };
        let children = match json.get("children") {
            None => Vec::new(),
            Some(Json::Array(children)) => children
                .iter()
                .map(|child| Tile::from_json(child, depth + 1))
                .collect::<InterchangeResult<Vec<_>>>()?,
            Some(_) => {
                return Err(InterchangeError::InvalidConfiguration(
                    "tile children must be an array".into(),
                ));
            }
        };
        let tile = Tile { bounding_volume, geometric_error, refine, content, children };
        tile.validate()?;
        Ok(tile)
    }
}

/// Root document of a 3D Tiles tileset.
#[derive(Clone, Debug, PartialEq)]
pub struct Tileset {
    /// Tileset-level geometric error.
    pub geometric_error: f64,
    /// Root tile.
    pub root: Tile,
}

impl Tileset {
    /// Serializes the tileset to a 3D Tiles 1.1 `tileset.json` document.
    #[must_use]
    pub fn to_json(&self) -> String {
        let document = Json::object(vec![
            ("asset", Json::object(vec![("version", Json::String("1.1".into()))])),
            ("geometricError", Json::Number(self.geometric_error.to_string())),
            ("root", self.root.to_json()),
        ]);
        serialize_json(&document)
    }
}

/// Serializes a tileset to a 3D Tiles 1.1 `tileset.json` string.
pub fn serialize_tileset_json(tileset: &Tileset) -> InterchangeResult<String> {
    validate_tileset(tileset)?;
    Ok(tileset.to_json())
}

/// Parses and validates a 3D Tiles 1.1 `tileset.json` string.
pub fn parse_tileset_json(document: &str) -> InterchangeResult<Tileset> {
    let json = parse_json(document)?;
    let asset = json
        .get("asset")
        .ok_or_else(|| InterchangeError::InvalidConfiguration("tileset missing asset".into()))?;
    let version = asset.get("version").and_then(Json::as_str).ok_or_else(|| {
        InterchangeError::InvalidConfiguration("tileset asset missing version".into())
    })?;
    if version != "1.1" && version != "1.0" {
        return Err(InterchangeError::InvalidConfiguration(format!(
            "unsupported tileset version {version}"
        )));
    }
    let geometric_error = json.get("geometricError").and_then(Json::as_f64).ok_or_else(|| {
        InterchangeError::InvalidConfiguration("tileset missing geometricError".into())
    })?;
    let root = Tile::from_json(
        json.get("root")
            .ok_or_else(|| InterchangeError::InvalidConfiguration("tileset missing root".into()))?,
        0,
    )?;
    let tileset = Tileset { geometric_error, root };
    validate_tileset(&tileset)?;
    Ok(tileset)
}

fn validate_tileset(tileset: &Tileset) -> InterchangeResult<()> {
    if !tileset.geometric_error.is_finite() || tileset.geometric_error < 0.0 {
        return Err(InterchangeError::InvalidConfiguration(
            "tileset geometric error must be finite and non-negative".into(),
        ));
    }
    tileset.root.validate()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        parse_tileset_json, serialize_tileset_json, BoundingVolume, Refinement, Tile, TileContent,
        Tileset,
    };

    fn sample_tileset() -> Tileset {
        let child = Tile {
            bounding_volume: BoundingVolume::box_from_bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
                .unwrap(),
            geometric_error: 0.0,
            refine: None,
            content: Some(TileContent { uri: "1.pnts".into() }),
            children: Vec::new(),
        };
        Tileset {
            geometric_error: 10.0,
            root: Tile {
                bounding_volume: BoundingVolume::box_from_bounds([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
                    .unwrap(),
                geometric_error: 10.0,
                refine: Some(Refinement::Replace),
                content: Some(TileContent { uri: "0.pnts".into() }),
                children: vec![child],
            },
        }
    }

    #[test]
    fn round_trips_hierarchy() {
        let tileset = sample_tileset();
        let document = serialize_tileset_json(&tileset).unwrap();
        let parsed = parse_tileset_json(&document).unwrap();
        assert_eq!(parsed, tileset);
    }

    #[test]
    fn rejects_negative_geometric_error() {
        let mut tileset = sample_tileset();
        tileset.root.geometric_error = -1.0;
        assert!(serialize_tileset_json(&tileset).is_err());
    }

    #[test]
    fn rejects_unknown_refine() {
        let document = r#"{"asset":{"version":"1.1"},"geometricError":1,"root":{"boundingVolume":{"box":[0,0,0,1,0,0,0,1,0,0,0,1]},"geometricError":0,"refine":"SWAP"}}"#;
        let parsed = parse_tileset_json(document).unwrap();
        assert_eq!(parsed.root.refine, None);
    }

    #[test]
    fn rejects_deep_nesting() {
        let mut document = String::from(
            r#"{"asset":{"version":"1.1"},"geometricError":1,"root":{"boundingVolume":{"box":[0,0,0,1,0,0,0,1,0,0,0,1]},"geometricError":0,"children":["#,
        );
        for _ in 0..80 {
            document.push_str(r#"{"boundingVolume":{"box":[0,0,0,1,0,0,0,1,0,0,0,1]},"geometricError":0,"children":["#);
        }
        for _ in 0..80 {
            document.push_str("]}");
        }
        document.push_str("]}}");
        assert!(parse_tileset_json(&document).is_err());
    }
}
