//! OpenUSD adapter contracts and USDA ASCII mesh interchange (no libusd).
//!
//! Native OpenUSD/Hydra bindings remain install-time optional. Enabling
//! `openusd` provides stage adapters plus a portable `.usda` ASCII codec for
//! triangle meshes that other USD tools can open.

use spatialrust_scene::TriangleMesh;

use crate::{InterchangeError, InterchangeResult};

/// Hierarchical USD prim path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UsdPrimPath(pub String);

impl UsdPrimPath {
    /// Creates a validated absolute prim path.
    pub fn try_new(path: impl Into<String>) -> InterchangeResult<Self> {
        let path = path.into();
        if !path.starts_with('/') || path.len() < 2 {
            return Err(InterchangeError::InvalidConfiguration(
                "USD prim path must be absolute like /World/Mesh".into(),
            ));
        }
        Ok(Self(path))
    }

    /// Returns the leaf prim name (`/World/Mesh` → `Mesh`).
    #[must_use]
    pub fn leaf_name(&self) -> &str {
        self.0.rsplit('/').next().unwrap_or(self.0.as_str())
    }
}

/// Host-side USD stage description used before optional bindings land.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UsdStageDescription {
    /// Root layer identifier.
    pub root_layer: String,
    /// Declared mesh prim paths.
    pub mesh_prims: Vec<UsdPrimPath>,
}

/// Adapter interface for composing / exporting OpenUSD stages.
pub trait UsdStageAdapter {
    /// Declares a mesh prim for a triangle mesh.
    fn declare_mesh(&mut self, path: UsdPrimPath, mesh: &TriangleMesh) -> InterchangeResult<()>;

    /// Returns the stage description.
    fn description(&self) -> &UsdStageDescription;
}

/// In-memory USD stage adapter (no OpenUSD native dependency).
#[derive(Clone, Debug, Default)]
pub struct MemoryUsdStageAdapter {
    description: UsdStageDescription,
    meshes: Vec<(UsdPrimPath, TriangleMesh)>,
}

impl MemoryUsdStageAdapter {
    /// Creates an adapter with a root layer id.
    #[must_use]
    pub fn new(root_layer: impl Into<String>) -> Self {
        Self {
            description: UsdStageDescription {
                root_layer: root_layer.into(),
                mesh_prims: Vec::new(),
            },
            meshes: Vec::new(),
        }
    }

    /// Returns declared meshes.
    #[must_use]
    pub fn meshes(&self) -> &[(UsdPrimPath, TriangleMesh)] {
        &self.meshes
    }

    /// Exports the stage as USDA ASCII text.
    pub fn export_usda(&self) -> InterchangeResult<String> {
        export_stage_usda(self)
    }
}

impl UsdStageAdapter for MemoryUsdStageAdapter {
    fn declare_mesh(&mut self, path: UsdPrimPath, mesh: &TriangleMesh) -> InterchangeResult<()> {
        if mesh.is_empty() {
            return Err(InterchangeError::InvalidConfiguration(
                "cannot declare an empty mesh prim".into(),
            ));
        }
        if mesh.positions.len() % 3 != 0 || mesh.indices.len() % 3 != 0 {
            return Err(InterchangeError::InvalidConfiguration(
                "mesh positions/indices must be multiples of 3".into(),
            ));
        }
        self.description.mesh_prims.push(path.clone());
        self.meshes.push((path, mesh.clone()));
        Ok(())
    }

    fn description(&self) -> &UsdStageDescription {
        &self.description
    }
}

/// Exports all meshes from a memory stage as a single USDA ASCII document.
pub fn export_stage_usda(stage: &MemoryUsdStageAdapter) -> InterchangeResult<String> {
    if stage.meshes.is_empty() {
        return Err(InterchangeError::InvalidConfiguration(
            "stage has no mesh prims to export".into(),
        ));
    }
    let mut out =
        String::from("#usda 1.0\n(\n    defaultPrim = \"World\"\n)\n\ndef Xform \"World\"\n{\n");
    for (path, mesh) in &stage.meshes {
        out.push_str(&format!("    def Mesh \"{}\"\n    {{\n", path.leaf_name()));
        out.push_str("        point3f[] points = [");
        for (i, chunk) in mesh.positions.chunks_exact(3).enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("({}, {}, {})", chunk[0], chunk[1], chunk[2]));
        }
        out.push_str("]\n");
        let tri_count = mesh.indices.len() / 3;
        out.push_str("        int[] faceVertexCounts = [");
        for i in 0..tri_count {
            if i > 0 {
                out.push_str(", ");
            }
            out.push('3');
        }
        out.push_str("]\n");
        out.push_str("        int[] faceVertexIndices = [");
        for (i, idx) in mesh.indices.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&idx.to_string());
        }
        out.push_str("]\n");
        out.push_str(&format!("        custom string spatialrust:primPath = \"{}\"\n", path.0));
        out.push_str("    }\n");
    }
    out.push_str("}\n");
    Ok(out)
}

/// Imports the first SpatialRust-authored Mesh prim from USDA ASCII.
pub fn import_mesh_from_usda(usda: &str) -> InterchangeResult<(UsdPrimPath, TriangleMesh)> {
    if !usda.contains("#usda") {
        return Err(InterchangeError::InvalidConfiguration("missing USDA header".into()));
    }
    let path = extract_quoted_after(usda, "spatialrust:primPath = ").or_else(|_| {
        let leaf = extract_mesh_leaf(usda)?;
        UsdPrimPath::try_new(format!("/World/{leaf}"))
    })?;
    let points_blob = extract_bracket_list(usda, "point3f[] points = ")?;
    let indices_blob = extract_bracket_list(usda, "int[] faceVertexIndices = ")?;
    let mut positions = Vec::new();
    for tok in points_blob.split(['(', ')', ',', ' ']).filter(|t| !t.is_empty()) {
        let v: f32 = tok.parse().map_err(|_| {
            InterchangeError::InvalidConfiguration(format!("bad point component `{tok}`"))
        })?;
        positions.push(v);
    }
    if positions.len() % 3 != 0 {
        return Err(InterchangeError::InvalidConfiguration(
            "points length must be a multiple of 3".into(),
        ));
    }
    let mut indices = Vec::new();
    for tok in indices_blob.split([',', ' ']).filter(|t| !t.is_empty()) {
        let v: u32 = tok.parse().map_err(|_| {
            InterchangeError::InvalidConfiguration(format!("bad face index `{tok}`"))
        })?;
        indices.push(v);
    }
    if indices.len() % 3 != 0 {
        return Err(InterchangeError::InvalidConfiguration(
            "faceVertexIndices length must be a multiple of 3".into(),
        ));
    }
    Ok((path, TriangleMesh { positions, indices }))
}

fn extract_mesh_leaf(usda: &str) -> InterchangeResult<String> {
    let key = "def Mesh \"";
    let start = usda
        .find(key)
        .ok_or_else(|| InterchangeError::InvalidConfiguration("missing Mesh prim".into()))?
        + key.len();
    let end = usda[start..]
        .find('"')
        .ok_or_else(|| InterchangeError::InvalidConfiguration("unterminated Mesh name".into()))?
        + start;
    Ok(usda[start..end].to_string())
}

fn extract_quoted_after(usda: &str, marker: &str) -> InterchangeResult<UsdPrimPath> {
    let start = usda
        .find(marker)
        .ok_or_else(|| InterchangeError::InvalidConfiguration(format!("missing {marker}")))?
        + marker.len();
    let rest = usda[start..].trim_start();
    if !rest.starts_with('"') {
        return Err(InterchangeError::InvalidConfiguration("expected quoted string".into()));
    }
    let end = rest[1..]
        .find('"')
        .ok_or_else(|| InterchangeError::InvalidConfiguration("unterminated string".into()))?
        + 1;
    UsdPrimPath::try_new(rest[1..end].to_string())
}

fn extract_bracket_list(usda: &str, marker: &str) -> InterchangeResult<String> {
    let start = usda
        .find(marker)
        .ok_or_else(|| InterchangeError::InvalidConfiguration(format!("missing {marker}")))?
        + marker.len();
    let rest = &usda[start..];
    let open = rest
        .find('[')
        .ok_or_else(|| InterchangeError::InvalidConfiguration("missing '['".into()))?;
    let close = rest[open..]
        .find(']')
        .ok_or_else(|| InterchangeError::InvalidConfiguration("missing ']'".into()))?
        + open;
    Ok(rest[open + 1..close].to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        export_stage_usda, import_mesh_from_usda, MemoryUsdStageAdapter, UsdPrimPath,
        UsdStageAdapter,
    };
    use spatialrust_scene::TriangleMesh;

    #[test]
    fn usda_roundtrip_preserves_mesh() {
        let mut stage = MemoryUsdStageAdapter::new("scene.usda");
        let mesh = TriangleMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            indices: vec![0, 1, 2],
        };
        stage.declare_mesh(UsdPrimPath::try_new("/World/Mesh").unwrap(), &mesh).unwrap();
        let usda = export_stage_usda(&stage).unwrap();
        assert!(usda.starts_with("#usda 1.0"));
        let (path, imported) = import_mesh_from_usda(&usda).unwrap();
        assert_eq!(path.0, "/World/Mesh");
        assert_eq!(imported.positions, mesh.positions);
        assert_eq!(imported.indices, mesh.indices);
    }
}
