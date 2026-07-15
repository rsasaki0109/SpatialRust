//! OpenUSD adapter contracts without USD libraries in the default dependency tree.

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
}

impl UsdStageAdapter for MemoryUsdStageAdapter {
    fn declare_mesh(&mut self, path: UsdPrimPath, mesh: &TriangleMesh) -> InterchangeResult<()> {
        if mesh.is_empty() {
            return Err(InterchangeError::InvalidConfiguration(
                "cannot declare an empty mesh prim".into(),
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

#[cfg(test)]
mod tests {
    use super::{MemoryUsdStageAdapter, UsdPrimPath, UsdStageAdapter};
    use spatialrust_scene::TriangleMesh;

    #[test]
    fn declares_mesh_prim() {
        let mut stage = MemoryUsdStageAdapter::new("scene.usda");
        let mesh = TriangleMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            indices: vec![0, 1, 2],
        };
        stage
            .declare_mesh(UsdPrimPath::try_new("/World/Mesh").unwrap(), &mesh)
            .unwrap();
        assert_eq!(stage.description().mesh_prims.len(), 1);
    }
}
