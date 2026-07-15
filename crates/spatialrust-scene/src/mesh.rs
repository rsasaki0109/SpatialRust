//! Triangle mesh storage for reconstructed surfaces.

/// Indexed triangle mesh with interleaved XYZ positions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriangleMesh {
    /// Interleaved XYZ vertex positions.
    pub positions: Vec<f32>,
    /// Triangle indices (groups of three).
    pub indices: Vec<u32>,
}

impl TriangleMesh {
    /// Returns the vertex count.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    /// Returns the triangle count.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Returns whether the mesh has no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}
