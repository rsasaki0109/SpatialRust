use spatialrust_viz::{DeviceIdentity, VisualResidency};

use crate::{runtime::BufferSlot, RenderResult, WgpuRenderer};

/// Device-resident geometry topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuGeometryKind {
    /// Independent points.
    Points,
    /// Independent line segments.
    Lines,
    /// Indexed triangles.
    Triangles,
}

/// Explicitly uploaded, device-resident visual geometry.
///
/// Dropping this value releases its buffers. Call [`Self::recycle`] to return
/// them to the originating renderer's bounded reuse pool.
pub struct GpuGeometry {
    pub(crate) renderer_id: u64,
    pub(crate) kind: GpuGeometryKind,
    pub(crate) vertex_count: u32,
    pub(crate) index_count: u32,
    pub(crate) positions: Option<BufferSlot>,
    pub(crate) rgb: Option<BufferSlot>,
    pub(crate) scalar: Option<BufferSlot>,
    pub(crate) indices: Option<BufferSlot>,
    residency: VisualResidency,
}

impl GpuGeometry {
    /// Geometry topology.
    #[must_use]
    pub const fn kind(&self) -> GpuGeometryKind {
        self.kind
    }

    /// Number of uploaded vertices.
    #[must_use]
    pub const fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Number of uploaded indices.
    #[must_use]
    pub const fn index_count(&self) -> u32 {
        self.index_count
    }

    /// Residency identifying the exact renderer runtime.
    #[must_use]
    pub const fn residency(&self) -> &VisualResidency {
        &self.residency
    }

    /// Logical position-buffer bytes uploaded for this geometry.
    #[must_use]
    pub fn position_bytes(&self) -> u64 {
        self.positions.as_ref().map_or(0, |slot| slot.logical_bytes)
    }

    /// Logical RGB-buffer bytes uploaded for this geometry.
    #[must_use]
    pub fn rgb_bytes(&self) -> u64 {
        self.rgb.as_ref().map_or(0, |slot| slot.logical_bytes)
    }

    /// Logical scalar-buffer bytes uploaded for this geometry.
    #[must_use]
    pub fn scalar_bytes(&self) -> u64 {
        self.scalar.as_ref().map_or(0, |slot| slot.logical_bytes)
    }

    /// Logical index-buffer bytes uploaded for this geometry.
    #[must_use]
    pub fn index_bytes(&self) -> u64 {
        self.indices.as_ref().map_or(0, |slot| slot.logical_bytes)
    }

    /// Returns all buffers to the originating renderer's bounded reuse pool.
    pub fn recycle(mut self, renderer: &WgpuRenderer) -> RenderResult<()> {
        renderer.recycle_geometry(&mut self)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        renderer_id: u64,
        kind: GpuGeometryKind,
        vertex_count: u32,
        index_count: u32,
        positions: Option<BufferSlot>,
        rgb: Option<BufferSlot>,
        scalar: Option<BufferSlot>,
        indices: Option<BufferSlot>,
        device: DeviceIdentity,
    ) -> Self {
        Self {
            renderer_id,
            kind,
            vertex_count,
            index_count,
            positions,
            rgb,
            scalar,
            indices,
            residency: VisualResidency::Device(device),
        }
    }
}
