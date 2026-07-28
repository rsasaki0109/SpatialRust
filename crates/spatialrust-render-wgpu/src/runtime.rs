use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use bytemuck::{Pod, Zeroable};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_viz::{
    DeviceIdentity, PointCloudView, TransferDirection, TransferEvent, TransferReceipt,
    TriangleMeshView, VisualPrimitive, VisualResidency,
};

use crate::{GpuGeometry, GpuGeometryKind, RenderError, RenderResult};

const MAX_CACHED_BUFFERS: usize = 32;
static NEXT_RENDERER_ID: AtomicU64 = AtomicU64::new(1);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PositionVertex {
    position: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Rgba8Vertex {
    color: [u8; 4],
}

pub(crate) struct BufferSlot {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) capacity: u64,
    pub(crate) logical_bytes: u64,
    usage: wgpu::BufferUsages,
}

#[derive(Default)]
struct RenderBufferPool {
    buffers: Vec<BufferSlot>,
}

impl RenderBufferPool {
    fn acquire(
        &mut self,
        device: &wgpu::Device,
        logical_bytes: u64,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> BufferSlot {
        let required_capacity = logical_bytes.max(wgpu::COPY_BUFFER_ALIGNMENT);
        if let Some(index) = self
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.usage == usage && slot.capacity >= required_capacity)
            .min_by_key(|(_, slot)| slot.capacity)
            .map(|(index, _)| index)
        {
            let mut slot = self.buffers.swap_remove(index);
            slot.logical_bytes = logical_bytes;
            return slot;
        }
        BufferSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: required_capacity,
                usage,
                mapped_at_creation: false,
            }),
            capacity: required_capacity,
            logical_bytes,
            usage,
        }
    }

    fn recycle(&mut self, slot: BufferSlot) {
        if self.buffers.len() < MAX_CACHED_BUFFERS {
            self.buffers.push(slot);
        } else {
            slot.buffer.destroy();
        }
    }
}

/// Renderer runtime sharing one explicit `spatialrust-gpu` wgpu device.
pub struct WgpuRenderer {
    id: u64,
    runtime: Arc<WgpuRuntime>,
    device_identity: DeviceIdentity,
    buffer_pool: Mutex<RenderBufferPool>,
}

impl WgpuRenderer {
    /// Creates a renderer on a caller-selected wgpu runtime.
    #[must_use]
    pub fn new(runtime: Arc<WgpuRuntime>) -> Self {
        let id = NEXT_RENDERER_ID.fetch_add(1, Ordering::Relaxed);
        let adapter = runtime.adapter_info();
        let backend = format!("wgpu-{}", adapter.backend);
        let device = format!("{}#renderer-{id}", adapter.name);
        let device_identity = DeviceIdentity::try_new(backend, device)
            .expect("wgpu adapter identity and renderer id are non-empty");
        Self { id, runtime, device_identity, buffer_pool: Mutex::new(RenderBufferPool::default()) }
    }

    /// Identity of this exact renderer runtime.
    #[must_use]
    pub const fn device_identity(&self) -> &DeviceIdentity {
        &self.device_identity
    }

    /// Number of buffers currently retained for reuse.
    #[must_use]
    pub fn cached_buffer_count(&self) -> usize {
        self.buffer_pool.lock().expect("render buffer pool poisoned").buffers.len()
    }

    /// Explicitly uploads one borrowed visual primitive.
    ///
    /// Structure-of-arrays positions are packed into a vertex buffer as part of
    /// this named operation. The returned receipt records every host/device
    /// crossing and its exact logical byte count.
    pub fn upload(
        &self,
        primitive: VisualPrimitive<'_>,
    ) -> RenderResult<(GpuGeometry, TransferReceipt)> {
        match primitive {
            VisualPrimitive::Points(points) => self.upload_points(points),
            VisualPrimitive::Lines(lines) => {
                let vertices = pack_interleaved_positions(lines.positions_xyz)?;
                self.upload_packed(GpuGeometryKind::Lines, vertices, None, None, None, "line")
            }
            VisualPrimitive::Triangles(mesh) => self.upload_triangles(mesh),
        }
    }

    fn upload_points(
        &self,
        points: PointCloudView<'_>,
    ) -> RenderResult<(GpuGeometry, TransferReceipt)> {
        let vertex_count = checked_u32(points.positions.len(), "point count")?;
        let mut positions = Vec::with_capacity(points.positions.len());
        for index in 0..points.positions.len() {
            positions.push(PositionVertex {
                position: [
                    points.positions.x[index],
                    points.positions.y[index],
                    points.positions.z[index],
                ],
            });
        }
        let rgb = points.rgb.map(|columns| {
            (0..points.positions.len())
                .map(|index| Rgba8Vertex {
                    color: [columns.red[index], columns.green[index], columns.blue[index], u8::MAX],
                })
                .collect::<Vec<_>>()
        });
        let scalar = points.scalar.map(|column| column.values.to_vec());
        self.upload_buffers(
            GpuGeometryKind::Points,
            vertex_count,
            0,
            &positions,
            rgb.as_deref(),
            scalar.as_deref(),
            None,
            "point",
        )
    }

    fn upload_triangles(
        &self,
        mesh: TriangleMeshView<'_>,
    ) -> RenderResult<(GpuGeometry, TransferReceipt)> {
        let vertices = pack_interleaved_positions(mesh.positions_xyz)?;
        let vertex_count = checked_u32(vertices.len(), "mesh vertex count")?;
        let index_count = checked_u32(mesh.indices.len(), "mesh index count")?;
        self.upload_buffers(
            GpuGeometryKind::Triangles,
            vertex_count,
            index_count,
            &vertices,
            None,
            None,
            Some(mesh.indices),
            "triangle",
        )
    }

    fn upload_packed(
        &self,
        kind: GpuGeometryKind,
        vertices: Vec<PositionVertex>,
        rgb: Option<&[Rgba8Vertex]>,
        scalar: Option<&[f32]>,
        indices: Option<&[u32]>,
        prefix: &'static str,
    ) -> RenderResult<(GpuGeometry, TransferReceipt)> {
        let vertex_count = checked_u32(vertices.len(), "vertex count")?;
        let index_count = checked_u32(indices.map_or(0, <[u32]>::len), "index count")?;
        self.upload_buffers(
            kind,
            vertex_count,
            index_count,
            &vertices,
            rgb,
            scalar,
            indices,
            prefix,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_buffers(
        &self,
        kind: GpuGeometryKind,
        vertex_count: u32,
        index_count: u32,
        positions: &[PositionVertex],
        rgb: Option<&[Rgba8Vertex]>,
        scalar: Option<&[f32]>,
        indices: Option<&[u32]>,
        prefix: &'static str,
    ) -> RenderResult<(GpuGeometry, TransferReceipt)> {
        let mut receipt = TransferReceipt::new();
        let position_slot = self.upload_slice(
            positions,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            "spatialrust render positions",
            &format!("{prefix}-positions-upload"),
            &mut receipt,
        )?;
        let rgb_slot = rgb
            .map(|values| {
                self.upload_slice(
                    values,
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    "spatialrust render rgb",
                    &format!("{prefix}-rgb-upload"),
                    &mut receipt,
                )
            })
            .transpose()?;
        let scalar_slot = scalar
            .map(|values| {
                self.upload_slice(
                    values,
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    "spatialrust render scalar",
                    &format!("{prefix}-scalar-upload"),
                    &mut receipt,
                )
            })
            .transpose()?;
        let index_slot = indices
            .map(|values| {
                self.upload_slice(
                    values,
                    wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    "spatialrust render indices",
                    &format!("{prefix}-indices-upload"),
                    &mut receipt,
                )
            })
            .transpose()?;
        Ok((
            GpuGeometry::new(
                self.id,
                kind,
                vertex_count,
                index_count,
                Some(position_slot),
                rgb_slot,
                scalar_slot,
                index_slot,
                self.device_identity.clone(),
            ),
            receipt,
        ))
    }

    fn upload_slice<T: Pod>(
        &self,
        values: &[T],
        usage: wgpu::BufferUsages,
        label: &'static str,
        stage: &str,
        receipt: &mut TransferReceipt,
    ) -> RenderResult<BufferSlot> {
        let bytes = bytemuck::cast_slice(values);
        let logical_bytes = u64::try_from(bytes.len())
            .map_err(|_| RenderError::GeometrySize("buffer byte length exceeds u64".into()))?;
        let mut slot = self.buffer_pool.lock().expect("render buffer pool poisoned").acquire(
            self.runtime.device(),
            logical_bytes,
            usage,
            label,
        );
        if !bytes.is_empty() {
            self.runtime.queue().write_buffer(&slot.buffer, 0, bytes);
        }
        slot.logical_bytes = logical_bytes;
        receipt.push(
            TransferEvent::try_new(
                stage,
                TransferDirection::Upload,
                VisualResidency::Host,
                VisualResidency::Device(self.device_identity.clone()),
                logical_bytes,
            )
            .map_err(|error| RenderError::Transfer(error.to_string()))?,
        );
        Ok(slot)
    }

    pub(crate) fn recycle_geometry(&self, geometry: &mut GpuGeometry) -> RenderResult<()> {
        if geometry.renderer_id != self.id {
            return Err(RenderError::RuntimeMismatch(
                "geometry was uploaded by another renderer".into(),
            ));
        }
        let mut pool = self.buffer_pool.lock().expect("render buffer pool poisoned");
        for slot in [
            geometry.positions.take(),
            geometry.rgb.take(),
            geometry.scalar.take(),
            geometry.indices.take(),
        ]
        .into_iter()
        .flatten()
        {
            pool.recycle(slot);
        }
        Ok(())
    }
}

fn pack_interleaved_positions(values: &[f32]) -> RenderResult<Vec<PositionVertex>> {
    if values.len() % 3 != 0 {
        return Err(RenderError::GeometrySize(
            "interleaved XYZ data must contain complete triples".into(),
        ));
    }
    Ok(values
        .chunks_exact(3)
        .map(|xyz| PositionVertex { position: [xyz[0], xyz[1], xyz[2]] })
        .collect())
}

fn checked_u32(value: usize, label: &str) -> RenderResult<u32> {
    u32::try_from(value)
        .map_err(|_| RenderError::GeometrySize(format!("{label} exceeds the u32 render limit")))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use spatialrust_gpu::WgpuRuntime;
    use spatialrust_viz::{
        PointCloudView, PositionColumns3, Rgb8Columns, ScalarColumn, TriangleMeshView,
        VisualPrimitive, VisualResidency,
    };

    use super::{pack_interleaved_positions, WgpuRenderer};

    #[test]
    fn packs_interleaved_positions_without_reordering() {
        let packed = pack_interleaved_positions(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(packed[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(packed[1].position, [4.0, 5.0, 6.0]);
        assert!(pack_interleaved_positions(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn explicit_upload_receipt_and_recycling() {
        let Ok(runtime) = WgpuRuntime::new_headless() else {
            eprintln!("skipping GPU upload test: no headless adapter");
            return;
        };
        let renderer = WgpuRenderer::new(Arc::new(runtime));
        let positions = PositionColumns3::try_new(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]).unwrap();
        let rgb = Rgb8Columns::try_new(&[1, 2], &[3, 4], &[5, 6], 2).unwrap();
        let scalar = ScalarColumn::try_new("intensity", &[0.25, 0.75], 2).unwrap();
        let points = PointCloudView::positions_only(positions)
            .with_rgb(rgb)
            .unwrap()
            .with_scalar(scalar)
            .unwrap();

        let (geometry, receipt) = renderer.upload(VisualPrimitive::Points(points)).unwrap();
        assert_eq!(geometry.vertex_count(), 2);
        assert_eq!(geometry.position_bytes(), 24);
        assert_eq!(geometry.rgb_bytes(), 8);
        assert_eq!(geometry.scalar_bytes(), 8);
        assert_eq!(receipt.total_bytes().unwrap(), 40);
        assert_eq!(receipt.events().len(), 3);
        assert_eq!(receipt.events()[0].stage, "point-positions-upload");
        assert_eq!(
            geometry.residency(),
            &VisualResidency::Device(renderer.device_identity().clone())
        );

        geometry.recycle(&renderer).unwrap();
        assert_eq!(renderer.cached_buffer_count(), 3);

        let (reused, _) = renderer.upload(VisualPrimitive::Points(points)).unwrap();
        assert_eq!(renderer.cached_buffer_count(), 0);
        reused.recycle(&renderer).unwrap();

        let mesh =
            TriangleMeshView::try_new(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0, 1, 2])
                .unwrap();
        let (triangles, triangle_receipt) =
            renderer.upload(VisualPrimitive::Triangles(mesh)).unwrap();
        assert_eq!(triangles.vertex_count(), 3);
        assert_eq!(triangles.index_count(), 3);
        assert_eq!(triangles.position_bytes(), 36);
        assert_eq!(triangles.index_bytes(), 12);
        assert_eq!(triangle_receipt.total_bytes().unwrap(), 48);
        assert_eq!(triangle_receipt.events().len(), 2);
        triangles.recycle(&renderer).unwrap();
    }

    #[test]
    fn recycle_rejects_another_renderer() {
        let Ok(runtime) = WgpuRuntime::new_headless() else {
            eprintln!("skipping GPU runtime identity test: no headless adapter");
            return;
        };
        let runtime = Arc::new(runtime);
        let first = WgpuRenderer::new(Arc::clone(&runtime));
        let second = WgpuRenderer::new(runtime);
        let positions = PositionColumns3::try_new(&[0.0], &[0.0], &[0.0]).unwrap();
        let points = PointCloudView::positions_only(positions);
        let (geometry, _) = first.upload(VisualPrimitive::Points(points)).unwrap();
        assert!(geometry.recycle(&second).is_err());
    }
}
