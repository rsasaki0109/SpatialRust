use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_viz::{
    Camera, ColorMap, DeviceIdentity, LinearRgba, PointColor, Projection, TransferDirection,
    TransferEvent, TransferReceipt, VisualResidency, VisualStyle,
};
use wgpu::util::DeviceExt;

use crate::{GpuGeometry, GpuGeometryKind, RenderError, RenderResult, WgpuRenderer};

/// Portable color format used by headless render targets.
pub const RENDER_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Configuration for one device-resident headless render.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderOptions {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Camera used for this render.
    pub camera: Camera,
    /// Primitive presentation style.
    pub style: VisualStyle,
    /// Linear clear color.
    pub clear_color: LinearRgba,
}

impl RenderOptions {
    /// Creates options with a validated non-zero target size.
    pub fn try_new(
        width: u32,
        height: u32,
        camera: Camera,
        style: VisualStyle,
        clear_color: LinearRgba,
    ) -> RenderResult<Self> {
        if width == 0 || height == 0 {
            return Err(RenderError::GeometrySize(
                "render target dimensions must be non-zero".into(),
            ));
        }
        Ok(Self { width, height, camera, style, clear_color })
    }
}

/// Named device-side execution evidence for one render.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderReceipt {
    /// Exact renderer adapter identity.
    pub adapter: DeviceIdentity,
    /// Named stages in submission order.
    pub stages: Vec<&'static str>,
    /// Number of submitted draw calls.
    pub draw_calls: u32,
    /// Number of vertices or point instances consumed.
    pub element_count: u32,
}

/// Device-resident color and depth attachments.
pub struct GpuRenderTarget {
    pub(crate) renderer_id: u64,
    pub(crate) color: wgpu::Texture,
    pub(crate) depth: wgpu::Texture,
    point_ids: Option<wgpu::Texture>,
    width: u32,
    height: u32,
    residency: VisualResidency,
}

/// Caller-requested tightly packed RGBA8 readback.
#[derive(Debug)]
pub struct ReadbackImage {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Tightly packed row-major RGBA8 pixels.
    pub rgba: Vec<u8>,
    /// Exact device-to-host transfer receipt.
    pub transfers: TransferReceipt,
}

/// Caller-requested point-picking result.
#[derive(Debug)]
pub struct PickResult {
    /// Uploaded point index, or `None` for the cleared background.
    pub point_index: Option<u32>,
    /// Exact four-byte device-to-host transfer receipt.
    pub transfers: TransferReceipt,
}

impl GpuRenderTarget {
    /// Target width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Target height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Device residency of both attachments.
    #[must_use]
    pub const fn residency(&self) -> &VisualResidency {
        &self.residency
    }

    /// Returns whether this target belongs to the supplied renderer runtime.
    #[must_use]
    pub fn is_owned_by(&self, renderer: &WgpuRenderer) -> bool {
        self.renderer_id == renderer.id
    }
}

/// Result of a device-resident headless render.
pub struct HeadlessRender {
    /// Device-resident target. No readback has occurred.
    pub target: GpuRenderTarget,
    /// Explicit transfers required to submit the render.
    pub transfers: TransferReceipt,
    /// Named render stages and draw counts.
    pub receipt: RenderReceipt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ColorMode {
    Uniform,
    Rgb,
    Scalar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PipelineKey {
    kind: GpuGeometryKind,
    color: ColorMode,
}

#[derive(Default)]
pub(crate) struct RenderPipelines {
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    point_id_pipeline: Option<wgpu::RenderPipeline>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RenderUniform {
    view_projection: [[f32; 4]; 4],
    color: [f32; 4],
    scalar_range: [f32; 2],
    viewport: [f32; 2],
    point_size: f32,
    color_map: u32,
    _padding: [u32; 2],
}

impl WgpuRenderer {
    /// Renders uploaded geometry into device-resident color and depth textures.
    ///
    /// The method never reads the target back to the host. Its only host/device
    /// crossing is the returned, byte-accounted uniform upload.
    pub fn render_headless(
        &self,
        geometry: &GpuGeometry,
        options: &RenderOptions,
    ) -> RenderResult<HeadlessRender> {
        if geometry.renderer_id != self.id {
            return Err(RenderError::RuntimeMismatch(
                "geometry was uploaded by another renderer".into(),
            ));
        }
        let max_dimension = self.runtime.device().limits().max_texture_dimension_2d;
        if options.width > max_dimension || options.height > max_dimension {
            return Err(RenderError::GeometrySize(format!(
                "render target {}x{} exceeds adapter limit {max_dimension}",
                options.width, options.height
            )));
        }
        let (color_mode, uniform_color, scalar_range, point_size, color_map) =
            resolve_style(geometry, &options.style)?;
        let key = PipelineKey { kind: geometry.kind, color: color_mode };
        let view_projection =
            view_projection(options.camera, options.width as f32 / options.height as f32);
        let uniform = RenderUniform {
            view_projection,
            color: [
                uniform_color.red,
                uniform_color.green,
                uniform_color.blue,
                uniform_color.alpha,
            ],
            scalar_range,
            viewport: [options.width as f32, options.height as f32],
            point_size,
            color_map,
            _padding: [0; 2],
        };
        let uniform_bytes = bytemuck::bytes_of(&uniform);
        let uniform_buffer =
            self.runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spatialrust render uniform"),
                contents: uniform_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut pipelines = self.render_pipelines.lock().expect("render pipeline cache poisoned");
        let RenderPipelines { bind_group_layout, pipelines: pipeline_map, point_id_pipeline } =
            &mut *pipelines;
        if bind_group_layout.is_none() {
            *bind_group_layout = Some(self.runtime.device().create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("spatialrust render uniform layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                },
            ));
        }
        let bind_group_layout = bind_group_layout.as_ref().expect("bind group layout was inserted");
        let bind_group = self.runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatialrust render uniform bind group"),
            layout: bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let target = create_target(
            self.runtime.device(),
            self.id,
            options.width,
            options.height,
            self.device_identity().clone(),
            geometry.kind == GpuGeometryKind::Points,
        );
        let color_view = target.color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = target.depth.create_view(&wgpu::TextureViewDescriptor::default());
        let pipeline = pipeline_map
            .entry(key)
            .or_insert_with(|| create_pipeline(self.runtime.device(), bind_group_layout, key));
        if geometry.kind == GpuGeometryKind::Points && point_id_pipeline.is_none() {
            *point_id_pipeline =
                Some(create_point_id_pipeline(self.runtime.device(), bind_group_layout));
        }

        let mut encoder =
            self.runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spatialrust headless render encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("spatialrust headless render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: options.clear_color.red as f64,
                            g: options.clear_color.green as f64,
                            b: options.clear_color.blue as f64,
                            a: options.clear_color.alpha as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(
                0,
                geometry
                    .positions
                    .as_ref()
                    .expect("uploaded geometry always owns positions")
                    .buffer
                    .slice(..),
            );
            match color_mode {
                ColorMode::Uniform => {}
                ColorMode::Rgb => pass.set_vertex_buffer(
                    1,
                    geometry.rgb.as_ref().expect("RGB style was validated").buffer.slice(..),
                ),
                ColorMode::Scalar => pass.set_vertex_buffer(
                    1,
                    geometry.scalar.as_ref().expect("scalar style was validated").buffer.slice(..),
                ),
            }
            match geometry.kind {
                GpuGeometryKind::Points => pass.draw(0..6, 0..geometry.vertex_count),
                GpuGeometryKind::Lines => pass.draw(0..geometry.vertex_count, 0..1),
                GpuGeometryKind::Triangles => {
                    let indices =
                        geometry.indices.as_ref().expect("triangle geometry owns indices");
                    pass.set_index_buffer(indices.buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..geometry.index_count, 0, 0..1);
                }
            }
        }
        if geometry.kind == GpuGeometryKind::Points {
            let id_view = target
                .point_ids
                .as_ref()
                .expect("point target owns an ID texture")
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("spatialrust point ID render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &id_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(
                point_id_pipeline.as_ref().expect("point ID pipeline was initialized"),
            );
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(
                0,
                geometry
                    .positions
                    .as_ref()
                    .expect("uploaded geometry always owns positions")
                    .buffer
                    .slice(..),
            );
            pass.draw(0..6, 0..geometry.vertex_count);
        }
        self.runtime.queue().submit(Some(encoder.finish()));
        drop(pipelines);

        let mut transfers = TransferReceipt::new();
        transfers.push(
            TransferEvent::try_new(
                "render-uniform-upload",
                TransferDirection::Upload,
                VisualResidency::Host,
                VisualResidency::Device(self.device_identity().clone()),
                uniform_bytes.len() as u64,
            )
            .map_err(|error| RenderError::Transfer(error.to_string()))?,
        );
        let stage = match geometry.kind {
            GpuGeometryKind::Points => "draw-points",
            GpuGeometryKind::Lines => "draw-lines",
            GpuGeometryKind::Triangles => "draw-triangles",
        };
        let element_count = match geometry.kind {
            GpuGeometryKind::Triangles => geometry.index_count,
            _ => geometry.vertex_count,
        };
        Ok(HeadlessRender {
            target,
            transfers,
            receipt: RenderReceipt {
                adapter: self.device_identity().clone(),
                stages: if geometry.kind == GpuGeometryKind::Points {
                    vec!["clear-color-depth", stage, "draw-point-ids", "submit"]
                } else {
                    vec!["clear-color-depth", stage, "submit"]
                },
                draw_calls: if geometry.kind == GpuGeometryKind::Points { 2 } else { 1 },
                element_count,
            },
        })
    }

    /// Number of initialized render-pipeline variants.
    #[must_use]
    pub fn initialized_render_pipeline_count(&self) -> usize {
        let pipelines = self.render_pipelines.lock().expect("render pipeline cache poisoned");
        pipelines.pipelines.len() + usize::from(pipelines.point_id_pipeline.is_some())
    }

    /// Explicitly reads one device-resident target into tightly packed RGBA8.
    pub fn readback_rgba(&self, target: &GpuRenderTarget) -> RenderResult<ReadbackImage> {
        self.validate_target(target)?;
        let unpadded_row_bytes = target
            .width
            .checked_mul(4)
            .ok_or_else(|| RenderError::Readback("RGBA row byte count overflowed".into()))?;
        let padded_row_bytes = align_up(unpadded_row_bytes, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)?;
        let buffer_bytes = u64::from(padded_row_bytes)
            .checked_mul(u64::from(target.height))
            .ok_or_else(|| RenderError::Readback("padded readback byte count overflowed".into()))?;
        let buffer = self.runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("spatialrust RGBA readback"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spatialrust RGBA readback encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(target.height),
                },
            },
            wgpu::Extent3d { width: target.width, height: target.height, depth_or_array_layers: 1 },
        );
        self.runtime.queue().submit(Some(encoder.finish()));
        let mapped = map_buffer(&self.runtime, &buffer)?;
        let logical_bytes = usize::try_from(unpadded_row_bytes)
            .ok()
            .and_then(|row| row.checked_mul(target.height as usize))
            .ok_or_else(|| RenderError::Readback("logical RGBA byte count overflowed".into()))?;
        let mut rgba = Vec::with_capacity(logical_bytes);
        for row in mapped.chunks_exact(padded_row_bytes as usize).take(target.height as usize) {
            rgba.extend_from_slice(&row[..unpadded_row_bytes as usize]);
        }
        drop(mapped);
        buffer.unmap();
        let mut transfers = TransferReceipt::new();
        transfers.push(self.readback_event("rgba-readback", logical_bytes as u64)?);
        Ok(ReadbackImage { width: target.width, height: target.height, rgba, transfers })
    }

    /// Explicitly reads the point ID at one target pixel.
    pub fn pick_point(&self, target: &GpuRenderTarget, x: u32, y: u32) -> RenderResult<PickResult> {
        self.validate_target(target)?;
        if x >= target.width || y >= target.height {
            return Err(RenderError::Readback("pick coordinates are outside the target".into()));
        }
        let texture = target.point_ids.as_ref().ok_or_else(|| {
            RenderError::Readback("point picking requires a target rendered from points".into())
        })?;
        let buffer = self.runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("spatialrust point ID readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spatialrust point ID readback encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        self.runtime.queue().submit(Some(encoder.finish()));
        let mapped = map_buffer(&self.runtime, &buffer)?;
        let encoded = u32::from_le_bytes(mapped[..4].try_into().expect("mapped ID is four bytes"));
        drop(mapped);
        buffer.unmap();
        let mut transfers = TransferReceipt::new();
        transfers.push(self.readback_event("point-id-readback", 4)?);
        Ok(PickResult { point_index: encoded.checked_sub(1), transfers })
    }

    fn validate_target(&self, target: &GpuRenderTarget) -> RenderResult<()> {
        if target.renderer_id != self.id {
            return Err(RenderError::RuntimeMismatch(
                "render target belongs to another renderer".into(),
            ));
        }
        Ok(())
    }

    fn readback_event(&self, stage: &str, bytes: u64) -> RenderResult<TransferEvent> {
        TransferEvent::try_new(
            stage,
            TransferDirection::Readback,
            VisualResidency::Device(self.device_identity().clone()),
            VisualResidency::Host,
            bytes,
        )
        .map_err(|error| RenderError::Transfer(error.to_string()))
    }
}

fn resolve_style(
    geometry: &GpuGeometry,
    style: &VisualStyle,
) -> RenderResult<(ColorMode, LinearRgba, [f32; 2], f32, u32)> {
    match (geometry.kind, style) {
        (GpuGeometryKind::Points, VisualStyle::Points(point_style)) => match point_style.color {
            PointColor::Uniform(color) => {
                Ok((ColorMode::Uniform, color, [0.0, 1.0], point_style.size, 0))
            }
            PointColor::Rgb if geometry.rgb.is_some() => {
                Ok((ColorMode::Rgb, LinearRgba::WHITE, [0.0, 1.0], point_style.size, 0))
            }
            PointColor::Scalar { min, max, map } if geometry.scalar.is_some() => Ok((
                ColorMode::Scalar,
                LinearRgba::WHITE,
                [min, max],
                point_style.size,
                color_map_id(map),
            )),
            PointColor::Rgb => {
                Err(RenderError::GeometrySize("RGB style requires an uploaded RGB buffer".into()))
            }
            PointColor::Scalar { .. } => Err(RenderError::GeometrySize(
                "scalar style requires an uploaded scalar buffer".into(),
            )),
        },
        (GpuGeometryKind::Points, VisualStyle::Uniform(color)) => {
            Ok((ColorMode::Uniform, *color, [0.0, 1.0], 1.0, 0))
        }
        (_, VisualStyle::Uniform(color)) => Ok((ColorMode::Uniform, *color, [0.0, 1.0], 1.0, 0)),
        (_, VisualStyle::Points(_)) => Err(RenderError::GeometrySize(
            "point style cannot be applied to line or triangle geometry".into(),
        )),
    }
}

const fn color_map_id(map: ColorMap) -> u32 {
    match map {
        ColorMap::Viridis => 0,
        ColorMap::Turbo => 1,
        ColorMap::Gray => 2,
    }
}

fn create_target(
    device: &wgpu::Device,
    renderer_id: u64,
    width: u32,
    height: u32,
    identity: DeviceIdentity,
    point_picking: bool,
) -> GpuRenderTarget {
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("spatialrust headless color"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RENDER_TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("spatialrust headless depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let point_ids = point_picking.then(|| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("spatialrust point IDs"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    });
    GpuRenderTarget {
        renderer_id,
        color,
        depth,
        point_ids,
        width,
        height,
        residency: VisualResidency::Device(identity),
    }
}

fn create_point_id_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("spatialrust point ID shader"),
        source: wgpu::ShaderSource::Wgsl(POINT_ID_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("spatialrust point ID pipeline layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("spatialrust point ID pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R32Uint,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

fn map_buffer<'a>(
    runtime: &WgpuRuntime,
    buffer: &'a wgpu::Buffer,
) -> RenderResult<wgpu::BufferView<'a>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|error| RenderError::Readback(format!("map callback dropped: {error}")))?
        .map_err(|error| RenderError::Readback(format!("buffer map: {error}")))?;
    Ok(slice.get_mapped_range())
}

fn align_up(value: u32, alignment: u32) -> RenderResult<u32> {
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .ok_or_else(|| RenderError::Readback("row alignment overflowed".into()))
}

fn create_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    key: PipelineKey,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("spatialrust headless shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source(key).into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("spatialrust headless pipeline layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });
    let position_step = if key.kind == GpuGeometryKind::Points {
        wgpu::VertexStepMode::Instance
    } else {
        wgpu::VertexStepMode::Vertex
    };
    let position = wgpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: position_step,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }],
    };
    let attribute = match key.color {
        ColorMode::Uniform => None,
        ColorMode::Rgb => Some(wgpu::VertexBufferLayout {
            array_stride: 4,
            step_mode: position_step,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Unorm8x4,
                offset: 0,
                shader_location: 1,
            }],
        }),
        ColorMode::Scalar => Some(wgpu::VertexBufferLayout {
            array_stride: 4,
            step_mode: position_step,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32,
                offset: 0,
                shader_location: 1,
            }],
        }),
    };
    let mut buffers = vec![position];
    if let Some(attribute) = attribute {
        buffers.push(attribute);
    }
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("spatialrust headless render pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: match key.kind {
                GpuGeometryKind::Points => wgpu::PrimitiveTopology::TriangleList,
                GpuGeometryKind::Lines => wgpu::PrimitiveTopology::LineList,
                GpuGeometryKind::Triangles => wgpu::PrimitiveTopology::TriangleList,
            },
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: RENDER_TARGET_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

fn shader_source(key: PipelineKey) -> &'static str {
    match (key.kind, key.color) {
        (GpuGeometryKind::Points, ColorMode::Uniform) => POINT_UNIFORM_SHADER,
        (GpuGeometryKind::Points, ColorMode::Rgb) => POINT_RGB_SHADER,
        (GpuGeometryKind::Points, ColorMode::Scalar) => POINT_SCALAR_SHADER,
        (_, ColorMode::Uniform) => PRIMITIVE_UNIFORM_SHADER,
        (_, _) => unreachable!("non-point styles are validated as uniform"),
    }
}

fn view_projection(camera: Camera, aspect: f32) -> [[f32; 4]; 4] {
    multiply4(projection_matrix(camera.projection, aspect), view_matrix(camera))
}

fn view_matrix(camera: Camera) -> [[f32; 4]; 4] {
    let forward = (camera.target - camera.eye).normalize();
    let side = forward.cross(camera.up).normalize();
    let up = side.cross(forward);
    [
        [side.x, up.x, -forward.x, 0.0],
        [side.y, up.y, -forward.y, 0.0],
        [side.z, up.z, -forward.z, 0.0],
        [-side.dot(camera.eye), -up.dot(camera.eye), forward.dot(camera.eye), 1.0],
    ]
}

fn projection_matrix(projection: Projection, aspect: f32) -> [[f32; 4]; 4] {
    match projection {
        Projection::Perspective { vertical_fov_radians, near, far } => {
            let f = 1.0 / (vertical_fov_radians * 0.5).tan();
            [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, far / (near - far), -1.0],
                [0.0, 0.0, far * near / (near - far), 0.0],
            ]
        }
        Projection::Orthographic { vertical_span, near, far } => {
            let horizontal_span = vertical_span * aspect;
            [
                [2.0 / horizontal_span, 0.0, 0.0, 0.0],
                [0.0, 2.0 / vertical_span, 0.0, 0.0],
                [0.0, 0.0, 1.0 / (near - far), 0.0],
                [0.0, 0.0, near / (near - far), 1.0],
            ]
        }
    }
}

fn multiply4(left: [[f32; 4]; 4], right: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut output = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            output[column][row] = (0..4).map(|index| left[index][row] * right[column][index]).sum();
        }
    }
    output
}

const POINT_ID_SHADER: &str = r#"
struct Uniforms {
    view_projection: mat4x4<f32>, color: vec4<f32>, scalar_range: vec2<f32>,
    viewport: vec2<f32>, point_size: f32, color_map: u32, padding: vec2<u32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) encoded_id: u32,
};
const OFFSETS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);
@vertex fn vs_main(
    @builtin(vertex_index) corner: u32,
    @builtin(instance_index) point_id: u32,
    @location(0) center: vec3<f32>
) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.view_projection * vec4<f32>(center, 1.0);
    let offset = OFFSETS[corner] * uniforms.point_size / uniforms.viewport * 2.0 * output.position.w;
    output.position = vec4<f32>(output.position.xy + offset, output.position.zw);
    output.encoded_id = point_id + 1u;
    return output;
}
@fragment fn fs_main(input: VertexOutput) -> @location(0) u32 {
    return input.encoded_id;
}
"#;

const POINT_UNIFORM_SHADER: &str = r#"
struct Uniforms {
    view_projection: mat4x4<f32>,
    color: vec4<f32>,
    scalar_range: vec2<f32>,
    viewport: vec2<f32>,
    point_size: f32,
    color_map: u32,
    padding: vec2<u32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> };
const OFFSETS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);
@vertex fn vs_main(@builtin(vertex_index) corner: u32, @location(0) center: vec3<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.view_projection * vec4<f32>(center, 1.0);
    let offset = OFFSETS[corner] * uniforms.point_size / uniforms.viewport * 2.0 * output.position.w;
    output.position = vec4<f32>(output.position.xy + offset, output.position.zw);
    output.color = uniforms.color;
    return output;
}
@fragment fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }
"#;

const POINT_RGB_SHADER: &str = r#"
struct Uniforms {
    view_projection: mat4x4<f32>, color: vec4<f32>, scalar_range: vec2<f32>,
    viewport: vec2<f32>, point_size: f32, color_map: u32, padding: vec2<u32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> };
const OFFSETS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);
@vertex fn vs_main(
    @builtin(vertex_index) corner: u32,
    @location(0) center: vec3<f32>,
    @location(1) color: vec4<f32>
) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.view_projection * vec4<f32>(center, 1.0);
    let offset = OFFSETS[corner] * uniforms.point_size / uniforms.viewport * 2.0 * output.position.w;
    output.position = vec4<f32>(output.position.xy + offset, output.position.zw);
    output.color = color;
    return output;
}
@fragment fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }
"#;

const POINT_SCALAR_SHADER: &str = r#"
struct Uniforms {
    view_projection: mat4x4<f32>, color: vec4<f32>, scalar_range: vec2<f32>,
    viewport: vec2<f32>, point_size: f32, color_map: u32, padding: vec2<u32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> };
const OFFSETS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);
fn color_map(t_in: f32, map: u32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    if map == 2u { return vec3<f32>(t); }
    if map == 1u {
        return clamp(vec3<f32>(
            1.5 - abs(4.0 * t - 3.0),
            1.5 - abs(4.0 * t - 2.0),
            1.5 - abs(4.0 * t - 1.0)
        ), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec3<f32>(
        0.267 + t * (0.993 - 0.267),
        0.005 + t * (0.906 - 0.005),
        0.329 + t * (0.144 - 0.329)
    );
}
@vertex fn vs_main(
    @builtin(vertex_index) corner: u32,
    @location(0) center: vec3<f32>,
    @location(1) scalar: f32
) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.view_projection * vec4<f32>(center, 1.0);
    let offset = OFFSETS[corner] * uniforms.point_size / uniforms.viewport * 2.0 * output.position.w;
    output.position = vec4<f32>(output.position.xy + offset, output.position.zw);
    let t = (scalar - uniforms.scalar_range.x) /
        (uniforms.scalar_range.y - uniforms.scalar_range.x);
    output.color = vec4<f32>(color_map(t, uniforms.color_map), 1.0);
    return output;
}
@fragment fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }
"#;

const PRIMITIVE_UNIFORM_SHADER: &str = r#"
struct Uniforms {
    view_projection: mat4x4<f32>, color: vec4<f32>, scalar_range: vec2<f32>,
    viewport: vec2<f32>, point_size: f32, color_map: u32, padding: vec2<u32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs_main(@location(0) position: vec3<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.view_projection * vec4<f32>(position, 1.0);
    output.color = uniforms.color;
    return output;
}
@fragment fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }
"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use spatialrust_gpu::WgpuRuntime;
    use spatialrust_math::Vec3;
    use spatialrust_viz::{
        Camera, ColorMap, LineListView, LinearRgba, PointCloudView, PointColor, PointStyle,
        PositionColumns3, Projection, Rgb8Columns, ScalarColumn, TriangleMeshView, VisualPrimitive,
        VisualStyle,
    };

    use super::{RenderError, RenderOptions, WgpuRenderer};

    fn camera() -> Camera {
        Camera::try_new(
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 10.0 },
        )
        .unwrap()
    }

    #[test]
    fn rejects_zero_target_size() {
        assert!(RenderOptions::try_new(
            0,
            64,
            camera(),
            VisualStyle::Uniform(LinearRgba::WHITE),
            LinearRgba::BLACK,
        )
        .is_err());
    }

    #[test]
    fn renders_all_topologies_and_point_color_modes() {
        let Ok(runtime) = WgpuRuntime::new_headless() else {
            eprintln!("skipping headless render test: no adapter");
            return;
        };
        let runtime = Arc::new(runtime);
        let renderer = WgpuRenderer::new(Arc::clone(&runtime));
        let other_renderer = WgpuRenderer::new(runtime);
        let positions = PositionColumns3::try_new(&[0.0, 0.5], &[0.0, 0.0], &[0.0, 0.0]).unwrap();
        let rgb = Rgb8Columns::try_new(&[255, 0], &[0, 255], &[0, 0], 2).unwrap();
        let scalar = ScalarColumn::try_new("intensity", &[0.0, 1.0], 2).unwrap();
        let points = PointCloudView::positions_only(positions)
            .with_rgb(rgb)
            .unwrap()
            .with_scalar(scalar)
            .unwrap();
        let (gpu_points, _) = renderer.upload(VisualPrimitive::Points(points)).unwrap();

        let styles = [
            VisualStyle::Points(
                PointStyle::try_new(
                    9.0,
                    PointColor::Uniform(LinearRgba::try_new(1.0, 0.0, 0.0, 1.0).unwrap()),
                )
                .unwrap(),
            ),
            VisualStyle::Points(PointStyle::try_new(5.0, PointColor::Rgb).unwrap()),
            VisualStyle::Points(
                PointStyle::try_new(
                    5.0,
                    PointColor::Scalar { min: 0.0, max: 1.0, map: ColorMap::Viridis },
                )
                .unwrap(),
            ),
        ];
        for (style_index, style) in styles.into_iter().enumerate() {
            let options =
                RenderOptions::try_new(64, 64, camera(), style, LinearRgba::BLACK).unwrap();
            let output = renderer.render_headless(&gpu_points, &options).unwrap();
            assert_eq!(output.target.width(), 64);
            assert_eq!(
                output.receipt.stages,
                ["clear-color-depth", "draw-points", "draw-point-ids", "submit"]
            );
            assert_eq!(output.transfers.events()[0].stage, "render-uniform-upload");
            assert_eq!(output.transfers.total_bytes().unwrap(), 112);
            if style_index == 0 {
                let image = renderer.readback_rgba(&output.target).unwrap();
                assert_eq!(image.rgba.len(), 64 * 64 * 4);
                let center = (32 * 64 + 32) * 4;
                assert_eq!(&image.rgba[center..center + 4], &[255, 0, 0, 255]);
                assert_eq!(&image.rgba[..4], &[0, 0, 0, 255]);
                assert_eq!(image.transfers.total_bytes().unwrap(), 64 * 64 * 4);
                let pick = renderer.pick_point(&output.target, 32, 32).unwrap();
                assert_eq!(pick.point_index, Some(0));
                assert_eq!(pick.transfers.total_bytes().unwrap(), 4);
                assert!(other_renderer.readback_rgba(&output.target).is_err());
                assert_eq!(
                    renderer.pick_point(&output.target, 64, 32).unwrap_err(),
                    RenderError::Readback("pick coordinates are outside the target".into())
                );
            }
        }

        let padded_options = RenderOptions::try_new(
            17,
            19,
            camera(),
            VisualStyle::Points(
                PointStyle::try_new(
                    7.0,
                    PointColor::Uniform(LinearRgba::try_new(1.0, 0.0, 0.0, 1.0).unwrap()),
                )
                .unwrap(),
            ),
            LinearRgba::BLACK,
        )
        .unwrap();
        let padded_output = renderer.render_headless(&gpu_points, &padded_options).unwrap();
        let padded_image = renderer.readback_rgba(&padded_output.target).unwrap();
        assert_eq!(padded_image.rgba.len(), 17 * 19 * 4);
        assert!(renderer.pick_point(&padded_output.target, 8, 9).unwrap().point_index.is_some());

        let lines = LineListView::try_new(&[-0.5, 0.0, 0.0, 0.5, 0.0, 0.0]).unwrap();
        let (gpu_lines, _) = renderer.upload(VisualPrimitive::Lines(lines)).unwrap();
        let uniform = VisualStyle::Uniform(LinearRgba::WHITE);
        let options =
            RenderOptions::try_new(64, 64, camera(), uniform.clone(), LinearRgba::BLACK).unwrap();
        let line_output = renderer.render_headless(&gpu_lines, &options).unwrap();
        assert_eq!(line_output.receipt.stages[1], "draw-lines");
        assert!(renderer.pick_point(&line_output.target, 32, 32).is_err());

        let mesh = TriangleMeshView::try_new(
            &[-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0],
            &[0, 1, 2],
        )
        .unwrap();
        let (gpu_mesh, _) = renderer.upload(VisualPrimitive::Triangles(mesh)).unwrap();
        let options = RenderOptions::try_new(64, 64, camera(), uniform, LinearRgba::BLACK).unwrap();
        assert_eq!(
            renderer.render_headless(&gpu_mesh, &options).unwrap().receipt.stages[1],
            "draw-triangles"
        );
        renderer.runtime.wait_idle();
        assert_eq!(renderer.initialized_render_pipeline_count(), 6);
    }
}
