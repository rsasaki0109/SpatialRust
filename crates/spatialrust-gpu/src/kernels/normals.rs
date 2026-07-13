use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NormalsUniform {
    point_count: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Per-point normal estimation output: `(nx, ny, nz, curvature)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuNormal {
    /// Unit normal x/y/z.
    pub normal: [f32; 3],
    /// Surface variation curvature in `[0, 1/3]`.
    pub curvature: f32,
}

pub(crate) const NORMALS_WGSL: &str = r#"
struct Params { point_count: u32, k: u32, pad0: u32, pad1: u32, };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read> ys: array<f32>;
@group(0) @binding(3) var<storage, read> zs: array<f32>;
@group(0) @binding(4) var<storage, read> neighbors: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_normals: array<vec4<f32>>;

fn rotate(a: ptr<function, array<vec3<f32>, 3>>,
          v: ptr<function, array<vec3<f32>, 3>>,
          p: u32, q: u32) {
    let apq = (*a)[p][q];
    if (abs(apq) < 1e-20) {
        return;
    }
    let app = (*a)[p][p];
    let aqq = (*a)[q][q];
    let phi = 0.5 * (aqq - app) / apq;
    var t: f32;
    if (phi >= 0.0) {
        t = 1.0 / (phi + sqrt(1.0 + phi * phi));
    } else {
        t = -1.0 / (-phi + sqrt(1.0 + phi * phi));
    }
    let c = 1.0 / sqrt(1.0 + t * t);
    let s = t * c;

    // Apply the Jacobi rotation to columns/rows p and q of the symmetric matrix.
    for (var r: u32 = 0u; r < 3u; r = r + 1u) {
        let arp = (*a)[r][p];
        let arq = (*a)[r][q];
        (*a)[r][p] = c * arp - s * arq;
        (*a)[r][q] = s * arp + c * arq;
    }
    for (var r: u32 = 0u; r < 3u; r = r + 1u) {
        let apr = (*a)[p][r];
        let aqr = (*a)[q][r];
        (*a)[p][r] = c * apr - s * aqr;
        (*a)[q][r] = s * apr + c * aqr;
    }
    for (var r: u32 = 0u; r < 3u; r = r + 1u) {
        let vrp = (*v)[r][p];
        let vrq = (*v)[r][q];
        (*v)[r][p] = c * vrp - s * vrq;
        (*v)[r][q] = s * vrp + c * vrq;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }
    let k = params.k;
    let base = i * k;

    var mean = vec3<f32>(0.0, 0.0, 0.0);
    var count = 0.0;
    for (var j: u32 = 0u; j < k; j = j + 1u) {
        let idx = neighbors[base + j];
        mean = mean + vec3<f32>(xs[idx], ys[idx], zs[idx]);
        count = count + 1.0;
    }
    if (count < 3.0) {
        out_normals[i] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
        return;
    }
    mean = mean / count;

    var c00 = 0.0; var c11 = 0.0; var c22 = 0.0;
    var c01 = 0.0; var c02 = 0.0; var c12 = 0.0;
    for (var j: u32 = 0u; j < k; j = j + 1u) {
        let idx = neighbors[base + j];
        let d = vec3<f32>(xs[idx], ys[idx], zs[idx]) - mean;
        c00 = c00 + d.x * d.x;
        c11 = c11 + d.y * d.y;
        c22 = c22 + d.z * d.z;
        c01 = c01 + d.x * d.y;
        c02 = c02 + d.x * d.z;
        c12 = c12 + d.y * d.z;
    }

    var a = array<vec3<f32>, 3>(
        vec3<f32>(c00, c01, c02),
        vec3<f32>(c01, c11, c12),
        vec3<f32>(c02, c12, c22),
    );
    var v = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );

    for (var sweep: u32 = 0u; sweep < 16u; sweep = sweep + 1u) {
        rotate(&a, &v, 0u, 1u);
        rotate(&a, &v, 0u, 2u);
        rotate(&a, &v, 1u, 2u);
    }

    let eig = vec3<f32>(a[0][0], a[1][1], a[2][2]);
    var min_idx = 0u;
    if (eig[1] < eig[min_idx]) { min_idx = 1u; }
    if (eig[2] < eig[min_idx]) { min_idx = 2u; }

    let normal = vec3<f32>(v[0][min_idx], v[1][min_idx], v[2][min_idx]);
    let length = max(sqrt(dot(normal, normal)), 1e-20);
    let unit = normal / length;

    let trace = eig[0] + eig[1] + eig[2];
    var curvature = 0.0;
    if (trace > 0.0) {
        curvature = eig[min_idx] / trace;
    }

    out_normals[i] = vec4<f32>(unit.x, unit.y, unit.z, curvature);
}
"#;

/// Estimates per-point normals and curvature on the GPU.
///
/// `neighbors` is a flattened `point_count * k` array of indices into the point
/// arrays, where row `i` lists the neighbors of point `i` (pad short rows by
/// repeating the point's own index). Normal orientation is arbitrary (sign is
/// not disambiguated); callers can flip toward a viewpoint on the CPU.
pub fn estimate_normals_gpu(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    neighbors: &[u32],
    k: u32,
) -> SpatialResult<Vec<GpuNormal>> {
    let point_count = x.len();
    if y.len() != point_count || z.len() != point_count {
        return Err(SpatialError::BufferLengthMismatch { expected: point_count, found: y.len() });
    }
    if point_count == 0 {
        return Ok(Vec::new());
    }
    if k == 0 || neighbors.len() != point_count * k as usize {
        return Err(SpatialError::InvalidArgument(format!(
            "neighbors must have point_count*k = {} entries, got {}",
            point_count * k as usize,
            neighbors.len()
        )));
    }

    let device = runtime.device();
    let queue = runtime.queue();

    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normals-x"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let y_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normals-y"),
        contents: bytemuck::cast_slice(y),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let z_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normals-z"),
        contents: bytemuck::cast_slice(z),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let neighbor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normals-neighbors"),
        contents: bytemuck::cast_slice(neighbors),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let uniform = NormalsUniform { point_count: point_count as u32, k, _pad0: 0, _pad1: 0 };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normals-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let output_len = (point_count * std::mem::size_of::<[f32; 4]>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("normals-output"),
        size: output_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("normals-shader"),
        source: wgpu::ShaderSource::Wgsl(NORMALS_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("normals-pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("normals-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: z_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: neighbor_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("normals") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("normals-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((point_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("normals-staging"),
        size: output_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("normals-readback"),
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_len);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|error| {
            SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {error}"))
        })?;

    let data = slice.get_mapped_range();
    let raw: &[[f32; 4]] = bytemuck::cast_slice(&data);
    let normals =
        raw.iter().map(|v| GpuNormal { normal: [v[0], v[1], v[2]], curvature: v[3] }).collect();
    drop(data);
    staging.unmap();

    Ok(normals)
}

#[cfg(test)]
mod tests {
    use super::estimate_normals_gpu;
    use crate::runtime::WgpuRuntime;

    #[test]
    fn planar_patch_has_vertical_normal() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        // 5x5 grid on the z=0 plane.
        let mut x: Vec<f32> = Vec::new();
        let mut y: Vec<f32> = Vec::new();
        let mut z: Vec<f32> = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                x.push(i as f32 * 0.1);
                y.push(j as f32 * 0.1);
                z.push(0.0);
            }
        }
        let n = x.len();
        let k = 8u32;

        // Brute-force k nearest neighbors per point (CPU).
        let mut neighbors = Vec::with_capacity(n * k as usize);
        for i in 0..n {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                let da = (x[a] - x[i]).powi(2) + (y[a] - y[i]).powi(2) + (z[a] - z[i]).powi(2);
                let db = (x[b] - x[i]).powi(2) + (y[b] - y[i]).powi(2) + (z[b] - z[i]).powi(2);
                da.total_cmp(&db)
            });
            for &idx in order.iter().take(k as usize) {
                neighbors.push(idx as u32);
            }
        }

        let normals =
            estimate_normals_gpu(&runtime, &x, &y, &z, &neighbors, k).expect("gpu normals");
        assert_eq!(normals.len(), n);
        for normal in &normals {
            // Normal should point along Z (up or down) and curvature ~0 on a plane.
            assert!(normal.normal[2].abs() > 0.99, "normal not vertical: {:?}", normal.normal);
            assert!(normal.curvature < 1e-3, "curvature too high: {}", normal.curvature);
        }
    }
}
