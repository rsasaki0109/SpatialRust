use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::kernels::normals::GpuNormal;
use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;
/// Upper bound on dense grid cells to avoid pathological memory use; callers
/// should fall back to the CPU/KD-tree path when exceeded.
const MAX_CELLS: u64 = 64_000_000;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GridUniform {
    origin: [f32; 4],
    dims: [u32; 4], // dimx, dimy, dimz, point_count
    inv_cell: f32,
    radius_sq: f32,
    _pad0: f32,
    _pad1: f32,
}

const NORMALS_GRID_WGSL: &str = r#"
struct Params {
    origin: vec4<f32>,
    dims: vec4<u32>,
    inv_cell: f32,
    radius_sq: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read> ys: array<f32>;
@group(0) @binding(3) var<storage, read> zs: array<f32>;
@group(0) @binding(4) var<storage, read> sorted: array<u32>;
@group(0) @binding(5) var<storage, read> cell_start: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_normals: array<vec4<f32>>;

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

fn cell_coord(value: f32, origin: f32, inv_cell: f32, dim: u32) -> i32 {
    let c = i32(floor((value - origin) * inv_cell));
    return clamp(c, 0, i32(dim) - 1);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.dims.w) {
        return;
    }
    let px = xs[i];
    let py = ys[i];
    let pz = zs[i];
    let dimx = params.dims.x;
    let dimy = params.dims.y;
    let dimz = params.dims.z;

    let cx = cell_coord(px, params.origin.x, params.inv_cell, dimx);
    let cy = cell_coord(py, params.origin.y, params.inv_cell, dimy);
    let cz = cell_coord(pz, params.origin.z, params.inv_cell, dimz);

    // First pass: mean over radius neighbors across the 27 adjacent cells.
    var mean = vec3<f32>(0.0, 0.0, 0.0);
    var count = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        let nz = cz + dz;
        if (nz < 0 || nz >= i32(dimz)) { continue; }
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let ny = cy + dy;
            if (ny < 0 || ny >= i32(dimy)) { continue; }
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nx = cx + dx;
                if (nx < 0 || nx >= i32(dimx)) { continue; }
                let cid = (u32(nz) * dimy + u32(ny)) * dimx + u32(nx);
                let begin = cell_start[cid];
                let end = cell_start[cid + 1u];
                for (var s = begin; s < end; s = s + 1u) {
                    let j = sorted[s];
                    let d = vec3<f32>(xs[j] - px, ys[j] - py, zs[j] - pz);
                    if (dot(d, d) <= params.radius_sq) {
                        mean = mean + vec3<f32>(xs[j], ys[j], zs[j]);
                        count = count + 1.0;
                    }
                }
            }
        }
    }

    if (count < 3.0) {
        out_normals[i] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
        return;
    }
    mean = mean / count;

    var c00 = 0.0; var c11 = 0.0; var c22 = 0.0;
    var c01 = 0.0; var c02 = 0.0; var c12 = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        let nz = cz + dz;
        if (nz < 0 || nz >= i32(dimz)) { continue; }
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let ny = cy + dy;
            if (ny < 0 || ny >= i32(dimy)) { continue; }
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nx = cx + dx;
                if (nx < 0 || nx >= i32(dimx)) { continue; }
                let cid = (u32(nz) * dimy + u32(ny)) * dimx + u32(nx);
                let begin = cell_start[cid];
                let end = cell_start[cid + 1u];
                for (var s = begin; s < end; s = s + 1u) {
                    let j = sorted[s];
                    let p = vec3<f32>(xs[j], ys[j], zs[j]);
                    let rel = p - vec3<f32>(px, py, pz);
                    if (dot(rel, rel) <= params.radius_sq) {
                        let dd = p - mean;
                        c00 = c00 + dd.x * dd.x;
                        c11 = c11 + dd.y * dd.y;
                        c22 = c22 + dd.z * dd.z;
                        c01 = c01 + dd.x * dd.y;
                        c02 = c02 + dd.x * dd.z;
                        c12 = c12 + dd.y * dd.z;
                    }
                }
            }
        }
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
    let len = max(sqrt(dot(normal, normal)), 1e-20);
    let unit = normal / len;
    let trace = eig[0] + eig[1] + eig[2];
    var curvature = 0.0;
    if (trace > 0.0) {
        curvature = eig[min_idx] / trace;
    }
    out_normals[i] = vec4<f32>(unit.x, unit.y, unit.z, curvature);
}
"#;

/// Estimates per-point normals and curvature with a fully GPU radius neighbor
/// search over a uniform grid.
///
/// The grid (cell size = `radius`) is built on the CPU with a counting sort
/// (O(n)); the per-point neighbor gather, covariance, and eigen-decomposition
/// all run on the GPU. Returns `SpatialError::InvalidArgument` when the bounding
/// grid would exceed an internal cell cap (caller should fall back to the CPU
/// KD-tree path).
pub fn estimate_normals_grid_gpu(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
) -> SpatialResult<Vec<GpuNormal>> {
    let point_count = x.len();
    if y.len() != point_count || z.len() != point_count {
        return Err(SpatialError::BufferLengthMismatch { expected: point_count, found: y.len() });
    }
    if point_count == 0 {
        return Ok(Vec::new());
    }
    if radius <= 0.0 || radius.is_nan() {
        return Err(SpatialError::InvalidArgument("grid radius must be positive".to_owned()));
    }

    let (origin, dims) = grid_bounds(x, y, z, radius)?;
    let (sorted, cell_start) = build_grid(x, y, z, origin, dims, radius);

    let device = runtime.device();
    let queue = runtime.queue();
    let inv_cell = 1.0 / radius;

    let storage = wgpu::BufferUsages::STORAGE;
    let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ng-x"),
        contents: bytemuck::cast_slice(x),
        usage: storage,
    });
    let y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ng-y"),
        contents: bytemuck::cast_slice(y),
        usage: storage,
    });
    let z_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ng-z"),
        contents: bytemuck::cast_slice(z),
        usage: storage,
    });
    let sorted_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ng-sorted"),
        contents: bytemuck::cast_slice(&sorted),
        usage: storage,
    });
    let cell_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ng-cell-start"),
        contents: bytemuck::cast_slice(&cell_start),
        usage: storage,
    });
    let uniform = GridUniform {
        origin: [origin[0], origin[1], origin[2], 0.0],
        dims: [dims[0], dims[1], dims[2], point_count as u32],
        inv_cell,
        radius_sq: radius * radius,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ng-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let output_len = (point_count * std::mem::size_of::<[f32; 4]>()) as u64;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ng-output"),
        size: output_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ng-shader"),
        source: wgpu::ShaderSource::Wgsl(NORMALS_GRID_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ng-pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ng-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: z_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: sorted_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: cell_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: output_buf.as_entire_binding() },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ng") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ng-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((point_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ng-staging"),
        size: output_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ng-rb") });
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_len);
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

pub(crate) fn grid_bounds(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
) -> SpatialResult<([f32; 3], [u32; 3])> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for index in 0..x.len() {
        for (axis, value) in [x[index], y[index], z[index]].into_iter().enumerate() {
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    let inv_cell = 1.0 / radius;
    let mut dims = [0u32; 3];
    for axis in 0..3 {
        let span = ((max[axis] - min[axis]) * inv_cell).floor() as i64 + 1;
        dims[axis] = span.max(1) as u32;
    }
    let cells = dims[0] as u64 * dims[1] as u64 * dims[2] as u64;
    if cells > MAX_CELLS {
        return Err(SpatialError::InvalidArgument(format!(
            "grid would need {cells} cells (cap {MAX_CELLS}); use a larger radius or the CPU path"
        )));
    }
    Ok((min, dims))
}

/// Returns whether a uniform grid with the given cell size fits the GPU cell cap.
pub fn uniform_grid_fits(x: &[f32], y: &[f32], z: &[f32], cell_size: f32) -> bool {
    grid_bounds(x, y, z, cell_size).is_ok()
}

/// Counting-sort points into grid cells, returning sorted indices and CSR offsets.
pub(crate) fn build_grid(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: [f32; 3],
    dims: [u32; 3],
    radius: f32,
) -> (Vec<u32>, Vec<u32>) {
    let inv_cell = 1.0 / radius;
    let n = x.len();
    let num_cells = dims[0] as usize * dims[1] as usize * dims[2] as usize;

    let cell_of = |index: usize| -> usize {
        let cx = (((x[index] - origin[0]) * inv_cell).floor() as i64).clamp(0, dims[0] as i64 - 1)
            as usize;
        let cy = (((y[index] - origin[1]) * inv_cell).floor() as i64).clamp(0, dims[1] as i64 - 1)
            as usize;
        let cz = (((z[index] - origin[2]) * inv_cell).floor() as i64).clamp(0, dims[2] as i64 - 1)
            as usize;
        (cz * dims[1] as usize + cy) * dims[0] as usize + cx
    };

    let mut counts = vec![0u32; num_cells + 1];
    for index in 0..n {
        counts[cell_of(index)] += 1;
    }
    // Prefix sum -> cell_start (CSR offsets).
    let mut acc = 0u32;
    for slot in counts.iter_mut() {
        let c = *slot;
        *slot = acc;
        acc += c;
    }
    let cell_start = counts; // now offsets, length num_cells+1, last == n

    let mut cursor = cell_start.clone();
    let mut sorted = vec![0u32; n];
    for index in 0..n {
        let cell = cell_of(index);
        let slot = cursor[cell];
        sorted[slot as usize] = index as u32;
        cursor[cell] = slot + 1;
    }
    (sorted, cell_start)
}

#[cfg(test)]
mod tests {
    use super::estimate_normals_grid_gpu;
    use crate::runtime::WgpuRuntime;

    #[test]
    fn planar_patch_has_vertical_normal() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let mut x: Vec<f32> = Vec::new();
        let mut y: Vec<f32> = Vec::new();
        let mut z: Vec<f32> = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                x.push(i as f32 * 0.1);
                y.push(j as f32 * 0.1);
                z.push(0.0);
            }
        }
        let normals = estimate_normals_grid_gpu(&runtime, &x, &y, &z, 0.25).expect("grid normals");
        assert_eq!(normals.len(), x.len());
        for normal in &normals {
            assert!(normal.normal[2].abs() > 0.99, "normal not vertical: {:?}", normal.normal);
            assert!(normal.curvature < 1e-3, "curvature too high: {}", normal.curvature);
        }
    }
}
