use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::runtime::WgpuRuntime;
use spatialrust_search::{build_grid, grid_bounds};

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CovUniform {
    origin: [f32; 4],
    dims: [u32; 4], // dimx, dimy, dimz, point_count
    inv_cell: f32,
    radius_sq: f32,
    epsilon: f32,
    _pad: f32,
}

/// Per-point plane-regularized covariance as 6 unique elements:
/// `[c00, c11, c22, c01, c02, c12]`.
pub type GpuCovariance = [f32; 6];

const COV_GRID_WGSL: &str = r#"
struct Params {
    origin: vec4<f32>,
    dims: vec4<u32>,
    inv_cell: f32,
    radius_sq: f32,
    epsilon: f32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read> ys: array<f32>;
@group(0) @binding(3) var<storage, read> zs: array<f32>;
@group(0) @binding(4) var<storage, read> sorted: array<u32>;
@group(0) @binding(5) var<storage, read> cell_start: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_cov: array<vec4<f32>>;

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
                for (var s = cell_start[cid]; s < cell_start[cid + 1u]; s = s + 1u) {
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

    // Too few neighbors: emit an isotropic epsilon-scaled covariance.
    if (count < 3.0) {
        out_cov[i] = vec4<f32>(params.epsilon, params.epsilon, params.epsilon, 0.0);
        // remaining elements default to zero
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
                for (var s = cell_start[cid]; s < cell_start[cid + 1u]; s = s + 1u) {
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

    // GICP plane regularization: rebuild covariance with eigenvalues (eps, 1, 1),
    // smallest eigenvalue (surface normal) -> eps.
    let eig = vec3<f32>(a[0][0], a[1][1], a[2][2]);
    var min_idx = 0u;
    if (eig[1] < eig[min_idx]) { min_idx = 1u; }
    if (eig[2] < eig[min_idx]) { min_idx = 2u; }

    var reg = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var col = 0u; col < 3u; col = col + 1u) {
        var lambda = 1.0;
        if (col == min_idx) {
            lambda = params.epsilon;
        }
        let ax = v[0][col];
        let ay = v[1][col];
        let az = v[2][col];
        reg[0] = reg[0] + lambda * ax * ax;
        reg[1] = reg[1] + lambda * ay * ay;
        reg[2] = reg[2] + lambda * az * az;
        reg[3] = reg[3] + lambda * ax * ay;
        reg[4] = reg[4] + lambda * ax * az;
        reg[5] = reg[5] + lambda * ay * az;
    }

    out_cov[i] = vec4<f32>(reg[0], reg[1], reg[2], reg[3]);
    out_cov[params.dims.w + i] = vec4<f32>(reg[4], reg[5], 0.0, 0.0);
}
"#;

/// Estimates per-point plane-regularized covariances on the GPU via a uniform
/// grid radius neighbor search.
///
/// Returns one [`GpuCovariance`] per point — the unique elements of a covariance
/// matrix whose eigenvalues have been set to `(epsilon, 1, 1)` (the GICP
/// plane-to-plane model). Grid construction (counting sort) runs on the CPU; the
/// neighbor gather, covariance, and eigen-decomposition run on the GPU.
pub fn estimate_plane_covariances_grid_gpu(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
    epsilon: f32,
) -> SpatialResult<Vec<GpuCovariance>> {
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
    let storage = wgpu::BufferUsages::STORAGE;

    let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cg-x"),
        contents: bytemuck::cast_slice(x),
        usage: storage,
    });
    let y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cg-y"),
        contents: bytemuck::cast_slice(y),
        usage: storage,
    });
    let z_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cg-z"),
        contents: bytemuck::cast_slice(z),
        usage: storage,
    });
    let sorted_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cg-sorted"),
        contents: bytemuck::cast_slice(&sorted),
        usage: storage,
    });
    let cell_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cg-cell-start"),
        contents: bytemuck::cast_slice(&cell_start),
        usage: storage,
    });
    let uniform = CovUniform {
        origin: [origin[0], origin[1], origin[2], 0.0],
        dims: [dims[0], dims[1], dims[2], point_count as u32],
        inv_cell: 1.0 / radius,
        radius_sq: radius * radius,
        epsilon,
        _pad: 0.0,
    };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cg-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Two vec4 rows per point: row0 = (c00,c11,c22,c01), row1 = (c02,c12,_,_).
    let output_len = (2 * point_count * std::mem::size_of::<[f32; 4]>()) as u64;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cg-output"),
        size: output_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cg-shader"),
        source: wgpu::ShaderSource::Wgsl(COV_GRID_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cg-pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cg-bind-group"),
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
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cg") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cg-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((point_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cg-staging"),
        size: output_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cg-rb") });
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
    let rows: &[[f32; 4]] = bytemuck::cast_slice(&data);
    let mut out = Vec::with_capacity(point_count);
    for i in 0..point_count {
        let row0 = rows[i];
        let row1 = rows[point_count + i];
        out.push([row0[0], row0[1], row0[2], row0[3], row1[0], row1[1]]);
    }
    drop(data);
    staging.unmap();

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::estimate_plane_covariances_grid_gpu;
    use crate::runtime::WgpuRuntime;

    #[test]
    fn planar_patch_covariance_is_disk() {
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
        let eps = 1e-3_f32;
        let cov = estimate_plane_covariances_grid_gpu(&runtime, &x, &y, &z, 0.25, eps)
            .expect("gpu covariances");
        assert_eq!(cov.len(), x.len());
        // For a z=0 plane, the regularized covariance ~ diag(1, 1, eps):
        // in-plane variance ~1, out-of-plane (z) ~eps.
        for c in &cov {
            let [c00, c11, c22, _c01, _c02, _c12] = *c;
            assert!((c22 - eps).abs() < 1e-2, "c22 not ~eps: {c22}");
            assert!(c00 > 0.5 && c11 > 0.5, "in-plane variance too small: {c00},{c11}");
        }
    }
}
