use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::kernels::normals_grid::{build_grid, grid_bounds};
use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;
/// Upper cap for label-propagation iterations (Jacobi min-label).
const MAX_LABEL_PROPAGATION_ITERS: u32 = 16_384;
/// Minimum iterations even for tiny grids.
const MIN_LABEL_PROPAGATION_ITERS: u32 = 128;
/// Check for convergence every N iterations (readback).
const CONVERGENCE_CHECK_INTERVAL: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GridUniform {
    origin: [f32; 4],
    dims: [u32; 4],
    inv_cell: f32,
    radius_sq: f32,
    _pad0: f32,
    _pad1: f32,
}

const EUCLIDEAN_CLUSTER_WGSL: &str = r#"
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
@group(0) @binding(6) var<storage, read> labels_in: array<u32>;
@group(0) @binding(7) var<storage, read_write> labels_out: array<u32>;

fn cell_coord(value: f32, origin: f32, inv_cell: f32, dim: u32) -> i32 {
    let c = i32(floor((value - origin) * inv_cell));
    return clamp(c, 0, i32(dim) - 1);
}

fn cell_index(cx: i32, cy: i32, cz: i32, dimx: u32, dimy: u32) -> u32 {
    return (u32(cz) * dimy + u32(cy)) * dimx + u32(cx);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let point_count = params.dims.w;
    if (i >= point_count) {
        return;
    }

    var best = labels_in[i];
    let px = xs[i];
    let py = ys[i];
    let pz = zs[i];
    let dimx = params.dims.x;
    let dimy = params.dims.y;
    let dimz = params.dims.z;

    let cx = cell_coord(px, params.origin.x, params.inv_cell, dimx);
    let cy = cell_coord(py, params.origin.y, params.inv_cell, dimy);
    let cz = cell_coord(pz, params.origin.z, params.inv_cell, dimz);

    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        let nz = cz + dz;
        if (nz < 0 || nz >= i32(dimz)) {
            continue;
        }
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            let ny = cy + dy;
            if (ny < 0 || ny >= i32(dimy)) {
                continue;
            }
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let nx = cx + dx;
                if (nx < 0 || nx >= i32(dimx)) {
                    continue;
                }
                let cell = cell_index(nx, ny, nz, dimx, dimy);
                let start = cell_start[cell];
                let end = cell_start[cell + 1u];
                for (var slot = start; slot < end; slot = slot + 1u) {
                    let j = sorted[slot];
                    let dxp = xs[j] - px;
                    let dyp = ys[j] - py;
                    let dzp = zs[j] - pz;
                    let dist_sq = dxp * dxp + dyp * dyp + dzp * dzp;
                    if (dist_sq <= params.radius_sq) {
                        best = min(best, labels_in[j]);
                    }
                }
            }
        }
    }

    labels_out[i] = best;
}
"#;

/// Connected-component labels via GPU grid label propagation.
///
/// Returns one root index per point (minimum index in each component). Callers
/// filter by cluster size and remap to sequential labels on the CPU.
pub fn euclidean_cluster_roots_gpu(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    cluster_tolerance: f32,
) -> SpatialResult<Vec<u32>> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(SpatialError::InvalidArgument("xyz arrays must have equal length".to_owned()));
    }
    let point_count = x.len();
    if point_count == 0 {
        return Ok(Vec::new());
    }
    if cluster_tolerance <= 0.0 || cluster_tolerance.is_nan() {
        return Err(SpatialError::InvalidArgument("cluster_tolerance must be positive".to_owned()));
    }

    let (origin_arr, dims) = grid_bounds(x, y, z, cluster_tolerance)?;
    let origin = [origin_arr[0], origin_arr[1], origin_arr[2], 0.0];
    let (sorted, cell_start) = build_grid(x, y, z, origin_arr, dims, cluster_tolerance);

    let device = runtime.device();
    let queue = runtime.queue();

    let x_buffer = runtime.upload_f32_storage("euclidean-cluster-x", x)?;
    let y_buffer = runtime.upload_f32_storage("euclidean-cluster-y", y)?;
    let z_buffer = runtime.upload_f32_storage("euclidean-cluster-z", z)?;
    let sorted_buffer = runtime.upload_u32_storage("euclidean-cluster-sorted", &sorted)?;
    let cell_start_buffer =
        runtime.upload_u32_storage("euclidean-cluster-cell-start", &cell_start)?;

    let labels_a: Vec<u32> = (0..point_count as u32).collect();
    let label_size = std::mem::size_of_val(labels_a.as_slice()) as u64;

    let label_buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("euclidean-cluster-labels-a"),
        contents: bytemuck::cast_slice(&labels_a),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let label_buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("euclidean-cluster-labels-b"),
        size: label_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniform = GridUniform {
        origin,
        dims: [dims[0], dims[1], dims[2], point_count as u32],
        inv_cell: 1.0 / cluster_tolerance,
        radius_sq: cluster_tolerance * cluster_tolerance,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("euclidean-cluster-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("euclidean-cluster-shader"),
        source: wgpu::ShaderSource::Wgsl(EUCLIDEAN_CLUSTER_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("euclidean-cluster-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let max_iters = label_propagation_iterations(dims, point_count);
    let mut read_from_a = true;
    let mut labels_host = labels_a;

    for iter in 0..max_iters {
        let (labels_in, labels_out) = if read_from_a {
            (&label_buffer_a, &label_buffer_b)
        } else {
            (&label_buffer_b, &label_buffer_a)
        };

        dispatch_label_pass(
            device,
            queue,
            &pipeline,
            &uniform_buffer,
            &x_buffer,
            &y_buffer,
            &z_buffer,
            &sorted_buffer,
            &cell_start_buffer,
            labels_in,
            labels_out,
            point_count,
        )?;
        read_from_a = !read_from_a;

        let is_last = iter + 1 == max_iters;
        let should_check = is_last || (iter + 1) % CONVERGENCE_CHECK_INTERVAL == 0;
        if should_check {
            let final_buffer = if read_from_a { &label_buffer_a } else { &label_buffer_b };
            let current = read_storage_u32(device, queue, final_buffer, point_count)?;
            if current == labels_host {
                recycle_cluster_buffers(
                    runtime,
                    x,
                    y,
                    z,
                    &sorted,
                    &cell_start,
                    x_buffer,
                    y_buffer,
                    z_buffer,
                    sorted_buffer,
                    cell_start_buffer,
                );
                return Ok(current);
            }
            labels_host = current;
        }
    }

    recycle_cluster_buffers(
        runtime,
        x,
        y,
        z,
        &sorted,
        &cell_start,
        x_buffer,
        y_buffer,
        z_buffer,
        sorted_buffer,
        cell_start_buffer,
    );

    Err(SpatialError::InvalidArgument(
        "gpu euclidean label propagation did not converge within iteration cap".to_owned(),
    ))
}

/// Iteration count for Jacobi min-label propagation.
///
/// Thin chains can need up to `point_count - 1` hops; dense volumes are bounded
/// by the grid span when cell size equals tolerance.
fn label_propagation_iterations(dims: [u32; 3], point_count: usize) -> u32 {
    let grid_span = dims[0].saturating_add(dims[1]).saturating_add(dims[2]);
    let path_bound = point_count.saturating_sub(1) as u32;
    grid_span.max(path_bound).max(MIN_LABEL_PROPAGATION_ITERS).min(MAX_LABEL_PROPAGATION_ITERS)
}

fn dispatch_label_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    uniform_buffer: &wgpu::Buffer,
    x_buffer: &wgpu::Buffer,
    y_buffer: &wgpu::Buffer,
    z_buffer: &wgpu::Buffer,
    sorted_buffer: &wgpu::Buffer,
    cell_start_buffer: &wgpu::Buffer,
    labels_in: &wgpu::Buffer,
    labels_out: &wgpu::Buffer,
    point_count: usize,
) -> SpatialResult<()> {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("euclidean-cluster-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: z_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: sorted_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: cell_start_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: labels_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: labels_out.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("euclidean-cluster"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("euclidean-cluster-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(point_count.div_ceil(WORKGROUP_SIZE as usize) as u32, 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}

fn recycle_cluster_buffers(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    sorted: &[u32],
    cell_start: &[u32],
    x_buffer: wgpu::Buffer,
    y_buffer: wgpu::Buffer,
    z_buffer: wgpu::Buffer,
    sorted_buffer: wgpu::Buffer,
    cell_start_buffer: wgpu::Buffer,
) {
    runtime.recycle_storage(std::mem::size_of_val(x) as u64, x_buffer);
    runtime.recycle_storage(std::mem::size_of_val(y) as u64, y_buffer);
    runtime.recycle_storage(std::mem::size_of_val(z) as u64, z_buffer);
    runtime.recycle_storage(std::mem::size_of_val(sorted) as u64, sorted_buffer);
    runtime.recycle_storage(std::mem::size_of_val(cell_start) as u64, cell_start_buffer);
}

fn read_storage_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<u32>> {
    let byte_len = std::mem::size_of_val(&[0u32; 1]) as u64 * len as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("euclidean-cluster-readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("euclidean-cluster-readback"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_len);
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
    let values: Vec<u32> = bytemuck::cast_slice(&data)[..len].to_vec();
    drop(data);
    staging.unmap();
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::label_propagation_iterations;

    #[test]
    fn iteration_count_scales_with_grid_span_and_point_count() {
        assert_eq!(label_propagation_iterations([1, 1, 1], 10), 128);
        assert_eq!(label_propagation_iterations([200, 200, 40], 460_400), 16_384);
        assert_eq!(label_propagation_iterations([1, 1, 1], 400), 400);
        assert_eq!(label_propagation_iterations([10_000, 1, 1], 500), 16_384);
    }
}
