use super::*;
use crate::kernels::{
    estimate_normals_grid_gpu, estimate_plane_covariances_grid_gpu, GpuCovariance, GpuNormal,
};
use bytemuck::{Pod, Zeroable};

// Pinned host replay changes only the CPU builder. Both modes use the same
// current runtime-owned pipelines, shaders, scratch executor and readback.
mod previous_grid {
    include!("../../tests/fixtures/grid_prefix_serial.rs");
}

fn fixture(n: usize, large: bool) -> [Vec<f32>; 3] {
    let dims = if large { [256_usize, 256, 128] } else { [256, 128, 128] };
    let cells = dims.into_iter().product::<usize>();
    let mut xyz: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(n));
    for i in 0..n {
        let cell = match i {
            0 => 0,
            1 => cells - 1,
            _ => i.wrapping_mul(104729).wrapping_add(i / 3) % cells,
        };
        let c = [cell % dims[0], cell / dims[0] % dims[1], cell / (dims[0] * dims[1])];
        for axis in 0..3 {
            let offset = if i % 3 == 0 { 0.0 } else { 0.5 };
            xyz[axis].push(c[axis] as f32 + offset - 4.0);
        }
    }
    xyz
}

fn normal_words(output: Vec<GpuNormal>) -> Vec<u32> {
    output
        .into_iter()
        .flat_map(|n| {
            [
                n.normal[0].to_bits(),
                n.normal[1].to_bits(),
                n.normal[2].to_bits(),
                n.curvature.to_bits(),
            ]
        })
        .collect()
}

#[test]
fn parallel_cpu_prefix_preserves_large_grid_gpu_output_words() {
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let mut words = 0;
    for n in [17_usize, 257, 4097, 65537] {
        for large in [false, true] {
            let xyz = fixture(n, large);
            let actual = normal_words(
                estimate_normals_grid_gpu(&runtime, &xyz[0], &xyz[1], &xyz[2], 1.0).unwrap(),
            );
            let expected = normal_words(
                previous_normals_grid(&runtime, &xyz[0], &xyz[1], &xyz[2], 1.0).unwrap(),
            );
            assert_eq!(actual, expected);
            words += actual.len();
            for epsilon in [0.001, -0.0] {
                let actual: Vec<_> = estimate_plane_covariances_grid_gpu(
                    &runtime, &xyz[0], &xyz[1], &xyz[2], 1.0, epsilon,
                )
                .unwrap()
                .into_iter()
                .flat_map(|c| c.map(f32::to_bits))
                .collect();
                let expected: Vec<_> =
                    previous_covariances_grid(&runtime, &xyz[0], &xyz[1], &xyz[2], 1.0, epsilon)
                        .unwrap()
                        .into_iter()
                        .flat_map(|c| c.map(f32::to_bits))
                        .collect();
                assert_eq!(actual, expected);
                words += actual.len();
            }
        }
    }
    eprintln!("GRID_PREFIX_GPU verified {words} exact large-grid output words with shared current shaders.");
}

#[path = "grid_prefix_benchmark.rs"]
mod benchmark;

fn prepare_previous(
    runtime: &WgpuRuntime,
    kind: Kind,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
) -> SpatialResult<Grid> {
    let limits = runtime.device().limits();
    let n = x.len() as u64;
    // Reject point/output limits before allocating the CPU counting-sort grid.
    plan(kind, n, &[n, 2], &limits)?;
    let (origin, dims) = spatialrust_search::grid_bounds(x, y, z, radius)?;
    let mut cells = 1_u64;
    for dim in dims {
        if dim == 0 || dim > i32::MAX as u32 {
            return Err(SpatialError::InvalidArgument(
                "GPU grid dimensions must fit positive i32 coordinates".to_owned(),
            ));
        }
        cells = cells.checked_mul(u64::from(dim)).ok_or_else(|| {
            SpatialError::InvalidArgument("GPU grid cell count overflows".to_owned())
        })?;
    }
    let cell_entries = cells.checked_add(1).ok_or_else(|| {
        SpatialError::InvalidArgument("GPU grid sentinel count overflows".to_owned())
    })?;
    plan(kind, n, &[n, cell_entries], &limits)?;
    let (sorted, cell_start) = previous_grid::build_grid(x, y, z, origin, dims, radius);
    Ok((origin, dims, sorted, cell_start))
}

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

fn previous_normals_grid(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
) -> SpatialResult<Vec<GpuNormal>> {
    let point_count = x.len();
    if y.len() != point_count {
        return Err(SpatialError::BufferLengthMismatch { expected: point_count, found: y.len() });
    }
    if z.len() != point_count {
        return Err(SpatialError::BufferLengthMismatch { expected: point_count, found: z.len() });
    }
    if point_count == 0 {
        return Ok(Vec::new());
    }
    if radius <= 0.0 || radius.is_nan() {
        return Err(SpatialError::InvalidArgument("grid radius must be positive".to_owned()));
    }
    let (origin, dims, sorted, cell_start) =
        prepare_previous(runtime, Kind::RadiusNormals, x, y, z, radius)?;
    let params = GridUniform {
        origin: [origin[0], origin[1], origin[2], 0.0],
        dims: [dims[0], dims[1], dims[2], point_count as u32],
        inv_cell: 1.0 / radius,
        radius_sq: radius * radius,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    run(
        runtime,
        Kind::RadiusNormals,
        point_count,
        bytemuck::bytes_of(&params),
        &[
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            bytemuck::cast_slice(z),
            bytemuck::cast_slice(&sorted),
            bytemuck::cast_slice(&cell_start),
        ],
        |rows| {
            rows.iter().map(|r| GpuNormal { normal: [r[0], r[1], r[2]], curvature: r[3] }).collect()
        },
    )
}

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

fn previous_covariances_grid(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
    epsilon: f32,
) -> SpatialResult<Vec<GpuCovariance>> {
    let point_count = x.len();
    if y.len() != point_count {
        return Err(SpatialError::BufferLengthMismatch { expected: point_count, found: y.len() });
    }
    if z.len() != point_count {
        return Err(SpatialError::BufferLengthMismatch { expected: point_count, found: z.len() });
    }
    if point_count == 0 {
        return Ok(Vec::new());
    }
    if radius <= 0.0 || radius.is_nan() {
        return Err(SpatialError::InvalidArgument("grid radius must be positive".to_owned()));
    }
    let (origin, dims, sorted, cell_start) =
        prepare_previous(runtime, Kind::PlaneCovariances, x, y, z, radius)?;
    let params = CovUniform {
        origin: [origin[0], origin[1], origin[2], 0.0],
        dims: [dims[0], dims[1], dims[2], point_count as u32],
        inv_cell: 1.0 / radius,
        radius_sq: radius * radius,
        epsilon,
        _pad: 0.0,
    };
    run(
        runtime,
        Kind::PlaneCovariances,
        point_count,
        bytemuck::bytes_of(&params),
        &[
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            bytemuck::cast_slice(z),
            bytemuck::cast_slice(&sorted),
            bytemuck::cast_slice(&cell_start),
        ],
        |rows| {
            (0..point_count)
                .map(|i| {
                    let a = rows[i];
                    let b = rows[point_count + i];
                    [a[0], a[1], a[2], a[3], b[0], b[1]]
                })
                .collect()
        },
    )
}
