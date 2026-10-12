use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};

use super::point_estimators::{self, Kind};
use crate::kernels::normals::GpuNormal;
use crate::runtime::WgpuRuntime;

pub use spatialrust_search::uniform_grid_fits;

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

pub(crate) const NORMALS_GRID_WGSL: &str = r#"
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
    // Adjacent x cells occupy one contiguous counting-sort interval.
    let first_x = u32(max(cx - 1, 0));
    let after_x = u32(min(cx + 1, i32(dimx) - 1)) + 1u;

    // First pass: mean over radius neighbors in up to nine contiguous cell rows.
    var mean = vec3<f32>(0.0, 0.0, 0.0);
    var count = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        let nz = cz + dz;
        if (nz < 0 || nz >= i32(dimz)) { continue; }
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let ny = cy + dy;
            if (ny < 0 || ny >= i32(dimy)) { continue; }
            let row = (u32(nz) * dimy + u32(ny)) * dimx;
            let begin = cell_start[row + first_x];
            let end = cell_start[row + after_x];
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
            let row = (u32(nz) * dimy + u32(ny)) * dimx;
            let begin = cell_start[row + first_x];
            let end = cell_start[row + after_x];
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
/// (O(n + cells)); the per-point neighbor gather, covariance, and eigen-decomposition
/// all run on the GPU. Returns `SpatialError::InvalidArgument` when the bounding
/// grid would exceed an internal cell cap (caller should fall back to the CPU
/// KD-tree path).
/// Nonempty inputs require finite coordinates and a positive radius with a
/// finite reciprocal; positive infinity retains the unbounded one-cell grid.
pub fn estimate_normals_grid_gpu(
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
        point_estimators::prepare_grid(runtime, Kind::RadiusNormals, x, y, z, radius)?;
    let params = GridUniform {
        origin: [origin[0], origin[1], origin[2], 0.0],
        dims: [dims[0], dims[1], dims[2], point_count as u32],
        inv_cell: 1.0 / radius,
        radius_sq: radius * radius,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    point_estimators::run(
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
