
struct Params {
    origin: vec4<f32>,
    dims: vec4<u32>,
    inv_cell: f32,
    radius_sq: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
struct SparseKey { ix: i32, iy: i32, iz: i32, pad: i32, }
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> keys: array<SparseKey>;
@group(0) @binding(3) var<storage, read> sorted: array<u32>;
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;
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

fn cell_coord(value: f32, origin: f32, inv_cell: f32, dim: u32) -> i32 { return i32(floor((value - origin) * inv_cell)); }
fn key_less(key: SparseKey, x: i32, y: i32, z: i32) -> bool {
    if (key.ix != x) { return key.ix < x; }
    if (key.iy != y) { return key.iy < y; }
    return key.iz < z;
}
fn find_cell(x: i32, y: i32, z: i32, count: u32) -> i32 {
    var low = 0u; var high = count;
    loop { if (low >= high) { break; } let mid = low + (high-low)/2u; if (key_less(keys[mid],x,y,z)) { low=mid+1u; } else { high=mid; } }
    if (low < count) { let key=keys[low]; if (key.ix==x && key.iy==y && key.iz==z) { return i32(low); } }
    return -1;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.dims.w) {
        return;
    }
    let px = positions[i * 3u];
    let py = positions[i * 3u + 1u];
    let pz = positions[i * 3u + 2u];
    let cell_count = params.dims.x;

    let cx = cell_coord(px, params.origin.x, params.inv_cell, cell_count);
    let cy = cell_coord(py, params.origin.y, params.inv_cell, cell_count);
    let cz = cell_coord(pz, params.origin.z, params.inv_cell, cell_count);

    // First pass: mean over radius neighbors across the 27 adjacent cells.
    var mean = vec3<f32>(0.0, 0.0, 0.0);
    var count = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        let nz = cz + dz;

        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let ny = cy + dy;

            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nx = cx + dx;

                let found = find_cell(nx, ny, nz, cell_count);
                if (found < 0) { continue; }
                let cid = u32(found);
                let begin = cell_start[cid];
                let end = select(params.dims.w, cell_start[cid + 1u], cid + 1u < cell_count);
                for (var s = begin; s < end; s = s + 1u) {
                    let j = sorted[s];
                    let d = vec3<f32>(positions[j * 3u] - px, positions[j * 3u + 1u] - py, positions[j * 3u + 2u] - pz);
                    if (dot(d, d) <= params.radius_sq) {
                        mean = mean + vec3<f32>(positions[j * 3u], positions[j * 3u + 1u], positions[j * 3u + 2u]);
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

        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let ny = cy + dy;

            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nx = cx + dx;

                let found = find_cell(nx, ny, nz, cell_count);
                if (found < 0) { continue; }
                let cid = u32(found);
                let begin = cell_start[cid];
                let end = select(params.dims.w, cell_start[cid + 1u], cid + 1u < cell_count);
                for (var s = begin; s < end; s = s + 1u) {
                    let j = sorted[s];
                    let p = vec3<f32>(positions[j * 3u], positions[j * 3u + 1u], positions[j * 3u + 2u]);
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
