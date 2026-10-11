
struct Params {
    point_count: u32,
    hypothesis_count: u32,
    distance_threshold: f32,
    groups: u32,
};

struct Hypothesis {
    i0: u32,
    i1: u32,
    i2: u32,
    pad: u32,
};

struct Score {
    inlier_count: u32,
    nx: f32,
    ny: f32,
    nz: f32,
    d: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read> ys: array<f32>;
@group(0) @binding(3) var<storage, read> zs: array<f32>;
@group(0) @binding(4) var<storage, read> hypotheses: array<Hypothesis>;
@group(0) @binding(5) var<storage, read_write> scores: array<Score>;

fn plane_from_indices(i0: u32, i1: u32, i2: u32) -> Score {
    let p0 = vec3<f32>(xs[i0], ys[i0], zs[i0]);
    let p1 = vec3<f32>(xs[i1], ys[i1], zs[i1]);
    let p2 = vec3<f32>(xs[i2], ys[i2], zs[i2]);
    let v1 = p1 - p0;
    let v2 = p2 - p0;
    var normal = cross(v1, v2);
    if (dot(normal, normal) < 1e-12) {
        return Score(0u, 0.0, 0.0, 0.0, 0.0);
    }
    normal = normalize(normal);
    let d = -dot(normal, p0);
    var count = 0u;
    for (var i: u32 = 0u; i < params.point_count; i = i + 1u) {
        let dist = abs(normal.x * xs[i] + normal.y * ys[i] + normal.z * zs[i] + d);
        if (dist <= params.distance_threshold) {
            count = count + 1u;
        }
    }
    return Score(count, normal.x, normal.y, normal.z, d);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var h = gid.x;
    loop {
        if h >= params.hypothesis_count { break; }
        let hyp = hypotheses[h];
        scores[h] = plane_from_indices(hyp.i0, hyp.i1, hyp.i2);
        let stride = params.groups * 256u;
        if params.hypothesis_count - h <= stride { break; }
        h += stride;
    }
}
