
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
    // The count field temporarily marks non-degenerate models in shared memory.
    return Score(1u, normal.x, normal.y, normal.z, d);
}

var<workgroup> plane: Score;
var<workgroup> counts: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    var h = group.x;
    loop {
        if h >= params.hypothesis_count { break; }
        if lane == 0u {
            let hyp = hypotheses[h];
            plane = plane_from_indices(hyp.i0, hyp.i1, hyp.i2);
        }
        workgroupBarrier();
        var count = 0u;
        if plane.inlier_count != 0u {
            var i = lane;
            loop {
                if i >= params.point_count { break; }
                let dist = abs(plane.nx * xs[i] + plane.ny * ys[i] + plane.nz * zs[i] + plane.d);
                if dist <= params.distance_threshold { count += 1u; }
                if params.point_count - i <= 256u { break; }
                i += 256u;
            }
        }
        counts[lane] = count;
        workgroupBarrier();
        // Four disjoint source blocks per stage; only the first block is written.
        // Integer partial sums stay bounded by point_count, independent of order.
        for (var stride = 64u; stride != 0u; stride /= 4u) {
            if lane < stride {
                counts[lane] += counts[lane + stride]
                    + counts[lane + 2u * stride] + counts[lane + 3u * stride];
            }
            workgroupBarrier();
        }
        if lane == 0u {
            scores[h] = Score(counts[0], plane.nx, plane.ny, plane.nz, plane.d);
        }
        // Every lane must finish reading the model before the next hypothesis.
        workgroupBarrier();
        if params.hypothesis_count - h <= params.groups { break; }
        h += params.groups;
    }
}
