struct Params { count: u32, root: u32, bound: f32, unused: u32 }
struct Node {
    split: f32, axis: u32, left: u32, right: u32,
    start: u32, end: u32, pad0: u32, pad1: u32,
}
struct Point { xyz: vec3<f32>, index: u32 }
struct Neighbor { index: u32, distance: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> points: array<Point>;
@group(0) @binding(3) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> results: array<Neighbor>;

@compute @workgroup_size(64)
fn nearest(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count { return; }
    let q = queries[i].xyz;
    var best = Neighbor(0xffffffffu, params.bound);
    if !all(abs(q) <= vec3<f32>(3.402823466e+38)) {
        results[i] = best;
        return;
    }
    // One deferred far child per balanced-tree level. The far test is made
    // AFTER the near subtree returns, exactly as in the CPU recursive search.
    var far_nodes: array<u32, 32>;
    var plane_distances: array<f32, 32>;
    var depth = 0u;
    var current = params.root;
    loop {
        let node = nodes[current];
        if node.axis != 0xffffffffu {
            let diff = q[node.axis] - node.split;
            far_nodes[depth] = select(node.left, node.right, diff <= 0.0);
            plane_distances[depth] = diff * diff;
            depth += 1u;
            current = select(node.right, node.left, diff <= 0.0);
            continue;
        }
        for (var j = node.start; j < node.end; j += 1u) {
            let point = points[j];
            let dx = point.xyz.x - q.x;
            let dy = point.xyz.y - q.y;
            let dz = point.xyz.z - q.z;
            let distance = dx * dx + dy * dy + dz * dz;
            if distance <= params.bound && (best.index == 0xffffffffu || distance < best.distance) {
                best = Neighbor(point.index, distance);
            }
        }
        var found = false;
        loop {
            if depth == 0u { break; }
            depth -= 1u;
            let plane = plane_distances[depth];
            if select(plane < best.distance, plane <= params.bound, best.index == 0xffffffffu) {
                current = far_nodes[depth];
                found = true;
                break;
            }
        }
        if !found { break; }
    }
    results[i] = best;
}
