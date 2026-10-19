struct Params { count: u32, root: u32, capacity: u32, bound: f32 }
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
// Row: stored_count, total_count, then capacity pairs of (index, distance bits).
@group(0) @binding(4) var<storage, read_write> results: array<u32>;

fn search(i: u32, radius: bool) {
    let q = queries[i].xyz;
    let base = i * (2u + 2u * params.capacity);
    var best: array<Neighbor, ACCUMULATOR_SIZE>;
    var count = 0u;
    var total = 0u;
    var worst = 0.0;
    var worst_index = 0u;
    var far_nodes: array<u32, 32>;
    var planes: array<f32, 32>;
    var depth = 0u;
    var current = params.root;
    loop {
        let node = nodes[current];
        if node.axis != 0xffffffffu {
            let diff = q[node.axis] - node.split;
            far_nodes[depth] = select(node.left, node.right, diff <= 0.0);
            planes[depth] = diff * diff;
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
            if radius {
                if distance <= params.bound {
                    total += 1u;
                    if count < params.capacity {
                        results[base + 2u + count * 2u] = point.index;
                        results[base + 3u + count * 2u] = bitcast<u32>(distance);
                        count += 1u;
                    }
                }
            } else {
                if count < params.capacity {
                    best[count] = Neighbor(point.index, distance);
                    count += 1u;
                    if count == 1u || distance > worst {
                        worst = distance;
                        worst_index = count - 1u;
                    }
                } else if distance < worst {
                    best[worst_index] = Neighbor(point.index, distance);
                    worst = best[0].distance;
                    worst_index = 0u;
                    for (var k = 1u; k < count; k += 1u) {
                        if best[k].distance > worst {
                            worst = best[k].distance;
                            worst_index = k;
                        }
                    }
                }
            }
        }
        var found = false;
        loop {
            if depth == 0u { break; }
            depth -= 1u;
            let visit = select(count < params.capacity || planes[depth] < worst,
                planes[depth] <= params.bound, radius);
            if visit {
                current = far_nodes[depth];
                found = true;
                break;
            }
        }
        if !found { break; }
    }
    if !radius {
        // Stable distance sort matches the CPU accumulator's final sort_by,
        // including ties between retained/replaced accumulator slots.
        for (var k = 1u; k < count; k += 1u) {
            let candidate = best[k];
            var j = k;
            loop {
                if j == 0u { break; }
                if best[j - 1u].distance <= candidate.distance { break; }
                best[j] = best[j - 1u];
                j -= 1u;
            }
            best[j] = candidate;
        }
        for (var k = 0u; k < count; k += 1u) {
            results[base + 2u + k * 2u] = best[k].index;
            results[base + 3u + k * 2u] = bitcast<u32>(best[k].distance);
        }
        total = count;
    }
    results[base] = count;
    results[base + 1u] = total;
}

@compute @workgroup_size(64)
fn nearest_k(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.count { search(id.x, false); }
}
@compute @workgroup_size(64)
fn radius_neighbors(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.count { search(id.x, true); }
}
