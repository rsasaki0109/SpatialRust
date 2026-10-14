struct Params { count: u32, unused0: u32, unused1: u32, unused2: u32 }
struct PointBits { x: u32, y: u32, z: u32, original: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> axes: array<u32>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read_write> points: array<PointBits>;
@group(0) @binding(4) var<storage, read_write> queries: array<PointBits>;

fn store_point(i: u32) {
    let original = order[i];
    points[i] = PointBits(axes[original], axes[params.count + original], axes[2u * params.count + original], original);
}

// Copy coordinate words as integers so zero signs and subnormals are retained.
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    for (var i = id.x; i < params.count; i += groups.x * 64u) {
        store_point(i);
    }
}

// The same uploaded axes also produce queries in original input order.
@compute @workgroup_size(64)
fn gather_with_queries(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    for (var i = id.x; i < params.count; i += groups.x * 64u) {
        store_point(i);
        queries[i] = PointBits(axes[i], axes[params.count + i], axes[2u * params.count + i], 0u);
    }
}
