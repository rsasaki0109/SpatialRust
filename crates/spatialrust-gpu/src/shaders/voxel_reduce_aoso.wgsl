struct Params {
    cell_count: u32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_indices: array<u32>;
@group(0) @binding(2) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read> positions: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.cell_count) {
        return;
    }
    let start = cell_starts[cell];
    let end = select(params.point_count, cell_starts[cell + 1u], cell + 1u < params.cell_count);
    let count = end - start;
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var j = start; j < end; j = j + 1u) {
        let base = point_indices[j] * 3u;
        sum = sum + vec3<f32>(positions[base], positions[base + 1u], positions[base + 2u]);
    }
    let centroid = select(vec3<f32>(0.0), sum / f32(count), count > 0u);
    output[cell] = vec4<f32>(centroid, 0.0);
}
