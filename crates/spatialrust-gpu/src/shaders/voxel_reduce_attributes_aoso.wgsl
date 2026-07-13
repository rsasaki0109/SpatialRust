struct Params {
    cell_count: u32,
    point_count: u32,
    stride: u32,
    first_mode: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_indices: array<u32>;
@group(0) @binding(2) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read> records: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_index = global_id.x;
    let output_count = params.cell_count * params.stride;
    if (output_index >= output_count) {
        return;
    }
    let cell = output_index / params.stride;
    let field = output_index % params.stride;
    let start = cell_starts[cell];
    let end = select(params.point_count, cell_starts[cell + 1u], cell + 1u < params.cell_count);
    if (start >= end) {
        output[output_index] = 0.0;
        return;
    }
    if (params.first_mode != 0u) {
        output[output_index] = records[point_indices[start] * params.stride + field];
        return;
    }
    var sum = 0.0;
    for (var index = start; index < end; index = index + 1u) {
        sum = sum + records[point_indices[index] * params.stride + field];
    }
    output[output_index] = sum / f32(end - start);
}
