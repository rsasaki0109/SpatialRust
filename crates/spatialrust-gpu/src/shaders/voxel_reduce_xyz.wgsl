struct Params {
    cell_count: u32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_indices: array<u32>;
@group(0) @binding(2) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read> values_x: array<f32>;
@group(0) @binding(4) var<storage, read> values_y: array<f32>;
@group(0) @binding(5) var<storage, read> values_z: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(8) var<storage, read_write> output_z: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if cell >= params.cell_count {
        return;
    }

    let start = cell_starts[cell];
    let end = select(
        params.point_count,
        cell_starts[cell + 1u],
        cell + 1u < params.cell_count,
    );
    let count = end - start;
    if count == 0u {
        output_x[cell] = 0.0;
        output_y[cell] = 0.0;
        output_z[cell] = 0.0;
        return;
    }

    var sum_x = 0.0;
    var sum_y = 0.0;
    var sum_z = 0.0;
    for (var j = 0u; j < count; j = j + 1u) {
        let index = point_indices[start + j];
        sum_x = sum_x + values_x[index];
        sum_y = sum_y + values_y[index];
        sum_z = sum_z + values_z[index];
    }
    let inv = 1.0 / f32(count);
    output_x[cell] = sum_x * inv;
    output_y[cell] = sum_y * inv;
    output_z[cell] = sum_z * inv;
}
