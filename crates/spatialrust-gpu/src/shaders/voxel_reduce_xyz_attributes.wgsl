struct Params {
    cell_count: u32,
    point_count: u32,
    attribute_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_indices: array<u32>;
@group(0) @binding(2) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read> values_x: array<f32>;
@group(0) @binding(4) var<storage, read> values_y: array<f32>;
@group(0) @binding(5) var<storage, read> values_z: array<f32>;
@group(0) @binding(6) var<storage, read> values_a0: array<f32>;
@group(0) @binding(7) var<storage, read> values_a1: array<f32>;
@group(0) @binding(8) var<storage, read> values_a2: array<f32>;
@group(0) @binding(9) var<storage, read> values_a3: array<f32>;
@group(0) @binding(10) var<storage, read_write> output: array<f32>;

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
    let cells = params.cell_count;
    if count == 0u {
        for (var channel = 0u; channel < 3u + params.attribute_count; channel = channel + 1u) {
            output[channel * cells + cell] = 0.0;
        }
        return;
    }

    var sum_x = 0.0;
    var sum_y = 0.0;
    var sum_z = 0.0;
    var sum_a0 = 0.0;
    var sum_a1 = 0.0;
    var sum_a2 = 0.0;
    var sum_a3 = 0.0;
    for (var j = 0u; j < count; j = j + 1u) {
        let index = point_indices[start + j];
        sum_x = sum_x + values_x[index];
        sum_y = sum_y + values_y[index];
        sum_z = sum_z + values_z[index];
        sum_a0 = sum_a0 + values_a0[index];
        if params.attribute_count > 1u {
            sum_a1 = sum_a1 + values_a1[index];
        }
        if params.attribute_count > 2u {
            sum_a2 = sum_a2 + values_a2[index];
        }
        if params.attribute_count > 3u {
            sum_a3 = sum_a3 + values_a3[index];
        }
    }
    let inv = 1.0 / f32(count);
    output[cell] = sum_x * inv;
    output[cells + cell] = sum_y * inv;
    output[2u * cells + cell] = sum_z * inv;
    // Retain the scalar attribute kernel's division, independently of XYZ's inverse.
    output[3u * cells + cell] = sum_a0 / f32(count);
    if params.attribute_count > 1u {
        output[4u * cells + cell] = sum_a1 / f32(count);
    }
    if params.attribute_count > 2u {
        output[5u * cells + cell] = sum_a2 / f32(count);
    }
    if params.attribute_count > 3u {
        output[6u * cells + cell] = sum_a3 / f32(count);
    }
}
