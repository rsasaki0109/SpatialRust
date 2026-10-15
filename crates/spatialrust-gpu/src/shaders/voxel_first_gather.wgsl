struct Params {
    cell_count: u32,
    point_count: u32,
    attribute_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_indices: array<u32>;
@group(0) @binding(2) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read> values_x: array<u32>;
@group(0) @binding(4) var<storage, read> values_y: array<u32>;
@group(0) @binding(5) var<storage, read> values_z: array<u32>;
@group(0) @binding(6) var<storage, read> values_a0: array<u32>;
@group(0) @binding(7) var<storage, read> values_a1: array<u32>;
@group(0) @binding(8) var<storage, read> values_a2: array<u32>;
@group(0) @binding(9) var<storage, read> values_a3: array<u32>;
@group(0) @binding(10) var<storage, read_write> output: array<u32>;

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
    let cells = params.cell_count;
    var first_index = 0u;
    if end > start {
        first_index = point_indices[start];
    }
    output[cell] = values_x[first_index];
    output[cells + cell] = values_y[first_index];
    output[2u * cells + cell] = values_z[first_index];
    // The scalar gather writes zero for an empty cell; multi2/multi4 read point zero.
    if end <= start && params.attribute_count == 1u {
        output[3u * cells + cell] = 0u;
    } else {
        output[3u * cells + cell] = values_a0[first_index];
    }
    if params.attribute_count > 1u {
        output[4u * cells + cell] = values_a1[first_index];
    }
    if params.attribute_count > 2u {
        output[5u * cells + cell] = values_a2[first_index];
    }
    if params.attribute_count > 3u {
        output[6u * cells + cell] = values_a3[first_index];
    }
}
