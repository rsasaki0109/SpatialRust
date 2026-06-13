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
@group(0) @binding(6) var<storage, read> values_a0: array<f32>;
@group(0) @binding(7) var<storage, read> values_a1: array<f32>;
@group(0) @binding(8) var<storage, read> values_a2: array<f32>;
@group(0) @binding(9) var<storage, read> values_a3: array<f32>;
@group(0) @binding(10) var<storage, read_write> packed_output: array<f32>;

fn first_point_index(cell: u32) -> u32 {
    let start = cell_starts[cell];
    let end = select(
        params.point_count,
        cell_starts[cell + 1u],
        cell + 1u < params.cell_count,
    );
    if end <= start {
        return 0u;
    }
    return point_indices[start];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if cell >= params.cell_count {
        return;
    }

    let index = first_point_index(cell);
    let cells = params.cell_count;
    packed_output[cell] = values_x[index];
    packed_output[cells + cell] = values_y[index];
    packed_output[2u * cells + cell] = values_z[index];
    packed_output[3u * cells + cell] = values_a0[index];
    packed_output[4u * cells + cell] = values_a1[index];
    packed_output[5u * cells + cell] = values_a2[index];
    packed_output[6u * cells + cell] = values_a3[index];
}
