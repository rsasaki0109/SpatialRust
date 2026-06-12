struct Params {
    cell_count: u32,
    point_count: u32,
    channel_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_indices: array<u32>;
@group(0) @binding(2) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read> values_c0: array<f32>;
@group(0) @binding(4) var<storage, read> values_c1: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_c0: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_c1: array<f32>;

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
    if params.channel_count >= 1u {
        output_c0[cell] = values_c0[index];
    }
    if params.channel_count >= 2u {
        output_c1[cell] = values_c1[index];
    }
}
