struct Params {
    cell_count: u32,
    point_count: u32,
    channel_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_indices: array<u32>;
@group(0) @binding(2) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read> values_c0: array<u32>;
@group(0) @binding(4) var<storage, read> values_c1: array<u32>;
@group(0) @binding(5) var<storage, read> values_c2: array<u32>;
@group(0) @binding(6) var<storage, read> values_c3: array<u32>;
@group(0) @binding(7) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if cell >= params.cell_count {
        return;
    }
    let start = cell_starts[cell];
    let end = select(params.point_count, cell_starts[cell + 1u], cell + 1u < params.cell_count);
    var index = 0u;
    if end > start {
        index = point_indices[start];
    }
    output[cell] = values_c0[index];
    if params.channel_count >= 2u {
        output[params.cell_count + cell] = values_c1[index];
    }
    if params.channel_count >= 3u {
        output[2u * params.cell_count + cell] = values_c2[index];
    }
    if params.channel_count >= 4u {
        output[3u * params.cell_count + cell] = values_c3[index];
    }
}
