struct Params {
    cell_count: u32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<u32>;
@group(0) @binding(3) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;

fn cell_value(cell: u32) -> u32 {
    let start = cell_starts[cell];
    let end = select(
        params.point_count,
        cell_starts[cell + 1u],
        cell + 1u < params.cell_count,
    );
    let count = end - start;
    if count == 0u {
        return 0u;
    }

    var sum = 0.0;
    for (var j = 0u; j < count; j = j + 1u) {
        let index = point_indices[start + j];
        let word_index = index >> 2u;
        let shift = (index & 3u) * 8u;
        let value = (values[word_index] >> shift) & 0xFFu;
        sum = sum + f32(value);
    }
    return u32(round(sum / f32(count))) & 0xFFu;
}

// Every lane writes scratch, including inactive lanes in the final workgroup.
var<workgroup> cell_values: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    var value = 0u;
    if global_id.x < params.cell_count {
        value = cell_value(global_id.x);
    }
    cell_values[local_index] = value;
    workgroupBarrier();

    // The first 64 lanes each own one complete four-cell output word.
    if local_index < 64u {
        let first_lane = local_index * 4u;
        let first_cell = global_id.x - local_index + first_lane;
        if first_cell < params.cell_count {
            output[first_cell >> 2u] = cell_values[first_lane]
                | (cell_values[first_lane + 1u] << 8u)
                | (cell_values[first_lane + 2u] << 16u)
                | (cell_values[first_lane + 3u] << 24u);
        }
    }
}
