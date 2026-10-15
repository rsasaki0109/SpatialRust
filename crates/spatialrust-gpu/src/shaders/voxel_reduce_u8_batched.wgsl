struct Params { cell_count: u32, point_count: u32, channels: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_indices: array<u32>;
@group(0) @binding(2) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read> values0: array<u32>;
@group(0) @binding(4) var<storage, read> values1: array<u32>;
@group(0) @binding(5) var<storage, read> values2: array<u32>;
@group(0) @binding(6) var<storage, read> values3: array<u32>;
@group(0) @binding(7) var<storage, read_write> output: array<u32>;

fn cell_value(cell: u32) -> vec4<u32> {
    let start = cell_starts[cell];
    let end = select(params.point_count, cell_starts[cell + 1u], cell + 1u < params.cell_count);
    let count = end - start;
    if count == 0u { return vec4<u32>(0u); }
    var sum0 = 0.0;
    var sum1 = 0.0;
    var sum2 = 0.0;
    var sum3 = 0.0;
    for (var j = 0u; j < count; j = j + 1u) {
        let index = point_indices[start + j];
        let word_index = index >> 2u;
        let shift = (index & 3u) * 8u;
        if params.channels > 0u { sum0 = sum0 + f32((values0[word_index] >> shift) & 0xFFu); }
        if params.channels > 1u { sum1 = sum1 + f32((values1[word_index] >> shift) & 0xFFu); }
        if params.channels > 2u { sum2 = sum2 + f32((values2[word_index] >> shift) & 0xFFu); }
        if params.channels > 3u { sum3 = sum3 + f32((values3[word_index] >> shift) & 0xFFu); }
    }
    return vec4<u32>(u32(round(sum0 / f32(count))) & 0xFFu, u32(round(sum1 / f32(count))) & 0xFFu, u32(round(sum2 / f32(count))) & 0xFFu, u32(round(sum3 / f32(count))) & 0xFFu);
}

// Every lane initializes all four arrays, including inactive tail lanes.
var<workgroup> cell_values0: array<u32, 256>;
var<workgroup> cell_values1: array<u32, 256>;
var<workgroup> cell_values2: array<u32, 256>;
var<workgroup> cell_values3: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
    var value = vec4<u32>(0u);
    if global_id.x < params.cell_count { value = cell_value(global_id.x); }
    cell_values0[local_index] = value[0];
    cell_values1[local_index] = value[1];
    cell_values2[local_index] = value[2];
    cell_values3[local_index] = value[3];
    workgroupBarrier();
    if local_index < 64u {
        let first_lane = local_index * 4u;
        let first_cell = global_id.x - local_index + first_lane;
        if first_cell < params.cell_count {
            let stride = (params.cell_count >> 2u) + select(0u, 1u, (params.cell_count & 3u) != 0u);
            if params.channels > 0u {
                output[0u * stride + (first_cell >> 2u)] = cell_values0[first_lane]
                    | (cell_values0[first_lane + 1u] << 8u)
                    | (cell_values0[first_lane + 2u] << 16u)
                    | (cell_values0[first_lane + 3u] << 24u);
            }
            if params.channels > 1u {
                output[1u * stride + (first_cell >> 2u)] = cell_values1[first_lane]
                    | (cell_values1[first_lane + 1u] << 8u)
                    | (cell_values1[first_lane + 2u] << 16u)
                    | (cell_values1[first_lane + 3u] << 24u);
            }
            if params.channels > 2u {
                output[2u * stride + (first_cell >> 2u)] = cell_values2[first_lane]
                    | (cell_values2[first_lane + 1u] << 8u)
                    | (cell_values2[first_lane + 2u] << 16u)
                    | (cell_values2[first_lane + 3u] << 24u);
            }
            if params.channels > 3u {
                output[3u * stride + (first_cell >> 2u)] = cell_values3[first_lane]
                    | (cell_values3[first_lane + 1u] << 8u)
                    | (cell_values3[first_lane + 2u] << 16u)
                    | (cell_values3[first_lane + 3u] << 24u);
            }
        }
    }
}
