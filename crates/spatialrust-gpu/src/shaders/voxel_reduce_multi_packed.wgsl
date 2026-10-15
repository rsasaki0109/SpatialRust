struct Params { cell_count: u32, point_count: u32, channels: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_indices: array<u32>;
@group(0) @binding(2) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read> values0: array<f32>;
@group(0) @binding(4) var<storage, read> values1: array<f32>;
@group(0) @binding(5) var<storage, read> values2: array<f32>;
@group(0) @binding(6) var<storage, read> values3: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if cell >= params.cell_count { return; }
    let start = cell_starts[cell];
    let end = select(params.point_count, cell_starts[cell + 1u], cell + 1u < params.cell_count);
    let count = end - start;
    if count == 0u {
        if params.channels > 0u { output[0u * params.cell_count + cell] = 0.0; }
        if params.channels > 1u { output[1u * params.cell_count + cell] = 0.0; }
        if params.channels > 2u { output[2u * params.cell_count + cell] = 0.0; }
        if params.channels > 3u { output[3u * params.cell_count + cell] = 0.0; }
        return;
    }
    var sum0 = 0.0;
    var sum1 = 0.0;
    var sum2 = 0.0;
    var sum3 = 0.0;
    for (var j = 0u; j < count; j = j + 1u) {
        let index = point_indices[start + j];
        if params.channels > 0u { sum0 = sum0 + values0[index]; }
        if params.channels > 1u { sum1 = sum1 + values1[index]; }
        if params.channels > 2u { sum2 = sum2 + values2[index]; }
        if params.channels > 3u { sum3 = sum3 + values3[index]; }
    }
    if params.channels > 0u { output[0u * params.cell_count + cell] = sum0 / f32(count); }
    if params.channels > 1u { output[1u * params.cell_count + cell] = sum1 / f32(count); }
    if params.channels > 2u { output[2u * params.cell_count + cell] = sum2 / f32(count); }
    if params.channels > 3u { output[3u * params.cell_count + cell] = sum3 / f32(count); }
}
