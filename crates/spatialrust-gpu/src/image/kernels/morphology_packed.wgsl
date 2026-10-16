struct Params {
    source_width: u32,
    source_height: u32,
    output_width: u32,
    output_height: u32,
    kernel_width: u32,
    kernel_height: u32,
    operation: u32,
    pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

fn horizontal_value(x: u32, y: u32) -> u32 {
    let radius = params.kernel_width / 2u;
    let first = max(i32(x) - i32(radius), 0);
    let last = min(i32(x) + i32(radius), i32(params.source_width) - 1);
    var value = select(255u, 0u, params.operation == 1u);
    for (var coordinate = first; coordinate <= last; coordinate = coordinate + 1) {
        let sample = textureLoad(input_px, vec2<i32>(coordinate, i32(y)), 0).r;
        if (params.operation == 0u) { value = min(value, sample); }
        else { value = max(value, sample); }
    }
    return value;
}

@compute @workgroup_size(16, 16)
fn horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let packed_width = params.output_width / 4u + select(0u, 1u, params.output_width % 4u != 0u);
    if (gid.x >= packed_width || gid.y >= params.output_height) { return; }
    let first = gid.x * 4u;
    var values = vec4<u32>(0u);
    // One invocation owns all four bytes; incomplete row tails stay zero.
    for (var lane = 0u; lane < 4u; lane += 1u) {
        let x = first + lane;
        if (x < params.output_width) { values[lane] = horizontal_value(x, gid.y); }
    }
    textureStore(output_px, vec2<i32>(gid.xy), values);
}

@compute @workgroup_size(16, 16)
fn vertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.output_width || gid.y >= params.output_height) { return; }
    let radius = params.kernel_height / 2u;
    let first = max(i32(gid.y) - i32(radius), 0);
    let last = min(i32(gid.y) + i32(radius), i32(params.source_height) - 1);
    var value = select(255u, 0u, params.operation == 1u);
    for (var coordinate = first; coordinate <= last; coordinate = coordinate + 1) {
        let packed = textureLoad(input_px, vec2<i32>(i32(gid.x / 4u), coordinate), 0);
        let sample = packed[gid.x % 4u];
        if (params.operation == 0u) { value = min(value, sample); }
        else { value = max(value, sample); }
    }
    textureStore(output_px, vec2<i32>(gid.xy), vec4<u32>(value, 0u, 0u, 0u));
}
