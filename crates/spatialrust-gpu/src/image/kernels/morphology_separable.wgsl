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

fn reduce_at(gid: vec3<u32>, horizontal: bool) {
    if (gid.x >= params.output_width || gid.y >= params.output_height) { return; }
    let position = select(gid.y, gid.x, horizontal);
    let dimension = select(params.source_height, params.source_width, horizontal);
    let radius = select(params.kernel_height, params.kernel_width, horizontal) / 2u;
    let first = max(i32(position) - i32(radius), 0);
    let last = min(i32(position) + i32(radius), i32(dimension) - 1);
    var value = select(255u, 0u, params.operation == 1u);
    // Repeated clamped edge values do not change either extremum.
    for (var coordinate = first; coordinate <= last; coordinate = coordinate + 1) {
        let point = select(vec2<i32>(i32(gid.x), coordinate),
            vec2<i32>(coordinate, i32(gid.y)), horizontal);
        let sample = textureLoad(input_px, point, 0).r;
        if (params.operation == 0u) { value = min(value, sample); }
        else { value = max(value, sample); }
    }
    textureStore(output_px, vec2<i32>(gid.xy), vec4<u32>(value, 0u, 0u, 0u));
}

@compute @workgroup_size(16, 16)
fn horizontal(@builtin(global_invocation_id) gid: vec3<u32>) { reduce_at(gid, true); }

@compute @workgroup_size(16, 16)
fn vertical(@builtin(global_invocation_id) gid: vec3<u32>) { reduce_at(gid, false); }
