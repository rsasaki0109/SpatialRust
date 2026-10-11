struct Params {
    width: u32,
    height: u32,
    groups_per_row: u32,
    group_count: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

fn flat_pixel(group: vec3<u32>, lane: u32) -> u32 {
    let flat_group = group.y * params.groups_per_row + group.x;
    // A balanced rectangle can contain padding groups. Reject them before
    // multiplying by 256 so the u32::MAX pixel boundary cannot wrap.
    if (flat_group >= params.group_count) { return 0xffffffffu; }
    return flat_group * 256u + lane;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = flat_pixel(group, lane);
    let pixel_count = params.width * params.height;
    if (index >= pixel_count) { return; }
    let xy = vec2<i32>(i32(index % params.width), i32(index / params.width));
    let rgb = textureLoad(input_px, xy, 0).rgb;
    let gray = (77u * rgb.r + 150u * rgb.g + 29u * rgb.b + 128u) >> 8u;
    textureStore(output_px, xy, vec4<u32>(gray, 0u, 0u, 0u));
}
