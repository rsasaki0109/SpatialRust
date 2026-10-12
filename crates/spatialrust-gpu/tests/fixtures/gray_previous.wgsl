
struct Params { width: u32, height: u32, pad0: u32, pad1: u32, };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let pixel_count = params.width * params.height;
    if (index >= pixel_count) {
        return;
    }
    let xy = vec2<i32>(i32(index % params.width), i32(index / params.width));
    let rgb = textureLoad(input_px, xy, 0).rgb;
    // Match CPU BT.601 fixed-point: (77*R + 150*G + 29*B + 128) >> 8
    let gray = (77u * rgb.r + 150u * rgb.g + 29u * rgb.b + 128u) >> 8u;
    textureStore(output_px, xy, vec4<u32>(gray, 0u, 0u, 0u));
}
