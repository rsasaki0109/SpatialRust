
struct Params {
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
    border_mode: u32,
    groups: u32,
    pixel_count: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

fn sample_gray(x: i32, y: i32) -> u32 {
    var sx = x;
    var sy = y;
    if (params.border_mode == 0u) {
        sx = clamp(sx, 0, i32(params.width) - 1);
        sy = clamp(sy, 0, i32(params.height) - 1);
    } else if (sx < 0 || sy < 0 || sx >= i32(params.width) || sy >= i32(params.height)) {
        return 0u;
    }
    return textureLoad(input_px, vec2<i32>(sx, sy), 0).r;
}

fn blur(index: u32) {
    let x = i32(index % params.width);
    let y = i32(index / params.width);
    let radius_x = i32(params.kernel_width / 2u);
    let radius_y = i32(params.kernel_height / 2u);
    var sum: u32 = 0u;
    var count: u32 = 0u;
    for (var dy: i32 = -radius_y; dy <= radius_y; dy = dy + 1) {
        for (var dx: i32 = -radius_x; dx <= radius_x; dx = dx + 1) {
            sum = sum + sample_gray(x + dx, y + dy);
            count = count + 1u;
        }
    }
    let value = (sum + count / 2u) / count;
    textureStore(output_px, vec2<i32>(x, y), vec4<u32>(value, 0u, 0u, 0u));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let stride = params.groups * 256u;
    var index = gid.x;
    loop {
        if (index >= params.pixel_count) { break; }
        blur(index);
        if (params.pixel_count - index <= stride) { break; }
        index += stride;
    }
}
