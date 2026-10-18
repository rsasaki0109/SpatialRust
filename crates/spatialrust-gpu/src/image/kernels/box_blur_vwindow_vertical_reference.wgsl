@group(0) @binding(1) var<storage, read> horizontal: array<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

fn horizontal_sum(index: u32) -> u32 {
    return (horizontal[index / 2u] >> ((index % 2u) * 16u)) & 65535u;
}

fn blur(index: u32) {
    let x = index % params.width;
    let y = index / params.width;
    let radius = params.kernel_height / 2u;
    let end = last(y, radius, params.height);
    var sum = 0u;
    var sy = first(y, radius);
    loop {
        sum += horizontal_sum(sy * params.width + x);
        if (sy == end) { break; }
        sy += 1u;
    }
    if (params.border_mode == 0u) {
        sum += left_mass(y, radius) * horizontal_sum(x);
        sum += right_mass(y, radius, params.height) *
            horizontal_sum((params.height - 1u) * params.width + x);
    }
    let count = params.kernel_width * params.kernel_height;
    let value = (sum + count / 2u) / count;
    textureStore(output_px, vec2<i32>(i32(x), i32(y)), vec4<u32>(value, 0u, 0u, 0u));
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
