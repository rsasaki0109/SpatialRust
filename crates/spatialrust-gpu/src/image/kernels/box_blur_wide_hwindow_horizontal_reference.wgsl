@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> horizontal: array<u32>;

fn blur(index: u32) {
    let x = index % params.width;
    let y = index / params.width;
    let radius = params.kernel_width / 2u;
    let end = last(x, radius, params.width);
    var sum = 0u;
    var sx = first(x, radius);
    loop {
        sum += textureLoad(input_px, vec2<i32>(i32(sx), i32(y)), 0).r;
        if (sx == end) { break; }
        sx += 1u;
    }
    if (params.border_mode == 0u) {
        sum += left_mass(x, radius) * textureLoad(input_px, vec2<i32>(0, i32(y)), 0).r;
        sum += right_mass(x, radius, params.width) *
            textureLoad(input_px, vec2<i32>(i32(params.width - 1u), i32(y)), 0).r;
    }
    horizontal[index] = sum;
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
