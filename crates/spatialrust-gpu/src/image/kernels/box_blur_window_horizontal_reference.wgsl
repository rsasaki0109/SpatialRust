@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> horizontal: array<u32>;

fn blur(index: u32) -> u32 {
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
    return sum;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let stride = params.groups * 256u;
    let words = params.pixel_count / 2u + params.pixel_count % 2u;
    var word = gid.x;
    loop {
        if (word >= words) { break; }
        let index = word * 2u;
        var value = blur(index);
        if (index + 1u < params.pixel_count) { value |= blur(index + 1u) << 16u; }
        // One invocation owns both halves; an odd tail keeps a zero high half.
        horizontal[word] = value;
        if (words - word <= stride) { break; }
        word += stride;
    }
}
