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

fn window_sample(x: i32, y: u32) -> u32 {
    if (params.border_mode != 0u && (x < 0 || x >= i32(params.width))) { return 0u; }
    return textureLoad(input_px, vec2<i32>(clamp(x, 0, i32(params.width) - 1), i32(y)), 0).r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let stride = params.groups * 256u;
    // Preserve the scalar schedule for workloads with limited parallel work.
    if (params.pixel_count < 65536u) {
        var index = gid.x;
        loop {
            if (index >= params.pixel_count) { break; }
            horizontal[index] = blur(index);
            if (params.pixel_count - index <= stride) { break; }
            index += stride;
        }
        return;
    }
    let pairs = params.pixel_count / 2u + params.pixel_count % 2u;
    var pair = gid.x;
    loop {
        if (pair >= pairs) { break; }
        let index = pair * 2u;
        let value = blur(index);
        horizontal[index] = value;
        if (index + 1u < params.pixel_count) {
            let x = index % params.width;
            var next = 0u;
            if (x + 1u < params.width) {
                let radius = i32(params.kernel_width / 2u);
                let y = index / params.width;
                // The outgoing byte belongs to the first nonnegative window.
                next = value - window_sample(i32(x) - radius, y) +
                    window_sample(i32(x) + radius + 1, y);
            } else {
                next = blur(index + 1u);
            }
            horizontal[index + 1u] = next;
        }
        if (pairs - pair <= stride) { break; }
        pair += stride;
    }
}
