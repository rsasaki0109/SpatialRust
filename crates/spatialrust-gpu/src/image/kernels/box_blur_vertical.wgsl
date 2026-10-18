@group(0) @binding(1) var<storage, read> horizontal: array<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

fn horizontal_sum(index: u32) -> u32 {
    return horizontal[index];
}

fn vertical_sum(index: u32) -> u32 {
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
    return sum;
}

fn horizontal_at(x: u32, row: i32) -> u32 {
    if (params.border_mode == 0u) {
        return horizontal_sum(u32(clamp(row, 0, i32(params.height) - 1)) * params.width + x);
    }
    if (row < 0 || row >= i32(params.height)) { return 0u; }
    return horizontal_sum(u32(row) * params.width + x);
}

fn store_value(x: u32, y: u32, sum: u32) {
    let count = params.kernel_width * params.kernel_height;
    let value = (sum + count / 2u) / count;
    textureStore(output_px, vec2<i32>(i32(x), i32(y)), vec4<u32>(value, 0u, 0u, 0u));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let stride = params.groups * 256u;
    let row_pairs = params.height / 2u + params.height % 2u;
    let pairs = params.width * row_pairs;
    var pair = gid.x;
    loop {
        if (pair >= pairs) { break; }
        let x = pair % params.width;
        let y = (pair / params.width) * 2u;
        let index = y * params.width + x;
        if (params.kernel_height == 1u) {
            store_value(x, y, horizontal_sum(index));
            if (y + 1u < params.height) {
                store_value(x, y + 1u, horizontal_sum(index + params.width));
            }
        } else {
            let sum = vertical_sum(index);
            store_value(x, y, sum);
            if (y + 1u < params.height) {
                let radius = i32(params.kernel_height / 2u);
                // The outgoing sample occurs in the first nonnegative window.
                let next = sum - horizontal_at(x, i32(y) - radius) +
                    horizontal_at(x, i32(y) + radius + 1);
                store_value(x, y + 1u, next);
            }
        }
        if (pairs - pair <= stride) { break; }
        pair += stride;
    }
}
