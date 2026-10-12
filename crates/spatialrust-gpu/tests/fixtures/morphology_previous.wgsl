
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

fn in_output(gid: vec3<u32>) -> bool {
    return gid.x < params.output_width && gid.y < params.output_height;
}

fn gray_at(x: i32, y: i32) -> i32 {
    let sx = clamp(x, 0, i32(params.source_width) - 1);
    let sy = clamp(y, 0, i32(params.source_height) - 1);
    return i32(textureLoad(input_px, vec2<i32>(sx, sy), 0).r);
}

@compute @workgroup_size(16, 16)
fn resize_nearest(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_output(gid)) { return; }
    let sx = min((gid.x * params.source_width) / params.output_width, params.source_width - 1u);
    let sy = min((gid.y * params.source_height) / params.output_height, params.source_height - 1u);
    textureStore(output_px, vec2<i32>(gid.xy), textureLoad(input_px, vec2<i32>(i32(sx), i32(sy)), 0));
}

@compute @workgroup_size(16, 16)
fn sobel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_output(gid)) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let gx = -gray_at(x - 1, y - 1) + gray_at(x + 1, y - 1)
        - 2 * gray_at(x - 1, y) + 2 * gray_at(x + 1, y)
        - gray_at(x - 1, y + 1) + gray_at(x + 1, y + 1);
    let gy = -gray_at(x - 1, y - 1) - 2 * gray_at(x, y - 1) - gray_at(x + 1, y - 1)
        + gray_at(x - 1, y + 1) + 2 * gray_at(x, y + 1) + gray_at(x + 1, y + 1);
    let magnitude = u32(clamp(abs(gx) + abs(gy), 0, 255));
    textureStore(output_px, vec2<i32>(x, y), vec4<u32>(magnitude, 0u, 0u, 0u));
}

@compute @workgroup_size(16, 16)
fn morphology(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_output(gid)) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let rx = i32(params.kernel_width / 2u);
    let ry = i32(params.kernel_height / 2u);
    var value = select(255, 0, params.operation == 1u);
    for (var dy = -ry; dy <= ry; dy = dy + 1) {
        for (var dx = -rx; dx <= rx; dx = dx + 1) {
            let sample = gray_at(x + dx, y + dy);
            if (params.operation == 0u) { value = min(value, sample); }
            else { value = max(value, sample); }
        }
    }
    textureStore(output_px, vec2<i32>(x, y), vec4<u32>(u32(value), 0u, 0u, 0u));
}
