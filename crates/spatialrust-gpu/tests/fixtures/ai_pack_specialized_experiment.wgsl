override PACK_CHANNELS: u32 = 4u;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
    scale: f32,
    mean0: f32,
    mean1: f32,
    mean2: f32,
    mean3: f32,
    std0: f32,
    std1: f32,
    std2: f32,
    std3: f32,
    _tail0: f32,
    _tail1: f32,
    _tail2: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var source_px: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn pack_ai_chw(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let pixel = textureLoad(source_px, vec2<i32>(gid.xy), 0);
    let values = vec4<f32>(pixel);
    let means = vec4<f32>(params.mean0, params.mean1, params.mean2, params.mean3);
    let stds = vec4<f32>(params.std0, params.std1, params.std2, params.std3);
    let index = gid.y * params.width + gid.x;
    let plane = params.width * params.height;
    for (var channel = 0u; channel < PACK_CHANNELS; channel++) {
        output[channel * plane + index] = (values[channel] * params.scale - means[channel]) / stds[channel];
    }
}
