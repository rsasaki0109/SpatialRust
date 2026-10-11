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

// A 16x16 output block shares an 18x18 clamped input block including its halo.
var<workgroup> samples: array<i32, 324>;

@compute @workgroup_size(16, 16)
fn sobel_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let origin = vec2<i32>(group.xy * 16u);
    for (var i = lane; i < 324u; i += 256u) {
        let position = origin + vec2<i32>(i32(i % 18u), i32(i / 18u)) - vec2<i32>(1);
        let clamped = clamp(position, vec2<i32>(0),
            vec2<i32>(i32(params.source_width) - 1, i32(params.source_height) - 1));
        samples[i] = i32(textureLoad(input_px, clamped, 0).r);
    }
    // Even lanes outside a partial output block must finish loading the halo.
    workgroupBarrier();
    if gid.x >= params.output_width || gid.y >= params.output_height { return; }
    let i = (lid.y + 1u) * 18u + lid.x + 1u;
    let gx = -samples[i - 19u] + samples[i - 17u]
        - 2 * samples[i - 1u] + 2 * samples[i + 1u]
        - samples[i + 17u] + samples[i + 19u];
    let gy = -samples[i - 19u] - 2 * samples[i - 18u] - samples[i - 17u]
        + samples[i + 17u] + 2 * samples[i + 18u] + samples[i + 19u];
    let magnitude = u32(clamp(abs(gx) + abs(gy), 0, 255));
    textureStore(output_px, vec2<i32>(gid.xy), vec4<u32>(magnitude, 0u, 0u, 0u));
}
