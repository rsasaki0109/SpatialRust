struct Params {
    width: u32,
    height: u32,
    border: u32,
    padding: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

fn sample_at(xy: vec2<i32>) -> vec4<u32> {
    let extent = vec2<i32>(i32(params.width), i32(params.height));
    if (params.border == 1u && (any(xy < vec2<i32>(0)) || any(xy >= extent))) {
        return vec4<u32>(0u);
    }
    return textureLoad(input_px, clamp(xy, vec2<i32>(0), extent - vec2<i32>(1)), 0);
}
