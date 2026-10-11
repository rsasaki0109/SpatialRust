struct Params { width: u32, height: u32, groups_x: u32, groups_y: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var source: texture_2d<u32>;
@group(0) @binding(2) var<storage, read> lookup: array<u32>;
@group(0) @binding(3) var output_px: texture_storage_2d<rgba8uint, write>;

@compute @workgroup_size(16, 16)
fn equalize_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let stride_x = params.groups_x * 16u;
    let stride_y = params.groups_y * 16u;
    var y = id.y;
    loop {
        if y >= params.height { break; }
        var x = id.x;
        loop {
            if x >= params.width { break; }
            let xy = vec2<i32>(i32(x), i32(y));
            let value = textureLoad(source, xy, 0).r;
            textureStore(output_px, xy, vec4<u32>(lookup[value], 0u, 0u, 0u));
            if params.width - x <= stride_x { break; }
            x += stride_x;
        }
        if params.height - y <= stride_y { break; }
        y += stride_y;
    }
}
