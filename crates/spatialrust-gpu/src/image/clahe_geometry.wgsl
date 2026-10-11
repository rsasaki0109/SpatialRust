@group(0) @binding(1) var<storage, read_write> geometry: array<vec4<u32>>;
fn axis_geometry(position: u32, tile_size: u32, tiles: u32) -> vec4<u32> {
    let p = fp_ratio(position, tile_size);
    let half = F64(vec2<u32>(0u, 1048576u), -1i);
    var left = 0u; var right = 0u; var weight: F64;
    if fp_less(p, half) {
        weight = fp_difference(fp_uint(1u), fp_difference(half, p));
    } else {
        let tx = fp_difference(p, half);
        let raw = fp_floor(tx);
        left = min(raw, tiles - 1u);
        right = min(raw + 1u, tiles - 1u);
        weight = fp_difference(tx, fp_uint(raw));
    }
    return vec4<u32>(left, right, fp_encode(weight));
}
@compute @workgroup_size(256)
fn clahe_geometry(@builtin(global_invocation_id) id: vec3<u32>) {
    let total = params.width + params.height;
    let stride = params.geometry_groups * 256u;
    var index = id.x;
    loop {
        if index >= total { break; }
        if index < params.width {
            geometry[index] = axis_geometry(index, params.tile_width, params.tiles_x);
        } else {
            geometry[index] = axis_geometry(index - params.width, params.tile_height, params.tiles_y);
        }
        if total - index <= stride { break; }
        index += stride;
    }
}
