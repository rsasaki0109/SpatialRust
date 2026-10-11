@group(0) @binding(1) var source: texture_2d<u32>;
@group(0) @binding(2) var<storage, read> geometry: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> lookups: array<u32>;
@group(0) @binding(4) var output_px: texture_storage_2d<rgba8uint, write>;
fn blend(a: u32, b: u32, weight: F64, complement: F64) -> F64 {
    return fp_sum(fp_product(fp_uint(a), complement), fp_product(fp_uint(b), weight));
}
@compute @workgroup_size(16, 16)
fn clahe_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let stride_x = params.groups_x * 16u;
    let stride_y = params.groups_y * 16u;
    var y = id.y;
    loop {
        if y >= params.height { break; }
        let vertical = geometry[params.width + y];
        let ya = fp_decode(vertical.zw);
        let yc = fp_difference(fp_uint(1u), ya);
        let top_row = vertical.x * params.tiles_x;
        let bottom_row = vertical.y * params.tiles_x;
        var x = id.x;
        loop {
            if x >= params.width { break; }
            let xy = vec2<i32>(i32(x), i32(y));
            let value = textureLoad(source, xy, 0).r;
            let horizontal = geometry[x];
            let a = lookups[(top_row + horizontal.x) * 256u + value];
            let b = lookups[(top_row + horizontal.y) * 256u + value];
            let c = lookups[(bottom_row + horizontal.x) * 256u + value];
            let d = lookups[(bottom_row + horizontal.y) * 256u + value];
            // Equal integer corners are far from any half-byte boundary.
            // CPU binary64 error is < 1e-12 here, so their rounded byte is exact.
            var result = a;
            if a != b || a != c || a != d {
                let xa = fp_decode(horizontal.zw);
                let xc = fp_difference(fp_uint(1u), xa);
                let top = blend(a, b, xa, xc);
                let bottom = blend(c, d, xa, xc);
                result = fp_byte(fp_sum(fp_product(top, yc), fp_product(bottom, ya)));
            }
            textureStore(output_px, xy, vec4<u32>(result, 0u, 0u, 0u));
            if params.width - x <= stride_x { break; }
            x += stride_x;
        }
        if params.height - y <= stride_y { break; }
        y += stride_y;
    }
}
