@group(0) @binding(1) var<storage, read> horizontal: array<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

// The host bounds height * kernel_width * 255 before selecting this shader.
// Interleave adjacent columns while keeping the complete scan in 16 KiB.
var<workgroup> prefix: array<u32, 4096>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    var extent = 1u;
    while extent < params.height { extent *= 2u; }
    let columns = min(8u, 4096u / extent);
    let stride = params.groups * columns;
    var base_x = group.x * columns;
    while base_x < params.width {
        for (var i = lane; i < extent * columns; i += 256u) {
            let y = i / columns;
            let x = base_x + i % columns;
            var value = 0u;
            if x < params.width && y < params.height {
                value = horizontal[y * params.width + x];
            }
            prefix[i] = value;
        }
        workgroupBarrier();
        var step = 1u;
        while step < extent {
            let nodes = extent / (2u * step) * columns;
            for (var node = lane; node < nodes; node += 256u) {
                let end = ((node / columns + 1u) * 2u * step - 1u) * columns + node % columns;
                prefix[end] += prefix[end - step * columns];
            }
            workgroupBarrier();
            step *= 2u;
        }
        if lane < columns { prefix[(extent - 1u) * columns + lane] = 0u; }
        workgroupBarrier();
        step = extent / 2u;
        while step > 0u {
            let nodes = extent / (2u * step) * columns;
            for (var node = lane; node < nodes; node += 256u) {
                let end = ((node / columns + 1u) * 2u * step - 1u) * columns + node % columns;
                let left = prefix[end - step * columns];
                let base = prefix[end];
                prefix[end - step * columns] = base;
                prefix[end] = base + left;
            }
            workgroupBarrier();
            step /= 2u;
        }
        let radius = params.kernel_height / 2u;
        let count = params.kernel_width * params.kernel_height;
        for (var i = lane; i < params.height * columns; i += 256u) {
            let y = i / columns;
            let column = i % columns;
            let x = base_x + column;
            if x < params.width {
                let lo = first(y, radius);
                let hi = last(y, radius, params.height);
                var sum = prefix[hi * columns + column] - prefix[lo * columns + column];
                sum += horizontal[hi * params.width + x];
                if params.border_mode == 0u {
                    sum += left_mass(y, radius) * horizontal[x];
                    sum += right_mass(y, radius, params.height) * horizontal[(params.height - 1u) * params.width + x];
                }
                textureStore(output_px, vec2<i32>(i32(x), i32(y)), vec4<u32>((sum + count / 2u) / count, 0u, 0u, 0u));
            }
        }
        workgroupBarrier();
        if params.width - base_x <= stride { break; }
        base_x += stride;
    }
}
