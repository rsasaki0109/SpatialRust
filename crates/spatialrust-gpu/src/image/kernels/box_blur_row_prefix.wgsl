@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> horizontal: array<u32>;

// One row per workgroup. Padding is zero and every tree node is overwritten.
var<workgroup> row_prefix: array<u32, 4096>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    var extent = 1u;
    loop {
        if (extent >= params.width) { break; }
        extent *= 2u;
    }
    var row = group.x;
    loop {
        if (row >= params.height) { break; }
        for (var x = lane; x < extent; x += 256u) {
            var value = 0u;
            if (x < params.width) {
                value = textureLoad(input_px, vec2<i32>(i32(x), i32(row)), 0).r;
            }
            row_prefix[x] = value;
        }
        workgroupBarrier();

        // Work-efficient exclusive scan. With width <= 4096 every subtotal
        // is at most 4096*255, well inside u32 even for an all-maximum row.
        var step = 1u;
        loop {
            if (step >= extent) { break; }
            for (var node = lane; node < extent / (2u * step); node += 256u) {
                let end = (node + 1u) * 2u * step - 1u;
                row_prefix[end] += row_prefix[end - step];
            }
            workgroupBarrier();
            step *= 2u;
        }
        if (lane == 0u) { row_prefix[extent - 1u] = 0u; }
        workgroupBarrier();
        step = extent / 2u;
        loop {
            if (step == 0u) { break; }
            for (var node = lane; node < extent / (2u * step); node += 256u) {
                let end = (node + 1u) * 2u * step - 1u;
                let left = row_prefix[end - step];
                let base = row_prefix[end];
                row_prefix[end - step] = base;
                row_prefix[end] = base + left;
            }
            workgroupBarrier();
            step /= 2u;
        }
        let radius = params.kernel_width / 2u;
        for (var x = lane; x < params.width; x += 256u) {
            let lo = first(x, radius);
            let hi = last(x, radius, params.width);
            var sum = row_prefix[hi] + textureLoad(input_px, vec2<i32>(i32(hi), i32(row)), 0).r
                - row_prefix[lo];
            if (params.border_mode == 0u) {
                sum += left_mass(x, radius) * textureLoad(input_px, vec2<i32>(0, i32(row)), 0).r;
                sum += right_mass(x, radius, params.width) *
                    textureLoad(input_px, vec2<i32>(i32(params.width - 1u), i32(row)), 0).r;
            }
            horizontal[row * params.width + x] = sum;
        }
        // Complete all prefix reads before another row reuses shared memory.
        workgroupBarrier();
        if (params.height - row <= params.groups) { break; }
        row += params.groups;
    }
}
