@group(0) @binding(1) var source: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> lookups: array<u32>;
var<workgroup> bins: array<atomic<u32>, 256>;
var<workgroup> scale: vec4<u32>;
fn reflect101(position: u32, length: u32) -> u32 {
    if length == 1u { return 0u; }
    let period = (length - 1u) * 2u;
    let value = position % period;
    return select(period - value, value, value < length);
}
@compute @workgroup_size(256)
fn clahe_lookup(@builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    let area = params.tile_width * params.tile_height;
    var tile = group.x;
    loop {
        if tile >= params.tile_count { break; }
        atomicStore(&bins[lane], 0u);
        workgroupBarrier();
        let origin = vec2<u32>((tile % params.tiles_x) * params.tile_width,
            (tile / params.tiles_x) * params.tile_height);
        var index = lane;
        loop {
            if index >= area { break; }
            let x = reflect101(origin.x + index % params.tile_width, params.width);
            let y = reflect101(origin.y + index / params.tile_width, params.height);
            atomicAdd(&bins[textureLoad(source, vec2<i32>(i32(x), i32(y)), 0).r], 1u);
            if area - index <= 256u { break; }
            index += 256u;
        }
        workgroupBarrier();
        let cumulative = clahe_cumulative(lane, atomicLoad(&bins[lane]), params.clip_count);
        if lane == 0u { scale = reciprocal_info(area); }
        workgroupBarrier();
        lookups[tile * 256u + lane] = mapped(cumulative, scale);
        workgroupBarrier();
        if params.tile_count - tile <= params.tile_groups { break; }
        tile += params.tile_groups;
    }
}
