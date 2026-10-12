@group(0) @binding(1) var source: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> partials: array<u32>;
var<workgroup> bins: array<atomic<u32>, 256>;
fn reflect101(position: u32, length: u32) -> u32 {
    if position < length { return position; }
    if length == 1u { return 0u; }
    let period = (length - 1u) * 2u;
    let value = position % period;
    return select(period - value, value, value < length);
}
@compute @workgroup_size(256)
fn clahe_partial_histogram(@builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    let area = params.tile_width * params.tile_height;
    let chunk_size = area / params.chunks + select(0u, 1u, area % params.chunks != 0u);
    let total = params.tile_count * params.chunks;
    var partial = group.x;
    loop {
        if partial >= total { break; }
        atomicStore(&bins[lane], 0u);
        workgroupBarrier();
        let tile = partial / params.chunks;
        let chunk = partial % params.chunks;
        let start = chunk * chunk_size;
        var length = 0u;
        if start < area { length = min(chunk_size, area - start); }
        let origin = vec2<u32>((tile % params.tiles_x) * params.tile_width,
            (tile / params.tiles_x) * params.tile_height);
        var offset = lane;
        loop {
            if offset >= length { break; }
            let index = start + offset;
            let x = reflect101(origin.x + index % params.tile_width, params.width);
            let y = reflect101(origin.y + index / params.tile_width, params.height);
            atomicAdd(&bins[textureLoad(source, vec2<i32>(i32(x), i32(y)), 0).r], 1u);
            if length - offset <= 256u { break; }
            offset += 256u;
        }
        workgroupBarrier();
        partials[partial * 256u + lane] = atomicLoad(&bins[lane]);
        workgroupBarrier();
        if total - partial <= params.partial_groups { break; }
        partial += params.partial_groups;
    }
}
