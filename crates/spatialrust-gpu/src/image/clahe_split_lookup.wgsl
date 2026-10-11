@group(0) @binding(1) var<storage, read> partials: array<u32>;
@group(0) @binding(2) var<storage, read_write> lookups: array<u32>;
var<workgroup> scale: vec4<u32>;
fn merged_count(tile: u32, lane: u32, chunks: u32) -> u32 {
    var count = 0u;
    for (var chunk = 0u; chunk < chunks; chunk += 1u) {
        count += partials[(tile * chunks + chunk) * 256u + lane];
    }
    // All nonnegative partial sums are bounded by the original u32 tile area.
    return count;
}
@compute @workgroup_size(256)
fn clahe_merged_lookup(@builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    let area = params.tile_width * params.tile_height;
    var tile = group.x;
    loop {
        if tile >= params.tile_count { break; }
        let cumulative = clahe_cumulative(lane, merged_count(tile, lane, params.chunks), params.clip_count);
        if lane == 0u { scale = reciprocal_info(area); }
        workgroupBarrier();
        lookups[tile * 256u + lane] = mapped(cumulative, scale);
        workgroupBarrier();
        if params.tile_count - tile <= params.tile_groups { break; }
        tile += params.tile_groups;
    }
}
