struct Params { width: u32, height: u32, channel: u32, groups: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var pixels: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> partials: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
// One padding word per replica separates equal-bin addresses.
var<workgroup> local_bins: array<atomic<u32>, 2056>;

fn sum_replicas(bin: u32) -> u32 {
    var count = 0u;
    for (var replica = 0u; replica < 8u; replica++) {
        count += atomicLoad(&local_bins[replica * 257u + bin]);
    }
    return count;
}

@compute @workgroup_size(256)
fn histogram_partials(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    for (var replica = 0u; replica < 8u; replica++) {
        atomicStore(&local_bins[replica * 257u + lane], 0u);
    }
    workgroupBarrier();
    let total = params.width * params.height;
    let stride = params.groups * 256u;
    var index = group.x * 256u + lane;
    loop {
        if (index >= total) { break; }
        let xy = vec2<i32>(i32(index % params.width), i32(index / params.width));
        let value = textureLoad(pixels, xy, 0)[params.channel];
        atomicAdd(&local_bins[(lane & 7u) * 257u + value], 1u);
        // Prevent u32 wrapping even at the maximum accepted pixel count.
        if (total - index <= stride) { break; }
        index += stride;
    }
    workgroupBarrier();
    // Every entry is overwritten, including zero bins in recycled scratch.
    partials[group.x * 256u + lane] = sum_replicas(lane);
}

@compute @workgroup_size(256)
fn histogram_reduce(@builtin(local_invocation_index) bin: u32) {
    var count = 0u;
    for (var group = 0u; group < params.groups; group++) {
        count += partials[group * 256u + bin];
    }
    output[bin] = count;
}
