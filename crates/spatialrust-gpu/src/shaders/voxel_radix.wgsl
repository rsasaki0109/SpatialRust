struct Params {
    point_count: u32,
    group_count: u32,
    axis: u32,
    shift: u32,
}

struct VoxelSortEntry {
    ix: i32,
    iy: i32,
    iz: i32,
    point_index: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_entries: array<VoxelSortEntry>;
@group(0) @binding(2) var<storage, read_write> output_entries: array<VoxelSortEntry>;
@group(0) @binding(3) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(4) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> ranks: array<u32>;

// Eight 32-bit lane masks per digit make ranks deterministic without relying
// on atomic insertion order or optional subgroup operations.
var<workgroup> masks: array<atomic<u32>, 128>;
var<workgroup> sums: array<vec2<u32>, 256>;

fn digit(entry: VoxelSortEntry) -> u32 {
    var coordinate = entry.iz;
    if params.axis == 0u {
        coordinate = entry.ix;
    } else if params.axis == 1u {
        coordinate = entry.iy;
    }
    // Flip the sign bit so unsigned radix order matches signed i32 order.
    return ((bitcast<u32>(coordinate) ^ 0x80000000u) >> params.shift) & 15u;
}

@compute @workgroup_size(256)
fn histogram(@builtin(local_invocation_index) lane: u32,
             @builtin(workgroup_id) group: vec3<u32>) {
    if lane < 128u {
        atomicStore(&masks[lane], 0u);
    }
    workgroupBarrier();

    let index = group.x * 256u + lane;
    var bucket = 0u;
    if index < params.point_count {
        bucket = digit(input_entries[index]);
        atomicOr(&masks[bucket * 8u + lane / 32u], 1u << (lane % 32u));
    }
    workgroupBarrier();

    if index < params.point_count {
        let word = lane / 32u;
        let preceding = (1u << (lane % 32u)) - 1u;
        var rank = countOneBits(atomicLoad(&masks[bucket * 8u + word]) & preceding);
        for (var i = 0u; i < word; i++) {
            rank += countOneBits(atomicLoad(&masks[bucket * 8u + i]));
        }
        ranks[index] = rank;
    }
    if lane < 16u {
        var count = 0u;
        for (var word = 0u; word < 8u; word++) {
            count += countOneBits(atomicLoad(&masks[lane * 8u + word]));
        }
        histograms[lane * params.group_count + group.x] = count;
    }
}

// One workgroup per digit. Each lane owns a contiguous chunk of group counts.
// Scan chunk totals in shared memory, then write exclusive group offsets. The
// second component sums earlier digits to give this digit's global base.
@compute @workgroup_size(256)
fn prefix(@builtin(local_invocation_index) lane: u32,
          @builtin(workgroup_id) group: vec3<u32>) {
    let bucket = group.x;
    let chunk = (params.group_count + 255u) / 256u;
    let start = lane * chunk;
    let end = min(start + chunk, params.group_count);
    var total = vec2<u32>(0u);
    for (var i = start; i < end; i++) {
        total.x += histograms[bucket * params.group_count + i];
        for (var earlier = 0u; earlier < bucket; earlier++) {
            total.y += histograms[earlier * params.group_count + i];
        }
    }
    sums[lane] = total;
    workgroupBarrier();

    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var preceding = vec2<u32>(0u);
        if lane >= stride {
            preceding = sums[lane - stride];
        }
        workgroupBarrier();
        sums[lane] += preceding;
        workgroupBarrier();
    }

    var offset = sums[255u].y + sums[lane].x - total.x;
    for (var i = start; i < end; i++) {
        offsets[bucket * params.group_count + i] = offset;
        offset += histograms[bucket * params.group_count + i];
    }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.point_count {
        return;
    }
    let entry = input_entries[index];
    let bucket = digit(entry);
    let destination = offsets[bucket * params.group_count + index / 256u] + ranks[index];
    output_entries[destination] = entry;
}
