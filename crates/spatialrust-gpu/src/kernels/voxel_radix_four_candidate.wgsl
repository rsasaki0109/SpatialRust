struct Params {
    point_count: u32,
    group_count: u32,
    axis: u32,
    shift: u32,
}

// Pipeline specialization keeps the legacy small-input path free of the extra
// histogram barrier/subtotal stores. The host selects the cached large path.
override cached_prefix: bool = false;

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
var<workgroup> bucket_counts: array<u32, 16>;
// Transposed word prefixes are produced by the existing per-digit count loop.
var<workgroup> word_prefix: array<u32, 128>;
var<workgroup> sums: array<vec2<u32>, 256>;
var<workgroup> bucket_base: u32;

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

    if !cached_prefix && index < params.point_count {
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
            if cached_prefix { word_prefix[word * 16u + lane] = count; }
            count += countOneBits(atomicLoad(&masks[lane * 8u + word]));
        }
        if cached_prefix { bucket_counts[lane] = count; }
        histograms[lane * params.group_count + group.x] = count;
    }
    if cached_prefix {
        workgroupBarrier();
        if index < params.point_count {
            let word = lane / 32u;
            let preceding = (1u << (lane % 32u)) - 1u;
            ranks[index] = word_prefix[word * 16u + bucket]
                + countOneBits(atomicLoad(&masks[bucket * 8u + word]) & preceding);
        }
        if lane < 16u {
            var earlier = 0u;
            for (var bucket = 0u; bucket < lane; bucket++) {
                earlier += bucket_counts[bucket];
            }
            // Temporarily hold earlier-digit counts for this input workgroup.
            // The ordered prefix pass consumes these before overwriting offsets.
            offsets[lane * params.group_count + group.x] = earlier;
        }
    }
}

// One workgroup per digit. Each lane owns a contiguous chunk of group counts.
// Scan chunk totals in shared memory, then write exclusive group offsets. The
// cached specialization's second component uses an earlier-digit subtotal. Each
// bucket reads only its own offset slice; all lanes finish reading at the first
// barrier, before any lane overwrites that slice with final scatter offsets.
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
        if cached_prefix {
            total.y += offsets[bucket * params.group_count + i];
        } else {
            for (var earlier = 0u; earlier < bucket; earlier++) {
                total.y += histograms[earlier * params.group_count + i];
            }
        }
    }
    sums[lane] = total;
    workgroupBarrier();

    if !cached_prefix {
        for (var stride = 1u; stride < 256u; stride *= 2u) {
            var preceding = vec2<u32>(0u);
            if lane >= stride { preceding = sums[lane - stride]; }
            workgroupBarrier();
            sums[lane] += preceding;
            workgroupBarrier();
        }
        var offset = sums[255u].y + sums[lane].x - total.x;
        for (var i = start; i < end; i++) {
            offsets[bucket * params.group_count + i] = offset;
            offset += histograms[bucket * params.group_count + i];
        }
        return;
    }
    // Four disjoint child subtrees per lane halve the cached tree's barriers.
    // Both integer components are bounded by the point count.
    for (var stride = 1u; stride < 256u; stride *= 4u) {
        let end = (lane + 1u) * stride * 4u - 1u;
        if end < 256u {
            sums[end] += sums[end - 3u * stride] + sums[end - 2u * stride]
                + sums[end - stride];
        }
        workgroupBarrier();
    }
    if lane == 0u {
        bucket_base = sums[255u].y;
        sums[255u] = vec2<u32>(0u);
    }
    workgroupBarrier();
    for (var stride = 64u; stride != 0u; stride /= 4u) {
        let end = (lane + 1u) * stride * 4u - 1u;
        if end < 256u {
            let a = sums[end - 3u * stride];
            let b = sums[end - 2u * stride];
            let c = sums[end - stride];
            let base = sums[end];
            let ab = base + a;
            let abc = ab + b;
            sums[end - 3u * stride] = base;
            sums[end - 2u * stride] = ab;
            sums[end - stride] = abc;
            sums[end] = abc + c;
        }
        workgroupBarrier();
    }

    var offset = bucket_base + sums[lane].x;
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
