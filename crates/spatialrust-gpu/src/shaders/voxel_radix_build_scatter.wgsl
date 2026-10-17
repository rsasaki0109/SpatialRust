// First radix digit reads keys and constructs indices only at stable destinations.
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
fn scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.point_count {
        return;
    }
    let entry = input_entries[index];
    let bucket = digit(entry);
    let destination = offsets[bucket * params.group_count + index / 256u] + ranks[index];
    output_entries[destination] = VoxelSortEntry(entry.ix, entry.iy, entry.iz, index);
}
