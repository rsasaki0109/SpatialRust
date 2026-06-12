struct Params {
    padded_count: u32,
    pair_distance: u32,
    block_width: u32,
    _pad: u32,
}

struct VoxelSortEntry {
    ix: i32,
    iy: i32,
    iz: i32,
    point_index: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> entries: array<VoxelSortEntry>;

fn less_than(a: VoxelSortEntry, b: VoxelSortEntry) -> bool {
    if a.ix != b.ix {
        return a.ix < b.ix;
    }
    if a.iy != b.iy {
        return a.iy < b.iy;
    }
    if a.iz != b.iz {
        return a.iz < b.iz;
    }
    return a.point_index < b.point_index;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.padded_count {
        return;
    }

    let j = i ^ params.pair_distance;
    if j <= i {
        return;
    }

    let ascending = (i & params.block_width) == 0u;
    let left = entries[i];
    let right = entries[j];
    let swap = select(less_than(left, right), less_than(right, left), ascending);
    if swap {
        entries[i] = right;
        entries[j] = left;
    }
}
