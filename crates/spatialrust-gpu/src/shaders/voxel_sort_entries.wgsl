struct Params {
    point_count: u32,
    padded_count: u32,
    _pad0: u32,
    _pad1: u32,
}

struct VoxelKeyOutput {
    ix: i32,
    iy: i32,
    iz: i32,
    _pad: i32,
}

struct VoxelSortEntry {
    ix: i32,
    iy: i32,
    iz: i32,
    point_index: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<VoxelKeyOutput>;
@group(0) @binding(2) var<storage, read_write> entries: array<VoxelSortEntry>;

@compute @workgroup_size(256)
fn build_sort_entries(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.padded_count {
        return;
    }

    if i < params.point_count {
        let key = keys[i];
        entries[i] = VoxelSortEntry(key.ix, key.iy, key.iz, i);
        return;
    }

    entries[i] = VoxelSortEntry(2147483647, 2147483647, 2147483647, 4294967295u);
}
