struct CompactParams {
    point_count: u32,
    scan_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

struct VoxelSortEntry {
    ix: i32,
    iy: i32,
    iz: i32,
    point_index: u32,
}

struct VoxelKeyOutput {
    ix: i32,
    iy: i32,
    iz: i32,
    _pad: i32,
}

@group(0) @binding(0) var<uniform> params: CompactParams;
@group(0) @binding(1) var<storage, read> entries: array<VoxelSortEntry>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> prefix: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys: array<VoxelKeyOutput>;
@group(0) @binding(5) var<storage, read_write> cell_starts: array<u32>;
@group(0) @binding(6) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> point_indices: array<u32>;
@group(0) @binding(8) var<storage, read_write> scan_out: array<u32>;

@compute @workgroup_size(256)
fn mark_boundaries(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.point_count {
        return;
    }

    if i == 0u {
        flags[i] = 1u;
        return;
    }

    let previous = entries[i - 1u];
    let current = entries[i];
    let changed = (current.ix != previous.ix)
        || (current.iy != previous.iy)
        || (current.iz != previous.iz);
    flags[i] = select(0u, 1u, changed);
}

@compute @workgroup_size(256)
fn init_inclusive(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.point_count {
        return;
    }
    prefix[i] = flags[i];
}

@compute @workgroup_size(256)
fn scan_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.point_count {
        return;
    }

    let stride = params.scan_stride;
    if stride == 0u || i < stride {
        scan_out[i] = prefix[i];
        return;
    }

    scan_out[i] = prefix[i] + prefix[i - stride];
}

@compute @workgroup_size(256)
fn write_segments(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.point_count {
        return;
    }

    let boundary = flags[i];
    let inclusive_count = prefix[i];
    let cell_id = inclusive_count - 1u;

    point_indices[i] = entries[i].point_index;

    if boundary == 1u {
        keys[cell_id] = VoxelKeyOutput(
            entries[i].ix,
            entries[i].iy,
            entries[i].iz,
            0,
        );
        cell_starts[cell_id] = i;
    }

    atomicAdd(&cell_counts[cell_id], 1u);
}
