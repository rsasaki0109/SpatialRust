struct FilterParams {
    point_count: u32,
    padded_count: u32,
    scan_stride: u32,
    _pad: u32,
}

struct VoxelSortEntry {
    ix: i32,
    iy: i32,
    iz: i32,
    point_index: u32,
}

@group(0) @binding(0) var<uniform> params: FilterParams;
@group(0) @binding(1) var<storage, read> sorted: array<VoxelSortEntry>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read> scan_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> scan_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<VoxelSortEntry>;

@compute @workgroup_size(256)
fn mark_valid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.padded_count {
        return;
    }

    flags[i] = select(0u, 1u, sorted[i].point_index < params.point_count);
}

@compute @workgroup_size(256)
fn init_inclusive(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.padded_count {
        return;
    }
    scan_out[i] = flags[i];
}

@compute @workgroup_size(256)
fn scan_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.padded_count {
        return;
    }

    let stride = params.scan_stride;
    if stride == 0u || i < stride {
        scan_out[i] = scan_in[i];
        return;
    }

    scan_out[i] = scan_in[i] + scan_in[i - stride];
}

@compute @workgroup_size(256)
fn scatter_valid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.padded_count {
        return;
    }

    if flags[i] == 0u {
        return;
    }

    let out_idx = scan_in[i] - flags[i];
    output[out_idx] = sorted[i];
}
