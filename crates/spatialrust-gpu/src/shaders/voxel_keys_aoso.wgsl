struct Params {
    origin: vec4<f32>,
    inv_leaf: f32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

struct VoxelKey {
    ix: i32,
    iy: i32,
    iz: i32,
    _pad: i32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> keys: array<VoxelKey>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.point_count) {
        return;
    }

    let base = index * 3u;
    let ix = i32(floor((positions[base] - params.origin.x) * params.inv_leaf));
    let iy = i32(floor((positions[base + 1u] - params.origin.y) * params.inv_leaf));
    let iz = i32(floor((positions[base + 2u] - params.origin.z) * params.inv_leaf));
    keys[index] = VoxelKey(ix, iy, iz, 0);
}
