struct Params {
    points: u32,
    cells: u32,
    radius_sq: f32,
    all_connected: u32,
};
struct Key { x: i32, y: i32, z: i32, pad: i32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> z: array<f32>;
@group(0) @binding(4) var<storage, read> keys: array<Key>;
@group(0) @binding(5) var<storage, read> sorted: array<u32>;
@group(0) @binding(6) var<storage, read> starts: array<u32>;
@group(0) @binding(7) var<storage, read_write> parent: array<atomic<u32>>;

fn root(index: u32) -> u32 {
    var node = index;
    loop {
        let next = atomicLoad(&parent[node]);
        if (next == node) { break; }
        let grandparent = atomicLoad(&parent[next]);
        atomicMin(&parent[node], grandparent);
        node = next;
    }
    return node;
}

fn unite(a: u32, b: u32) {
    loop {
        let ra = root(a);
        let rb = root(b);
        if (ra == rb) { return; }
        let lower = min(ra, rb);
        let upper = max(ra, rb);
        // Retry even a spurious weak-CAS failure. Only a current root is hooked,
        // always toward a smaller index, so no cycle or lock owner can arise.
        let result = atomicCompareExchangeWeak(&parent[upper], upper, lower);
        if (result.exchanged) { return; }
    }
}

fn key_less(key: Key, nx: i32, ny: i32, nz: i32) -> bool {
    if (key.x != nx) { return key.x < nx; }
    if (key.y != ny) { return key.y < ny; }
    return key.z < nz;
}

fn find_cell(nx: i32, ny: i32, nz: i32) -> u32 {
    var low = 0u;
    var high = params.cells;
    loop {
        if (low >= high) { break; }
        let mid = low + (high - low) / 2u;
        if (key_less(keys[mid], nx, ny, nz)) { low = mid + 1u; }
        else { high = mid; }
    }
    if (low < params.cells) {
        let key = keys[low];
        if (key.x == nx && key.y == ny && key.z == nz) { return low; }
    }
    return params.cells;
}

fn offset_fits(key: i32, delta: i32) -> bool {
    return !((key == 2147483647 && delta == 1) || (key == (-2147483647 - 1) && delta == -1));
}

@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.points) {
        atomicStore(&parent[gid.x], select(gid.x, 0u, params.all_connected != 0u));
    }
}

@compute @workgroup_size(256)
fn connect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.points || params.all_connected != 0u) { return; }
    // Compacted starts are ascending. Recover this sorted point's actual cell
    // rather than recomputing floating voxel coordinates or downloading keys.
    var low = 0u;
    var high = params.cells;
    loop {
        if (low >= high) { break; }
        let mid = low + (high - low) / 2u;
        if (starts[mid] <= s) { low = mid + 1u; } else { high = mid; }
    }
    let key = keys[low - 1u];
    let i = sorted[s];
    for (var dz = -1; dz <= 1; dz++) {
        if (!offset_fits(key.z, dz)) { continue; }
        for (var dy = -1; dy <= 1; dy++) {
            if (!offset_fits(key.y, dy)) { continue; }
            for (var dx = -1; dx <= 1; dx++) {
                if (!offset_fits(key.x, dx)) { continue; }
                let cell = find_cell(key.x + dx, key.y + dy, key.z + dz);
                if (cell == params.cells) { continue; }
                var end = params.points;
                if (cell + 1u < params.cells) { end = starts[cell + 1u]; }
                for (var t = starts[cell]; t < end; t++) {
                    let j = sorted[t];
                    if (j <= i) { continue; }
                    let vx = x[j] - x[i];
                    let vy = y[j] - y[i];
                    let vz = z[j] - z[i];
                    if (vx * vx + vy * vy + vz * vz <= params.radius_sq) { unite(i, j); }
                }
            }
        }
    }
}

@compute @workgroup_size(256)
fn compress(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.points) { atomicStore(&parent[gid.x], root(gid.x)); }
}
