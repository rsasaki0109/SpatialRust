struct ScanParams {
    element_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ScanParams;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> prefix: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;

var<workgroup> tree: array<u32, 256>;

// Blelloch exclusive scan plus the original element gives an inclusive scan.
// Every lane participates in every barrier, including the final partial block.
@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let i = global_id.x;
    let lane = local_id.x;
    var value = 0u;
    if i < params.element_count { value = source[i]; }
    tree[lane] = value;
    workgroupBarrier();

    for (var stride = 1u; stride < 256u; stride *= 2u) {
        let slot = (lane + 1u) * stride * 2u - 1u;
        if slot < 256u { tree[slot] += tree[slot - stride]; }
        workgroupBarrier();
    }
    if lane == 0u {
        block_sums[group_id.x] = tree[255u];
        tree[255u] = 0u;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride /= 2u) {
        let slot = (lane + 1u) * stride * 2u - 1u;
        if slot < 256u {
            let left = tree[slot - stride];
            tree[slot - stride] = tree[slot];
            tree[slot] += left;
        }
        workgroupBarrier();
    }
    if i < params.element_count { prefix[i] = tree[lane] + value; }
}

// source contains globally inclusive block totals from the next hierarchy level.
@compute @workgroup_size(256)
fn add_block_offsets(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.element_count { return; }
    let block = i / 256u;
    if block > 0u { prefix[i] += source[block - 1u]; }
}
