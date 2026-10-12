use super::sparse_grid_normals_shader;

#[test]
fn sparse_radius_shader_preserves_parent_expansion() {
    // Rust normalizes inline CRLF; replacement guards also leave blank indents.
    let canonical = |s: &str| s.lines().map(str::trim_end).collect::<Vec<_>>().join("\n");
    assert_eq!(
        canonical(sparse_grid_normals_shader()),
        canonical(&parent_sparse_grid_normals_shader())
    );
}

fn parent_sparse_grid_normals_shader() -> String {
    let declarations = "@group(0) @binding(1) var<storage, read> xs: array<f32>;\n@group(0) @binding(2) var<storage, read> ys: array<f32>;\n@group(0) @binding(3) var<storage, read> zs: array<f32>;\n@group(0) @binding(4) var<storage, read> sorted: array<u32>;\n@group(0) @binding(5) var<storage, read> cell_start: array<u32>;\n@group(0) @binding(6) var<storage, read_write> out_normals: array<vec4<f32>>;";
    let sparse_declarations = r#"struct SparseKey { ix: i32, iy: i32, iz: i32, pad: i32, }
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> keys: array<SparseKey>;
@group(0) @binding(3) var<storage, read> sorted: array<u32>;
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_normals: array<vec4<f32>>;"#;
    let cell_coord = r#"fn cell_coord(value: f32, origin: f32, inv_cell: f32, dim: u32) -> i32 {
    let c = i32(floor((value - origin) * inv_cell));
    return clamp(c, 0, i32(dim) - 1);
}"#;
    let lookup = r#"fn cell_coord(value: f32, origin: f32, inv_cell: f32, dim: u32) -> i32 { return i32(floor((value - origin) * inv_cell)); }
fn key_less(key: SparseKey, x: i32, y: i32, z: i32) -> bool {
    if (key.ix != x) { return key.ix < x; }
    if (key.iy != y) { return key.iy < y; }
    return key.iz < z;
}
fn find_cell(x: i32, y: i32, z: i32, count: u32) -> i32 {
    var low = 0u; var high = count;
    loop { if (low >= high) { break; } let mid = low + (high-low)/2u; if (key_less(keys[mid],x,y,z)) { low=mid+1u; } else { high=mid; } }
    if (low < count) { let key=keys[low]; if (key.ix==x && key.iy==y && key.iz==z) { return i32(low); } }
    return -1;
}"#;
    let dense = "let cid = (u32(nz) * dimy + u32(ny)) * dimx + u32(nx);\n                let begin = cell_start[cid];\n                let end = cell_start[cid + 1u];";
    let sparse = "let found = find_cell(nx, ny, nz, cell_count);\n                if (found < 0) { continue; }\n                let cid = u32(found);\n                let begin = cell_start[cid];\n                let end = select(params.dims.w, cell_start[cid + 1u], cid + 1u < cell_count);";
    include_str!("../../tests/fixtures/grid_radius_previous.wgsl")
        .replace("\r\n", "\n")
        .replace(declarations, sparse_declarations)
        .replace(cell_coord, lookup)
        .replace("let px = xs[i];", "let px = positions[i * 3u];")
        .replace("let py = ys[i];", "let py = positions[i * 3u + 1u];")
        .replace("let pz = zs[i];", "let pz = positions[i * 3u + 2u];")
        .replace("let dimx = params.dims.x;\n    let dimy = params.dims.y;\n    let dimz = params.dims.z;", "let cell_count = params.dims.x;")
        .replace(", dimx);", ", cell_count);")
        .replace(", dimy);", ", cell_count);")
        .replace(", dimz);", ", cell_count);")
        .replace("if (nz < 0 || nz >= i32(dimz)) { continue; }", "")
        .replace("if (ny < 0 || ny >= i32(dimy)) { continue; }", "")
        .replace("if (nx < 0 || nx >= i32(dimx)) { continue; }", "")
        .replace(dense, sparse)
        .replace("xs[j]", "positions[j * 3u]")
        .replace("ys[j]", "positions[j * 3u + 1u]")
        .replace("zs[j]", "positions[j * 3u + 2u]")
}
