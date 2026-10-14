struct Params { count: u32, unused0: u32, unused1: u32, unused2: u32 }
struct SampleBits { position: vec4<u32>, normal: vec4<u32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> normals: array<u32>;
@group(0) @binding(3) var<storage, read_write> samples: array<SampleBits>;

// Integer copies retain every coordinate/normal bit, with explicit zero padding.
@compute @workgroup_size(64)
fn gather_samples(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i < params.count {
        samples[i] = SampleBits(vec4<u32>(positions[i].xyz, 0u),
            vec4<u32>(normals[3u*i], normals[3u*i+1u], normals[3u*i+2u], 0u));
    }
}
