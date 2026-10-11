@group(0) @binding(0) var<storage, read> cdf: array<u32>;
@group(0) @binding(1) var<storage, read_write> lookup: array<u32>;
var<workgroup> minimum: u32;
// denominator, reciprocal-rounding error magnitude/direction, exponent + 25
var<workgroup> scale_info: vec4<u32>;

@compute @workgroup_size(256)
fn equalize_lookup(@builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        minimum = 0u;
        for (var i = 0u; i < 256u; i++) {
            if cdf[i] != 0u { minimum = cdf[i]; break; }
        }
        let denominator = cdf[255u] - minimum;
        scale_info = vec4<u32>(0u);
        if denominator != 0u { scale_info = reciprocal_info(denominator); }
    }
    workgroupBarrier();
    if scale_info.x == 0u {
        lookup[lane] = lane; // The CPU preserves a constant image's original byte.
    } else if cdf[lane] <= minimum {
        lookup[lane] = 0u;
    } else {
        lookup[lane] = mapped(cdf[lane] - minimum, scale_info);
    }
}
