@group(0) @binding(0) var<storage, read> cdf: array<u32>;
@group(0) @binding(1) var<storage, read_write> lookup: array<u32>;
var<workgroup> minimum: u32;
// denominator, reciprocal-rounding error magnitude/direction, exponent + 25
var<workgroup> scale_info: vec4<u32>;

fn less64(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

fn shift64(a: u32, shift: u32) -> vec2<u32> {
    if shift == 0u { return vec2<u32>(a, 0u); }
    if shift == 32u { return vec2<u32>(0u, a); }
    return vec2<u32>(a << shift, a >> (32u - shift));
}

fn multiply64(a: u32, b: u32) -> vec2<u32> {
    let p0 = (a & 65535u) * (b & 65535u);
    let p1 = (a >> 16u) * (b & 65535u);
    let p2 = (a & 65535u) * (b >> 16u);
    let middle = (p0 >> 16u) + (p1 & 65535u) + (p2 & 65535u);
    return vec2<u32>((p0 & 65535u) | (middle << 16u),
        (a >> 16u) * (b >> 16u) + (p1 >> 16u) + (p2 >> 16u) + (middle >> 16u));
}

// Exact IEEE binary64 division rounding of 255/d, represented only by its
// exponent and signed rational error. This needs no shader f64 or f32.
fn reciprocal_info(d: u32) -> vec4<u32> {
    var exponent = 7i;
    loop {
        var fits = false;
        if exponent >= 0i {
            fits = !less64(vec2<u32>(255u, 0u), shift64(d, u32(exponent)));
        } else {
            fits = !less64(shift64(255u, u32(-exponent)), vec2<u32>(d, 0u));
        }
        if fits { break; }
        exponent -= 1i;
    }
    var remainder = 255u % d;
    var last_bit = (255u / d) & 1u;
    for (var i = 0u; i < u32(52i - exponent); i++) {
        let doubled = remainder << 1u;
        let bit = (remainder >> 31u) != 0u || doubled >= d;
        // Modular u32 subtraction also handles the carried high bit.
        remainder = select(doubled, doubled - d, bit);
        last_bit = select(0u, 1u, bit);
    }
    let above = remainder > d - remainder ||
        (remainder == d - remainder && last_bit != 0u);
    return vec4<u32>(d, select(remainder, d - remainder, above),
        select(0u, 1u, above), u32(exponent + 25i));
}

fn mapped(n: u32, info: vec4<u32>) -> u32 {
    let d = info.x;
    let product = multiply64(n, 255u);
    var quotient = 0u;
    for (var bit = 128u; bit > 0u; bit >>= 1u) {
        let candidate = quotient | bit;
        if !less64(product, multiply64(d, candidate)) { quotient = candidate; }
    }
    let remainder = product.x - multiply64(d, quotient).x;
    if remainder < d - remainder { return quotient; }
    if remainder > d - remainder { return quotient + 1u; }

    // Non-half rational values are farther from a rounding boundary than the
    // binary64 error for u32 counts. At an exact half, emulate the CPU's rounded
    // reciprocal then rounded multiply, including a one-ulp drift off the half.
    var exponent = -1i;
    var bits = quotient;
    loop {
        if bits == 0u { break; }
        exponent += 1i;
        bits >>= 1u;
    }
    let error = multiply64(n, info.y);
    let doubled_error = vec2<u32>(error.x << 1u, (error.y << 1u) | (error.x >> 31u));
    let half_ulp = shift64(d, u32(exponent - i32(info.w) + 25i));
    if less64(half_ulp, doubled_error) { return quotient + info.z; }
    return quotient + (quotient & 1u);
}

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
