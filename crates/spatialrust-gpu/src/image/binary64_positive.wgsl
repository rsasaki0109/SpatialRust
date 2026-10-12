// Finite non-negative binary64 subset for CLAHE geometry and interpolation.
// Values here are normal or zero; no sign, subnormal, infinity or NaN inputs.
// mantissa is an exact normalized 53-bit integer, exponent is unbiased.
struct F64 { mantissa: vec2<u32>, exponent: i32, };
fn wide_left(a: vec2<u32>, shift: u32) -> vec2<u32> {
    if shift == 0u { return a; }
    if shift >= 32u { return vec2<u32>(0u, a.x << (shift - 32u)); }
    return vec2<u32>(a.x << shift, (a.y << shift) | (a.x >> (32u - shift)));
}
fn wide_right(a: vec2<u32>, shift: u32) -> vec2<u32> {
    if shift == 0u { return a; }
    if shift >= 32u { return vec2<u32>(a.y >> (shift - 32u), 0u); }
    return vec2<u32>((a.x >> shift) | (a.y << (32u - shift)), a.y >> shift);
}
fn wide_jam(a: vec2<u32>, shift: u32) -> vec2<u32> {
    if shift == 0u { return a; }
    if shift >= 64u { return vec2<u32>(select(0u, 1u, any(a != vec2<u32>(0u))), 0u); }
    if shift == 32u { return vec2<u32>(a.y | select(0u, 1u, a.x != 0u), 0u); }
    if shift > 32u {
        let lost = a.x != 0u || (a.y << (64u - shift)) != 0u;
        return vec2<u32>((a.y >> (shift - 32u)) | select(0u, 1u, lost), 0u);
    }
    var result = wide_right(a, shift);
    result.x |= select(0u, 1u, (a.x << (32u - shift)) != 0u);
    return result;
}
fn wide_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}
fn wide_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn fp_rounded(mantissa: vec2<u32>, exponent: i32) -> F64 {
    var m = mantissa;
    var e = exponent;
    if any(m != vec2<u32>(0u)) {
        let leading = select(firstLeadingBit(m.x), 32u + firstLeadingBit(m.y), m.y != 0u);
        if leading < 55u { m = wide_left(m, 55u - leading); e -= i32(55u - leading); }
        if leading > 55u { m = wide_jam(m, 1u); e += 1i; }
        let remainder = m.x & 7u;
        m = wide_right(m, 3u);
        if remainder > 4u || (remainder == 4u && (m.x & 1u) != 0u) {
            m = wide_add(m, vec2<u32>(1u, 0u));
        }
        if (m.y & 2097152u) != 0u { m = wide_right(m, 1u); e += 1i; }
        return F64(m, e);
    }
    return F64(vec2<u32>(0u), 0i);
}
fn fp_uint(value: u32) -> F64 {
    if value == 0u { return F64(vec2<u32>(0u), 0i); }
    let e = firstLeadingBit(value);
    return F64(wide_left(vec2<u32>(value, 0u), 52u - e), i32(e));
}
fn fp_less(a: F64, b: F64) -> bool {
    if all(a.mantissa == vec2<u32>(0u)) { return any(b.mantissa != vec2<u32>(0u)); }
    if all(b.mantissa == vec2<u32>(0u)) { return false; }
    return a.exponent < b.exponent ||
        (a.exponent == b.exponent && less64(a.mantissa, b.mantissa));
}
fn fp_sum(a: F64, b: F64) -> F64 {
    if all(a.mantissa == vec2<u32>(0u)) { return b; }
    if all(b.mantissa == vec2<u32>(0u)) { return a; }
    var large = a; var small = b;
    if fp_less(a, b) { large = b; small = a; }
    return fp_rounded(wide_add(wide_left(large.mantissa, 3u),
        wide_jam(wide_left(small.mantissa, 3u), u32(large.exponent - small.exponent))), large.exponent);
}
// Magnitude of the difference; CLAHE callers pass ordered non-negative values.
fn fp_difference(a: F64, b: F64) -> F64 {
    if all(a.mantissa == vec2<u32>(0u)) { return b; }
    if all(b.mantissa == vec2<u32>(0u)) { return a; }
    var large = a; var small = b;
    if fp_less(a, b) { large = b; small = a; }
    return fp_rounded(wide_sub(wide_left(large.mantissa, 3u),
        wide_jam(wide_left(small.mantissa, 3u), u32(large.exponent - small.exponent))), large.exponent);
}
fn fp_product(a: F64, b: F64) -> F64 {
    if all(a.mantissa == vec2<u32>(0u)) || all(b.mantissa == vec2<u32>(0u)) {
        return F64(vec2<u32>(0u), 0i);
    }
    let p0 = multiply64(a.mantissa.x, b.mantissa.x);
    let p1 = multiply64(a.mantissa.x, b.mantissa.y);
    let p2 = multiply64(a.mantissa.y, b.mantissa.x);
    let p3 = multiply64(a.mantissa.y, b.mantissa.y);
    let mid0 = p0.y + p1.x;
    let mid1 = mid0 + p2.x;
    let carry = select(0u, 1u, mid0 < p0.y) + select(0u, 1u, mid1 < mid0);
    let high0 = p1.y + p2.y;
    let high1 = high0 + p3.x;
    let high2 = high1 + carry;
    let high_carry = select(0u, 1u, high1 < high0) + select(0u, 1u, high2 < high1);
    let product = vec4<u32>(p0.x, mid1, high2, p3.y + high_carry);
    let top = (product.w & 512u) != 0u;
    let shift = select(20u, 21u, top);
    var m = vec2<u32>((product.y >> shift) | (product.z << (32u - shift)),
        (product.z >> shift) | (product.w << (32u - shift)));
    let guard = (product.y & (1u << (shift - 1u))) != 0u;
    let sticky = product.x != 0u || (product.y & ((1u << (shift - 1u)) - 1u)) != 0u;
    var e = a.exponent + b.exponent + select(0i, 1i, top);
    if guard && (sticky || (m.x & 1u) != 0u) { m = wide_add(m, vec2<u32>(1u, 0u)); }
    if (m.y & 2097152u) != 0u { m = wide_right(m, 1u); e += 1i; }
    return F64(m, e);
}
fn fp_ratio(n: u32, d: u32) -> F64 {
    if n == 0u { return F64(vec2<u32>(0u), 0i); }
    // Positive dyadic divisors only shift the exponent of an exact u32 value.
    // The smallest supported ratio is still normal, so no rounding is needed.
    if d != 0u && (d & (d - 1u)) == 0u {
        var value = fp_uint(n);
        value.exponent -= i32(firstLeadingBit(d));
        return value;
    }
    var e = i32(firstLeadingBit(n)) - i32(firstLeadingBit(d));
    if e >= 0i {
        if less64(vec2<u32>(n, 0u), shift64(d, u32(e))) { e -= 1i; }
    } else {
        if less64(shift64(n, u32(-e)), vec2<u32>(d, 0u)) { e -= 1i; }
    }
    var m = vec2<u32>(n / d, 0u);
    var remainder = n % d;
    for (var i = 0u; i < u32(52i - e); i++) {
        let doubled = remainder << 1u;
        let bit = (remainder >> 31u) != 0u || doubled >= d;
        remainder = select(doubled, doubled - d, bit);
        m = wide_left(m, 1u);
        m.x |= select(0u, 1u, bit);
    }
    if remainder > d - remainder || (remainder == d - remainder && (m.x & 1u) != 0u) {
        m = wide_add(m, vec2<u32>(1u, 0u));
    }
    if (m.y & 2097152u) != 0u { m = wide_right(m, 1u); e += 1i; }
    return F64(m, e);
}
fn fp_encode(a: F64) -> vec2<u32> {
    if all(a.mantissa == vec2<u32>(0u)) { return vec2<u32>(0u); }
    return vec2<u32>(a.mantissa.x, (a.mantissa.y & 1048575u) | (u32(a.exponent + 1023i) << 20u));
}
fn fp_decode(bits: vec2<u32>) -> F64 {
    if all(bits == vec2<u32>(0u)) { return F64(vec2<u32>(0u), 0i); }
    return F64(vec2<u32>(bits.x, (bits.y & 1048575u) | 1048576u), i32(bits.y >> 20u) - 1023i);
}
fn fp_floor(a: F64) -> u32 {
    if a.exponent < 0i || all(a.mantissa == vec2<u32>(0u)) { return 0u; }
    return wide_right(a.mantissa, u32(52i - a.exponent)).x;
}
fn fp_byte(a: F64) -> u32 {
    let q = fp_floor(a);
    let fraction = fp_difference(a, fp_uint(q));
    let half = F64(vec2<u32>(0u, 1048576u), -1i);
    if fp_less(fraction, half) { return min(q, 255u); }
    if fp_less(half, fraction) { return min(q + 1u, 255u); }
    return min(q + (q & 1u), 255u);
}
