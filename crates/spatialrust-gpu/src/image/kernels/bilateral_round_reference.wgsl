// Positive finite normal numerator/denominator; output matches
// (n/d).round().clamp(0,255) after CPU binary64 round-to-nearest-even division.
fn product128(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p0 = multiply64(a.x,b.x); let p1=multiply64(a.x,b.y);
    let p2=multiply64(a.y,b.x); let p3=multiply64(a.y,b.y);
    let m0=p0.y+p1.x; let m1=m0+p2.x;
    let carry=select(0u,1u,m0<p0.y)+select(0u,1u,m1<m0);
    let h0=p1.y+p2.y; let h1=h0+p3.x; let h2=h1+carry;
    let hc=select(0u,1u,h1<h0)+select(0u,1u,h2<h1);
    return vec4<u32>(p0.x,m1,h2,p3.y+hc);
}
fn at_least128(a:vec4<u32>, b:vec4<u32>)->bool {
    if a.w!=b.w {return a.w>b.w;} if a.z!=b.z {return a.z>b.z;}
    if a.y!=b.y {return a.y>b.y;} return a.x>=b.x;
}
fn rounded_ratio_reaches(n:F64,d:F64,k:u32)->bool {
    var boundary=fp_uint(2u*k+1u); boundary.exponent-=1i;
    let gap=n.exponent-d.exponent-boundary.exponent;
    if gap>=2i {return true;} if gap<=-2i {return false;}
    // Half-byte thresholds have an even significand. A midpoint below the
    // threshold therefore rounds upward to it, then byte rounding is half-up.
    // 0.5 is a power-of-two boundary: the predecessor has half the ordinary ULP.
    let extra=select(1u,2u,k==0u);
    let midpoint=wide_sub(wide_left(boundary.mantissa,extra),vec2<u32>(1u,0u));
    let q=u32(gap+52i+i32(extra))-32u;
    let left=vec4<u32>(0u,n.mantissa.x<<q,(n.mantissa.y<<q)|(n.mantissa.x>>(32u-q)),n.mantissa.y>>(32u-q));
    return at_least128(left,product128(d.mantissa,midpoint));
}
fn bilateral_byte(n:F64,d:F64)->u32 {
    if all(n.mantissa==vec2<u32>(0u)) {return 0u;}
    var lo=0u;var hi=255u;
    loop {if lo==hi {break;} let middle=(lo+hi)/2u;
        if rounded_ratio_reaches(n,d,middle) {lo=middle+1u;} else {hi=middle;}
    }
    return lo;
}
