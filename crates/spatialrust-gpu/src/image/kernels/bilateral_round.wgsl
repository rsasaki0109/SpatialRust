// Exact byte result; floating arithmetic only proposes a search starting point.
fn product128(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p0=multiply64(a.x,b.x);let p1=multiply64(a.x,b.y);
    let p2=multiply64(a.y,b.x);let p3=multiply64(a.y,b.y);
    let m0=p0.y+p1.x;let m1=m0+p2.x;
    let carry=select(0u,1u,m0<p0.y)+select(0u,1u,m1<m0);
    let h0=p1.y+p2.y;let h1=h0+p3.x;let h2=h1+carry;
    let hc=select(0u,1u,h1<h0)+select(0u,1u,h2<h1);
    return vec4<u32>(p0.x,m1,h2,p3.y+hc);
}
fn at_least128(a:vec4<u32>,b:vec4<u32>)->bool {
    if a.w!=b.w {return a.w>b.w;}if a.z!=b.z {return a.z>b.z;}
    if a.y!=b.y {return a.y>b.y;}return a.x>=b.x;
}
fn rounded_ratio_reaches(n:F64,d:F64,k:u32)->bool {
    var boundary=fp_uint(2u*k+1u);boundary.exponent-=1i;
    let gap=n.exponent-d.exponent-boundary.exponent;
    if gap>=2i {return true;}if gap<=-2i {return false;}
    let extra=select(1u,2u,k==0u);
    let midpoint=wide_sub(wide_left(boundary.mantissa,extra),vec2<u32>(1u,0u));
    let q=u32(gap+52i+i32(extra))-32u;
    let left=vec4<u32>(0u,n.mantissa.x<<q,(n.mantissa.y<<q)|(n.mantissa.x>>(32u-q)),n.mantissa.y>>(32u-q));
    return at_least128(left,product128(d.mantissa,midpoint));
}
// Any candidate in 0..=255 is safe: only exact midpoint comparisons narrow
// the interval or return a result. Failed guesses use bounded integer search.
fn bilateral_byte_bracket(n:F64,d:F64,candidate:u32)->u32 {
    var lo=0u;var hi=255u;
    if candidate>0u {
        if rounded_ratio_reaches(n,d,candidate-1u) {lo=candidate;}
        else {hi=candidate-1u;}
    }
    if lo==candidate {
        if candidate==255u || !rounded_ratio_reaches(n,d,candidate) {return candidate;}
        lo=candidate+1u;
    }
    loop {if lo==hi {break;}let middle=(lo+hi)/2u;
        if rounded_ratio_reaches(n,d,middle) {lo=middle+1u;}else {hi=middle;}
    }
    return lo;
}
fn bilateral_byte(n:F64,d:F64)->u32 {
    if all(n.mantissa==vec2<u32>(0u)) {return 0u;}
    let gap=n.exponent-d.exponent;
    if gap< -1i {return select(0u,1u,rounded_ratio_reaches(n,d,0u));}
    if gap>8i {return 255u;}
    // Normalized high limbs are positive exact 21-bit integers. Scale only
    // within normal f32 range; estimation error never determines the output.
    let scale=bitcast<f32>(u32(gap+127i)<<23u);
    let estimate=f32(n.mantissa.y)/f32(d.mantissa.y)*scale;
    let candidate=u32(clamp(floor(estimate+0.5),0.0,255.0));
    return bilateral_byte_bracket(n,d,candidate);
}
