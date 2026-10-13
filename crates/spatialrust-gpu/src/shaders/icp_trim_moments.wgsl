// Carry two f32 words through normalization, mean division and products.
// Matching/queries stay f32. Existing add/merge retain compensated reductions.
// Host always supplies params.unused=0. The runtime integer operation preserves
// each rounded f32 value while preventing elimination of error-free transforms
// through floating-point reassociation across their dependent expressions.
fn ds_round(value: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(value)^params.unused);
}
fn precise_add(a: vec2<f32>, b: f32) -> vec2<f32> {
    let s=ds_round(a.x+b); let v=ds_round(s-a.x);
    let e=ds_round(ds_round(a.x-ds_round(s-v))+ds_round(b-v))+a.y;
    let h=ds_round(s+e);
    return vec2<f32>(h,ds_round(e-ds_round(h-s)));
}
fn precise_merge(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return precise_add(precise_add(a,b.x),b.y);
}
fn ds_multiply_words(a: u32, b: u32) -> vec2<u32> {
    let a0=a&65535u; let a1=a>>16u; let b0=b&65535u; let b1=b>>16u;
    let w0=a0*b0; let t=a1*b0+(w0>>16u); let w1=a0*b1+(t&65535u);
    return vec2<u32>((w1<<16u)|(w0&65535u),a1*b1+(t>>16u)+(w1>>16u));
}
fn ds_scale_word(mantissa: u32, exponent: i32) -> f32 {
    if mantissa==0u { return 0.0; }
    let bits=firstLeadingBit(mantissa)+1u;
    let fraction=f32(mantissa)*bitcast<f32>((127u-bits)<<23u);
    let combined=exponent+i32(bits);
    if combined>128 { return bitcast<f32>(0x7f800000u); }
    return ldexp(fraction,combined);
}
// Integer 24x24 multiplication prevents floating-point contraction or an
// unfused WGSL fma from erasing the exact product's low word.
fn two_product(a: f32, b: f32) -> vec2<f32> {
    let x=bitcast<u32>(a); let y=bitcast<u32>(b);
    let xe=(x>>23u)&255u; let ye=(y>>23u)&255u;
    if xe==255u || ye==255u { return vec2<f32>(bitcast<f32>(0x7f800000u),0.0); }
    let xm=(x&0x7fffffu)|select(0x800000u,0u,xe==0u);
    let ym=(y&0x7fffffu)|select(0x800000u,0u,ye==0u);
    let exponent=select(i32(xe)-150,-149,xe==0u)+select(i32(ye)-150,-149,ye==0u);
    let product=ds_multiply_words(xm,ym);
    let upper=(product.x>>24u)|(product.y<<8u);
    let lower=product.x&0xffffffu;
    let sign=(x^y)&0x80000000u;
    let hi=bitcast<f32>(bitcast<u32>(ds_scale_word(upper,exponent+24))|sign);
    let lo=bitcast<f32>(bitcast<u32>(ds_scale_word(lower,exponent))|sign);
    return vec2<f32>(hi,lo);
}
fn ds_divide(a: vec2<f32>, divisor: f32) -> vec2<f32> {
    let quotient=a.x/divisor;
    let product=two_product(quotient,divisor);
    let remainder=precise_merge(a,-product);
    let correction=(remainder.x+remainder.y)/divisor;
    return precise_add(vec2<f32>(quotient,0.0),correction);
}
fn ds_offset(point: f32, anchor: f32, scale: f32) -> vec2<f32> {
    return ds_divide(precise_add(vec2<f32>(point,0.0),-anchor),scale);
}
fn ds_product(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    var total=two_product(a.x,b.x);
    total=precise_add(total,a.x*b.y);
    total=precise_add(total,a.y*b.x);
    return precise_add(total,a.y*b.y);
}
