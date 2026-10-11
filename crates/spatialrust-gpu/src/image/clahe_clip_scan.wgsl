// Exclusive tree scan of (bounded count, clipped excess). Integer sums stay
// within tile area. Redistribution's inclusive prefix is computed directly,
// preserving the CPU's residual positions without a second scan.
var<workgroup> clip_prefix: array<vec2<u32>, 256>;
var<workgroup> clip_excess: u32;
fn clahe_cumulative(lane: u32, count: u32, limit: u32) -> u32 {
    let bounded = select(count, min(count, limit), limit != 0u);
    clip_prefix[lane] = vec2<u32>(bounded, count - bounded);
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        let end = (lane + 1u) * stride * 2u - 1u;
        if end < 256u { clip_prefix[end] += clip_prefix[end - stride]; }
        workgroupBarrier();
    }
    if lane == 0u {
        clip_excess = clip_prefix[255].y;
        clip_prefix[255] = vec2<u32>(0u);
    }
    workgroupBarrier();
    for (var stride = 128u; stride != 0u; stride /= 2u) {
        let end = (lane + 1u) * stride * 2u - 1u;
        if end < 256u {
            let left = clip_prefix[end - stride];
            clip_prefix[end - stride] = clip_prefix[end];
            clip_prefix[end] += left;
        }
        workgroupBarrier();
    }
    let batch = clip_excess / 256u;
    let residual = clip_excess % 256u;
    var step = 1u;
    if residual != 0u { step = 256u / residual; }
    return clip_prefix[lane].x + bounded + batch * (lane + 1u)
        + min(residual, lane / step + 1u);
}
