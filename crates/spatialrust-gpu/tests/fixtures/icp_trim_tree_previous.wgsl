// Stable MSD selection of (nonnegative distance, original source index).
// Only the selected prefix is counted; matches never leave the device.
struct Params {
    count: u32, groups: u32, step: u32, mantissa_lo: u32,
    mantissa_hi: u32, shift: u32, bound: f32, targets: u32,
}
struct Neighbor { index: u32, distance: f32 }
struct Selection {
    gated: u32, retained: u32, rank: u32, distance: u32,
    source_index: u32, unused0: u32, unused1: u32, unused2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> matches: array<Neighbor>;
@group(0) @binding(2) var<storage,read_write> histograms: array<u32>;
@group(0) @binding(3) var<storage,read_write> selection: Selection;
var<workgroup> histogram: array<atomic<u32>,256>;
var<workgroup> totals: array<u32,256>;
override select_lanes: u32 = 64u;

fn multiply_wide(a: u32, b: u32) -> vec2<u32> {
    let a0=a&65535u; let a1=a>>16u; let b0=b&65535u; let b1=b>>16u;
    let w0=a0*b0; let t=a1*b0+(w0>>16u);
    let w1=a0*b1+(t&65535u);
    return vec2<u32>((w1<<16u)|(w0&65535u),a1*b1+(t>>16u)+(w1>>16u));
}

// Emulate floor(f64(count) * fraction), including the binary64 product's
// round-to-nearest-even before floor. Casting the fraction to f32 changes K.
fn retained_count(count: u32) -> u32 {
    let lo=multiply_wide(count,params.mantissa_lo);
    let hi=multiply_wide(count,params.mantissa_hi);
    let middle=lo.y+hi.x;
    var p=vec3<u32>(lo.x,middle,hi.y+select(0u,1u,middle<lo.y));
    var bits=0u;
    if p.z!=0u { bits=65u+firstLeadingBit(p.z); }
    else if p.y!=0u { bits=33u+firstLeadingBit(p.y); }
    else if p.x!=0u { bits=1u+firstLeadingBit(p.x); }
    if bits>53u {
        let discarded_bits=bits-53u; // At most 32 bits for a u32 count.
        var mask=0xffffffffu;
        var unit=0u;
        var odd=(p.y&1u)!=0u;
        if discarded_bits<32u { unit=1u<<discarded_bits; mask=unit-1u; odd=(p.x&unit)!=0u; }
        let remainder=p.x&mask;
        let half=1u<<(discarded_bits-1u);
        let up=remainder>half || (remainder==half && odd);
        p.x=p.x&~mask;
        if up {
            let before=p.x; p.x+=unit;
            if discarded_bits==32u || p.x<before {
                p.y+=1u;
                if p.y==0u { p.z+=1u; }
            }
        }
    }
    // A fraction in (0,1] has shift >=52. Subnormal fractions retain zero.
    if params.shift>=96u { return 0u; }
    if params.shift>=64u { return p.z>>(params.shift-64u); }
    return (p.y>>(params.shift-32u))|(p.z<<(64u-params.shift));
}

fn distance_key(distance: f32) -> u32 {
    return bitcast<u32>(distance)&0x7fffffffu;
}

@compute @workgroup_size(64)
fn histogram_prefix(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    for (var digit=lane;digit<256u;digit+=64u) { atomicStore(&histogram[digit],0u); }
    workgroupBarrier();
    let shift=24u-(params.step%4u)*8u;
    for (var i=group.x*64u+lane;i<params.count;i+=params.groups*64u) {
        let m=matches[i];
        if m.index>=params.targets || !(m.distance>=0.0 && m.distance<=params.bound) { continue; }
        let distance=distance_key(m.distance);
        var key=distance;
        var prefix=selection.distance;
        if params.step>=4u {
            if distance!=selection.distance { continue; }
            key=i; prefix=selection.source_index;
        }
        if params.step!=0u {
            if selection.retained==0u { continue; }
            // Avoid the undefined/implementation-dependent shift by 32.
            if shift<24u && (key>>(shift+8u))!=(prefix>>(shift+8u)) { continue; }
        }
        atomicAdd(&histogram[(key>>shift)&255u],1u);
    }
    workgroupBarrier();
    for (var digit=lane;digit<256u;digit+=64u) {
        histograms[group.x*256u+digit]=atomicLoad(&histogram[digit]);
    }
}

@compute @workgroup_size(select_lanes)
fn select_digit(@builtin(local_invocation_index) lane: u32) {
    // A capable device can assign one lane per bin instead of four bins per
    // lane, without changing integer sums, scratch, barriers or pass count.
    for (var digit=lane;digit<256u;digit+=select_lanes) {
        var total=0u;
        for (var g=0u;g<params.groups;g+=1u) { total+=histograms[g*256u+digit]; }
        totals[digit]=total;
    }
    workgroupBarrier();
    if lane!=0u { return; }
    if params.step==0u {
        var gated=0u;
        for (var digit=0u;digit<256u;digit+=1u) { gated+=totals[digit]; }
        selection.gated=gated; selection.retained=retained_count(gated);
        selection.rank=select(0u,selection.retained-1u,selection.retained>0u);
    }
    if selection.retained==0u { return; }
    var preceding=0u;
    for (var digit=0u;digit<256u;digit+=1u) {
        if selection.rank-preceding<totals[digit] {
            selection.rank-=preceding;
            let shift=24u-(params.step%4u)*8u;
            if params.step<4u { selection.distance|=digit<<shift; }
            else { selection.source_index|=digit<<shift; }
            break;
        }
        preceding+=totals[digit];
    }
}
