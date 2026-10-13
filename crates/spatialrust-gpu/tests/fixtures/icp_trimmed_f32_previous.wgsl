struct Params {
    row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>,
    source_anchor: vec4<f32>, target_anchor: vec4<f32>,
    count: u32, targets: u32, groups: u32, unused: u32,
    bound: f32, max_distance: f32, scale: f32, padding: f32,
}
struct Neighbor { index: u32, distance: f32 }
struct Partial { values: array<vec2<f32>,16>, count: u32, pad0: u32, pad1: u32, pad2: u32 }
struct Statistics {
    status: atomic<u32>, count: u32, reserved0: u32, reserved1: u32,
    source_anchor: vec4<f32>, target_anchor: vec4<f32>, values: array<vec2<f32>,16>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> source: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> target_points: array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> queries: array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> matches: array<Neighbor>;
@group(0) @binding(5) var<storage,read_write> partials: array<Partial>;
@group(0) @binding(6) var<storage,read_write> result: Statistics;
var<workgroup> sums: array<vec2<f32>,64>;
var<workgroup> counts: array<u32,64>;

fn finite(v: f32) -> bool { return abs(v)<=3.4028234663852886e38; }
fn finite3(v: vec3<f32>) -> bool { return finite(v.x) && finite(v.y) && finite(v.z); }
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 { return a.x*b.x+a.y*b.y+a.z*b.z; }
fn transform_point(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot3(params.row0.xyz,p)+params.row0.w,
        dot3(params.row1.xyz,p)+params.row1.w,dot3(params.row2.xyz,p)+params.row2.w);
}
fn add(a: vec2<f32>, b: f32) -> vec2<f32> {
    let s=a.x+b; let v=s-a.x; let e=(a.x-(s-v))+(b-v)+a.y; let h=s+e;
    return vec2<f32>(h,e-(h-s));
}
fn merge(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return add(add(a,b.x),b.y); }
fn valid_match(i: u32) -> bool {
    let m=matches[i];
    if m.index==0xffffffffu {
        if m.distance!=params.bound { atomicOr(&result.status,2u); }
        return false;
    }
    if m.index>=params.targets || !(m.distance>=0.0 && m.distance<=params.bound) {
        atomicOr(&result.status,2u); return false;
    }
    return true;
}

@compute @workgroup_size(64)
fn transform_source(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x>=params.count { return; }
    let p=transform_point(source[id.x].xyz);
    if !finite3(p) { atomicOr(&result.status,1u); queries[id.x]=vec4<f32>(0.0); }
    else { queries[id.x]=vec4<f32>(p,0.0); }
}

@compute @workgroup_size(64)
fn centroid_partials(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let term=select(group.y,15u,group.y==6u);
    let anchor=transform_point(params.source_anchor.xyz);
    var total=vec2<f32>(0.0); var count=0u;
    for (var i=group.x*64u+lane;i<params.count;i+=params.groups*64u) {
        if !valid_match(i) { continue; }
        if term!=15u && !retained_match(i) { continue; }
        let p=queries[i].xyz; let q=target_points[matches[i].index].xyz;
        var value=0.0;
        if term<3u { value=((p-anchor)/params.scale)[term]; }
        else if term<6u { value=((q-params.target_anchor.xyz)/params.scale)[term-3u]; }
        else { let d=(p-q)/params.max_distance; value=dot3(d,d); }
        if !finite(value) { atomicOr(&result.status,4u); }
        total=add(total,value); count+=1u;
    }
    sums[lane]=total; counts[lane]=count; workgroupBarrier();
    for (var stride=32u;stride>0u;stride/=2u) {
        if lane<stride { sums[lane]=merge(sums[lane],sums[lane+stride]); counts[lane]+=counts[lane+stride]; }
        workgroupBarrier();
    }
    if lane==0u {
        partials[group.x].values[term]=sums[0];
        if group.y==0u { partials[group.x].count=counts[0]; }
    }
}

@compute @workgroup_size(64)
fn finish_centroids(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x!=0u { return; }
    var count=0u;
    for (var g=0u;g<params.groups;g+=1u) { count+=partials[g].count; }
    result.count=count; result.reserved0=selection.gated;
    if count!=selection.retained { atomicOr(&result.status,2u); }
    result.source_anchor=vec4<f32>(transform_point(params.source_anchor.xyz),0.0);
    result.target_anchor=params.target_anchor;
    if !finite3(result.source_anchor.xyz) { atomicOr(&result.status,1u); }
    for (var term=0u;term<16u;term+=1u) {
        result.values[term]=vec2<f32>(0.0);
        if term<6u || term==15u {
            var total=vec2<f32>(0.0);
            for (var g=0u;g<params.groups;g+=1u) { total=merge(total,partials[g].values[term]); }
            if count>0u || term==15u {
                if term<6u { result.values[term]=merge(vec2<f32>(total.x/f32(count),0.0),vec2<f32>(total.y/f32(count),0.0)); }
                else { result.values[term]=total; }
            }
        }
    }
}

@compute @workgroup_size(64)
fn covariance_partials(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let term=group.y; let row=term/3u; let col=term%3u;
    let mean_s=result.values[row].x+result.values[row].y;
    let mean_t=result.values[col+3u].x+result.values[col+3u].y;
    var total=vec2<f32>(0.0);
    if result.count>0u && atomicLoad(&result.status)==0u {
        for (var i=group.x*64u+lane;i<params.count;i+=params.groups*64u) {
            if !valid_match(i) || !retained_match(i) { continue; }
            let s=(queries[i].xyz-result.source_anchor.xyz)/params.scale;
            let t=(target_points[matches[i].index].xyz-result.target_anchor.xyz)/params.scale;
            total=add(total,(s[row]-mean_s)*(t[col]-mean_t));
        }
    }
    sums[lane]=total; workgroupBarrier();
    for (var stride=32u;stride>0u;stride/=2u) {
        if lane<stride { sums[lane]=merge(sums[lane],sums[lane+stride]); }
        workgroupBarrier();
    }
    if lane==0u { partials[group.x].values[term+6u]=sums[0]; }
}

@compute @workgroup_size(64)
fn finish_covariance(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x!=0u { return; }
    for (var term=6u;term<15u;term+=1u) {
        var total=vec2<f32>(0.0);
        for (var g=0u;g<params.groups;g+=1u) { total=merge(total,partials[g].values[term]); }
        if !finite(total.x) || !finite(total.y) { atomicOr(&result.status,4u); }
        result.values[term]=total;
    }
    if result.count==0u { result.source_anchor=vec4<f32>(0.0); result.target_anchor=vec4<f32>(0.0); }
    if atomicLoad(&result.status)!=0u || result.reserved0==0u {
        result.count=0u; result.reserved0=0u; result.source_anchor=vec4<f32>(0.0); result.target_anchor=vec4<f32>(0.0);
        for (var term=0u;term<16u;term+=1u) { result.values[term]=vec2<f32>(0.0); }
    }
}

struct Selection { gated:u32, retained:u32, rank:u32, distance:u32, source_index:u32, unused0:u32, unused1:u32, unused2:u32 }
@group(0) @binding(7) var<storage,read> selection: Selection;
fn retained_match(i:u32)->bool { let distance=bitcast<u32>(matches[i].distance)&0x7fffffffu; return selection.retained>0u && (distance<selection.distance || (distance==selection.distance && i<=selection.source_index)); }
