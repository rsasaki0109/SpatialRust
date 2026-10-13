struct Params {
    row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>, anchor: vec4<f32>,
    count: u32, targets: u32, groups: u32, unused: u32,
    bound: f32, max_distance: f32, scale: f32, normal_scale: f32,
}
struct Neighbor { index: u32, distance: f32 }
struct Pair { rotation: vec4<f32>, normal: vec4<f32> }
struct Partial { values: array<vec2<f32>,28>, counts: vec2<u32>, reserved: vec2<u32> }
struct Equations {
    status: atomic<u32>, count: u32, fitness_count: u32, reserved: u32,
    anchor: vec4<f32>, values: array<vec2<f32>,28>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> source: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> target_points: array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> queries: array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> matches: array<Neighbor>;
@group(0) @binding(5) var<storage,read> normals: array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> pairs: array<Pair>;
@group(0) @binding(7) var<storage,read_write> partials: array<Partial>;
@group(0) @binding(8) var<storage,read_write> result: Equations;
var<workgroup> sums: array<vec2<f32>,64>;
var<workgroup> counts: array<vec2<u32>,64>;

fn finite(v: f32) -> bool {return abs(v)<=3.4028234663852886e38;}
fn finite3(v: vec3<f32>) -> bool {return finite(v.x) && finite(v.y) && finite(v.z);}
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {return a.x*b.x+a.y*b.y+a.z*b.z;}
fn transform_point(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot3(params.row0.xyz,p)+params.row0.w,dot3(params.row1.xyz,p)+params.row1.w,dot3(params.row2.xyz,p)+params.row2.w);
}
fn add(a: vec2<f32>, b: f32) -> vec2<f32> {
    let s=a.x+b;let v=s-a.x;let e=(a.x-(s-v))+(b-v)+a.y;let h=s+e;
    return vec2<f32>(h,e-(h-s));
}
fn merge(a: vec2<f32>,b: vec2<f32>) -> vec2<f32> {return add(add(a,b.x),b.y);}

@compute @workgroup_size(64)
fn transform_source(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;if i>=params.count {return;}
    let p=transform_point(source[i].xyz);
    if !finite3(p) {atomicOr(&result.status,1u);queries[i]=vec4<f32>(0.0);}
    else {queries[i]=vec4<f32>(p,0.0);}
}

@compute @workgroup_size(64)
fn prepare_pairs(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;if i>=params.count {return;}
    pairs[i]=Pair(vec4<f32>(0.0),vec4<f32>(0.0));
    let m=matches[i];
    if m.index==0xffffffffu {
        if m.distance!=bitcast<f32>(0x7f800000u) {atomicOr(&result.status,2u);}
        return;
    }
    if m.index>=params.targets || !(m.distance>=0.0) {atomicOr(&result.status,2u);return;}
    if m.distance>params.bound || atomicLoad(&result.status)!=0u {return;}
    let p=queries[i].xyz;let q=target_points[m.index].xyz;
    let anchor=transform_point(params.anchor.xyz);
    let d=(p-anchor)/params.scale;let e=(p-q)/params.max_distance;
    if !finite3(d) || !finite3(e) {atomicOr(&result.status,4u);return;}
    let n=normals[m.index].xyz;
    let rotation=cross(d,n);let residual=dot3(e,n);
    if !finite3(rotation) || !finite(residual) {atomicOr(&result.status,4u);return;}
    pairs[i]=Pair(vec4<f32>(rotation,residual),vec4<f32>(n,1.0));
}
fn row(pair: Pair,i:u32) -> f32 {
    if i<3u {return pair.rotation[i];} return pair.normal[i-3u];
}
fn indices(term: u32) -> vec2<u32> {
    var remaining=term;
    for (var a=0u;a<6u;a+=1u) {if remaining<6u-a {return vec2<u32>(a,a+remaining);} remaining-=6u-a;}
    return vec2<u32>(0u);
}
@compute @workgroup_size(64)
fn reduce_partials(@builtin(workgroup_id) group: vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let term=group.y;let ab=indices(term);var total=vec2<f32>(0.0);var count=vec2<u32>(0u);
    for (var i=group.x*64u+lane;i<params.count;i+=params.groups*64u) {
        let pair=pairs[i];count+=vec2<u32>(u32(pair.normal.w));
        var value=0.0;
        if term<21u {value=row(pair,ab.x)*row(pair,ab.y);}
        else if term<27u {value=row(pair,term-21u)*pair.rotation.w;}
        else {value=pair.rotation.w*pair.rotation.w;}
        total=add(total,value);
    }
    sums[lane]=total;counts[lane]=count;workgroupBarrier();
    for (var stride=32u;stride>0u;stride/=2u) {
        if lane<stride {sums[lane]=merge(sums[lane],sums[lane+stride]);counts[lane]+=counts[lane+stride];}
        workgroupBarrier();
    }
    if lane==0u {partials[group.x].values[term]=sums[0];if term==0u {partials[group.x].counts=counts[0];}}
}
@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x!=0u {return;}
    var count=vec2<u32>(0u);
    for (var g=0u;g<params.groups;g+=1u) {count+=partials[g].counts;}
    result.count=count.x;result.fitness_count=count.y;result.anchor=vec4<f32>(transform_point(params.anchor.xyz),0.0);
    if !finite3(result.anchor.xyz) {atomicOr(&result.status,1u);}
    for (var term=0u;term<28u;term+=1u) {
        var total=vec2<f32>(0.0);
        for (var g=0u;g<params.groups;g+=1u) {total=merge(total,partials[g].values[term]);}
        if !finite(total.x) || !finite(total.y) {atomicOr(&result.status,4u);}
        result.values[term]=total;
    }
    if atomicLoad(&result.status)!=0u || count.y==0u {
        result.count=0u;result.fitness_count=0u;result.anchor=vec4<f32>(0.0);
        for (var term=0u;term<28u;term+=1u) {result.values[term]=vec2<f32>(0.0);}
    }
}
