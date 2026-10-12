struct Params {
    row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>, anchor: vec4<f32>,
    count: u32, targets: u32, groups: u32, unused: u32,
    bound: f32, max_distance: f32, scale: f32, weight_scale: f32,
}
struct Covariance { diagonal: vec4<f32>, off: vec4<f32> }
struct Neighbor { index: u32, distance: f32 }
struct Pair { w0: vec4<f32>, w1: vec4<f32>, p: vec4<f32>, residual: vec4<f32> }
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
@group(0) @binding(5) var<storage,read> covariances: array<Covariance>;
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
fn covariance(c: Covariance) -> mat3x3<f32> {
    return mat3x3<f32>(vec3<f32>(c.diagonal.x,c.diagonal.w,c.off.x),vec3<f32>(c.diagonal.w,c.diagonal.y,c.off.y),vec3<f32>(c.off.x,c.off.y,c.diagonal.z));
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
    pairs[i]=Pair(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0));
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
    let r=mat3x3<f32>(vec3<f32>(params.row0.x,params.row1.x,params.row2.x),vec3<f32>(params.row0.y,params.row1.y,params.row2.y),vec3<f32>(params.row0.z,params.row1.z,params.row2.z));
    let c=covariance(covariances[params.count+m.index])+r*covariance(covariances[i])*transpose(r);
    let scale=max(c[0][0],max(c[1][1],c[2][2]));
    let factor=scale*params.weight_scale;
    if !finite(scale) || scale<=0.0 || !finite(factor) || factor<=0.0 {atomicOr(&result.status,4u);return;}
    let xx=c[0][0]/scale;let yy=c[1][1]/scale;let zz=c[2][2]/scale;
    let xy=(0.5*c[0][1]+0.5*c[1][0])/scale;
    let xz=(0.5*c[0][2]+0.5*c[2][0])/scale;
    let yz=(0.5*c[1][2]+0.5*c[2][1])/scale;
    if xx<=1e-6 {atomicOr(&result.status,8u);return;}
    let l00=sqrt(xx);let l10=xy/l00;let l20=xz/l00;
    let pivot1=yy-l10*l10;
    if pivot1<=1e-6 {atomicOr(&result.status,8u);return;}
    let l11=sqrt(pivot1);let l21=(yz-l20*l10)/l11;
    let pivot2=zz-l20*l20-l21*l21;
    if pivot2<=1e-6 {atomicOr(&result.status,8u);return;}
    let l22=sqrt(pivot2);
    let a=1.0/l00;let b=-l10/(l00*l11);let cc=1.0/l11;
    let dd=(l10*l21/l11-l20)/(l00*l22);let ee=-l21/(l11*l22);let f=1.0/l22;
    let w0=vec4<f32>((a*a+b*b+dd*dd)/factor,(cc*cc+ee*ee)/factor,f*f/factor,(b*cc+dd*ee)/factor);
    let w1=vec4<f32>(dd*f/factor,ee*f/factor,dot3(e,e),0.0);
    if !finite3(w0.xyz) || !finite(w0.w) || !finite3(w1.xyz) || any(w0.xyz<vec3<f32>(1.1754943508222875e-38)) {atomicOr(&result.status,4u);return;}
    pairs[i]=Pair(w0,w1,vec4<f32>(d,1.0),vec4<f32>(e,1.0));
}
fn column(p: vec3<f32>, i: u32) -> vec3<f32> {
    switch i {case 0u:{return vec3<f32>(0.0,-p.z,p.y);} case 1u:{return vec3<f32>(p.z,0.0,-p.x);} case 2u:{return vec3<f32>(-p.y,p.x,0.0);}
        case 3u:{return vec3<f32>(1.0,0.0,0.0);} case 4u:{return vec3<f32>(0.0,1.0,0.0);} default:{return vec3<f32>(0.0,0.0,1.0);}}
}
fn weighted(pair: Pair,v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot3(vec3<f32>(pair.w0.x,pair.w0.w,pair.w1.x),v),dot3(vec3<f32>(pair.w0.w,pair.w0.y,pair.w1.y),v),dot3(vec3<f32>(pair.w1.x,pair.w1.y,pair.w0.z),v));
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
        let pair=pairs[i];count+=vec2<u32>(u32(pair.p.w),u32(pair.residual.w));
        var value=0.0;
        if term<21u {value=dot3(column(pair.p.xyz,ab.x),weighted(pair,column(pair.p.xyz,ab.y)));}
        else if term<27u {value=dot3(column(pair.p.xyz,term-21u),weighted(pair,pair.residual.xyz));}
        else {value=pair.w1.z;}
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
