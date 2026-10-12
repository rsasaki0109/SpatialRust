struct Params {
    count: u32, salient_capacity: u32, non_max_capacity: u32, min_neighbors: u32,
    gamma_21: f32, gamma_32: f32, radius: f32, non_max_radius_squared: f32,
}
struct Status {
    bad_salient: atomic<u32>, bad_non_max: atomic<u32>,
    required_salient: atomic<u32>, required_non_max: atomic<u32>,
    malformed: atomic<u32>, last_bad_plus_one: atomic<u32>,
    numerical: atomic<u32>, reserved: atomic<u32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> salient: array<u32>;
@group(0) @binding(3) var<storage,read> non_max: array<u32>;
@group(0) @binding(4) var<storage,read_write> scores: array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> flags: array<u32>;
@group(0) @binding(6) var<storage,read_write> status: Status;

// Compensated scalar accumulation reduces covariance summation error.
fn add(sum: vec2<f32>, value: f32) -> vec2<f32> {
    let s=sum.x+value; let v=s-sum.x;
    let e=(sum.x-(s-v))+(value-v)+sum.y;
    let h=s+e;
    return vec2<f32>(h,e-(h-s));
}
fn finite(value: f32) -> bool { return abs(value)<=3.4028234663852886e38; }
fn record_bad(i: u32, total: u32, second: bool, malformed: bool) {
    if second {
        atomicAdd(&status.bad_non_max,1u); atomicMax(&status.required_non_max,total);
    } else {
        atomicAdd(&status.bad_salient,1u); atomicMax(&status.required_salient,total);
    }
    if malformed { atomicAdd(&status.malformed,1u); }
    atomicMax(&status.last_bad_plus_one,i+1u);
}
fn validate_row(i: u32, second: bool) -> bool {
    var capacity=params.salient_capacity;
    if second { capacity=params.non_max_capacity; }
    let base=i*(2u+2u*capacity);
    var stored=0u; var total=0u;
    if second { stored=non_max[base]; total=non_max[base+1u]; }
    else { stored=salient[base]; total=salient[base+1u]; }
    if stored>capacity || total<stored || total>params.count {
        record_bad(i,total,second,true); return false;
    }
    if stored!=total { record_bad(i,total,second,false); return false; }
    var bound=params.radius*params.radius;
    if second { bound=params.non_max_radius_squared; }
    for (var n=0u;n<stored;n+=1u) {
        var j=0u; var bits=0u;
        if second { j=non_max[base+2u+2u*n]; bits=non_max[base+3u+2u*n]; }
        else { j=salient[base+2u+2u*n]; bits=salient[base+3u+2u*n]; }
        let distance=bitcast<f32>(bits);
        if j>=params.count || !(distance>=0.0 && distance<=bound) {
            record_bad(i,total,second,true); return false;
        }
    }
    return true;
}

// Max-pivot Jacobi, with the CPU's 32-iteration/physical 1e-12 stopping rule.
// A scaled hypot avoids overflow in the tangent calculation; no eigenvectors
// are computed because ISS only consumes eigenvalues.
fn eigenvalues(matrix: array<vec3<f32>,3>, epsilon: f32) -> vec3<f32> {
    var a=matrix;
    for (var iteration=0u;iteration<32u;iteration+=1u) {
        var p=0u; var q=1u; var off=abs(a[0][1]);
        if abs(a[0][2])>off { p=0u;q=2u;off=abs(a[0][2]); }
        if abs(a[1][2])>off { p=1u;q=2u;off=abs(a[1][2]); }
        if off==0.0 || off<epsilon { break; }
        let app=a[p][p]; let aqq=a[q][q]; let apq=a[p][q];
        let delta=0.5*(aqq-app);
        let scale=max(abs(delta),abs(apq));
        let d=delta/scale; let h=apq/scale;
        let sign_delta=select(1.0,-1.0,(bitcast<u32>(delta)&0x80000000u)!=0u);
        let t=sign_delta*h/(abs(d)+sqrt(d*d+h*h));
        let c=1.0/sqrt(1.0+t*t); let s=t*c;
        a[p][p]=c*c*app-2.0*s*c*apq+s*s*aqq;
        a[q][q]=s*s*app+2.0*s*c*apq+c*c*aqq;
        a[p][q]=0.0;a[q][p]=0.0;
        for (var r=0u;r<3u;r+=1u) {
            if r!=p && r!=q {
                let arp=a[r][p]; let arq=a[r][q];
                a[r][p]=c*arp-s*arq;a[p][r]=a[r][p];
                a[r][q]=s*arp+c*arq;a[q][r]=a[r][q];
            }
        }
    }
    var values=vec3<f32>(a[0][0],a[1][1],a[2][2]);
    if values.x>values.y { let t=values.x;values.x=values.y;values.y=t; }
    if values.y>values.z { let t=values.y;values.y=values.z;values.z=t; }
    if values.x>values.y { let t=values.x;values.x=values.y;values.y=t; }
    return values;
}

@compute @workgroup_size(64)
fn build_saliency(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;
    if i>=params.count { return; }
    scores[i]=vec4<f32>(0.0);
    // Validate BOTH graphs before any selection pass; no late status races.
    let valid_salient=validate_row(i,false);
    let valid_non_max=validate_row(i,true);
    if !valid_salient || !valid_non_max { return; }
    let base=i*(2u+2u*params.salient_capacity); let count=salient[base];
    if count<3u || count<params.min_neighbors { return; }
    let origin=positions[i].xyz;
    // Scale by actual local extent, not the requested search radius. A huge
    // radius around a small cloud must not underflow its normalized covariance.
    var scale=0.0;
    for (var n=0u;n<count;n+=1u) {
        let relative=abs(positions[salient[base+2u+2u*n]].xyz-origin);
        scale=max(scale,max(relative.x,max(relative.y,relative.z)));
    }
    if scale==0.0 { return; }
    let scale_squared=scale*scale;
    if scale_squared<1.1754943508222875e-38 {
        atomicAdd(&status.numerical,1u);atomicMax(&status.last_bad_plus_one,i+1u);return;
    }
    var mean: array<vec2<f32>,3>;
    for (var n=0u;n<count;n+=1u) {
        let j=salient[base+2u+2u*n];
        let relative=(positions[j].xyz-origin)/scale;
        for (var axis=0u;axis<3u;axis+=1u) { mean[axis]=add(mean[axis],relative[axis]); }
    }
    for (var axis=0u;axis<3u;axis+=1u) { mean[axis]/=f32(count); }
    var covariance: array<vec2<f32>,6>;
    for (var n=0u;n<count;n+=1u) {
        let j=salient[base+2u+2u*n];
        let relative=(positions[j].xyz-origin)/scale;
        var d: vec3<f32>;
        for (var axis=0u;axis<3u;axis+=1u) { d[axis]=(relative[axis]-mean[axis].x)-mean[axis].y; }
        let products=array<f32,6>(d.x*d.x,d.y*d.y,d.z*d.z,d.x*d.y,d.x*d.z,d.y*d.z);
        for (var b=0u;b<6u;b+=1u) { covariance[b]=add(covariance[b],products[b]); }
    }
    var cov: array<f32,6>;
    for (var b=0u;b<6u;b+=1u) { cov[b]=(covariance[b].x+covariance[b].y)/f32(count); }
    let matrix=array<vec3<f32>,3>(vec3<f32>(cov[0],cov[3],cov[4]),vec3<f32>(cov[3],cov[1],cov[5]),vec3<f32>(cov[4],cov[5],cov[2]));
    let values=eigenvalues(matrix,1e-12/scale_squared);
    let physical=values*scale_squared;
    if !finite(physical.x) || !finite(physical.y) || !finite(physical.z) {
        atomicAdd(&status.numerical,1u);atomicMax(&status.last_bad_plus_one,i+1u);return;
    }
    var eligible=0.0;
    if values.z>0.0 && values.y>0.0 && values.y/values.z<params.gamma_21 && values.x/values.y<params.gamma_32 {
        eligible=1.0;
    }
    scores[i]=vec4<f32>(physical,eligible);
}

@compute @workgroup_size(64)
fn select_keypoints(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;
    if i>=params.count { return; }
    flags[i]=0u;
    if atomicLoad(&status.bad_salient)>0u || atomicLoad(&status.bad_non_max)>0u || atomicLoad(&status.numerical)>0u {
        scores[i]=vec4<f32>(0.0);return;
    }
    let score=scores[i];
    if score.w==0.0 { return; }
    let base=i*(2u+2u*params.non_max_capacity); let count=non_max[base];
    for (var n=0u;n<count;n+=1u) {
        let j=non_max[base+2u+2u*n]; let other=scores[j];
        if j!=i && other.w!=0.0 && score.x<other.x { return; }
    }
    flags[i]=1u;
}
