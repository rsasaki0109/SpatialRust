struct Params { count: u32, capacity: u32, pad0: u32, pad1: u32 }
struct Sample { position: vec4<f32>, normal: vec4<f32> }
struct Status {
    bad_rows: atomic<u32>, last_bad_plus_one: atomic<u32>,
    required_capacity: atomic<u32>, invalid_rows: atomic<u32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> samples: array<Sample>;
@group(0) @binding(2) var<storage, read> neighbors: array<u32>;
@group(0) @binding(3) var<storage, read_write> spfh: array<f32>;
@group(0) @binding(4) var<storage, read_write> descriptors: array<f32>;
@group(0) @binding(5) var<storage, read_write> status: Status;

fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 { return a.x*b.x + a.y*b.y + a.z*b.z; }
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
fn bin(value: f32, lo: f32, hi: f32) -> u32 {
    return u32(clamp((value-lo)/(hi-lo),0.0,0.999999)*11.0);
}
fn angle(y: f32, x: f32) -> f32 {
    // Explicit signed-zero handling avoids atan2(0,0) backend differences.
    if y == 0.0 {
        let sign = bitcast<u32>(y) & 0x80000000u;
        if (bitcast<u32>(x) & 0x80000000u) != 0u {
            return bitcast<f32>(bitcast<u32>(3.1415927410125732) | sign);
        }
        return bitcast<f32>(sign);
    }
    return atan2(y,x);
}
fn reject(i: u32, total: u32, invalid: bool) {
    atomicAdd(&status.bad_rows,1u);
    atomicMax(&status.last_bad_plus_one,i+1u);
    atomicMax(&status.required_capacity,total);
    if invalid { atomicAdd(&status.invalid_rows,1u); }
}

@compute @workgroup_size(64)
fn build_spfh(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;
    if i>=params.count { return; }
    let base=i*(2u+2u*params.capacity);
    let stored=neighbors[base]; let total=neighbors[base+1u];
    for (var b=0u;b<33u;b+=1u) { spfh[i*33u+b]=0.0; }
    if stored>params.capacity || total<stored || total>params.count {
        reject(i,total,true);return;
    }
    if stored!=total { reject(i,total,false);return; }
    var histogram: array<f32,33>;
    let p=samples[i].position.xyz; let np=samples[i].normal.xyz;
    for (var n=0u;n<stored;n+=1u) {
        let j=neighbors[base+2u+2u*n];
        if j>=params.count { reject(i,total,true);return; }
        if j==i { continue; }
        let diff=samples[j].position.xyz-p;
        let distance=sqrt(dot3(diff,diff));
        if distance<1e-9 { continue; }
        let pq=diff/distance;
        let raw_v=cross3(pq,np);
        let v_length=sqrt(dot3(raw_v,raw_v));
        if v_length<1e-9 { continue; }
        let v=raw_v/v_length;
        let w=cross3(np,v);
        let nq=samples[j].normal.xyz;
        let alpha=dot3(v,nq);let phi=dot3(np,pq);
        let theta=angle(dot3(w,nq),dot3(np,nq));
        histogram[bin(alpha,-1.0,1.0)]+=1.0;
        histogram[11u+bin(phi,-1.0,1.0)]+=1.0;
        histogram[22u+bin(theta,-3.1415927410125732,3.1415927410125732)]+=1.0;
    }
    for (var sub=0u;sub<3u;sub+=1u) {
        var sum=0.0;
        for (var b=0u;b<11u;b+=1u) { sum+=histogram[sub*11u+b]; }
        for (var b=0u;b<11u;b+=1u) {
            let value=histogram[sub*11u+b];
            if sum>0.0 { spfh[i*33u+sub*11u+b]=value/sum*100.0; }
        }
    }
}

@compute @workgroup_size(64)
fn build_fpfh(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;
    if i>=params.count { return; }
    if atomicLoad(&status.bad_rows)>0u {
        for (var b=0u;b<33u;b+=1u) { descriptors[i*33u+b]=0.0; }
        return;
    }
    let base=i*(2u+2u*params.capacity);
    let stored=neighbors[base];
    var acc: array<f32,33>;
    var sum_weight=0.0;
    for (var n=0u;n<stored;n+=1u) {
        let j=neighbors[base+2u+2u*n];
        if j==i { continue; }
        let distance=sqrt(bitcast<f32>(neighbors[base+3u+2u*n]));
        if distance<1e-9 { continue; }
        let weight=1.0/distance;
        sum_weight+=weight;
        for (var b=0u;b<33u;b+=1u) { acc[b]+=weight*spfh[j*33u+b]; }
    }
    for (var b=0u;b<33u;b+=1u) {
        var contribution=0.0;
        if sum_weight>0.0 { contribution=acc[b]/sum_weight; }
        descriptors[i*33u+b]=spfh[i*33u+b]+contribution;
    }
}
