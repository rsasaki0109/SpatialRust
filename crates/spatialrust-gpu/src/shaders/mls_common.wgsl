struct Params {
    count: u32, capacity: u32, min_neighbors: u32, padding: u32,
    radius_squared: f32, weight_scale_squared: f32, pad0: f32, pad1: f32,
}
struct Status {
    bad_rows: atomic<u32>, required_capacity: atomic<u32>, malformed: atomic<u32>,
    last_bad_plus_one: atomic<u32>, numerical: atomic<u32>,
    unchanged: atomic<u32>, plane: atomic<u32>, polynomial: atomic<u32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> neighbors: array<u32>;
struct Frame { centroid: vec4<f32>, normal: vec4<f32>, u: vec4<f32>, v: vec4<f32> }
@group(0) @binding(3) var<storage,read_write> frames: array<Frame>;
@group(0) @binding(4) var<storage,read_write> systems: array<f32>;
@group(0) @binding(5) var<storage,read_write> output: array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> status: Status;

fn add(sum: vec2<f32>, value: f32) -> vec2<f32> {
    let s=sum.x+value;let v=s-sum.x;
    let e=(sum.x-(s-v))+(value-v)+sum.y;let h=s+e;
    return vec2<f32>(h,e-(h-s));
}
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {return a.x*b.x+a.y*b.y+a.z*b.z;}
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
}
fn unit(v: vec3<f32>) -> vec3<f32> {return v/sqrt(dot3(v,v));}
fn finite(value: f32) -> bool {return abs(value)<=3.4028234663852886e38;}
fn reject(i: u32, total: u32, malformed: bool) {
    atomicAdd(&status.bad_rows,1u);atomicMax(&status.required_capacity,total);
    atomicMax(&status.last_bad_plus_one,i+1u);
    if malformed {atomicAdd(&status.malformed,1u);}
}
fn numerical(i: u32) {atomicAdd(&status.numerical,1u);atomicMax(&status.last_bad_plus_one,i+1u);}
fn weight(base: u32, n: u32) -> f32 {
    return exp(-bitcast<f32>(neighbors[base+3u+2u*n])/params.weight_scale_squared);
}
fn write_result(i: u32, value: vec3<f32>, mode: u32) {
    if !finite(value.x) || !finite(value.y) || !finite(value.z) {numerical(i);return;}
    output[i]=vec4<f32>(value,f32(mode));
    if mode==0u {atomicAdd(&status.unchanged,1u);}
    else if mode==1u {atomicAdd(&status.plane,1u);}
    else {atomicAdd(&status.polynomial,1u);}
}
