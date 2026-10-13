// Storage adapter/validation for the canonical radius-grid covariance kernel.
struct Params {
    origin:vec4<f32>, dims:vec4<u32>, inv_cell:f32, radius_sq:f32,
    epsilon:f32, capacity:u32, offset:u32, total:u32, padding:vec2<u32>,
}
struct Covariance { diagonal:vec4<f32>, off:vec4<f32> }
struct Covariances { status:atomic<u32>, reserved:array<u32,7>, rows:array<Covariance> }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> points:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> sorted:array<u32>;
@group(0) @binding(3) var<storage,read> cell_start:array<u32>;
@group(0) @binding(4) var<storage,read_write> result:Covariances;
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
fn store_covariance(i:u32,diagonal:vec4<f32>,off:vec4<f32>) {
    // Packed Cholesky and inverse trace validate exactly the stored values.
    let xx=diagonal.x;let yy=diagonal.y;let zz=diagonal.z;
    let xy=diagonal.w;let xz=off.x;let yz=off.y;
    if !finite(xx)||!finite(yy)||!finite(zz)||!finite(xy)||!finite(xz)||!finite(yz)||xx<=0.0 {atomicOr(&result.status,4u);return;}
    let a=sqrt(xx);let b=xy/a;let d=xz/a;let pivot1=yy-b*b;
    if pivot1<=0.0 {atomicOr(&result.status,4u);return;}
    let e=sqrt(pivot1);let f=(yz-d*b)/e;let pivot2=zz-d*d-f*f;
    if pivot2<=0.0 {atomicOr(&result.status,4u);return;}
    let g=sqrt(pivot2);
    let i00=1.0/a;let i10=-b/(a*e);let i11=1.0/e;let i20=(b*f/e-d)/(a*g);let i21=-f/(e*g);let i22=1.0/g;
    let trace=i00*i00+i10*i10+i11*i11+i20*i20+i21*i21+i22*i22;
    if !finite(trace)||trace>6.0/params.epsilon {atomicOr(&result.status,4u);return;}
    result.rows[params.offset+i]=Covariance(diagonal,off);
}
@compute @workgroup_size(256)
fn finalize(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x<params.total && atomicLoad(&result.status)!=0u {result.rows[id.x]=Covariance(vec4<f32>(0.0),vec4<f32>(0.0));}
}
