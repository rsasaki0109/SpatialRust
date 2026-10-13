struct Params { sources:u32, targets:u32, neighbors:u32, epsilon:f32 }
struct Covariance { diagonal:vec4<f32>, off:vec4<f32> }
struct Covariances { status:atomic<u32>, reserved:array<u32,7>, rows:array<Covariance> }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> source:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> target_points:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> source_neighbors:array<u32>;
@group(0) @binding(4) var<storage,read> target_neighbors:array<u32>;
@group(0) @binding(5) var<storage,read_write> result:Covariances;
fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}
fn finite3(v:vec3<f32>)->bool {return finite(v.x)&&finite(v.y)&&finite(v.z);}
fn word(index:u32,second:bool)->u32 {if second {return target_neighbors[index];}return source_neighbors[index];}
fn point(index:u32,second:bool)->vec3<f32> {if second {return target_points[index].xyz;}return source[index].xyz;}
fn add(sum:vec2<f32>,value:f32)->vec2<f32> {let s=sum.x+value;let v=s-sum.x;let e=(sum.x-(s-v))+(value-v)+sum.y;let h=s+e;return vec2<f32>(h,e-(h-s));}

// Max-pivot Jacobi with eigenvectors, matching the CPU iteration/tie order and
// physical absolute stop. Scaled hypot avoids phi*phi overflow in f32.
fn normal(matrix:array<vec3<f32>,3>,threshold:f32)->vec3<f32> {
    var a=matrix;
    var v=array<vec3<f32>,3>(vec3<f32>(1.0,0.0,0.0),vec3<f32>(0.0,1.0,0.0),vec3<f32>(0.0,0.0,1.0));
    for(var iteration=0u;iteration<32u;iteration+=1u) {
        var p=0u;var q=1u;var off=abs(a[0][1]);
        if abs(a[0][2])>off {p=0u;q=2u;off=abs(a[0][2]);}
        if abs(a[1][2])>off {p=1u;q=2u;off=abs(a[1][2]);}
        if off==0.0||off<threshold {break;}
        let app=a[p][p];let aqq=a[q][q];let apq=a[p][q];let delta=0.5*(aqq-app);
        let scale=max(abs(delta),abs(apq));let d=delta/scale;let h=apq/scale;
        let sign_delta=select(1.0,-1.0,(bitcast<u32>(delta)&0x80000000u)!=0u);
        let t=sign_delta*h/(abs(d)+sqrt(d*d+h*h));let c=1.0/sqrt(1.0+t*t);let s=t*c;
        a[p][p]=c*c*app-2.0*s*c*apq+s*s*aqq;a[q][q]=s*s*app+2.0*s*c*apq+c*c*aqq;
        a[p][q]=0.0;a[q][p]=0.0;
        for(var r=0u;r<3u;r+=1u) {
            if r!=p&&r!=q {let arp=a[r][p];let arq=a[r][q];a[r][p]=c*arp-s*arq;a[p][r]=a[r][p];a[r][q]=s*arp+c*arq;a[q][r]=a[r][q];}
            let vp=v[r][p];let vq=v[r][q];v[r][p]=c*vp-s*vq;v[r][q]=s*vp+c*vq;
        }
    }
    var smallest=0u;if a[1][1]<a[smallest][smallest] {smallest=1u;}if a[2][2]<a[smallest][smallest] {smallest=2u;}
    let n=vec3<f32>(v[0][smallest],v[1][smallest],v[2][smallest]);return normalize(n);
}
fn emit(i:u32,n:vec3<f32>) {
    let delta=params.epsilon-1.0;
    let squared=n*n;let diagonal=(vec3<f32>(1.0)-squared)+params.epsilon*squared;
    let c=Covariance(vec4<f32>(diagonal,delta*n.x*n.y),vec4<f32>(delta*n.x*n.z,delta*n.y*n.z,0.0,0.0));
    // Check exact packed row's Cholesky pivots and analytic inverse-weight bound.
    let xx=c.diagonal.x;let yy=c.diagonal.y;let zz=c.diagonal.z;let xy=c.diagonal.w;let xz=c.off.x;let yz=c.off.y;
    if !finite3(c.diagonal.xyz)||!finite(c.diagonal.w)||!finite3(c.off.xyz)||xx<=0.0 {atomicOr(&result.status,4u);return;}
    let a=sqrt(xx);let b=xy/a;let d=xz/a;let pivot1=yy-b*b;
    if pivot1<=0.0 {atomicOr(&result.status,4u);return;}
    let e=sqrt(pivot1);let f=(yz-d*b)/e;let pivot2=zz-d*d-f*f;
    if pivot2<=0.0 {atomicOr(&result.status,4u);return;}
    let g=sqrt(pivot2);
    let i00=1.0/a;let i10=-b/(a*e);let i11=1.0/e;let i20=(b*f/e-d)/(a*g);let i21=-f/(e*g);let i22=1.0/g;
    let trace=i00*i00+i10*i10+i11*i11+i20*i20+i21*i21+i22*i22;
    if !finite(trace)||trace>6.0/params.epsilon {atomicOr(&result.status,4u);return;}
    result.rows[i]=c;
}

@compute @workgroup_size(64)
fn build_covariances(@builtin(global_invocation_id) id:vec3<u32>) {
    let index=id.x;if index>=params.sources+params.targets {return;}
    result.rows[index]=Covariance(vec4<f32>(0.0),vec4<f32>(0.0));
    let second=index>=params.sources;let i=select(index,index-params.sources,second);let size=select(params.sources,params.targets,second);
    let base=i*(2u+2u*params.neighbors);let stored=word(base,second);let total=word(base+1u,second);
    if stored!=min(size,params.neighbors)||total!=stored {atomicOr(&result.status,1u);return;}
    for(var j=0u;j<stored;j+=1u) {let neighbor=word(base+2u+2u*j,second);let distance=bitcast<f32>(word(base+3u+2u*j,second));if neighbor>=size||!(distance>=0.0)||!finite(distance) {atomicOr(&result.status,1u);return;}}
    // CPU's single-neighbor fallback uses identity covariance, regularized on X.
    if stored<2u {emit(index,vec3<f32>(1.0,0.0,0.0));return;}
    let origin=point(i,second);var scale=0.0;
    for(var j=0u;j<stored;j+=1u) {let d=abs(point(word(base+2u+2u*j,second),second)-origin);scale=max(scale,max(d.x,max(d.y,d.z)));}
    if scale==0.0 {emit(index,vec3<f32>(1.0,0.0,0.0));return;}
    let scale_squared=scale*scale;
    if !finite(scale_squared)||scale_squared<1.1754943508222875e-38 {atomicOr(&result.status,2u);return;}
    var mean:array<vec2<f32>,3>;
    for(var j=0u;j<stored;j+=1u) {let d=(point(word(base+2u+2u*j,second),second)-origin)/scale;for(var axis=0u;axis<3u;axis+=1u) {mean[axis]=add(mean[axis],d[axis]);}}
    for(var axis=0u;axis<3u;axis+=1u) {mean[axis]/=f32(stored);}
    var sums:array<vec2<f32>,6>;
    for(var j=0u;j<stored;j+=1u) {
        let relative=(point(word(base+2u+2u*j,second),second)-origin)/scale;var d:vec3<f32>;
        for(var axis=0u;axis<3u;axis+=1u) {d[axis]=(relative[axis]-mean[axis].x)-mean[axis].y;}
        let products=array<f32,6>(d.x*d.x,d.y*d.y,d.z*d.z,d.x*d.y,d.x*d.z,d.y*d.z);
        for(var a=0u;a<6u;a+=1u) {sums[a]=add(sums[a],products[a]);}
    }
    var cov:array<f32,6>;for(var a=0u;a<6u;a+=1u) {cov[a]=(sums[a].x+sums[a].y)/f32(stored-1u);if !finite(cov[a]) {atomicOr(&result.status,2u);return;}}
    let matrix=array<vec3<f32>,3>(vec3<f32>(cov[0],cov[3],cov[4]),vec3<f32>(cov[3],cov[1],cov[5]),vec3<f32>(cov[4],cov[5],cov[2]));
    let n=normal(matrix,1e-12/scale_squared);
    if !finite3(n) {atomicOr(&result.status,2u);return;}emit(index,n);
}

@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=params.sources+params.targets {return;}
    if atomicLoad(&result.status)!=0u {result.rows[i]=Covariance(vec4<f32>(0.0),vec4<f32>(0.0));}
}
