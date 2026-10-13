struct Params {
    width:u32,height:u32,pixels:u32,groups:u32,
    aperture:u32,l2:u32,low:u32,high:u32,
    smoothing_weights:array<vec4<i32>,2>,derivative:array<vec4<i32>,2>,
}
// Scaled u8 Sobel components are bounded by 1020/12240/10200 (3/5/7).
// Preserve both signed components exactly in one word, alongside u32 magnitude.
struct Gradient { components:u32,magnitude:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var source:texture_2d<u32>;
@group(0) @binding(2) var<storage,read_write> gradients:array<Gradient>;
@group(0) @binding(3) var<storage,read_write> horizontal:array<vec2<i32>>;
@group(0) @binding(4) var<storage,read_write> states:array<u32>;
@group(0) @binding(5) var<storage,read_write> parent:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> strong:array<atomic<u32>>;
@group(0) @binding(7) var output:texture_storage_2d<rgba8uint,write>;

fn pixel(x:i32,y:i32) -> i32 {
    return i32(textureLoad(source,vec2<i32>(clamp(x,0,i32(params.width)-1),clamp(y,0,i32(params.height)-1)),0).x);
}
fn magnitude(g:vec2<i32>) -> u32 {
    if params.l2!=0u { return u32(g.x*g.x+g.y*g.y); }
    return u32(abs(g.x)+abs(g.y));
}
fn pack_components(g:vec2<i32>) -> u32 {
    return (u32(g.x)&65535u) | ((u32(g.y)&65535u)<<16u);
}
fn unpack_components(value:u32) -> vec2<i32> {
    return vec2<i32>(i32(value<<16u)>>16u,i32(value)>>16u);
}
fn scaled(value:i32) -> i32 {
    if params.aperture!=7u { return value; }
    let a=u32(abs(value)); let q=a>>4u; let r=a&15u;
    let rounded=q+u32(r>8u || (r==8u && (q&1u)!=0u));
    return select(i32(rounded),-i32(rounded),value<0);
}
fn root(index:u32) -> u32 {
    var node=index;
    loop {
        let next=atomicLoad(&parent[node]);
        if next==node { break; }
        let grandparent=atomicLoad(&parent[next]);
        atomicMin(&parent[node],grandparent);
        node=next;
    }
    return node;
}
fn unite(a:u32,b:u32) {
    loop {
        let ra=root(a); let rb=root(b);
        if ra==rb { return; }
        let lower=min(ra,rb); let upper=max(ra,rb);
        // Only a current root is linked, and every link strictly decreases.
        // Retry spurious weak-CAS failures; there is no cycle or lock owner.
        let result=atomicCompareExchangeWeak(&parent[upper],upper,lower);
        if result.exchanged { return; }
    }
}
fn advance(index:u32) -> u32 {
    let stride=params.groups*256u;
    if params.pixels-index<=stride { return params.pixels; }
    return index+stride;
}

@compute @workgroup_size(256)
fn gradient_direct(@builtin(global_invocation_id) id:vec3<u32>) {
    for(var index=id.x;index<params.pixels;index=advance(index)) {
        let x=i32(index%params.width); let y=i32(index/params.width);
        let a=pixel(x-1,y-1); let b=pixel(x,y-1); let c=pixel(x+1,y-1);
        let d=pixel(x-1,y); let f=pixel(x+1,y);
        let g=pixel(x-1,y+1); let h=pixel(x,y+1); let i=pixel(x+1,y+1);
        let gx=c-a+2*(f-d)+i-g; let gy=g+2*h+i-a-2*b-c;
        let derivative=vec2<i32>(gx,gy);
        gradients[index]=Gradient(pack_components(derivative),magnitude(derivative));
    }
}
@compute @workgroup_size(256)
fn gradient_horizontal(@builtin(global_invocation_id) id:vec3<u32>) {
    for(var index=id.x;index<params.pixels;index=advance(index)) {
        let x=i32(index%params.width); let y=i32(index/params.width);
        var total=vec2<i32>(0);
        for(var tap=0u;tap<params.aperture;tap+=1u) {
            let value=pixel(x+i32(tap)-i32(params.aperture/2u),y);
            total+=value*vec2<i32>(params.derivative[tap/4u][tap%4u],params.smoothing_weights[tap/4u][tap%4u]);
        }
        horizontal[index]=total;
    }
}
@compute @workgroup_size(256)
fn gradient_vertical(@builtin(global_invocation_id) id:vec3<u32>) {
    for(var index=id.x;index<params.pixels;index=advance(index)) {
        let x=index%params.width; let y=i32(index/params.width);
        var total=vec2<i32>(0);
        for(var tap=0u;tap<params.aperture;tap+=1u) {
            let sy=u32(clamp(y+i32(tap)-i32(params.aperture/2u),0,i32(params.height)-1));
            total+=horizontal[sy*params.width+x]*vec2<i32>(params.smoothing_weights[tap/4u][tap%4u],params.derivative[tap/4u][tap%4u]);
        }
        let g=vec2<i32>(scaled(total.x),scaled(total.y));
        gradients[index]=Gradient(pack_components(g),magnitude(g));
    }
}
fn neighbor_magnitude(x:i32,y:i32) -> u32 {
    if x<0 || y<0 || x>=i32(params.width) || y>=i32(params.height) { return 0u; }
    return gradients[u32(y)*params.width+u32(x)].magnitude;
}
fn directional_maximum(x:i32,y:i32,g:Gradient) -> bool {
    let direction=unpack_components(g.components);
    let ax=u32(abs(direction.x)); let ay=u32(abs(direction.y))<<15u; let tg22=ax*13573u;
    if ay<tg22 { return g.magnitude>neighbor_magnitude(x-1,y) && g.magnitude>=neighbor_magnitude(x+1,y); }
    if ay>tg22+(ax<<16u) { return g.magnitude>neighbor_magnitude(x,y-1) && g.magnitude>=neighbor_magnitude(x,y+1); }
    let sign=select(1,-1,(direction.x^direction.y)<0);
    return g.magnitude>neighbor_magnitude(x-sign,y-1) && g.magnitude>neighbor_magnitude(x+sign,y+1);
}
@compute @workgroup_size(256)
fn classify(@builtin(global_invocation_id) id:vec3<u32>) {
    for(var index=id.x;index<params.pixels;index=advance(index)) {
        let g=gradients[index]; var state=0u;
        if g.magnitude>params.low && directional_maximum(i32(index%params.width),i32(index/params.width),g) {
            state=select(1u,2u,g.magnitude>params.high);
        }
        states[index]=state;
        atomicStore(&parent[index],index);
        atomicStore(&strong[index],0u);
    }
}
@compute @workgroup_size(256)
fn connect(@builtin(global_invocation_id) id:vec3<u32>) {
    for(var index=id.x;index<params.pixels;index=advance(index)) {
        if states[index]!=1u { continue; }
        let x=index%params.width; let y=index/params.width;
        // Four forward edges enumerate every undirected eight-neighbor edge.
        if x+1u<params.width && states[index+1u]==1u { unite(index,index+1u); }
        if y+1u<params.height {
            let below=index+params.width;
            if states[below]==1u { unite(index,below); }
            if x>0u && states[below-1u]==1u { unite(index,below-1u); }
            if x+1u<params.width && states[below+1u]==1u { unite(index,below+1u); }
        }
    }
}
@compute @workgroup_size(256)
fn mark_roots(@builtin(global_invocation_id) id:vec3<u32>) {
    for(var index=id.x;index<params.pixels;index=advance(index)) {
        if states[index]!=1u { continue; }
        let r=root(index);
        atomicStore(&parent[index],r);
        let x=index%params.width; let y=index/params.width;
        let x0=x-u32(x>0u); let y0=y-u32(y>0u);
        let x1=min(x+1u,params.width-1u); let y1=min(y+1u,params.height-1u);
        var seeded=false;
        for(var ny=y0;ny<=y1;ny+=1u) {
            for(var nx=x0;nx<=x1;nx+=1u) { seeded=seeded || states[ny*params.width+nx]==2u; }
        }
        if seeded { atomicMax(&strong[r],1u); }
    }
}
@compute @workgroup_size(256)
fn pack_edges(@builtin(global_invocation_id) id:vec3<u32>) {
    for(var index=id.x;index<params.pixels;index=advance(index)) {
        let r=atomicLoad(&parent[index]);
        let edge=select(0u,255u,states[index]==2u || (states[index]==1u && atomicLoad(&strong[r])!=0u));
        textureStore(output,vec2<i32>(i32(index%params.width),i32(index/params.width)),vec4<u32>(edge,0u,0u,0u));
    }
}
