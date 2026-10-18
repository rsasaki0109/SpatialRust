// Q15 multi-channel vertical filtering; original per-channel digit bounds apply.
@group(0) @binding(1) var output:texture_storage_2d<rgba8uint,write>;
@group(0) @binding(2) var<storage,read> sums:array<u32>;

fn pair_at(x:u32, row:i32, c:u32) -> vec2<u32> {
    if row<0 { return vec2<u32>(params.constant[c],params.constant[c+1u])<<vec2<u32>(15u); }
    let base=(u32(row)*params.width+x)*params.channels+c;
    var result=vec2<u32>(sums[base],0u);
    if c+1u<params.channels { result.y=sums[base+1u]; }
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    var index=id.x;
    loop {
        if index>=params.pixels { break; }
        let x=index%params.width; let y=index/params.width;
        let radius=params.kernel_height/2u;
        var value=vec4<u32>(0u);
        for(var c=0u;c<params.channels;c+=2u) {
            let center=pair_at(x,i32(y),c);
            let center_weight=params.weights_y[radius/4u][radius%4u];
            var lo=(center&vec2<u32>(65535u))*center_weight;
            var hi=(center>>vec2<u32>(16u))*center_weight;
            for(var tap=0u;tap<radius;tap+=1u) {
                let top=mapped(i32(y)+i32(tap)-i32(radius),params.height);
                let bottom=mapped(i32(y)+i32(params.kernel_height-1u-tap)-i32(radius),params.height);
                let a=pair_at(x,top,c); let b=pair_at(x,bottom,c);
                let weight=params.weights_y[tap/4u][tap%4u];
                lo+=((a&vec2<u32>(65535u))+(b&vec2<u32>(65535u)))*weight;
                hi+=((a>>vec2<u32>(16u))+(b>>vec2<u32>(16u)))*weight;
            }
            let rounded=(hi+(lo>>vec2<u32>(16u))+vec2<u32>(8192u))>>vec2<u32>(14u);
            value[c]=min(255u,rounded.x);
            if c+1u<params.channels { value[c+1u]=min(255u,rounded.y); }
        }
        textureStore(output,vec2<i32>(i32(x),i32(y)),value);
        let stride=params.groups*256u;
        if params.pixels-index<=stride { break; }
        index+=stride;
    }
}
