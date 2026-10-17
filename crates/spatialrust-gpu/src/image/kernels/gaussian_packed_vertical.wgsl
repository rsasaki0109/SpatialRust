@group(0) @binding(1) var output:texture_storage_2d<rgba8uint,write>;
@group(0) @binding(2) var<storage,read> sums:array<u32>;

// The packed planner selects Q8 only: horizontal <= 255*256, weight mass 256.
// Both channels share one word load, and their rounded sums fit in u32.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    var index=id.x;
    loop {
        if index>=params.pixels { break; }
        let x=index%params.width; let y=index/params.width;
        var value=vec4<u32>(0u);
        for(var pair=0u;pair<params.padding0;pair+=1u) {
            let c=2u*pair;
            let radius=params.kernel_height/2u;
            let center=sums[(y*params.width+x)*params.padding0+pair];
            var total=vec2<u32>(center&65535u,center>>16u)*params.weights_y[radius/4u][radius%4u];
            for(var tap=0u;tap<radius;tap+=1u) {
                let top=mapped(i32(y)+i32(tap)-i32(radius),params.height);
                let bottom=mapped(i32(y)+i32(params.kernel_height-1u-tap)-i32(radius),params.height);
                var a=vec2<u32>(params.constant[c],params.constant[c+1u])<<vec2<u32>(8u);
                var b=a;
                if top>=0 {
                    let word=sums[(u32(top)*params.width+x)*params.padding0+pair];
                    a=vec2<u32>(word&65535u,word>>16u);
                }
                if bottom>=0 {
                    let word=sums[(u32(bottom)*params.width+x)*params.padding0+pair];
                    b=vec2<u32>(word&65535u,word>>16u);
                }
                // Decode each half before addition to avoid cross-channel carries.
                total+=(a+b)*params.weights_y[tap/4u][tap%4u];
            }
            let rounded=(total+vec2<u32>(32768u))>>vec2<u32>(16u);
            value[c]=min(255u,rounded.x);
            if c+1u<params.channels { value[c+1u]=min(255u,rounded.y); }
        }
        textureStore(output,vec2<i32>(i32(x),i32(y)),value);
        let stride=params.groups*256u;
        if params.pixels-index<=stride { break; }
        index+=stride;
    }
}
