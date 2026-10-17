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
            var total=vec2<u32>(0u);
            for(var tap=0u;tap<params.kernel_height;tap+=1u) {
                let sy=mapped(i32(y)+i32(tap)-i32(params.kernel_height/2u),params.height);
                var horizontal=vec2<u32>(params.constant[c],params.constant[c+1u])<<vec2<u32>(8u);
                if sy>=0 {
                    let word=sums[(u32(sy)*params.width+x)*params.padding0+pair];
                    horizontal=vec2<u32>(word&65535u,word>>16u);
                }
                let weight=params.weights_y[tap/4u][tap%4u];
                total+=horizontal*weight;
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
