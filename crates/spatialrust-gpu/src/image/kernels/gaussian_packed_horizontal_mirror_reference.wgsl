@group(0) @binding(1) var source:texture_2d<u32>;
@group(0) @binding(2) var<storage,read_write> sums:array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    var index=id.x;
    loop {
        if index>=params.pixels { break; }
        let x=index%params.width; let y=index/params.width;
        var total=vec4<u32>(0u);
        for(var tap=0u;tap<params.kernel_width;tap+=1u) {
            let sx=mapped(i32(x)+i32(tap)-i32(params.kernel_width/2u),params.width);
            var value=params.constant;
            if sx>=0 { value=textureLoad(source,vec2<i32>(sx,i32(y)),0); }
            total+=value*params.weights_x[tap/4u][tap%4u];
        }
        // One invocation owns all words of this pixel; odd channels zero the high half.
        for(var pair=0u;pair<params.padding0;pair+=1u) {
            let c=2u*pair;
            var word=total[c];
            if c+1u<params.channels { word|=total[c+1u]<<16u; }
            sums[index*params.padding0+pair]=word;
        }
        let stride=params.groups*256u;
        if params.pixels-index<=stride { break; }
        index+=stride;
    }
}
