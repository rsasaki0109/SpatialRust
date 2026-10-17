@group(0) @binding(1) var output:texture_storage_2d<rgba8uint,write>;
@group(0) @binding(2) var<storage,read> sums:array<u32>;

// Q15 weights are nonnegative with mass 32768; horizontal sums are < 2^23.
// Accumulate their base-65536 digits separately, then carry only once.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    var index=id.x;
    loop {
        if index>=params.pixels { break; }
        let x=index%params.width; let y=index/params.width;
        var value=vec4<u32>(0u);
        for(var c=0u;c<params.channels;c+=1u) {
            let radius=params.kernel_height/2u;
            let center=sums[(y*params.width+x)*params.channels+c];
            let center_weight=params.weights_y[radius/4u][radius%4u];
            var lo=0u; var hi=0u;
            if params.bits==8u { lo=center*center_weight; }
            else {
                lo=(center&65535u)*center_weight;
                hi=(center>>16u)*center_weight;
            }
            for(var tap=0u;tap<radius;tap+=1u) {
                let top=mapped(i32(y)+i32(tap)-i32(radius),params.height);
                let bottom=mapped(i32(y)+i32(params.kernel_height-1u-tap)-i32(radius),params.height);
                var a=params.constant[c]<<params.bits; var b=a;
                if top>=0 { a=sums[(u32(top)*params.width+x)*params.channels+c]; }
                if bottom>=0 { b=sums[(u32(bottom)*params.width+x)*params.channels+c]; }
                let weight=params.weights_y[tap/4u][tap%4u];
                if params.bits==8u { lo+=(a+b)*weight; }
                else {
                    // Split before pairing: each digit's weighted mass stays bounded.
                    lo+=((a&65535u)+(b&65535u))*weight;
                    hi+=((a>>16u)+(b>>16u))*weight;
                }
            }
            if params.bits==8u { value[c]=min(255u,(lo+32768u)>>16u); }
            else {
                // lo <= 65535*32768 < 2^31; hi <= 127*32768 < 2^22.
                // Dividing by 2^30 with half-up rounding discards the low 16 bits.
                value[c]=min(255u,(hi+(lo>>16u)+8192u)>>14u);
            }
        }
        textureStore(output,vec2<i32>(i32(x),i32(y)),value);
        let stride=params.groups*256u;
        if params.pixels-index<=stride { break; }
        index+=stride;
    }
}
