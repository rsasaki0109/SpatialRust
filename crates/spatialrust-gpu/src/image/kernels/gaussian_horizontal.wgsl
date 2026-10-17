@group(0) @binding(1) var source:texture_2d<u32>;
@group(0) @binding(2) var<storage,read_write> sums:array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    var index=id.x;
    loop {
        if index>=params.pixels { break; }
        let x=index%params.width; let y=index/params.width;
        // Symmetric nonnegative weights share one multiply per mirrored pair.
        // Pair values are at most 510; total mass bounds every u32 partial sum.
        let radius=params.kernel_width/2u;
        let center=textureLoad(source,vec2<i32>(i32(x),i32(y)),0);
        var total=center*params.weights_x[radius/4u][radius%4u];
        for(var tap=0u;tap<radius;tap+=1u) {
            let left=mapped(i32(x)+i32(tap)-i32(radius),params.width);
            let right=mapped(i32(x)+i32(params.kernel_width-1u-tap)-i32(radius),params.width);
            var a=params.constant; var b=params.constant;
            if left>=0 { a=textureLoad(source,vec2<i32>(left,i32(y)),0); }
            if right>=0 { b=textureLoad(source,vec2<i32>(right,i32(y)),0); }
            total+=(a+b)*params.weights_x[tap/4u][tap%4u];
        }
        for(var c=0u;c<params.channels;c+=1u) { sums[index*params.channels+c]=total[c]; }
        let stride=params.groups*256u;
        if params.pixels-index<=stride { break; }
        index+=stride;
    }
}
