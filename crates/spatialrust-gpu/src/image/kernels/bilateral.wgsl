struct Params { width:u32,height:u32,channels:u32,taps:u32,border:u32,pad0:u32,pad1:u32,pad2:u32, };
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var source:texture_2d<u32>;
@group(0) @binding(2) var<storage,read> coefficients:array<vec2<u32>>;
@group(0) @binding(3) var destination:texture_storage_2d<rgba8uint,write>;
fn bilateral_sample(xy:vec2<i32>)->vec4<u32> {
    let dimensions=vec2<i32>(i32(params.width),i32(params.height));
    if params.border==1u && (any(xy<vec2<i32>(0i)) || any(xy>=dimensions)) {return vec4<u32>(0u);}
    return textureLoad(source,clamp(xy,vec2<i32>(0i),dimensions-vec2<i32>(1i)),0);
}
struct BilateralSums {r:F64,g:F64,b:F64,a:F64,total:F64,};
fn bilateral_accumulate(xy:vec2<i32>)->BilateralSums {
    let center=bilateral_sample(xy);
    var r=fp_uint(0u);var g=fp_uint(0u);var b=fp_uint(0u);var a=fp_uint(0u);var total=fp_uint(0u);
    for(var tap=0u;tap<params.taps;tap++) {
        let offset=coefficients[2u*tap];let row=coefficients[2u*tap+1u].x;
        let pixel=bilateral_sample(xy+vec2<i32>(bitcast<i32>(offset.x),bitcast<i32>(offset.y)));
        let delta=vec4<u32>(abs(vec4<i32>(pixel)-vec4<i32>(center)));
        var l1=delta.x;if params.channels>1u {l1+=delta.y;} if params.channels>2u {l1+=delta.z;} if params.channels>3u {l1+=delta.w;}
        let bits=coefficients[row+l1];if all(bits==vec2<u32>(0u)) {continue;}
        let weight=fp_decode(bits);
        r=fp_sum(r,fp_product(fp_uint(pixel.x),weight));
        if params.channels>1u {g=fp_sum(g,fp_product(fp_uint(pixel.y),weight));}
        if params.channels>2u {b=fp_sum(b,fp_product(fp_uint(pixel.z),weight));}
        if params.channels>3u {a=fp_sum(a,fp_product(fp_uint(pixel.w),weight));}
        total=fp_sum(total,weight);
    }
    return BilateralSums(r,g,b,a,total);
}
@compute @workgroup_size(16,16)
fn bilateral(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.width || id.y>=params.height {return;}
    let xy=vec2<i32>(id.xy);let values=bilateral_accumulate(xy);
    var result=vec4<u32>(bilateral_byte(values.r,values.total),0u,0u,0u);
    if params.channels>1u {result.y=bilateral_byte(values.g,values.total);} if params.channels>2u {result.z=bilateral_byte(values.b,values.total);} if params.channels>3u {result.w=bilateral_byte(values.a,values.total);}
    textureStore(destination,xy,result);
}
