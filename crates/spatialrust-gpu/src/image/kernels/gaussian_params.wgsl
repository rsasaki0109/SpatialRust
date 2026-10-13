struct Params {
    width:u32, height:u32, pixels:u32, groups:u32,
    kernel_width:u32, kernel_height:u32, border:u32, bits:u32,
    channels:u32, padding0:u32, padding1:u32, padding2:u32,
    weights_x:array<vec4<u32>,2>, weights_y:array<vec4<u32>,2>, constant:vec4<u32>,
}
@group(0) @binding(0) var<uniform> params:Params;

// Coordinates differ from the image by at most three taps. Reflections can
// repeat on one/two-pixel strips; the validated signed arithmetic stays bounded.
fn mapped(index:i32, length:u32) -> i32 {
    let n=i32(length);
    if index>=0 && index<n { return index; }
    if params.border==4u { return -1; }
    if params.border==0u { return clamp(index,0,n-1); }
    if n==1 { return 0; }
    if params.border==3u {
        // Keep remainder operands unsigned/nonnegative, including negative taps.
        if index<0 { return i32(length-1u-u32(-(index+1))%length); }
        return i32(u32(index)%length);
    }
    var value=index;
    loop {
        if value>=0 && value<n { break; }
        if value<0 {
            value=select(-value-1,-value,params.border==2u);
        } else {
            value=select(2*n-value-1,2*n-value-2,params.border==2u);
        }
    }
    return value;
}
