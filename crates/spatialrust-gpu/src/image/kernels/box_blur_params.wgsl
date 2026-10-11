struct Params {
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
    border_mode: u32,
    groups: u32,
    pixel_count: u32,
    pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;

fn first(center: u32, radius: u32) -> u32 {
    return center - min(center, radius);
}
fn last(center: u32, radius: u32, length: u32) -> u32 {
    return center + min(radius, length - 1u - center);
}
fn left_mass(center: u32, radius: u32) -> u32 {
    return radius - min(center, radius);
}
fn right_mass(center: u32, radius: u32, length: u32) -> u32 {
    return radius - min(length - 1u - center, radius);
}
