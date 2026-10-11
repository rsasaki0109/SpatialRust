struct SplitParams {
    width: u32, height: u32, tile_width: u32, tile_height: u32,
    tiles_x: u32, tile_count: u32, chunks: u32, partial_groups: u32,
    clip_count: u32, tile_groups: u32, padding: vec2<u32>,
};
@group(0) @binding(0) var<uniform> params: SplitParams;
