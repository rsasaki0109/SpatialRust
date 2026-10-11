struct ClaheParams {
    width: u32, height: u32, tile_width: u32, tile_height: u32,
    tiles_x: u32, tiles_y: u32, clip_count: u32, tile_count: u32,
    tile_groups: u32, geometry_groups: u32, groups_x: u32, groups_y: u32,
};
@group(0) @binding(0) var<uniform> params: ClaheParams;
