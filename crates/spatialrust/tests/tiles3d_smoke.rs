#[cfg(all(feature = "interchange-tiles3d-copc", feature = "io-copc"))]
#[test]
fn tiles3d_copc_public_api_end_to_end() {
    use spatialrust::interchange::{decode_pnts, export_copc_tileset, CopcTilesetOptions};
    use spatialrust::{write_copc_file_with_params, CopcWriterParams, PointCloudBuilder};

    let mut builder = PointCloudBuilder::xyz();
    for index in 0..7_000 {
        let x = (index % 31) as f32 - 15.0;
        let y = ((index / 31) % 29) as f32 - 14.0;
        let z = ((index / (31 * 29)) % 23) as f32 - 11.0;
        builder.push_point([x, y, z]).unwrap();
    }
    let cloud = builder.build().unwrap();

    let copc_path = std::env::temp_dir()
        .join(format!("spatialrust_tiles3d_copc_it_{}.copc.laz", std::process::id()));
    write_copc_file_with_params(
        &copc_path,
        &cloud,
        &CopcWriterParams { max_points_per_node: 96, max_depth: 8 },
    )
    .unwrap();

    let out_dir = std::env::temp_dir()
        .join(format!("spatialrust_tiles3d_copc_it_out_{}", std::process::id()));
    let receipt =
        export_copc_tileset(&copc_path, &out_dir, &CopcTilesetOptions::default()).unwrap();
    assert_eq!(receipt.point_count, cloud.len() as u64);
    assert!(receipt.tile_count > 1);
    for tile in 0..receipt.tile_count {
        let pnts = std::fs::read(out_dir.join(format!("{tile}.pnts"))).unwrap();
        assert!(decode_pnts(&pnts).is_ok());
    }
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_file(&copc_path);
}

#[cfg(all(feature = "interchange-tiles3d", feature = "io-pcd"))]
#[test]
fn tiles3d_public_api_end_to_end() {
    use std::io::Cursor;

    use spatialrust::interchange::{
        build_point_tileset, decode_pnts, parse_tileset_json, write_point_tileset,
        TilesetBuilderOptions,
    };
    use spatialrust::{read_pcd, write_pcd, HasPositions3, PointCloudBuilder};

    let mut builder = PointCloudBuilder::xyz();
    for x in 0..10 {
        for y in 0..10 {
            for z in 0..10 {
                builder.push_point([x as f32, y as f32, z as f32]).unwrap();
            }
        }
    }
    let cloud = builder.build().unwrap();

    let mut bytes = Vec::new();
    write_pcd(&mut bytes, &cloud, spatialrust::PcdWriteFormat::Ascii).unwrap();
    let loaded = read_pcd(&mut Cursor::new(bytes)).unwrap();
    assert_eq!(loaded.len(), 1000);

    let (x, y, z) = loaded.positions3().unwrap();
    let mut positions = Vec::with_capacity(loaded.len() * 3);
    for index in 0..loaded.len() {
        positions.push(x[index]);
        positions.push(y[index]);
        positions.push(z[index]);
    }

    let built = build_point_tileset(
        &positions,
        None,
        &TilesetBuilderOptions { max_points_per_tile: 32, ..Default::default() },
    )
    .unwrap();
    assert!(built.tiles.len() > 1);
    let total: usize = built.tiles.iter().map(|tile| tile.point_count).sum();
    assert_eq!(total, 1000);

    let json = spatialrust::interchange::serialize_tileset_json(&built.tileset).unwrap();
    let parsed = parse_tileset_json(&json).unwrap();
    assert_eq!(parsed, built.tileset);

    let dir = std::env::temp_dir().join(format!("spatialrust-tiles3d-it-{}", std::process::id()));
    let receipt = write_point_tileset(&dir, &built).unwrap();
    assert_eq!(receipt.point_count, 1000);
    assert!(dir.join("tileset.json").exists());
    for tile in &built.tiles {
        assert!(dir.join(&tile.uri).exists());
        let pnts = std::fs::read(dir.join(&tile.uri)).unwrap();
        let decoded = decode_pnts(&pnts).unwrap();
        assert_eq!(decoded.point_count(), tile.point_count);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
