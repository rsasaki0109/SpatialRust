// Exercises the HTTP COPC reader; only compiled when the feature is enabled so
// `cargo test --workspace` (default features) does not fail to build it.
#![cfg(feature = "io-copc-http")]

mod common;

#[test]
fn read_copc_url_with_query_matches_local_file() {
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};
    use spatialrust_io::{
        read_copc_file_with_query, read_copc_url_info, read_copc_url_with_query,
        write_copc_file_with_params, CopcBounds, CopcQuery, CopcWriterParams,
    };

    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
    for index in 0..2_000 {
        let x = (index % 64) as f32 * 0.1;
        let y = (index / 64) as f32 * 0.1;
        builder.push_point([x, y, 0.0, (index % 256) as f32]).unwrap();
    }
    let cloud = builder.build().unwrap();

    let path = std::env::temp_dir()
        .join(format!("spatialrust_io_http_copc_{}.copc.laz", std::process::id()));
    write_copc_file_with_params(
        &path,
        &cloud,
        &CopcWriterParams { max_points_per_node: 256, max_depth: 8 },
    )
    .unwrap();
    let payload = std::fs::read(&path).unwrap();
    let (_server, url) = common::LocalHttpFileServer::start(payload);

    let info = read_copc_url_info(&url).expect("http copc info");
    let roi = CopcBounds::from_ranges((0.0, 3.0), (0.0, 3.0), (-0.01, 0.01));
    let resolution = info.spacing * 4.0;
    let query = CopcQuery::with_resolution(roi, resolution);
    let local = read_copc_file_with_query(&path, &query).expect("local query");
    let remote = read_copc_url_with_query(&url, Some(&query)).expect("http query");

    assert_eq!(remote.len(), local.len());
    assert!(remote.len() < cloud.len());

    let _ = std::fs::remove_file(path);
}
