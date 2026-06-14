#[cfg(feature = "io-copc")]
#[test]
fn copc_public_api_and_format_detection() {
    use spatialrust::{
        detect_point_cloud_format, read_copc_file, write_copc_file, PointCloudBuilder,
        PointCloudFileFormat,
    };

    let mut builder = PointCloudBuilder::xyz();
    builder.push_point([1.0, 2.0, 3.0]).unwrap();
    builder.push_point([4.0, 5.0, 6.0]).unwrap();
    let cloud = builder.build().unwrap();

    assert_eq!(detect_point_cloud_format("scan.copc.laz"), Some(PointCloudFileFormat::Copc));

    let path = std::env::temp_dir()
        .join(format!("spatialrust_copc_smoke_{}.copc.laz", std::process::id()));
    write_copc_file(&path, &cloud).expect("write copc");
    let loaded = read_copc_file(&path).expect("read copc");
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded.len(), cloud.len());
}

#[cfg(feature = "io-copc")]
#[test]
fn copc_query_api_surface() {
    use spatialrust::{
        copc_level_for_resolution, read_copc_file_in_bounds, read_copc_file_with_query, CopcBounds,
        CopcQuery,
    };

    assert_eq!(copc_level_for_resolution(10.0, 2.5), 2);

    let bounds = CopcBounds::from_ranges((0.0, 10.0), (0.0, 10.0), (0.0, 10.0));
    let query = CopcQuery::with_resolution(bounds, 0.5);
    assert!(query.validate().is_ok());

    let path = std::env::temp_dir()
        .join(format!("spatialrust_copc_query_{}.copc.laz", std::process::id()));
    let bounds_error = read_copc_file_in_bounds(
        &path,
        CopcBounds::from_ranges((1.0, 0.0), (0.0, 1.0), (0.0, 1.0)),
    )
    .unwrap_err();
    assert!(matches!(bounds_error, spatialrust::IoError::CopcFormat(_)));

    let query_error = read_copc_file_with_query(&path, &query).unwrap_err();
    assert!(matches!(
        query_error,
        spatialrust::IoError::CopcFormat(_) | spatialrust::IoError::CopcParse(_)
    ));
}

#[cfg(feature = "io-copc-http")]
#[test]
fn copc_http_api_surface() {
    use spatialrust::{read_copc_url, HttpByteSource};

    assert!(read_copc_url("/tmp/local.copc.laz").is_err());
    let source = HttpByteSource::new("https://example.com/cloud.copc.laz").unwrap();
    assert_eq!(source.url(), "https://example.com/cloud.copc.laz");
}
