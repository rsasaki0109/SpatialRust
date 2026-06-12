#[cfg(feature = "io-e57")]
#[test]
fn e57_public_api_roundtrip() {
    use spatialrust::{
        read_e57_file, read_point_cloud_file, write_e57_file, write_point_cloud_file,
        PointCloudBuilder,
    };

    let mut builder = PointCloudBuilder::xyz();
    builder.push_point([1.0, 2.0, 3.0]).unwrap();
    builder.push_point([4.0, 5.0, 6.0]).unwrap();
    let cloud = builder.build().unwrap();

    let path = std::env::temp_dir().join(format!("spatialrust_e57_smoke_{}.e57", std::process::id()));
    write_e57_file(&path, &cloud).unwrap();
    let loaded = read_e57_file(&path).unwrap();
    assert_eq!(loaded.len(), 2);

    write_point_cloud_file(&path, &cloud).unwrap();
    let auto = read_point_cloud_file(&path).unwrap();
    let _ = std::fs::remove_file(path);
    assert_eq!(auto.len(), 2);
}
