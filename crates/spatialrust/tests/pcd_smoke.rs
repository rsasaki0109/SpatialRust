#[cfg(feature = "io-pcd")]
#[test]
fn pcd_public_api_roundtrip() {
    use spatialrust::{read_pcd, write_pcd, PcdWriteFormat, PointCloudBuilder};
    use std::io::Cursor;

    let mut builder = PointCloudBuilder::xyz();
    builder.push_point([1.0, 2.0, 3.0]).unwrap();
    let cloud = builder.build().unwrap();

    let mut bytes = Vec::new();
    write_pcd(&mut bytes, &cloud, PcdWriteFormat::Ascii).unwrap();
    let loaded = read_pcd(&mut Cursor::new(bytes)).unwrap();
    assert_eq!(loaded.len(), 1);
}
