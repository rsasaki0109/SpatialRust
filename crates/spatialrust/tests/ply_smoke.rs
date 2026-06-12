#[cfg(feature = "io-ply")]
#[test]
fn ply_public_api_roundtrip() {
    use spatialrust::{read_ply, write_ply, PlyWriteFormat, PointCloudBuilder};
    use std::io::Cursor;

    let mut builder = PointCloudBuilder::xyz();
    builder.push_point([1.0, 2.0, 3.0]).unwrap();
    let cloud = builder.build().unwrap();

    let mut bytes = Vec::new();
    write_ply(&mut bytes, &cloud, PlyWriteFormat::Ascii).unwrap();
    let loaded = read_ply(&mut Cursor::new(bytes)).unwrap();
    assert_eq!(loaded.len(), 1);
}
