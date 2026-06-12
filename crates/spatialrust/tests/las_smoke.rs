#[cfg(feature = "io-las")]
#[test]
fn las_public_api_roundtrip() {
    use spatialrust::{read_las, write_las, LasWriteFormat, PointCloudBuilder};
    use std::io::Cursor;

    let mut builder = PointCloudBuilder::xyz();
    builder.push_point([1.0, 2.0, 3.0]).unwrap();
    let cloud = builder.build().unwrap();

    let mut cursor = write_las(Cursor::new(Vec::new()), &cloud, LasWriteFormat::Las).unwrap();
    cursor.set_position(0);
    let loaded = read_las(cursor).unwrap();
    assert_eq!(loaded.len(), 1);
}
