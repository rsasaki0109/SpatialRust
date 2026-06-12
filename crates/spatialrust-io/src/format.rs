//! Extension-based point cloud file format detection and dispatch.

use std::path::Path;

use spatialrust_core::PointCloud;

use crate::IoError;

/// Supported point cloud file formats detected from file extensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointCloudFileFormat {
    /// PCD (Point Cloud Data).
    Pcd,
    /// PLY (Polygon File Format).
    Ply,
    /// LAS (ASPRS LiDAR).
    Las,
    /// LAZ (compressed LAS).
    Laz,
    /// E57 (ASTM E2807).
    E57,
    /// COPC (Cloud Optimized Point Cloud).
    Copc,
}

impl PointCloudFileFormat {
    /// Returns the canonical lowercase file extension without a leading dot.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Pcd => "pcd",
            Self::Ply => "ply",
            Self::Las => "las",
            Self::Laz => "laz",
            Self::E57 => "e57",
            Self::Copc => "copc.laz",
        }
    }
}

/// Detects a point cloud format from a file path extension.
#[must_use]
pub fn detect_point_cloud_format(path: impl AsRef<Path>) -> Option<PointCloudFileFormat> {
    let path = path.as_ref();
    let file_name = path.file_name()?.to_str()?.to_ascii_lowercase();
    if file_name.ends_with(".copc.laz") || file_name.ends_with(".copc.las") {
        return Some(PointCloudFileFormat::Copc);
    }

    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "pcd" => Some(PointCloudFileFormat::Pcd),
        "ply" => Some(PointCloudFileFormat::Ply),
        "las" => Some(PointCloudFileFormat::Las),
        "laz" => Some(PointCloudFileFormat::Laz),
        "e57" => Some(PointCloudFileFormat::E57),
        _ => None,
    }
}

/// Reads a point cloud from disk, dispatching on the file extension.
pub fn read_point_cloud_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    let path = path.as_ref();
    let format = detect_point_cloud_format(path).ok_or_else(|| {
        IoError::Io(format!(
            "unsupported or missing point cloud extension: {}",
            path.display()
        ))
    })?;
    read_point_cloud_file_with_format(path, format)
}

/// Reads a point cloud using an explicit format.
pub fn read_point_cloud_file_with_format(
    path: impl AsRef<Path>,
    format: PointCloudFileFormat,
) -> Result<PointCloud, IoError> {
    match format {
        PointCloudFileFormat::Pcd => read_pcd_file(path),
        PointCloudFileFormat::Ply => read_ply_file(path),
        PointCloudFileFormat::Las => read_las_file(path),
        PointCloudFileFormat::Laz => read_laz_file(path),
        PointCloudFileFormat::E57 => read_e57_file(path),
        PointCloudFileFormat::Copc => read_copc_file(path),
    }
}

/// Writes a point cloud to disk, dispatching on the file extension.
pub fn write_point_cloud_file(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    let path = path.as_ref();
    let format = detect_point_cloud_format(path).ok_or_else(|| {
        IoError::Io(format!(
            "unsupported or missing point cloud extension: {}",
            path.display()
        ))
    })?;
    write_point_cloud_file_with_format(path, cloud, format)
}

/// Writes a point cloud using an explicit format.
pub fn write_point_cloud_file_with_format(
    path: impl AsRef<Path>,
    cloud: &PointCloud,
    format: PointCloudFileFormat,
) -> Result<(), IoError> {
    match format {
        PointCloudFileFormat::Pcd => write_pcd_file(path, cloud),
        PointCloudFileFormat::Ply => write_ply_file(path, cloud),
        PointCloudFileFormat::Las => write_las_file(path, cloud),
        PointCloudFileFormat::Laz => write_laz_file(path, cloud),
        PointCloudFileFormat::E57 => write_e57_file(path, cloud),
        PointCloudFileFormat::Copc => write_copc_file(path, cloud),
    }
}

#[cfg(feature = "io-pcd")]
fn read_pcd_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    crate::pcd::read_pcd_file(path)
}

#[cfg(not(feature = "io-pcd"))]
fn read_pcd_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    Err(missing_feature_error("io-pcd", path.as_ref()))
}

#[cfg(feature = "io-ply")]
fn read_ply_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    crate::ply::read_ply_file(path)
}

#[cfg(not(feature = "io-ply"))]
fn read_ply_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    Err(missing_feature_error("io-ply", path.as_ref()))
}

#[cfg(feature = "io-las")]
fn read_las_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    crate::las::read_las_file(path)
}

#[cfg(not(feature = "io-las"))]
fn read_las_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    Err(missing_feature_error("io-las", path.as_ref()))
}

#[cfg(feature = "io-laz")]
fn read_laz_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    crate::las::read_las_file(path)
}

#[cfg(not(feature = "io-laz"))]
fn read_laz_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    Err(missing_feature_error("io-laz", path.as_ref()))
}

#[cfg(feature = "io-e57")]
fn read_e57_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    crate::e57::read_e57_file(path)
}

#[cfg(not(feature = "io-e57"))]
fn read_e57_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    Err(missing_feature_error("io-e57", path.as_ref()))
}

#[cfg(feature = "io-copc")]
fn read_copc_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    crate::copc::read_copc_file(path)
}

#[cfg(not(feature = "io-copc"))]
fn read_copc_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    Err(missing_feature_error("io-copc", path.as_ref()))
}

#[cfg(feature = "io-pcd")]
fn write_pcd_file(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    crate::pcd::write_pcd_file(path, cloud, crate::pcd::PcdWriteFormat::Binary)
}

#[cfg(not(feature = "io-pcd"))]
fn write_pcd_file(path: impl AsRef<Path>, _cloud: &PointCloud) -> Result<(), IoError> {
    Err(missing_feature_error("io-pcd", path.as_ref()))
}

#[cfg(feature = "io-ply")]
fn write_ply_file(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    crate::ply::write_ply_file(path, cloud, crate::ply::PlyWriteFormat::BinaryLittleEndian)
}

#[cfg(not(feature = "io-ply"))]
fn write_ply_file(path: impl AsRef<Path>, _cloud: &PointCloud) -> Result<(), IoError> {
    Err(missing_feature_error("io-ply", path.as_ref()))
}

#[cfg(feature = "io-las")]
fn write_las_file(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    crate::las::write_las_file(path, cloud, crate::las::LasWriteFormat::Las)
}

#[cfg(not(feature = "io-las"))]
fn write_las_file(path: impl AsRef<Path>, _cloud: &PointCloud) -> Result<(), IoError> {
    Err(missing_feature_error("io-las", path.as_ref()))
}

#[cfg(feature = "io-laz")]
fn write_laz_file(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    crate::las::write_las_file(path, cloud, crate::las::LasWriteFormat::Laz)
}

#[cfg(not(feature = "io-laz"))]
fn write_laz_file(path: impl AsRef<Path>, _cloud: &PointCloud) -> Result<(), IoError> {
    Err(missing_feature_error("io-laz", path.as_ref()))
}

#[cfg(feature = "io-e57")]
fn write_e57_file(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    crate::e57::write_e57_file(path, cloud)
}

#[cfg(not(feature = "io-e57"))]
fn write_e57_file(path: impl AsRef<Path>, _cloud: &PointCloud) -> Result<(), IoError> {
    Err(missing_feature_error("io-e57", path.as_ref()))
}

#[cfg(feature = "io-copc")]
fn write_copc_file(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    crate::copc::write_copc_file(path, cloud)
}

#[cfg(not(feature = "io-copc"))]
fn write_copc_file(path: impl AsRef<Path>, _cloud: &PointCloud) -> Result<(), IoError> {
    Err(missing_feature_error("io-copc", path.as_ref()))
}

fn missing_feature_error(feature: &str, path: &Path) -> IoError {
    IoError::Io(format!(
        "reading or writing `{}` requires the `{feature}` feature",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::{detect_point_cloud_format, PointCloudFileFormat};
    use std::path::PathBuf;

    #[test]
    fn detects_known_extensions() {
        assert_eq!(
            detect_point_cloud_format("scan.pcd"),
            Some(PointCloudFileFormat::Pcd)
        );
        assert_eq!(
            detect_point_cloud_format(PathBuf::from("/tmp/cloud.PLY")),
            Some(PointCloudFileFormat::Ply)
        );
        assert_eq!(detect_point_cloud_format("data.xyz"), None);
        assert_eq!(
            detect_point_cloud_format("scan.copc.laz"),
            Some(PointCloudFileFormat::Copc)
        );
    }

    #[cfg(feature = "io-pcd")]
    #[test]
    fn roundtrip_via_auto_dispatch() {
        use super::{read_point_cloud_file, write_point_cloud_file};
        use spatialrust_core::PointCloudBuilder;

        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        let cloud = builder.build().unwrap();

        let path = std::env::temp_dir().join(format!("spatialrust_auto_{}.pcd", std::process::id()));
        write_point_cloud_file(&path, &cloud).unwrap();
        let loaded = read_point_cloud_file(&path).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(loaded.len(), 1);
    }
}
