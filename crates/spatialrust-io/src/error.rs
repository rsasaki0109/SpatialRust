use thiserror::Error;

use spatialrust_core::SpatialError;

/// IO-specific error type for SpatialRust.
#[derive(Debug, Error, PartialEq)]
pub enum IoError {
    /// Underlying IO failure.
    #[error("io failure: {0}")]
    Io(String),

    /// PCD header or payload parsing failed.
    #[error("pcd parse error: {0}")]
    PcdParse(String),

    /// Unsupported or invalid PCD encoding.
    #[error("pcd format error: {0}")]
    PcdFormat(String),

    /// PLY header or payload parsing failed.
    #[error("ply parse error: {0}")]
    PlyParse(String),

    /// Unsupported or invalid PLY encoding.
    #[error("ply format error: {0}")]
    PlyFormat(String),

    /// LAS header or payload parsing failed.
    #[error("las parse error: {0}")]
    LasParse(String),

    /// Unsupported or invalid LAS encoding.
    #[error("las format error: {0}")]
    LasFormat(String),

    /// LAZ compression support is not enabled.
    #[error("laz format error: {0}")]
    LazFormat(String),

    /// E57 header or payload parsing failed.
    #[error("e57 parse error: {0}")]
    E57Parse(String),

    /// Unsupported or invalid E57 encoding.
    #[error("e57 format error: {0}")]
    E57Format(String),

    /// COPC header or payload parsing failed.
    #[error("copc parse error: {0}")]
    CopcParse(String),

    /// Unsupported or invalid COPC encoding.
    #[error("copc format error: {0}")]
    CopcFormat(String),

    /// Core data model error propagated from `spatialrust-core`.
    #[error(transparent)]
    Core(#[from] SpatialError),
}

impl From<std::io::Error> for IoError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

#[cfg_attr(not(feature = "io-pcd"), allow(dead_code))]
pub(crate) fn pcd_parse(message: impl Into<String>) -> IoError {
    IoError::PcdParse(message.into())
}

#[cfg_attr(not(feature = "io-pcd"), allow(dead_code))]
pub(crate) fn pcd_format(message: impl Into<String>) -> IoError {
    IoError::PcdFormat(message.into())
}

#[cfg_attr(not(feature = "io-ply"), allow(dead_code))]
pub(crate) fn ply_parse(message: impl Into<String>) -> IoError {
    IoError::PlyParse(message.into())
}

#[cfg_attr(not(feature = "io-ply"), allow(dead_code))]
pub(crate) fn ply_format(message: impl Into<String>) -> IoError {
    IoError::PlyFormat(message.into())
}

#[cfg_attr(not(feature = "io-las"), allow(dead_code))]
pub(crate) fn las_parse(message: impl Into<String>) -> IoError {
    IoError::LasParse(message.into())
}

#[cfg_attr(not(feature = "io-las"), allow(dead_code))]
pub(crate) fn las_format(message: impl Into<String>) -> IoError {
    IoError::LasFormat(message.into())
}

#[allow(dead_code)]
pub(crate) fn laz_format(message: impl Into<String>) -> IoError {
    IoError::LazFormat(message.into())
}

#[cfg_attr(not(feature = "io-e57"), allow(dead_code))]
pub(crate) fn e57_parse(message: impl Into<String>) -> IoError {
    IoError::E57Parse(message.into())
}

#[cfg_attr(not(feature = "io-e57"), allow(dead_code))]
pub(crate) fn e57_format(message: impl Into<String>) -> IoError {
    IoError::E57Format(message.into())
}

#[cfg_attr(not(feature = "io-copc"), allow(dead_code))]
pub(crate) fn copc_parse(message: impl Into<String>) -> IoError {
    IoError::CopcParse(message.into())
}

#[cfg_attr(not(feature = "io-copc"), allow(dead_code))]
pub(crate) fn copc_format(message: impl Into<String>) -> IoError {
    IoError::CopcFormat(message.into())
}
