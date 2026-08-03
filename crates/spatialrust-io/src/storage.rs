//! Explicit input/output roots for local spatial data.

use std::path::{Component, Path, PathBuf};

#[cfg(feature = "storage-preflight")]
use fs2::available_space;

use crate::IoError;

/// Initial v1.3 minimum free-space floor for an external output root.
#[cfg(feature = "storage-preflight")]
pub const DEFAULT_MIN_OUTPUT_FREE_BYTES: u64 = 20 * 1024 * 1024 * 1024;

/// Result of a fail-closed output-root free-space check.
#[cfg(feature = "storage-preflight")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoragePreflight {
    /// Absolute output root that was inspected.
    pub root: PathBuf,
    /// Bytes available on the filesystem containing [`Self::root`].
    pub available_bytes: u64,
    /// Minimum bytes required by the caller.
    pub required_free_bytes: u64,
}

#[cfg(feature = "storage-preflight")]
impl StoragePreflight {
    /// Checks an existing absolute directory before opening a data source.
    pub fn check(root: impl AsRef<Path>, required_free_bytes: u64) -> Result<Self, IoError> {
        let root = root.as_ref();
        if !root.is_absolute() {
            return Err(IoError::Storage(format!(
                "output preflight root `{}` must be absolute",
                root.display()
            )));
        }
        let metadata = std::fs::metadata(root).map_err(|error| {
            IoError::Storage(format!(
                "cannot inspect output preflight root `{}`: {error}",
                root.display()
            ))
        })?;
        if !metadata.is_dir() {
            return Err(IoError::Storage(format!(
                "output preflight root `{}` is not a directory",
                root.display()
            )));
        }
        let available_bytes = available_space(root).map_err(|error| {
            IoError::Storage(format!("cannot read free space for `{}`: {error}", root.display()))
        })?;
        if available_bytes < required_free_bytes {
            return Err(IoError::Storage(format!(
                "output root `{}` has {available_bytes} available bytes, below required {required_free_bytes}",
                root.display()
            )));
        }
        Ok(Self { root: root.to_path_buf(), available_bytes, required_free_bytes })
    }
}

/// Optional roots used to resolve logical input and output paths.
///
/// Relative paths are joined to their corresponding root. Absolute paths are
/// always treated as explicit locations and bypass the roots. Relative paths
/// containing `..` are rejected so a logical dataset path cannot escape its
/// configured root.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StorageRoots {
    input_root: Option<PathBuf>,
    output_root: Option<PathBuf>,
}

impl StorageRoots {
    /// Creates a root mapping. Either root may be omitted.
    #[must_use]
    pub const fn new(input_root: Option<PathBuf>, output_root: Option<PathBuf>) -> Self {
        Self { input_root, output_root }
    }

    /// Returns the configured input root, if any.
    #[must_use]
    pub fn input_root(&self) -> Option<&Path> {
        self.input_root.as_deref()
    }

    /// Returns the configured output root, if any.
    #[must_use]
    pub fn output_root(&self) -> Option<&Path> {
        self.output_root.as_deref()
    }

    /// Resolves a logical input path against the configured input root.
    pub fn resolve_input(&self, path: impl AsRef<Path>) -> Result<PathBuf, IoError> {
        resolve_path(self.input_root.as_deref(), path.as_ref(), "input")
    }

    /// Resolves a logical output path against the configured output root.
    pub fn resolve_output(&self, path: impl AsRef<Path>) -> Result<PathBuf, IoError> {
        resolve_path(self.output_root.as_deref(), path.as_ref(), "output")
    }

    /// Creates the parent directory for an explicit output path when needed.
    pub fn ensure_output_parent(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        Ok(())
    }

    /// Checks the configured output root before opening an external data source.
    #[cfg(feature = "storage-preflight")]
    pub fn preflight_output(&self, required_free_bytes: u64) -> Result<StoragePreflight, IoError> {
        let root = self
            .output_root
            .as_deref()
            .ok_or_else(|| IoError::Storage("output preflight requires an output root".into()))?;
        StoragePreflight::check(root, required_free_bytes)
    }
}

fn resolve_path(root: Option<&Path>, path: &Path, kind: &str) -> Result<PathBuf, IoError> {
    if path.is_absolute() || root.is_none() {
        return Ok(path.to_path_buf());
    }

    if path.components().any(|component| component == Component::ParentDir) {
        return Err(IoError::Storage(format!(
            "relative {kind} path `{}` must not contain `..`",
            path.display()
        )));
    }

    Ok(root.expect("root checked above").join(path))
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "storage-preflight")]
    use super::StoragePreflight;
    use super::StorageRoots;
    use std::path::PathBuf;

    #[test]
    fn resolves_relative_paths_per_direction() {
        let roots = StorageRoots::new(
            Some(PathBuf::from("/mnt/input")),
            Some(PathBuf::from("/mnt/output")),
        );
        assert_eq!(roots.resolve_input("scan.las").unwrap(), PathBuf::from("/mnt/input/scan.las"));
        assert_eq!(
            roots.resolve_output("runs/result.las").unwrap(),
            PathBuf::from("/mnt/output/runs/result.las")
        );
    }

    #[test]
    fn absolute_paths_bypass_roots() {
        let roots = StorageRoots::new(
            Some(PathBuf::from("/mnt/input")),
            Some(PathBuf::from("/mnt/output")),
        );
        assert_eq!(roots.resolve_input("/tmp/scan.las").unwrap(), PathBuf::from("/tmp/scan.las"));
        assert_eq!(
            roots.resolve_output("/tmp/result.las").unwrap(),
            PathBuf::from("/tmp/result.las")
        );
    }

    #[test]
    fn rejects_relative_root_escape() {
        let roots = StorageRoots::new(Some(PathBuf::from("/mnt/input")), None);
        let error = roots.resolve_input("../private/scan.las").unwrap_err();
        assert!(error.to_string().contains("must not contain `..`"));
    }

    #[cfg(feature = "storage-preflight")]
    #[test]
    fn preflight_reports_available_space_for_absolute_directory() {
        let directory = tempfile::tempdir().unwrap();
        let roots = StorageRoots::new(None, Some(directory.path().to_path_buf()));
        let report = roots.preflight_output(1).unwrap();
        assert_eq!(report.root, directory.path());
        assert!(report.available_bytes >= report.required_free_bytes);
    }

    #[cfg(feature = "storage-preflight")]
    #[test]
    fn preflight_rejects_missing_output_root_and_relative_path() {
        let roots = StorageRoots::default();
        assert!(roots
            .preflight_output(1)
            .unwrap_err()
            .to_string()
            .contains("requires an output root"));
        let error = StoragePreflight::check("relative", 1).unwrap_err();
        assert!(error.to_string().contains("must be absolute"));
    }
}
