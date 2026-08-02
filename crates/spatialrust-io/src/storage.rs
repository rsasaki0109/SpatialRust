//! Explicit input/output roots for local spatial data.

use std::path::{Component, Path, PathBuf};

use crate::IoError;

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
}
