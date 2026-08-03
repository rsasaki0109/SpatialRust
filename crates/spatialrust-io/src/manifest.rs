//! Checksummed file receipts and dataset manifests.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::IoError;

/// Current JSON schema version for [`DatasetManifest`].
pub const DATASET_MANIFEST_VERSION: u32 = 1;

/// Logical role of a file in a dataset operation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptRole {
    /// File consumed by an operation.
    Input,
    /// File produced by an operation.
    Output,
    /// File associated with an operation but not consumed or produced by it.
    Auxiliary,
}

/// Size and SHA-256 receipt for one local file or URI source.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FileReceipt {
    /// Role of the source in the operation.
    pub role: ReceiptRole,
    /// Resolved local path or URI written as a path-like JSON string.
    pub path: PathBuf,
    /// Number of bytes observed while hashing a local file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Lowercase hexadecimal SHA-256 digest for a local file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

impl FileReceipt {
    /// Hashes a local file and returns its size/checksum receipt.
    pub fn from_path(role: ReceiptRole, path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut bytes = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];

        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            bytes = bytes.checked_add(read as u64).ok_or_else(|| {
                IoError::Manifest(format!("file size overflow while hashing `{}`", path.display()))
            })?;
        }

        let digest = hasher.finalize();
        Ok(Self {
            role,
            path: path.to_path_buf(),
            size_bytes: Some(bytes),
            sha256: Some(hex_digest(&digest)),
        })
    }

    /// Records a URI source whose bytes were not materialized locally.
    #[must_use]
    pub fn from_uri(role: ReceiptRole, uri: impl Into<PathBuf>) -> Self {
        Self { role, path: uri.into(), size_bytes: None, sha256: None }
    }
}

/// JSON manifest containing the files associated with one dataset operation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DatasetManifest {
    /// Version of the manifest JSON schema.
    pub version: u32,
    /// File and URI receipts in operation order.
    pub entries: Vec<FileReceipt>,
}

/// Summary returned after re-hashing every local manifest entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManifestValidation {
    /// Number of local files re-hashed successfully.
    pub checked_local_files: u64,
    /// Number of URI entries that intentionally have no local checksum.
    pub uri_entries: u64,
    /// Total bytes observed across checked local files.
    pub total_bytes: u64,
}

impl Default for DatasetManifest {
    fn default() -> Self {
        Self::new()
    }
}

impl DatasetManifest {
    /// Creates an empty manifest at the current schema version.
    #[must_use]
    pub const fn new() -> Self {
        Self { version: DATASET_MANIFEST_VERSION, entries: Vec::new() }
    }

    /// Adds a checksummed local file receipt.
    pub fn add_file(&mut self, role: ReceiptRole, path: impl AsRef<Path>) -> Result<(), IoError> {
        self.entries.push(FileReceipt::from_path(role, path)?);
        Ok(())
    }

    /// Adds a URI receipt without claiming a local byte count or checksum.
    pub fn add_uri(&mut self, role: ReceiptRole, uri: impl Into<PathBuf>) {
        self.entries.push(FileReceipt::from_uri(role, uri));
    }

    /// Reads a JSON manifest from disk without trusting its file receipts.
    pub fn read_json(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|error| {
            IoError::Manifest(format!("cannot read dataset manifest `{}`: {error}", path.display()))
        })?;
        serde_json::from_str(&text).map_err(|error| {
            IoError::Manifest(format!(
                "cannot parse dataset manifest `{}`: {error}",
                path.display()
            ))
        })
    }

    /// Re-hashes local entries and rejects missing, changed, or partial receipts.
    pub fn validate_local_files(&self) -> Result<ManifestValidation, IoError> {
        if self.version != DATASET_MANIFEST_VERSION {
            return Err(IoError::Manifest(format!(
                "unsupported dataset manifest version {}; expected {}",
                self.version, DATASET_MANIFEST_VERSION
            )));
        }
        let mut checked_local_files = 0_u64;
        let mut uri_entries = 0_u64;
        let mut total_bytes = 0_u64;
        for entry in &self.entries {
            match (&entry.size_bytes, &entry.sha256) {
                (Some(expected_size), Some(expected_sha256)) => {
                    let actual = FileReceipt::from_path(entry.role, &entry.path)?;
                    if actual.size_bytes != entry.size_bytes {
                        return Err(IoError::Manifest(format!(
                            "size mismatch for `{}`: expected {:?}, observed {:?}",
                            entry.path.display(),
                            entry.size_bytes,
                            actual.size_bytes
                        )));
                    }
                    if actual.sha256.as_deref() != Some(expected_sha256.as_str()) {
                        return Err(IoError::Manifest(format!(
                            "checksum mismatch for `{}`: expected {}, observed {}",
                            entry.path.display(),
                            expected_sha256,
                            actual.sha256.as_deref().unwrap_or("<missing>")
                        )));
                    }
                    checked_local_files = checked_local_files.checked_add(1).ok_or_else(|| {
                        IoError::Manifest(
                            "local file count overflow while validating manifest".into(),
                        )
                    })?;
                    total_bytes = total_bytes.checked_add(*expected_size).ok_or_else(|| {
                        IoError::Manifest(
                            "total byte count overflow while validating manifest".into(),
                        )
                    })?;
                }
                (None, None) if is_uri(&entry.path) => {
                    uri_entries = uri_entries.checked_add(1).ok_or_else(|| {
                        IoError::Manifest("URI count overflow while validating manifest".into())
                    })?;
                }
                (None, None) => {
                    return Err(IoError::Manifest(format!(
                        "manifest entry `{}` has no checksum and is not a URI",
                        entry.path.display()
                    )));
                }
                _ => {
                    return Err(IoError::Manifest(format!(
                        "manifest entry `{}` has a partial size/checksum receipt",
                        entry.path.display()
                    )));
                }
            }
        }
        Ok(ManifestValidation { checked_local_files, uri_entries, total_bytes })
    }

    /// Serializes this manifest as pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, IoError> {
        serde_json::to_string_pretty(self).map_err(|error| {
            IoError::Manifest(format!("cannot serialize dataset manifest: {error}"))
        })
    }

    /// Writes this manifest as JSON, creating its parent directory if needed.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, format!("{}\n", self.to_json()?))?;
        Ok(())
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn is_uri(path: &Path) -> bool {
    path.to_string_lossy().contains("://")
}

#[cfg(test)]
mod tests {
    use super::{DatasetManifest, FileReceipt, ReceiptRole, DATASET_MANIFEST_VERSION};

    #[test]
    fn hashes_file_and_records_size() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("scan.bin");
        std::fs::write(&path, b"hello").unwrap();

        let receipt = FileReceipt::from_path(ReceiptRole::Input, &path).unwrap();
        assert_eq!(receipt.size_bytes, Some(5));
        assert_eq!(
            receipt.sha256.as_deref(),
            Some("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
        );
    }

    #[test]
    fn serializes_local_and_uri_entries() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("scan.bin");
        std::fs::write(&path, b"data").unwrap();

        let mut manifest = DatasetManifest::new();
        manifest.add_file(ReceiptRole::Input, &path).unwrap();
        manifest.add_uri(ReceiptRole::Auxiliary, "https://example.test/scan.copc.laz");
        let json = manifest.to_json().unwrap();
        let decoded: DatasetManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.version, DATASET_MANIFEST_VERSION);
        assert_eq!(decoded.entries.len(), 2);
        assert_eq!(decoded.entries[0].role, ReceiptRole::Input);
        assert_eq!(decoded.entries[0].size_bytes, Some(4));
        assert_eq!(decoded.entries[1].sha256, None);
        let validation = decoded.validate_local_files().unwrap();
        assert_eq!(validation.checked_local_files, 1);
        assert_eq!(validation.uri_entries, 1);
        assert_eq!(validation.total_bytes, 4);

        let manifest_path = directory.path().join("manifest.json");
        decoded.write_json(&manifest_path).unwrap();
        assert_eq!(DatasetManifest::read_json(&manifest_path).unwrap(), decoded);
    }

    #[test]
    fn rejects_changed_local_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("scan.bin");
        std::fs::write(&path, b"before").unwrap();
        let mut manifest = DatasetManifest::new();
        manifest.add_file(ReceiptRole::Input, &path).unwrap();
        std::fs::write(&path, b"after").unwrap();

        let error = manifest.validate_local_files().unwrap_err();
        assert!(error.to_string().contains("mismatch"));
    }

    #[test]
    fn rejects_partial_or_unchecked_local_receipts() {
        let path = std::path::PathBuf::from("/tmp/scan.bin");
        let partial = DatasetManifest {
            version: DATASET_MANIFEST_VERSION,
            entries: vec![FileReceipt {
                role: ReceiptRole::Input,
                path: path.clone(),
                size_bytes: Some(1),
                sha256: None,
            }],
        };
        assert!(partial.validate_local_files().unwrap_err().to_string().contains("partial"));

        let unchecked = DatasetManifest {
            version: DATASET_MANIFEST_VERSION,
            entries: vec![FileReceipt {
                role: ReceiptRole::Input,
                path,
                size_bytes: None,
                sha256: None,
            }],
        };
        assert!(unchecked.validate_local_files().unwrap_err().to_string().contains("not a URI"));
    }
}
