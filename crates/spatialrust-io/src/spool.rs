//! Explicit, size-limited temporary disk spools for streaming formats.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::IoError;

static NEXT_SPOOL_ID: AtomicU64 = AtomicU64::new(0);

/// Configuration for a recoverably temporary, size-limited spool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpoolOptions {
    directory: PathBuf,
    limit_bytes: u64,
}

impl SpoolOptions {
    /// Creates a positive spool limit in an existing directory.
    pub fn new(directory: impl Into<PathBuf>, limit_bytes: u64) -> Result<Self, IoError> {
        let directory = directory.into();
        if limit_bytes == 0 {
            return Err(IoError::Streaming("spool limit must be positive".into()));
        }
        if !directory.is_dir() {
            return Err(IoError::Streaming(format!(
                "spool directory does not exist: {}",
                directory.display()
            )));
        }
        Ok(Self { directory, limit_bytes })
    }

    /// Returns the temporary directory.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Returns the hard maximum file extent.
    #[must_use]
    pub const fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }
}

/// Seekable temporary file that rejects writes before its extent exceeds the
/// configured disk budget and removes its `.part` file unless committed.
pub struct BoundedSpool {
    file: Option<File>,
    path: PathBuf,
    limit_bytes: u64,
    extent_bytes: u64,
    committed: bool,
}

impl BoundedSpool {
    /// Creates a uniquely named `.part` file with exclusive creation.
    pub fn create(options: &SpoolOptions, stem: &str) -> Result<Self, IoError> {
        let safe_stem: String = stem
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                    character
                } else {
                    '_'
                }
            })
            .collect();
        for _ in 0..100 {
            let id = NEXT_SPOOL_ID.fetch_add(1, Ordering::Relaxed);
            let path =
                options.directory.join(format!(".{safe_stem}.{}.{}.part", std::process::id(), id));
            match OpenOptions::new().read(true).write(true).create_new(true).open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        file: Some(file),
                        path,
                        limit_bytes: options.limit_bytes,
                        extent_bytes: 0,
                        committed: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(IoError::Streaming("could not allocate a unique temporary spool".into()))
    }

    /// Returns the temporary path, suitable for format libraries requiring a path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the largest written file extent.
    #[must_use]
    pub const fn extent_bytes(&self) -> u64 {
        self.extent_bytes
    }

    /// Flushes, closes, and atomically renames the spool to a new destination.
    ///
    /// Existing destinations are rejected rather than overwritten.
    pub fn commit(mut self, destination: impl AsRef<Path>) -> Result<PathBuf, IoError> {
        let destination = destination.as_ref();
        if destination.exists() {
            return Err(IoError::Streaming(format!(
                "spool destination already exists: {}",
                destination.display()
            )));
        }
        if let Some(mut file) = self.file.take() {
            file.flush()?;
            file.sync_all()?;
        }
        std::fs::rename(&self.path, destination)?;
        self.committed = true;
        Ok(destination.to_path_buf())
    }

    fn file_mut(&mut self) -> std::io::Result<&mut File> {
        self.file.as_mut().ok_or_else(|| std::io::Error::other("spool is closed"))
    }
}

impl Write for BoundedSpool {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let position = self.file_mut()?.stream_position()?;
        let requested_end = position
            .checked_add(buffer.len() as u64)
            .ok_or_else(|| std::io::Error::other("spool extent overflow"))?;
        if requested_end > self.limit_bytes {
            return Err(std::io::Error::other(format!(
                "spool limit exceeded: requested extent {requested_end}, limit {}",
                self.limit_bytes
            )));
        }
        let written = self.file_mut()?.write(buffer)?;
        self.extent_bytes = self.extent_bytes.max(position + written as u64);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file_mut()?.flush()
    }
}

impl Read for BoundedSpool {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.file_mut()?.read(buffer)
    }
}

impl Seek for BoundedSpool {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.file_mut()?.seek(position)
    }
}

impl Drop for BoundedSpool {
    fn drop(&mut self) {
        self.file.take();
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BoundedSpool, SpoolOptions};
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn rejects_growth_before_limit_is_crossed_and_cleans_up() {
        let options = SpoolOptions::new(std::env::temp_dir(), 4).unwrap();
        let path;
        {
            let mut spool = BoundedSpool::create(&options, "bounded-test").unwrap();
            path = spool.path().to_path_buf();
            spool.write_all(b"1234").unwrap();
            assert!(spool.write_all(b"5").is_err());
            spool.seek(SeekFrom::Start(1)).unwrap();
            spool.write_all(b"x").unwrap();
            assert_eq!(spool.extent_bytes(), 4);
        }
        assert!(!path.exists());
    }
}
