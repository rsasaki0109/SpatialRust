//! Exact producer readback, without allocating from untrusted JSON fields.

use super::{metadata_bytes, Coverage, LimitWriter, Result, WRITE_BUFFER_BYTES};
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    path::Path,
};

pub(super) fn verify_file(expected: &Coverage, path: &Path) -> Result<u64> {
    expected.limits.validate()?;
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > expected.limits.json_bytes {
        return Err("coverage readback requires a regular file within the byte limit".into());
    }
    verify_reader(expected, file)
}

pub(super) fn verify_reader(expected: &Coverage, source: impl Read) -> Result<u64> {
    expected.limits.validate()?;
    if expected.summary.is_none()
        || metadata_bytes(
            expected.records.capacity(),
            expected.associations.capacity(),
            expected.records.len(),
        )? > expected.limits.metadata_bytes
    {
        return Err("coverage readback requires a completed bounded ledger".into());
    }
    // Take precedes buffering: even read-ahead cannot exceed the JSON limit
    // plus one byte needed to distinguish exact EOF from appended content.
    let read_limit =
        expected.limits.json_bytes.checked_add(1).ok_or("coverage read limit overflow")?;
    let reader = BufReader::with_capacity(WRITE_BUFFER_BYTES, source.take(read_limit));
    if reader.capacity() > WRITE_BUFFER_BYTES {
        return Err("coverage read buffer exceeds fixed capacity".into());
    }
    let comparator = CompareWriter { reader, scratch: [0; WRITE_BUFFER_BYTES], failed: false };
    let sink = LimitWriter { inner: comparator, bytes: 0, limit: expected.limits.json_bytes };
    let mut writer = BufWriter::with_capacity(WRITE_BUFFER_BYTES, sink);
    if writer.capacity() > WRITE_BUFFER_BYTES {
        return Err("coverage comparison buffer exceeds fixed capacity".into());
    }
    serde_json::to_writer(&mut writer, expected)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_mut().inner.finish()?;
    Ok(writer.get_ref().bytes)
}

struct CompareWriter<R> {
    reader: R,
    scratch: [u8; WRITE_BUFFER_BYTES],
    failed: bool,
}

impl<R: Read> CompareWriter<R> {
    fn finish(&mut self) -> io::Result<()> {
        if self.failed {
            return Err(io::Error::other("coverage comparator already failed"));
        }
        loop {
            match self.reader.read(&mut self.scratch[..1]) {
                Ok(0) => return Ok(()),
                Ok(_) => {
                    self.failed = true;
                    return Err(io::Error::other("coverage readback has trailing bytes"));
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    self.failed = true;
                    return Err(error);
                }
            }
        }
    }
}

impl<R: Read> Write for CompareWriter<R> {
    fn write(&mut self, expected: &[u8]) -> io::Result<usize> {
        if self.failed {
            return Err(io::Error::other("coverage comparator already failed"));
        }
        for chunk in expected.chunks(WRITE_BUFFER_BYTES) {
            if let Err(error) = self.reader.read_exact(&mut self.scratch[..chunk.len()]) {
                self.failed = true;
                return Err(error);
            }
            if &self.scratch[..chunk.len()] != chunk {
                self.failed = true;
                return Err(io::Error::other("coverage readback differs from completed ledger"));
            }
        }
        Ok(expected.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, io::Cursor, rc::Rc};

    #[test]
    fn mismatch_poison_prevents_further_reads() {
        let mut writer = CompareWriter {
            reader: Cursor::new(b"bad trailing bytes"),
            scratch: [0; WRITE_BUFFER_BYTES],
            failed: false,
        };
        assert!(writer.write_all(b"good").is_err());
        let position = writer.reader.position();
        assert!(writer.write_all(b"more").is_err());
        assert!(writer.finish().is_err());
        assert_eq!(writer.reader.position(), position);
    }

    #[test]
    fn truncation_poison_prevents_further_reads() {
        let mut writer = CompareWriter {
            reader: Cursor::new(b"x"),
            scratch: [0; WRITE_BUFFER_BYTES],
            failed: false,
        };
        assert!(writer.write_all(b"long").is_err());
        let position = writer.reader.position();
        assert!(writer.write_all(b"x").is_err());
        assert!(writer.finish().is_err());
        assert_eq!(writer.reader.position(), position);
    }

    struct Broken(Rc<Cell<usize>>);
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            self.0.set(self.0.get() + 1);
            Err(io::Error::other("injected comparator failure"))
        }
    }

    #[test]
    fn buffered_drop_does_not_retry_a_failed_source() {
        let calls = Rc::new(Cell::new(0));
        let comparator = CompareWriter {
            reader: Broken(calls.clone()),
            scratch: [0; WRITE_BUFFER_BYTES],
            failed: false,
        };
        let mut writer = BufWriter::with_capacity(4, comparator);
        writer.write_all(b"abc").unwrap();
        assert!(writer.flush().is_err());
        assert_eq!(calls.get(), 1);
        drop(writer);
        assert_eq!(calls.get(), 1);
    }
}
