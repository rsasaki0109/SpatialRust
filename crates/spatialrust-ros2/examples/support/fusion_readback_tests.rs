//! Synthetic canonical byte and bounded IO controls; no physical evidence.

use super::super::{readback::verify_reader, *};
use super::{fixture, source, visit_all};
use std::{
    cell::Cell,
    io::{self, Cursor, Read},
    rc::Rc,
};

fn bytes(completed: &CompletedCoverage) -> Vec<u8> {
    let mut value = serde_json::to_vec(&completed.0).unwrap();
    value.push(b'\n');
    value
}

fn changed(mut edit: impl FnMut(&mut Coverage)) -> (CompletedCoverage, Vec<u8>) {
    let (_, _, _, completed) = fixture();
    let mut decoded: Coverage = serde_json::from_slice(&bytes(&completed)).unwrap();
    edit(&mut decoded);
    let mut encoded = serde_json::to_vec(&decoded).unwrap();
    encoded.push(b'\n');
    (completed, encoded)
}

struct Observed<R> {
    inner: R,
    read_bytes: Rc<Cell<usize>>,
    max_request: Rc<Cell<usize>>,
    short: usize,
    interrupts: usize,
    fail_at: Option<usize>,
}

impl<R: Read> Read for Observed<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.max_request.set(self.max_request.get().max(buffer.len()));
        if self.interrupts != 0 {
            self.interrupts -= 1;
            return Err(io::ErrorKind::Interrupted.into());
        }
        if self.fail_at.is_some_and(|limit| self.read_bytes.get() >= limit) {
            return Err(io::Error::other("injected source failure"));
        }
        let size = buffer.len().min(self.short);
        let n = self.inner.read(&mut buffer[..size])?;
        self.read_bytes.set(self.read_bytes.get() + n);
        Ok(n)
    }
}

fn observed<R: Read>(inner: R) -> Observed<R> {
    Observed {
        inner,
        read_bytes: Rc::new(Cell::new(0)),
        max_request: Rc::new(Cell::new(0)),
        short: usize::MAX,
        interrupts: 0,
        fail_at: None,
    }
}

#[test]
fn canonical_publication_returns_the_existing_file_checksum() {
    let (_, _, _, completed) = fixture();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("coverage.json");
    completed.publish(&path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), bytes(&completed));
    assert_eq!(
        completed.verified_receipt(&path).unwrap(),
        FileReceipt::from_path(ReceiptRole::Output, &path).unwrap()
    );
}

#[test]
fn missing_artifact_is_not_recreated() {
    let (_, _, _, completed) = fixture();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("missing.json");
    assert!(completed.verified_receipt(&path).is_err());
    assert!(!path.exists());
}

#[test]
fn directory_is_not_an_artifact() {
    let (_, _, _, completed) = fixture();
    let directory = tempfile::tempdir().unwrap();
    assert!(completed.verified_receipt(directory.path()).is_err());
}

#[test]
fn truncation_is_rejected_at_header_record_and_final_byte() {
    let (_, _, _, completed) = fixture();
    let encoded = bytes(&completed);
    for length in [0, 1, encoded.len() / 2, encoded.len() - 2] {
        assert!(verify_reader(&completed.0, Cursor::new(&encoded[..length])).is_err(), "{length}");
    }
}

#[test]
fn final_newline_is_required() {
    let (_, _, _, completed) = fixture();
    let mut encoded = bytes(&completed);
    encoded.pop();
    assert!(verify_reader(&completed.0, Cursor::new(encoded)).is_err());
}

#[test]
fn appended_whitespace_is_rejected() {
    let (_, _, _, completed) = fixture();
    let mut encoded = bytes(&completed);
    encoded.extend_from_slice(b" \n\t");
    assert!(verify_reader(&completed.0, Cursor::new(encoded)).is_err());
}

#[test]
fn appended_json_document_is_rejected() {
    let (_, _, _, completed) = fixture();
    let mut encoded = bytes(&completed);
    encoded.extend_from_slice(b"{}\n");
    assert!(verify_reader(&completed.0, Cursor::new(encoded)).is_err());
}

#[test]
fn equivalent_reformatting_is_outside_the_producer_contract() {
    let (_, _, _, completed) = fixture();
    let mut formatted = serde_json::to_vec_pretty(&completed.0).unwrap();
    formatted.push(b'\n');
    assert!(serde_json::from_slice::<Coverage>(&formatted).is_ok());
    assert!(verify_reader(&completed.0, Cursor::new(formatted)).is_err());
    let mut leading = vec![b' '];
    leading.extend_from_slice(&bytes(&completed));
    assert!(verify_reader(&completed.0, Cursor::new(leading)).is_err());
}

#[test]
fn source_and_calibration_identity_edits_are_rejected() {
    for mode in 0..4 {
        let (completed, encoded) = changed(|report| match mode {
            0 => report.context.identity.source_sha256 = "c".repeat(64),
            1 => report.context.identity.calibration_sha256 = "d".repeat(64),
            2 => report.context.identity.source_bytes += 1,
            _ => report.context.identity.calibration_bytes += 1,
        });
        assert!(verify_reader(&completed.0, Cursor::new(encoded)).is_err());
    }
}

#[test]
fn registered_frame_and_clock_edits_are_rejected() {
    for mode in 0..9 {
        let (completed, encoded) = changed(|report| match mode {
            0 => report.context.front_frame = "other".into(),
            1 => report.context.rear_frame = "other".into(),
            2 => report.context.root_frame = "other".into(),
            3 => report.context.clock_id = "other".into(),
            4 => report.context.offset_ns += 1.0,
            5 => report.context.drift_ppm += 1.0,
            6 => report.context.uncertainty_ns += 1,
            7 => report.context.anchor_nanos += 1,
            _ => report.context.max_delta_ns += 1,
        });
        assert!(verify_reader(&completed.0, Cursor::new(encoded)).is_err());
    }
}

#[test]
fn record_identity_points_and_visits_are_compared() {
    for mode in 0..6 {
        let (completed, encoded) = changed(|report| match mode {
            0 => report.records[0].episode_index += 1,
            1 => report.records[0].raw_nanos += 1,
            2 => report.records[0].corrected_nanos += 1,
            3 => report.records[0].source_points += 1,
            4 => report.records[0].visits += 1,
            _ => report.records[0].point_visits += 1,
        });
        assert!(verify_reader(&completed.0, Cursor::new(encoded)).is_err());
    }
}

#[test]
fn association_and_summary_edits_are_rejected() {
    for mode in 0..7 {
        let (completed, encoded) = changed(|report| match mode {
            0 => report.associations[0].front_index += 1,
            1 => report.associations[0].rear_index += 1,
            2 => report.associations[0].delta_ns += 1,
            3 => report.associations[0].visited_points = None,
            4 => report.summary.as_mut().unwrap().total_visits += 1,
            5 => report.summary.as_mut().unwrap().unique_visited_records += 1,
            _ => report.summary.as_mut().unwrap().total_point_visits += 1,
        });
        assert!(verify_reader(&completed.0, Cursor::new(encoded)).is_err());
    }
}

#[test]
fn schema_and_unknown_fields_cannot_be_added() {
    let (completed, encoded) = changed(|report| report.schema_version += 1);
    assert!(verify_reader(&completed.0, Cursor::new(encoded)).is_err());
    let mut encoded = bytes(&completed);
    encoded.splice(1..1, b"\"untrusted\":[],".iter().copied());
    assert!(verify_reader(&completed.0, Cursor::new(encoded)).is_err());
}

#[test]
fn a_different_expected_ledger_cannot_self_certify_saved_bytes() {
    let (mut completed, encoded) = changed(|report| report.context.identity.source_bytes += 1);
    assert!(verify_reader(&completed.0, Cursor::new(&encoded)).is_err());
    completed.0.context.identity.source_bytes += 1;
    assert_eq!(verify_reader(&completed.0, Cursor::new(&encoded)).unwrap(), encoded.len() as u64);
}

#[test]
fn failed_verification_preserves_artifact_and_preexisting_temporary_file() {
    let (_, _, _, completed) = fixture();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("coverage.json");
    let temporary = temp_path(&path);
    fs::write(&temporary, b"another attempt").unwrap();
    fs::write(&path, b"damaged").unwrap();
    assert!(completed.verified_receipt(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"damaged");
    assert_eq!(fs::read(&temporary).unwrap(), b"another attempt");
}

#[test]
fn invalid_limits_are_rejected_before_any_input_read() {
    for value in [0, MAX_JSON_BYTES + 1] {
        let (_, _, _, mut completed) = fixture();
        completed.0.limits.json_bytes = value;
        let input = observed(io::repeat(0));
        let count = input.read_bytes.clone();
        assert!(verify_reader(&completed.0, input).is_err());
        assert_eq!(count.get(), 0);
    }
}

#[test]
fn short_reads_preserve_exact_matching() {
    let (_, _, _, completed) = fixture();
    let encoded = bytes(&completed);
    for short in [1, 3, 17] {
        let mut input = observed(Cursor::new(&encoded));
        input.short = short;
        assert_eq!(verify_reader(&completed.0, input).unwrap(), encoded.len() as u64);
    }
}

#[test]
fn interrupted_reads_are_retried() {
    let (_, _, _, completed) = fixture();
    let encoded = bytes(&completed);
    let mut input = observed(Cursor::new(&encoded));
    input.interrupts = 3;
    input.short = 7;
    assert_eq!(verify_reader(&completed.0, input).unwrap(), encoded.len() as u64);
}

#[test]
fn source_io_failure_is_not_accepted_as_truncation_or_eof() {
    let (_, _, _, completed) = fixture();
    let encoded = bytes(&completed);
    let mut input = observed(Cursor::new(&encoded));
    input.short = 1;
    input.fail_at = Some(20);
    let error = verify_reader(&completed.0, input).unwrap_err();
    assert!(error.to_string().contains("injected source failure"), "{error}");
}

fn exact_limit(completed: &mut CompletedCoverage) -> Vec<u8> {
    let mut limit = 1;
    loop {
        completed.0.limits.json_bytes = limit;
        let encoded = bytes(completed);
        if encoded.len() as u64 == limit {
            return encoded;
        }
        limit = encoded.len() as u64;
    }
}

#[test]
fn exact_limit_includes_newline_and_one_bounded_trailing_probe() {
    let (_, _, _, mut completed) = fixture();
    let encoded = exact_limit(&mut completed);
    assert_eq!(verify_reader(&completed.0, Cursor::new(&encoded)).unwrap(), encoded.len() as u64);
    let input = observed(Cursor::new(&encoded).chain(io::repeat(b' ')));
    let count = input.read_bytes.clone();
    assert!(verify_reader(&completed.0, input).is_err());
    assert_eq!(count.get(), encoded.len() + 1);
}

#[test]
fn untrusted_long_string_does_not_allocate_from_json_fields() {
    let (_, _, _, completed) = fixture();
    let input = observed(Cursor::new(b"{\"oversized\":\"").chain(io::repeat(b'x')));
    let count = input.read_bytes.clone();
    let request = input.max_request.clone();
    assert!(verify_reader(&completed.0, input).is_err());
    assert!(count.get() <= WRITE_BUFFER_BYTES * 2);
    assert!(request.get() <= WRITE_BUFFER_BYTES);
}

#[test]
fn expected_output_over_the_limit_cannot_read_unbounded_input() {
    let (_, _, _, mut completed) = fixture();
    completed.0.limits.json_bytes = 8;
    let input = observed(io::repeat(b'x'));
    let count = input.read_bytes.clone();
    assert!(verify_reader(&completed.0, input).is_err());
    assert!(count.get() <= 9);
}

#[test]
fn error_on_the_eof_probe_is_propagated() {
    let (_, _, _, completed) = fixture();
    let encoded = bytes(&completed);
    let mut input = observed(Cursor::new(&encoded));
    input.short = 1;
    input.fail_at = Some(encoded.len());
    let error = verify_reader(&completed.0, input).unwrap_err();
    assert!(error.to_string().contains("injected source failure"), "{error}");
}

#[test]
fn oversized_artifact_is_rejected_without_repair() {
    let (_, _, _, mut completed) = fixture();
    completed.0.limits.json_bytes = 8;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("coverage.json");
    fs::write(&path, b"123456789").unwrap();
    assert!(completed.verified_receipt(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"123456789");
}

#[test]
fn complete_report_crosses_multiple_fixed_buffers() {
    let items: Vec<_> =
        (1..=256).map(|stamp| ("/front", stamp)).chain(std::iter::once(("/rear", 128))).collect();
    let episode = super::super::super::coverage_characterization::episode(&items);
    let mut context = super::context();
    context.max_delta_ns = 1000;
    let mut coverage = Coverage::prepare(&episode, &context).unwrap();
    visit_all(&mut coverage);
    let completed = coverage.finish(&episode, &context, &source(&episode), 512, 2048).unwrap();
    let encoded = bytes(&completed);
    assert!(encoded.len() > WRITE_BUFFER_BYTES * 2);
    let input = observed(Cursor::new(&encoded));
    let request = input.max_request.clone();
    assert_eq!(verify_reader(&completed.0, input).unwrap(), encoded.len() as u64);
    assert!(request.get() <= WRITE_BUFFER_BYTES);
    let mut damaged = encoded;
    damaged[WRITE_BUFFER_BYTES + 1] ^= 1;
    assert!(verify_reader(&completed.0, Cursor::new(damaged)).is_err());
}
