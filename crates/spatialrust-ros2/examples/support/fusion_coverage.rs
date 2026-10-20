//! Bounded metadata for successful TSDF submissions, not voxel coverage.

use super::{ClockCorrection, Config};
use serde::{Deserialize, Serialize};
use spatialrust_io::{FileReceipt, ReceiptRole};
use spatialrust_sync::{
    ClockDomain, DeterministicReplayer, MemoryEpisode, StampedRecord, SyncWindow, TopicId,
};
use spatialrust_viewer::MappingSourceSummary;
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{self, BufWriter, Write},
    mem::size_of,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MAX_RECORDS: usize = 65_536;
const MAX_METADATA_BYTES: usize = 8 * 1024 * 1024;
const MAX_JSON_BYTES: u64 = 32 * 1024 * 1024;
const MAX_LABEL_BYTES: usize = 1024;
const MAX_HEADER_BYTES: usize = 8192;
const WRITE_BUFFER_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct InputIdentity {
    source_sha256: String,
    source_bytes: u64,
    calibration_sha256: String,
    calibration_bytes: u64,
}

impl InputIdentity {
    pub(super) fn from_receipts(source: &FileReceipt, calibration: &FileReceipt) -> Result<Self> {
        if source.role != ReceiptRole::Input || calibration.role != ReceiptRole::Auxiliary {
            return Err("coverage requires verified input and calibration receipts".into());
        }
        let sha = |receipt: &FileReceipt| -> Result<String> {
            let value = receipt.sha256.as_deref().ok_or("coverage receipt has no SHA-256")?;
            super::validate_sha256(value)?;
            Ok(value.to_owned())
        };
        let identity = Self {
            source_sha256: sha(source)?,
            source_bytes: source.size_bytes.ok_or("coverage receipt has no size")?,
            calibration_sha256: sha(calibration)?,
            calibration_bytes: calibration.size_bytes.ok_or("coverage receipt has no size")?,
        };
        identity.validate()?;
        Ok(identity)
    }

    fn validate(&self) -> Result<()> {
        super::validate_sha256(&self.source_sha256)?;
        super::validate_sha256(&self.calibration_sha256)?;
        if self.source_bytes == 0 || self.calibration_bytes == 0 {
            return Err("coverage receipt size must be nonzero".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Context {
    identity: InputIdentity,
    front_topic: String,
    rear_topic: String,
    front_frame: String,
    rear_frame: String,
    root_frame: String,
    clock_id: String,
    offset_ns: f64,
    drift_ppm: f64,
    uncertainty_ns: u64,
    anchor_nanos: u64,
    chunk_points: u64,
    max_delta_ns: u64,
}

impl Context {
    pub(super) fn new(
        config: &Config,
        identity: InputIdentity,
        clock: &ClockCorrection,
        front_frame: &str,
        rear_frame: &str,
    ) -> Result<Self> {
        for value in [
            &config.front_topic,
            &config.rear_topic,
            front_frame,
            rear_frame,
            &config.root_frame,
            &clock.clock_id,
        ] {
            check_label(value)?;
        }
        let value = Self {
            identity,
            front_topic: config.front_topic.clone(),
            rear_topic: config.rear_topic.clone(),
            front_frame: front_frame.into(),
            rear_frame: rear_frame.into(),
            root_frame: config.root_frame.clone(),
            clock_id: clock.clock_id.clone(),
            offset_ns: clock.offset_ns,
            drift_ppm: clock.drift_ppm,
            uncertainty_ns: clock.uncertainty_ns,
            anchor_nanos: clock.anchor_nanos.ok_or("coverage clock has no collection anchor")?,
            chunk_points: u64::try_from(config.chunk_points)?,
            max_delta_ns: config.max_delta_ns,
        };
        value.validate()?;
        Ok(value)
    }

    fn strings(&self) -> [&String; 8] {
        [
            &self.identity.source_sha256,
            &self.identity.calibration_sha256,
            &self.front_topic,
            &self.rear_topic,
            &self.front_frame,
            &self.rear_frame,
            &self.root_frame,
            &self.clock_id,
        ]
    }

    fn validate(&self) -> Result<()> {
        self.identity.validate()?;
        let mut capacity = 0_usize;
        for value in self.strings() {
            check_label(value)?;
            capacity = capacity.checked_add(value.capacity()).ok_or("coverage header overflow")?;
        }
        if capacity > MAX_HEADER_BYTES
            || self.front_topic == self.rear_topic
            || self.front_frame == self.rear_frame
            || !self.offset_ns.is_finite()
            || !self.drift_ppm.is_finite()
            || self.chunk_points == 0
        {
            return Err("invalid or oversized coverage context".into());
        }
        self.clock().offset_i64()?;
        Ok(())
    }

    fn clock(&self) -> ClockCorrection {
        ClockCorrection {
            clock_id: self.clock_id.clone(),
            uncertainty_ns: self.uncertainty_ns,
            offset_ns: self.offset_ns,
            drift_ppm: self.drift_ppm,
            anchor_nanos: Some(self.anchor_nanos),
        }
    }

    fn sensor(&self, record: &StampedRecord) -> Result<Sensor> {
        match record.topic.as_str() {
            topic if topic == self.front_topic => Ok(Sensor::Front),
            topic if topic == self.rear_topic => Ok(Sensor::Rear),
            _ => Err("coverage episode has an unselected topic".into()),
        }
    }
}

fn check_label(value: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > MAX_LABEL_BYTES {
        return Err("coverage label length out of bounds".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Limits {
    records: usize,
    metadata_bytes: usize,
    json_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            records: MAX_RECORDS,
            metadata_bytes: MAX_METADATA_BYTES,
            json_bytes: MAX_JSON_BYTES,
        }
    }
}

impl Limits {
    fn validate(self) -> Result<()> {
        if self.records == 0
            || self.records > MAX_RECORDS
            || self.metadata_bytes == 0
            || self.metadata_bytes > MAX_METADATA_BYTES
            || self.json_bytes == 0
            || self.json_bytes > MAX_JSON_BYTES
        {
            return Err("coverage limits exceed fixed bounds".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Sensor {
    Front,
    Rear,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    episode_index: u64,
    sensor: Sensor,
    raw_nanos: u64,
    corrected_nanos: u64,
    source_points: u64,
    visits: u64,
    point_visits: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Association {
    pub(super) front_index: u64,
    pub(super) rear_index: u64,
    delta_ns: u64,
    visited_points: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Summary {
    front_records: u64,
    rear_records: u64,
    associations: u64,
    matched_front_records: u64,
    unmatched_front_records: u64,
    matched_rear_records: u64,
    unmatched_rear_records: u64,
    reused_rear_visits: u64,
    unique_visited_records: u64,
    unvisited_records: u64,
    total_visits: u64,
    total_point_visits: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Coverage {
    schema_version: u32,
    limits: Limits,
    context: Context,
    records: Vec<Record>,
    associations: Vec<Association>,
    summary: Option<Summary>,
}

#[derive(Debug)]
pub(super) struct CompletedCoverage(Coverage);

fn metadata_bytes(records: usize, associations: usize, scratch: usize) -> Result<usize> {
    // Account for actual vector capacities, a validation counter per record,
    // bounded context/clock headers, and the streaming writer buffer. This is
    // an explicit metadata bound, not an allocator/process RSS guarantee.
    records
        .checked_mul(size_of::<Record>())
        .and_then(|n| {
            associations.checked_mul(size_of::<Association>()).and_then(|a| n.checked_add(a))
        })
        .and_then(|n| scratch.checked_mul(size_of::<u64>()).and_then(|s| n.checked_add(s)))
        .and_then(|n| {
            n.checked_add(
                3 * (size_of::<Context>() + MAX_HEADER_BYTES)
                    + size_of::<Coverage>()
                    + WRITE_BUFFER_BYTES,
            )
        })
        .ok_or_else(|| "coverage metadata size overflow".into())
}

pub(super) fn check_record_bound(records: usize) -> Result<()> {
    if records > MAX_RECORDS {
        return Err("coverage retained-record limit exceeded before odometry".into());
    }
    Ok(())
}

pub(super) fn record_index(episode: &MemoryEpisode, record: &StampedRecord) -> Result<usize> {
    let key = (record.stamp.as_nanos(), record.topic.as_str());
    let index = episode
        .records()
        .binary_search_by_key(&key, |r| (r.stamp.as_nanos(), r.topic.as_str()))
        .map_err(|_| "coverage record is absent from episode")?;
    if !std::ptr::eq(&episode.records()[index], record) {
        return Err("coverage record is foreign or ambiguous".into());
    }
    Ok(index)
}

fn replay(
    episode: &MemoryEpisode,
    context: &Context,
    mut visit: impl FnMut(Association) -> Result<()>,
) -> Result<()> {
    let front = TopicId::new(context.front_topic.clone());
    let rear = TopicId::new(context.rear_topic.clone());
    let window = SyncWindow {
        max_delta_ns: context.max_delta_ns,
        max_uncertainty_ns: context.uncertainty_ns,
    };
    let mut replay = DeterministicReplayer::new(episode);
    while let Some(bundle) = replay.next_bundle(&front, std::slice::from_ref(&rear), window)? {
        let front = *bundle.get(&front).ok_or("coverage front bundle member missing")?;
        let rear = *bundle.get(&rear).ok_or("coverage rear bundle member missing")?;
        visit(Association {
            front_index: u64::try_from(record_index(episode, front)?)?,
            rear_index: u64::try_from(record_index(episode, rear)?)?,
            delta_ns: front.stamp.as_nanos().abs_diff(rear.stamp.as_nanos()),
            visited_points: None,
        })?;
    }
    Ok(())
}

impl Coverage {
    pub(super) fn prepare(episode: &MemoryEpisode, context: &Context) -> Result<Self> {
        Self::with_limits(episode, context, Limits::default())
    }

    fn with_limits(episode: &MemoryEpisode, context: &Context, limits: Limits) -> Result<Self> {
        limits.validate()?;
        context.validate()?;
        let n = episode.records().len();
        check_record_bound(n)?;
        let front_count =
            episode.records().iter().filter(|r| r.topic.as_str() == context.front_topic).count();
        if n > limits.records || metadata_bytes(n, front_count, n)? > limits.metadata_bytes {
            return Err("coverage metadata limit exceeded".into());
        }
        let mut records = Vec::new();
        records.try_reserve_exact(n)?;
        let mut associations = Vec::new();
        associations.try_reserve_exact(front_count)?;
        if metadata_bytes(records.capacity(), associations.capacity(), n)? > limits.metadata_bytes {
            return Err("coverage allocated capacity exceeds limit".into());
        }
        let mut clock = context.clock();
        let mut last = [None, None];
        for (index, record) in episode.records().iter().enumerate() {
            let descriptor = describe(index, record, context, &mut clock)?;
            let slot = match descriptor.sensor {
                Sensor::Front => 0,
                Sensor::Rear => 1,
            };
            if last[slot].is_some_and(|old| descriptor.corrected_nanos <= old) {
                return Err("coverage requires unique increasing topic times".into());
            }
            last[slot] = Some(descriptor.corrected_nanos);
            records.push(descriptor);
        }
        replay(episode, context, |association| {
            if associations.len() >= front_count {
                return Err("coverage association capacity exceeded".into());
            }
            associations.push(association);
            Ok(())
        })?;
        if associations.is_empty() {
            return Err("no front/rear records matched within the configured sync window".into());
        }
        Ok(Self {
            schema_version: 1,
            limits,
            context: context.clone(),
            records,
            associations,
            summary: None,
        })
    }

    pub(super) fn associations(&self) -> &[Association] {
        &self.associations
    }

    pub(super) fn note_front(&mut self, index: usize, points: usize) -> Result<()> {
        let record = self.records.get_mut(index).ok_or("coverage front index out of range")?;
        if record.sensor != Sensor::Front || record.visits != 0 {
            return Err("duplicate or invalid coverage front visit".into());
        }
        note(record, points)
    }

    pub(super) fn note_rear(&mut self, association: usize, points: usize) -> Result<()> {
        let association = self
            .associations
            .get_mut(association)
            .ok_or("coverage association index out of range")?;
        if association.visited_points.is_some() {
            return Err("duplicate coverage rear visit".into());
        }
        let record = self
            .records
            .get_mut(usize::try_from(association.rear_index)?)
            .ok_or("coverage rear index out of range")?;
        if record.sensor != Sensor::Rear {
            return Err("invalid coverage rear role".into());
        }
        note(record, points)?;
        association.visited_points = Some(u64::try_from(points)?);
        Ok(())
    }

    pub(super) fn finish(
        mut self,
        episode: &MemoryEpisode,
        expected: &Context,
        source: &MappingSourceSummary,
        records: u64,
        points: u64,
    ) -> Result<CompletedCoverage> {
        self.summary = Some(self.validated_summary(episode, expected, source, records, points)?);
        Ok(CompletedCoverage(self))
    }

    fn validated_summary(
        &self,
        episode: &MemoryEpisode,
        expected: &Context,
        source: &MappingSourceSummary,
        visits: u64,
        point_visits: u64,
    ) -> Result<Summary> {
        self.limits.validate()?;
        self.context.validate()?;
        expected.validate()?;
        source.validate()?;
        if self.schema_version != 1 || &self.context != expected {
            return Err("coverage schema/context mismatch".into());
        }
        let n = episode.records().len();
        check_record_bound(n)?;
        if self.records.len() != n
            || n > self.limits.records
            || self.associations.len() > n
            || metadata_bytes(self.records.capacity(), self.associations.capacity(), n)?
                > self.limits.metadata_bytes
        {
            return Err("coverage shape/capacity mismatch".into());
        }
        let mut expected_visits = Vec::<u64>::new();
        expected_visits.try_reserve_exact(n)?;
        if metadata_bytes(
            self.records.capacity(),
            self.associations.capacity(),
            expected_visits.capacity(),
        )? > self.limits.metadata_bytes
        {
            return Err("coverage validation capacity exceeded".into());
        }
        expected_visits.resize(n, 0);
        let mut front = 0_u64;
        let mut source_points = 0_u64;
        let mut clock = expected.clock();
        let mut last = [None, None];
        for (index, (record, original)) in self.records.iter().zip(episode.records()).enumerate() {
            let descriptor = describe(index, original, expected, &mut clock)?;
            if (
                record.episode_index,
                record.sensor,
                record.raw_nanos,
                record.corrected_nanos,
                record.source_points,
            ) != (
                descriptor.episode_index,
                descriptor.sensor,
                descriptor.raw_nanos,
                descriptor.corrected_nanos,
                descriptor.source_points,
            ) {
                return Err("coverage record identity mismatch".into());
            }
            let slot = match record.sensor {
                Sensor::Front => 0,
                Sensor::Rear => 1,
            };
            if last[slot].is_some_and(|old| record.corrected_nanos <= old) {
                return Err("coverage duplicate topic timestamp".into());
            }
            last[slot] = Some(record.corrected_nanos);
            source_points = source_points
                .checked_add(record.source_points)
                .ok_or("coverage source point overflow")?;
            if record.sensor == Sensor::Front {
                front += 1;
                expected_visits[index] = 1;
            }
        }
        let mut association_index = 0;
        replay(episode, expected, |association| {
            let actual =
                self.associations.get(association_index).ok_or("coverage association missing")?;
            if (actual.front_index, actual.rear_index, actual.delta_ns)
                != (association.front_index, association.rear_index, association.delta_ns)
            {
                return Err("coverage association policy/index/delta mismatch".into());
            }
            let rear = usize::try_from(actual.rear_index)?;
            if actual.visited_points != Some(self.records[rear].source_points) {
                return Err("coverage rear integration missing or point count differs".into());
            }
            expected_visits[rear] =
                expected_visits[rear].checked_add(1).ok_or("coverage visit overflow")?;
            association_index += 1;
            Ok(())
        })?;
        if association_index == 0 || association_index != self.associations.len() {
            return Err("coverage association count mismatch".into());
        }
        let total = u64::try_from(n)?;
        let rear = total.checked_sub(front).ok_or("coverage role count overflow")?;
        if source.front_topic != expected.front_topic
            || source.rear_topic != expected.rear_topic
            || source.front_record_count != front
            || source.rear_record_count != rear
            || source.total_record_count != total
            || source.total_point_count != source_points
            || !source.full_bag_processed
            || source.truncated
            || source.start_nanos != episode.records().first().map(|r| r.stamp.as_nanos())
            || source.end_nanos != episode.records().last().map(|r| r.stamp.as_nanos())
        {
            return Err("coverage source summary mismatch".into());
        }
        let (mut unique, mut matched_rear, mut actual_visits, mut actual_points) =
            (0_u64, 0_u64, 0_u64, 0_u64);
        for (record, count) in self.records.iter().zip(expected_visits) {
            if record.visits != count
                || record.point_visits
                    != record
                        .source_points
                        .checked_mul(count)
                        .ok_or("coverage point multiplication overflow")?
            {
                return Err("coverage missing/extra integration visit".into());
            }
            unique += u64::from(count != 0);
            matched_rear += u64::from(count != 0 && record.sensor == Sensor::Rear);
            actual_visits =
                actual_visits.checked_add(count).ok_or("coverage total visit overflow")?;
            actual_points = actual_points
                .checked_add(record.point_visits)
                .ok_or("coverage total point overflow")?;
        }
        if actual_visits != visits || actual_points != point_visits {
            return Err("coverage TSDF counters mismatch".into());
        }
        let associations = u64::try_from(association_index)?;
        let summary = Summary {
            front_records: front,
            rear_records: rear,
            associations,
            matched_front_records: associations,
            unmatched_front_records: front
                .checked_sub(associations)
                .ok_or("coverage front count overflow")?,
            matched_rear_records: matched_rear,
            unmatched_rear_records: rear - matched_rear,
            reused_rear_visits: associations - matched_rear,
            unique_visited_records: unique,
            unvisited_records: total - unique,
            total_visits: actual_visits,
            total_point_visits: actual_points,
        };
        if self.summary.as_ref().is_some_and(|existing| existing != &summary) {
            return Err("coverage stored summary mismatch".into());
        }
        Ok(summary)
    }
}

fn describe(
    index: usize,
    record: &StampedRecord,
    context: &Context,
    clock: &mut ClockCorrection,
) -> Result<Record> {
    let sensor = context.sensor(record)?;
    let frame = match sensor {
        Sensor::Front => &context.front_frame,
        Sensor::Rear => &context.rear_frame,
    };
    let raw_nanos = record.record.metadata().timestamp.as_nanos();
    let source_points = u64::try_from(record.record.cloud().len())?;
    if &record.record.metadata().frame_id.0 != frame
        || record.stamp.clock.as_str() != context.clock_id
        || record.stamp.domain != ClockDomain::External
        || record.stamp.quality.offset_ns != clock.offset_i64()?
        || record.stamp.quality.uncertainty_ns != context.uncertainty_ns
        || !record.stamp.quality.estimated
        || record.stamp.as_nanos() != clock.correct(raw_nanos)?
        || source_points == 0
    {
        return Err("coverage record frame/clock/time/points mismatch".into());
    }
    Ok(Record {
        episode_index: u64::try_from(index)?,
        sensor,
        raw_nanos,
        corrected_nanos: record.stamp.as_nanos(),
        source_points,
        visits: 0,
        point_visits: 0,
    })
}

fn note(record: &mut Record, points: usize) -> Result<()> {
    let points = u64::try_from(points)?;
    if points != record.source_points {
        return Err("coverage returned point count differs from submitted cloud".into());
    }
    let visits = record.visits.checked_add(1).ok_or("coverage visit overflow")?;
    let point_visits =
        record.point_visits.checked_add(points).ok_or("coverage point visit overflow")?;
    record.visits = visits;
    record.point_visits = point_visits;
    Ok(())
}

struct LimitWriter<W> {
    inner: W,
    bytes: u64,
    limit: u64,
}
impl<W: Write> Write for LimitWriter<W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.bytes.checked_add(buffer.len() as u64).map_or(true, |n| n > self.limit) {
            return Err(io::Error::other("coverage JSON byte limit exceeded"));
        }
        let n = self.inner.write(buffer)?;
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn temp_path(path: &Path) -> PathBuf {
    path.with_extension(format!("coverage.{}.tmp", std::process::id()))
}

impl CompletedCoverage {
    pub(super) fn publish(&self, path: &Path) -> Result<()> {
        let temporary = temp_path(path);
        let file = OpenOptions::new().write(true).create_new(true).open(&temporary)?;
        let result = (|| -> Result<()> {
            let sink = LimitWriter { inner: file, bytes: 0, limit: self.0.limits.json_bytes };
            let mut writer = BufWriter::with_capacity(WRITE_BUFFER_BYTES, sink);
            serde_json::to_writer(&mut writer, &self.0)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
            writer.get_ref().inner.sync_all()?;
            drop(writer);
            // An atomic no-clobber publication on supported local filesystems.
            // Unsupported hard links fail closed; rename could replace a file.
            fs::hard_link(&temporary, path)?;
            Ok(())
        })();
        let cleanup = fs::remove_file(&temporary);
        result?;
        cleanup?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "fusion_coverage_tests.rs"]
mod tests;
