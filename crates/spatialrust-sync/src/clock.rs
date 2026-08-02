//! Clock domains and sync quality attached to observation times.

use spatialrust_core::Timestamp;

/// Named clock source (host, sensor board, ROS time, ...).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ClockId(pub String);

impl ClockId {
    /// Creates a clock identifier.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrows the identifier.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ClockId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// Semantic class of a clock domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ClockDomain {
    /// Host monotonic / steady clock.
    #[default]
    HostSteady,
    /// Host wall / UTC-aligned clock.
    HostWall,
    /// Sensor-local free-running clock.
    Sensor,
    /// Externally synchronized domain (PTP/NTP/ROS `/clock`).
    External,
}

/// Quality of time synchronization relative to a reference clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SyncQuality {
    /// Estimated offset from the reference clock in nanoseconds.
    pub offset_ns: i64,
    /// Estimated 1-sigma uncertainty in nanoseconds.
    pub uncertainty_ns: u64,
    /// Whether the offset is an estimate rather than a hard measurement.
    pub estimated: bool,
}

impl SyncQuality {
    /// Exact sync with zero offset/uncertainty.
    #[must_use]
    pub const fn exact() -> Self {
        Self { offset_ns: 0, uncertainty_ns: 0, estimated: false }
    }

    /// Returns whether this stamp is usable for tight multimodal fusion.
    #[must_use]
    pub const fn is_tight(self, max_uncertainty_ns: u64) -> bool {
        self.uncertainty_ns <= max_uncertainty_ns
    }
}

/// Timestamp anchored to an explicit clock domain and sync quality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StampedTime {
    /// Clock that produced the timestamp.
    pub clock: ClockId,
    /// Domain class for the clock.
    pub domain: ClockDomain,
    /// Raw timestamp on `clock`.
    pub timestamp: Timestamp,
    /// Sync quality versus an episode reference clock.
    pub quality: SyncQuality,
}

impl StampedTime {
    /// Creates a stamped time with exact sync quality.
    #[must_use]
    pub fn exact(clock: impl Into<ClockId>, domain: ClockDomain, timestamp: Timestamp) -> Self {
        Self { clock: clock.into(), domain, timestamp, quality: SyncQuality::exact() }
    }

    /// Returns nanoseconds on the owning clock.
    #[must_use]
    pub const fn as_nanos(&self) -> u64 {
        self.timestamp.as_nanos()
    }

    /// Absolute nanosecond distance to another stamp (ignores clock identity).
    #[must_use]
    pub fn abs_delta_ns(&self, other: &Self) -> u64 {
        let a = self.as_nanos();
        let b = other.as_nanos();
        a.abs_diff(b)
    }
}

#[cfg(test)]
mod tests {
    use super::{ClockDomain, StampedTime, SyncQuality};
    use spatialrust_core::Timestamp;

    #[test]
    fn tight_sync_rejects_high_uncertainty() {
        let quality = SyncQuality { offset_ns: 0, uncertainty_ns: 5_000_000, estimated: true };
        assert!(!quality.is_tight(1_000_000));
        let stamp = StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(10));
        assert_eq!(stamp.as_nanos(), 10);
    }
}
