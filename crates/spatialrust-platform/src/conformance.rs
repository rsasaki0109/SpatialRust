//! Conformance suite markers and reports.

use crate::{PlatformError, PlatformResult};

/// Outcome of one conformance case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConformanceStatus {
    /// Passed.
    Pass,
    /// Failed.
    Fail,
    /// Skipped (missing optional feature).
    Skip,
}

/// One conformance case result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConformanceCase {
    /// Case id.
    pub id: String,
    /// Status.
    pub status: ConformanceStatus,
    /// Optional detail.
    pub detail: Option<String>,
}

/// Aggregated conformance report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConformanceReport {
    cases: Vec<ConformanceCase>,
}

impl ConformanceReport {
    /// Creates an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a case.
    pub fn record(&mut self, id: impl Into<String>, status: ConformanceStatus, detail: Option<String>) {
        self.cases.push(ConformanceCase { id: id.into(), status, detail });
    }

    /// Returns cases.
    #[must_use]
    pub fn cases(&self) -> &[ConformanceCase] {
        &self.cases
    }

    /// Fails if any case failed.
    pub fn assert_no_failures(&self) -> PlatformResult<()> {
        if self.cases.iter().any(|case| case.status == ConformanceStatus::Fail) {
            return Err(PlatformError::InvalidConfiguration(
                "conformance report contains failures".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ConformanceReport, ConformanceStatus};

    #[test]
    fn rejects_failures() {
        let mut report = ConformanceReport::new();
        report.record("arrow-roundtrip", ConformanceStatus::Pass, None);
        report.record("mcap-optional", ConformanceStatus::Skip, Some("feature off".into()));
        assert!(report.assert_no_failures().is_ok());
        report.record("bad", ConformanceStatus::Fail, None);
        assert!(report.assert_no_failures().is_err());
    }
}
