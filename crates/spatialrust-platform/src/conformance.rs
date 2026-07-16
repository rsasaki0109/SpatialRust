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
    pub fn record(
        &mut self,
        id: impl Into<String>,
        status: ConformanceStatus,
        detail: Option<String>,
    ) {
        self.cases.push(ConformanceCase { id: id.into(), status, detail });
    }

    /// Returns cases.
    #[must_use]
    pub fn cases(&self) -> &[ConformanceCase] {
        &self.cases
    }

    /// Counts passes.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.count(ConformanceStatus::Pass)
    }

    /// Counts failures.
    #[must_use]
    pub fn fail_count(&self) -> usize {
        self.count(ConformanceStatus::Fail)
    }

    /// Counts skips.
    #[must_use]
    pub fn skip_count(&self) -> usize {
        self.count(ConformanceStatus::Skip)
    }

    fn count(&self, status: ConformanceStatus) -> usize {
        self.cases.iter().filter(|case| case.status == status).count()
    }

    /// Compact summary string for logs/docs.
    #[must_use]
    pub fn summary(&self) -> String {
        format!("pass={} fail={} skip={}", self.pass_count(), self.fail_count(), self.skip_count())
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
    fn rejects_failures_and_summarizes() {
        let mut report = ConformanceReport::new();
        report.record("arrow-roundtrip", ConformanceStatus::Pass, None);
        report.record("mcap-optional", ConformanceStatus::Skip, Some("feature off".into()));
        assert!(report.assert_no_failures().is_ok());
        assert_eq!(report.summary(), "pass=1 fail=0 skip=1");
        report.record("bad", ConformanceStatus::Fail, None);
        assert!(report.assert_no_failures().is_err());
        assert_eq!(report.fail_count(), 1);
    }
}
