//! Backpressure contracts for distributed queues.

/// Signal indicating consumer pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackpressureSignal {
    /// Accept more work.
    Ok,
    /// Soft warn at high watermark.
    SoftLimit,
    /// Hard reject.
    HardLimit,
}

/// Watermark policy for explicit backpressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BackpressurePolicy {
    /// Soft watermark.
    pub soft_limit: usize,
    /// Hard watermark.
    pub hard_limit: usize,
}

impl BackpressurePolicy {
    /// Creates a validated policy.
    pub fn try_new(soft_limit: usize, hard_limit: usize) -> crate::DistributeResult<Self> {
        if soft_limit == 0 || hard_limit < soft_limit {
            return Err(crate::DistributeError::InvalidConfiguration(
                "require 0 < soft_limit <= hard_limit".into(),
            ));
        }
        Ok(Self { soft_limit, hard_limit })
    }

    /// Evaluates queue depth against watermarks.
    #[must_use]
    pub fn evaluate(self, depth: usize) -> BackpressureSignal {
        if depth >= self.hard_limit {
            BackpressureSignal::HardLimit
        } else if depth >= self.soft_limit {
            BackpressureSignal::SoftLimit
        } else {
            BackpressureSignal::Ok
        }
    }
}
