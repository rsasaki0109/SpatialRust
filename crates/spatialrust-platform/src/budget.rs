//! Named performance budgets and measured samples.

use crate::{PlatformError, PlatformResult};

/// Dimension of a performance budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BudgetKind {
    /// Wall-clock latency upper bound in microseconds.
    LatencyMicros,
    /// Wall-clock latency upper bound.
    LatencyMillis,
    /// Explicit transfer / copy volume upper bound.
    BytesCopied,
    /// Host memory residency upper bound.
    MemoryBytes,
    /// Number of dynamic allocations in one measured operation.
    AllocationCount,
    /// Number of worker threads permitted by the measured policy.
    ThreadCount,
}

/// One declared budget ceiling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PerformanceBudget {
    /// Budget id, e.g. `north-star-e2e-latency`.
    pub id: String,
    /// Dimension being constrained.
    pub kind: BudgetKind,
    /// Inclusive maximum allowed measurement.
    pub ceiling: u64,
}

/// One measured sample against a budget id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PerformanceSample {
    /// Matching budget id.
    pub budget_id: String,
    /// Observed value in the budget's units.
    pub observed: u64,
}

/// Collection of budgets plus measured samples.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PerformanceBudgetReport {
    budgets: Vec<PerformanceBudget>,
    samples: Vec<PerformanceSample>,
}

impl PerformanceBudgetReport {
    /// Creates an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a budget ceiling.
    pub fn declare(&mut self, budget: PerformanceBudget) {
        self.budgets.push(budget);
    }

    /// Records one measurement.
    pub fn sample(&mut self, budget_id: impl Into<String>, observed: u64) {
        self.samples.push(PerformanceSample { budget_id: budget_id.into(), observed });
    }

    /// Returns budgets.
    #[must_use]
    pub fn budgets(&self) -> &[PerformanceBudget] {
        &self.budgets
    }

    /// Returns samples.
    #[must_use]
    pub fn samples(&self) -> &[PerformanceSample] {
        &self.samples
    }

    /// Fails when any sample exceeds its ceiling, or samples a missing budget.
    pub fn assert_within_budgets(&self) -> PlatformResult<()> {
        for sample in &self.samples {
            let Some(budget) = self.budgets.iter().find(|b| b.id == sample.budget_id) else {
                return Err(PlatformError::InvalidConfiguration(format!(
                    "sample references unknown budget `{}`",
                    sample.budget_id
                )));
            };
            if sample.observed > budget.ceiling {
                return Err(PlatformError::BudgetExceeded {
                    budget_id: budget.id.clone(),
                    observed: sample.observed,
                    ceiling: budget.ceiling,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BudgetKind, PerformanceBudget, PerformanceBudgetReport};

    #[test]
    fn rejects_over_ceiling() {
        let mut report = PerformanceBudgetReport::new();
        report.declare(PerformanceBudget {
            id: "latency".into(),
            kind: BudgetKind::LatencyMillis,
            ceiling: 100,
        });
        report.sample("latency", 50);
        assert!(report.assert_within_budgets().is_ok());
        report.sample("latency", 101);
        assert!(report.assert_within_budgets().is_err());
    }
}
