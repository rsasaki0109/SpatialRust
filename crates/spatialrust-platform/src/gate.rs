//! Aggregated release-gate decision across platform surfaces.

use crate::{
    ApiStabilityClass, ConformanceReport, LtsPolicy, PerformanceBudgetReport, PlatformError,
    PlatformResult, SecurityChecklist, StabilityRegistry,
};

/// Outcomes of evaluating a [`ReleaseGate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseGateDecision {
    /// True when every configured surface passes.
    pub allowed: bool,
    /// Human-readable reasons for denial (empty when allowed).
    pub reasons: Vec<String>,
}

/// Release gate combining stability, conformance, security, LTS, and budgets.
#[derive(Clone, Debug, Default)]
pub struct ReleaseGate {
    /// API surface registry (optional for gate evaluation).
    pub stability: Option<StabilityRegistry>,
    /// Conformance report.
    pub conformance: Option<ConformanceReport>,
    /// Security checklist.
    pub security: Option<SecurityChecklist>,
    /// LTS policy (must declare at least one window when present).
    pub lts: Option<LtsPolicy>,
    /// Performance budgets.
    pub budgets: Option<PerformanceBudgetReport>,
    /// When true, Experimental APIs block the gate.
    pub reject_experimental: bool,
}

impl ReleaseGate {
    /// Creates an empty gate (nothing configured ⇒ denied).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Seeds a gate with SpatialRust 1.x defaults used by north-star checks.
    #[must_use]
    pub fn north_star_defaults() -> Self {
        Self {
            stability: Some(StabilityRegistry::north_star_surface()),
            conformance: Some(ConformanceReport::new()),
            security: Some(SecurityChecklist::north_star_baseline()),
            lts: Some(LtsPolicy::spatialrust_v1()),
            budgets: Some({
                let mut budgets = PerformanceBudgetReport::new();
                budgets.declare(crate::PerformanceBudget {
                    id: "north-star-e2e-latency-ms".into(),
                    kind: crate::BudgetKind::LatencyMillis,
                    ceiling: 5_000,
                });
                budgets.declare(crate::PerformanceBudget {
                    id: "north-star-e2e-bytes-copied".into(),
                    kind: crate::BudgetKind::BytesCopied,
                    ceiling: 64 * 1024 * 1024,
                });
                budgets
            }),
            reject_experimental: true,
        }
    }

    /// Evaluates configured surfaces and returns a decision.
    pub fn evaluate(&self) -> ReleaseGateDecision {
        let mut reasons = Vec::new();

        if self.stability.is_none()
            && self.conformance.is_none()
            && self.security.is_none()
            && self.lts.is_none()
            && self.budgets.is_none()
        {
            reasons.push("release gate has no configured surfaces".into());
            return ReleaseGateDecision {
                allowed: false,
                reasons,
            };
        }

        if let Some(stability) = &self.stability {
            if stability.items().is_empty() {
                reasons.push("stability registry is empty".into());
            }
            if self.reject_experimental {
                let experimental = stability
                    .items()
                    .iter()
                    .filter(|item| item.class == ApiStabilityClass::Experimental)
                    .count();
                if experimental > 0 {
                    reasons.push(format!(
                        "{experimental} experimental API(s) present while reject_experimental=true"
                    ));
                }
            }
        }

        if let Some(conformance) = &self.conformance {
            if conformance.cases().is_empty() {
                reasons.push("conformance report has no cases".into());
            }
            if let Err(err) = conformance.assert_no_failures() {
                reasons.push(err.to_string());
            }
            if conformance.pass_count() == 0 && !conformance.cases().is_empty() {
                reasons.push("conformance report has zero passes".into());
            }
        }

        if let Some(security) = &self.security {
            if !security.all_satisfied() {
                reasons.push("security checklist is not fully satisfied".into());
            }
        }

        if let Some(lts) = &self.lts {
            if lts.windows().is_empty() {
                reasons.push("LTS policy has no support windows".into());
            }
        }

        if let Some(budgets) = &self.budgets {
            if let Err(err) = budgets.assert_within_budgets() {
                reasons.push(err.to_string());
            }
        }

        ReleaseGateDecision {
            allowed: reasons.is_empty(),
            reasons,
        }
    }

    /// Convenience wrapper returning [`PlatformResult`].
    pub fn assert_allowed(&self) -> PlatformResult<()> {
        let decision = self.evaluate();
        if decision.allowed {
            Ok(())
        } else {
            Err(PlatformError::ReleaseGateDenied {
                reasons: decision.reasons,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReleaseGate;
    use crate::{ConformanceStatus, SecurityChecklist};

    #[test]
    fn north_star_defaults_need_conformance_passes() {
        let mut gate = ReleaseGate::north_star_defaults();
        assert!(!gate.evaluate().allowed);
        if let Some(report) = gate.conformance.as_mut() {
            report.record("smoke", ConformanceStatus::Pass, None);
        }
        // Security checklist from baseline starts unsatisfied until marked.
        if let Some(security) = gate.security.as_mut() {
            *security = SecurityChecklist::north_star_baseline_satisfied();
        }
        assert!(gate.assert_allowed().is_ok());
    }
}
