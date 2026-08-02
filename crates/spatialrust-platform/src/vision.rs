//! SpatialRust Vision 1.0 release evidence and mandatory gate policy.

use crate::{
    BudgetKind, ConformanceReport, ConformanceStatus, LtsPolicy, PerformanceBudget,
    PerformanceBudgetReport, ReleaseGate, ReleaseGateDecision, SecurityChecklist,
    StabilityRegistry,
};

const REQUIRED_CASES: &[&str] = &[
    "vision-full-rust",
    "vision-properties",
    "vision-python-bindings",
    "vision-opencv-correctness",
    "vision-opencv-performance",
    "vision-gpu-explicit-transfers",
    "vision-linux",
    "vision-windows",
    "vision-macos",
    "vision-unsafe-audit",
];

const REQUIRED_COMPARISONS: &[&str] = &[
    "opencv-vision-correctness",
    "opencv-vision-performance",
    "opencv-rgbd-performance",
    "opencv-calibration",
    "opencv-video",
    "opencv-odometry",
    "opencv-photography",
];

const REQUIRED_EXAMPLES: &[&str] = &["vision_1_cpu", "vision_1_release_gate"];

/// Fixed-dimension performance measurements consumed by the Vision 1.0 gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vision1Measurements {
    /// VGA CPU end-to-end latency in microseconds (budget: 50 ms).
    pub cpu_vga_latency_us: u64,
    /// 1080p CPU end-to-end latency in microseconds (budget: 250 ms).
    pub cpu_1080p_latency_us: u64,
    /// 4K resident GPU chain latency in microseconds (budget: 20 ms).
    pub gpu_4k_chain_latency_us: u64,
    /// Maximum explicit bytes copied by one canonical graph run (budget: 64 MiB).
    pub explicit_copy_bytes: u64,
}

/// Evidence gathered by CI/release tooling for a Vision 1.0 candidate.
#[derive(Clone, Debug)]
pub struct Vision1ReleaseEvidence {
    /// Required test/audit cases and their status.
    pub conformance: ConformanceReport,
    /// Satisfied security audit evidence.
    pub security: SecurityChecklist,
    /// Measured canonical performance values.
    pub measurements: Vision1Measurements,
    /// Comparison suite identifiers whose reports passed.
    pub passed_comparison_suites: Vec<String>,
    /// Cargo example names compiled and exercised by the candidate.
    pub verified_examples: Vec<String>,
    /// Migration policy identifier; must equal `vision-1`.
    pub migration_policy: String,
}

/// Mandatory SpatialRust Vision 1.0 release policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vision1ReleaseGate;

impl Vision1ReleaseGate {
    /// Returns conformance ids that must be present with `Pass` status.
    pub const fn required_conformance_cases() -> &'static [&'static str] {
        REQUIRED_CASES
    }
    /// Returns OpenCV comparison suite ids required by the release receipt.
    pub const fn required_comparison_suites() -> &'static [&'static str] {
        REQUIRED_COMPARISONS
    }
    /// Returns examples that must compile and run for the release candidate.
    pub const fn required_examples() -> &'static [&'static str] {
        REQUIRED_EXAMPLES
    }

    /// Evaluates all mandatory evidence and returns every denial reason.
    pub fn evaluate(evidence: &Vision1ReleaseEvidence) -> ReleaseGateDecision {
        let budgets = vision_budgets(evidence.measurements);
        let base = ReleaseGate {
            stability: Some(StabilityRegistry::vision_v1_surface()),
            conformance: Some(evidence.conformance.clone()),
            security: Some(evidence.security.clone()),
            lts: Some(LtsPolicy::spatialrust_v1()),
            budgets: Some(budgets),
            reject_experimental: true,
        };
        let mut decision = base.evaluate();
        for required in REQUIRED_CASES {
            let matching = evidence
                .conformance
                .cases()
                .iter()
                .filter(|case| case.id == *required)
                .collect::<Vec<_>>();
            match matching.as_slice() {
                [case] if case.status == ConformanceStatus::Pass => {}
                [case] => decision
                    .reasons
                    .push(format!("required conformance `{required}` is {:?}", case.status)),
                [] => {
                    decision.reasons.push(format!("required conformance `{required}` is missing"))
                }
                _ => decision
                    .reasons
                    .push(format!("required conformance `{required}` is duplicated")),
            }
        }
        require_names(
            &mut decision.reasons,
            "comparison suite",
            REQUIRED_COMPARISONS,
            &evidence.passed_comparison_suites,
        );
        require_names(
            &mut decision.reasons,
            "example",
            REQUIRED_EXAMPLES,
            &evidence.verified_examples,
        );
        if evidence.migration_policy != "vision-1" {
            decision.reasons.push("migration policy `vision-1` was not acknowledged".into());
        }
        decision.allowed = decision.reasons.is_empty();
        decision
    }
}

fn require_names(reasons: &mut Vec<String>, kind: &str, required: &[&str], actual: &[String]) {
    for name in required {
        if !actual.iter().any(|value| value == name) {
            reasons.push(format!("required {kind} `{name}` is missing"));
        }
    }
}

fn vision_budgets(measurements: Vision1Measurements) -> PerformanceBudgetReport {
    let mut report = PerformanceBudgetReport::new();
    for (id, kind, ceiling, observed) in [
        ("vision-cpu-vga-us", BudgetKind::LatencyMicros, 50_000, measurements.cpu_vga_latency_us),
        (
            "vision-cpu-1080p-us",
            BudgetKind::LatencyMicros,
            250_000,
            measurements.cpu_1080p_latency_us,
        ),
        (
            "vision-gpu-4k-chain-us",
            BudgetKind::LatencyMicros,
            20_000,
            measurements.gpu_4k_chain_latency_us,
        ),
        (
            "vision-explicit-copy-bytes",
            BudgetKind::BytesCopied,
            64 * 1024 * 1024,
            measurements.explicit_copy_bytes,
        ),
    ] {
        report.declare(PerformanceBudget { id: id.into(), kind, ceiling });
        report.sample(id, observed);
    }
    report
}

#[cfg(test)]
mod tests {
    use super::{Vision1Measurements, Vision1ReleaseEvidence, Vision1ReleaseGate};
    use crate::{ConformanceReport, ConformanceStatus, SecurityChecklist};

    fn passing() -> Vision1ReleaseEvidence {
        let mut conformance = ConformanceReport::new();
        for &id in Vision1ReleaseGate::required_conformance_cases() {
            conformance.record(id, ConformanceStatus::Pass, None);
        }
        Vision1ReleaseEvidence {
            conformance,
            security: SecurityChecklist::north_star_baseline_satisfied(),
            measurements: Vision1Measurements {
                cpu_vga_latency_us: 10_000,
                cpu_1080p_latency_us: 100_000,
                gpu_4k_chain_latency_us: 13_523,
                explicit_copy_bytes: 8 * 1024 * 1024,
            },
            passed_comparison_suites: Vision1ReleaseGate::required_comparison_suites()
                .iter()
                .map(ToString::to_string)
                .collect(),
            verified_examples: Vision1ReleaseGate::required_examples()
                .iter()
                .map(ToString::to_string)
                .collect(),
            migration_policy: "vision-1".into(),
        }
    }

    #[test]
    fn vision1_complete_evidence_is_allowed() {
        assert!(Vision1ReleaseGate::evaluate(&passing()).allowed);
    }

    #[test]
    fn vision1_rejects_missing_case_and_over_budget() {
        let mut evidence = passing();
        evidence.conformance = ConformanceReport::new();
        evidence.measurements.gpu_4k_chain_latency_us = 20_001;
        let decision = Vision1ReleaseGate::evaluate(&evidence);
        assert!(!decision.allowed);
        assert!(decision.reasons.iter().any(|reason| reason.contains("vision-linux")));
        assert!(decision.reasons.iter().any(|reason| reason.contains("budget")));
    }

    #[test]
    fn vision1_rejects_skipped_mandatory_case() {
        let mut evidence = passing();
        let mut conformance = ConformanceReport::new();
        for &id in Vision1ReleaseGate::required_conformance_cases() {
            let status = if id == "vision-macos" {
                ConformanceStatus::Skip
            } else {
                ConformanceStatus::Pass
            };
            conformance.record(id, status, None);
        }
        evidence.conformance = conformance;
        let decision = Vision1ReleaseGate::evaluate(&evidence);
        assert!(!decision.allowed);
        assert!(decision
            .reasons
            .iter()
            .any(|reason| reason.contains("vision-macos") && reason.contains("Skip")));
    }
}
