//! SpatialRust Vision 2 fail-closed performance and resource release gate.

use std::fmt::Write;

use crate::{
    BudgetKind, ConformanceReport, ConformanceStatus, LtsPolicy, PerformanceBudget,
    PerformanceBudgetReport, ReleaseGate, ReleaseGateDecision, SecurityChecklist,
    StabilityRegistry,
};

const REQUIRED_CASES: &[&str] = &[
    "vision2-rust-accuracy",
    "vision2-python-accuracy",
    "vision2-linux",
    "vision2-windows",
    "vision2-macos",
    "vision2-native-performance",
    "vision2-python-performance",
    "vision2-resource-budgets",
    "vision2-gpu-transfer",
    "vision2-pages",
    "vision2-unsafe-audit",
];

const REQUIRED_RECEIPTS: &[&str] = &[
    "vision2-component-baseline",
    "vision2-resize-color",
    "vision2-gaussian-sobel",
    "vision2-morphology",
    "vision2-canny",
    "vision2-gpu-resident-chain",
];

const REQUIRED_EXAMPLES: &[&str] = &["vision_2_release_gate"];

/// Typed canonical measurements consumed by the Vision 2 release gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vision2Measurements {
    /// Native RGB-to-gray allocating latency at 1080p.
    pub native_allocate_1080p_us: u64,
    /// Native RGB-to-gray caller-output latency at 1080p.
    pub native_reuse_1080p_us: u64,
    /// Python RGB-to-gray allocating latency at 1080p.
    pub python_allocate_1080p_us: u64,
    /// Python RGB-to-gray caller-output latency at 1080p.
    pub python_reuse_1080p_us: u64,
    /// Peak explicitly accounted host bytes in the canonical CPU receipt.
    pub peak_host_memory_bytes: u64,
    /// Dynamic allocations in the canonical caller-output operation.
    pub steady_state_allocations: u64,
    /// Worker count recorded by the default-thread receipt.
    pub worker_threads: u64,
    /// Explicit upload bytes in the canonical 4K GPU-resident chain.
    pub gpu_host_to_device_bytes: u64,
    /// Device-to-host bytes before an optional final readback.
    pub gpu_device_to_host_bytes: u64,
}

/// Evidence gathered by CI and release tooling for a Vision 2 candidate.
#[derive(Clone, Debug)]
pub struct Vision2ReleaseEvidence {
    /// Required accuracy, platform, audit, and documentation cases.
    pub conformance: ConformanceReport,
    /// Satisfied security audit evidence.
    pub security: SecurityChecklist,
    /// Typed performance, memory, allocation, thread, and transfer values.
    pub measurements: Vision2Measurements,
    /// Dated benchmark/correctness receipt identifiers.
    pub passed_receipts: Vec<String>,
    /// Cargo examples compiled and exercised by the candidate.
    pub verified_examples: Vec<String>,
    /// Migration policy identifier; must equal `vision-2`.
    pub migration_policy: String,
}

/// Mandatory SpatialRust Vision 2 release policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vision2ReleaseGate;

impl Vision2ReleaseGate {
    /// Returns conformance ids that must be present exactly once with `Pass`.
    pub const fn required_conformance_cases() -> &'static [&'static str] {
        REQUIRED_CASES
    }

    /// Returns dated receipt ids required by the release candidate.
    pub const fn required_receipts() -> &'static [&'static str] {
        REQUIRED_RECEIPTS
    }

    /// Returns runnable examples required by the release candidate.
    pub const fn required_examples() -> &'static [&'static str] {
        REQUIRED_EXAMPLES
    }

    /// Evaluates every mandatory item and returns all denial reasons.
    pub fn evaluate(evidence: &Vision2ReleaseEvidence) -> ReleaseGateDecision {
        let base = ReleaseGate {
            stability: Some(StabilityRegistry::vision_v2_surface()),
            conformance: Some(evidence.conformance.clone()),
            security: Some(evidence.security.clone()),
            lts: Some(LtsPolicy::spatialrust_v1()),
            budgets: Some(vision2_budgets(evidence.measurements)),
            reject_experimental: true,
        };
        let mut decision = base.evaluate();
        require_passing_cases(&mut decision.reasons, &evidence.conformance);
        require_names(
            &mut decision.reasons,
            "receipt",
            REQUIRED_RECEIPTS,
            &evidence.passed_receipts,
        );
        require_names(
            &mut decision.reasons,
            "example",
            REQUIRED_EXAMPLES,
            &evidence.verified_examples,
        );
        if evidence.migration_policy != "vision-2" {
            decision.reasons.push("migration policy `vision-2` was not acknowledged".into());
        }
        decision.allowed = decision.reasons.is_empty();
        decision
    }

    /// Generates the auditable Markdown receipt embedded in release docs.
    #[must_use]
    pub fn render_markdown(evidence: &Vision2ReleaseEvidence) -> String {
        let decision = Self::evaluate(evidence);
        let values = evidence.measurements;
        let mut output = String::from("# Vision 2 release receipt\n\n");
        let _ = writeln!(
            output,
            "Decision: **{}**\n",
            if decision.allowed { "allowed" } else { "denied" }
        );
        output.push_str("| Measurement | Observed | Ceiling |\n");
        output.push_str("| --- | ---: | ---: |\n");
        for (label, observed, ceiling) in measurement_rows(values) {
            let _ = writeln!(output, "| {label} | {observed} | {ceiling} |");
        }
        output.push_str("\nRequired receipts:\n\n");
        for receipt in REQUIRED_RECEIPTS {
            let present = evidence.passed_receipts.iter().any(|value| value == receipt);
            let _ = writeln!(output, "- [{}] `{receipt}`", if present { "x" } else { " " });
        }
        if !decision.reasons.is_empty() {
            output.push_str("\nDenial reasons:\n\n");
            for reason in decision.reasons {
                let _ = writeln!(output, "- {reason}");
            }
        }
        output
    }
}

fn require_passing_cases(reasons: &mut Vec<String>, conformance: &ConformanceReport) {
    for required in REQUIRED_CASES {
        let matching =
            conformance.cases().iter().filter(|case| case.id == *required).collect::<Vec<_>>();
        match matching.as_slice() {
            [case] if case.status == ConformanceStatus::Pass => {}
            [case] => {
                reasons.push(format!("required conformance `{required}` is {:?}", case.status))
            }
            [] => reasons.push(format!("required conformance `{required}` is missing")),
            _ => reasons.push(format!("required conformance `{required}` is duplicated")),
        }
    }
}

fn require_names(reasons: &mut Vec<String>, kind: &str, required: &[&str], actual: &[String]) {
    for name in required {
        let count = actual.iter().filter(|value| value.as_str() == *name).count();
        match count {
            1 => {}
            0 => reasons.push(format!("required {kind} `{name}` is missing")),
            _ => reasons.push(format!("required {kind} `{name}` is duplicated")),
        }
    }
}

fn measurement_rows(values: Vision2Measurements) -> [(&'static str, u64, u64); 9] {
    [
        ("native allocate 1080p (us)", values.native_allocate_1080p_us, 1_000),
        ("native reuse 1080p (us)", values.native_reuse_1080p_us, 400),
        ("Python allocate 1080p (us)", values.python_allocate_1080p_us, 1_200),
        ("Python reuse 1080p (us)", values.python_reuse_1080p_us, 400),
        ("peak host memory (bytes)", values.peak_host_memory_bytes, 64 * 1024 * 1024),
        ("steady-state allocations", values.steady_state_allocations, 0),
        ("worker threads", values.worker_threads, 12),
        ("GPU upload (bytes)", values.gpu_host_to_device_bytes, 3840 * 2160 * 4),
        ("GPU resident readback (bytes)", values.gpu_device_to_host_bytes, 0),
    ]
}

fn vision2_budgets(values: Vision2Measurements) -> PerformanceBudgetReport {
    let kinds = [
        BudgetKind::LatencyMicros,
        BudgetKind::LatencyMicros,
        BudgetKind::LatencyMicros,
        BudgetKind::LatencyMicros,
        BudgetKind::MemoryBytes,
        BudgetKind::AllocationCount,
        BudgetKind::ThreadCount,
        BudgetKind::BytesCopied,
        BudgetKind::BytesCopied,
    ];
    let mut report = PerformanceBudgetReport::new();
    for (((label, observed, ceiling), kind), id) in
        measurement_rows(values).into_iter().zip(kinds).zip([
            "vision2-native-allocate-1080p-us",
            "vision2-native-reuse-1080p-us",
            "vision2-python-allocate-1080p-us",
            "vision2-python-reuse-1080p-us",
            "vision2-peak-host-memory-bytes",
            "vision2-steady-state-allocations",
            "vision2-worker-threads",
            "vision2-gpu-upload-bytes",
            "vision2-gpu-resident-readback-bytes",
        ])
    {
        let _ = label;
        report.declare(PerformanceBudget { id: id.into(), kind, ceiling });
        report.sample(id, observed);
    }
    report
}

#[cfg(test)]
mod tests {
    use super::{Vision2Measurements, Vision2ReleaseEvidence, Vision2ReleaseGate};
    use crate::{ConformanceReport, ConformanceStatus, SecurityChecklist};

    fn passing() -> Vision2ReleaseEvidence {
        let mut conformance = ConformanceReport::new();
        for &id in Vision2ReleaseGate::required_conformance_cases() {
            conformance.record(id, ConformanceStatus::Pass, Some("CI receipt".into()));
        }
        Vision2ReleaseEvidence {
            conformance,
            security: SecurityChecklist::north_star_baseline_satisfied(),
            measurements: Vision2Measurements {
                native_allocate_1080p_us: 648,
                native_reuse_1080p_us: 195,
                python_allocate_1080p_us: 825,
                python_reuse_1080p_us: 232,
                peak_host_memory_bytes: 6_220_800,
                steady_state_allocations: 0,
                worker_threads: 12,
                gpu_host_to_device_bytes: 3840 * 2160 * 4,
                gpu_device_to_host_bytes: 0,
            },
            passed_receipts: Vision2ReleaseGate::required_receipts()
                .iter()
                .map(ToString::to_string)
                .collect(),
            verified_examples: Vision2ReleaseGate::required_examples()
                .iter()
                .map(ToString::to_string)
                .collect(),
            migration_policy: "vision-2".into(),
        }
    }

    #[test]
    fn vision2_complete_evidence_is_allowed_and_rendered() {
        let evidence = passing();
        assert!(Vision2ReleaseGate::evaluate(&evidence).allowed);
        let markdown = Vision2ReleaseGate::render_markdown(&evidence);
        assert!(markdown.contains("Decision: **allowed**"));
        assert!(markdown.contains("vision2-gpu-resident-chain"));
    }

    #[test]
    fn vision2_rejects_missing_skipped_and_duplicate_evidence() {
        let mut evidence = passing();
        let mut conformance = ConformanceReport::new();
        for &id in Vision2ReleaseGate::required_conformance_cases() {
            if id == "vision2-macos" {
                conformance.record(id, ConformanceStatus::Skip, None);
            } else if id == "vision2-pages" {
                conformance.record(id, ConformanceStatus::Pass, None);
                conformance.record(id, ConformanceStatus::Pass, None);
            } else if id != "vision2-python-accuracy" {
                conformance.record(id, ConformanceStatus::Pass, None);
            }
        }
        evidence.conformance = conformance;
        evidence.passed_receipts.pop();
        evidence.verified_examples.push("vision_2_release_gate".into());
        evidence.migration_policy = "vision-1".into();
        let decision = Vision2ReleaseGate::evaluate(&evidence);
        assert!(!decision.allowed);
        for needle in [
            "vision2-python-accuracy",
            "vision2-macos",
            "vision2-pages",
            "vision2-gpu-resident-chain",
            "vision_2_release_gate",
            "migration policy",
        ] {
            assert!(decision.reasons.iter().any(|reason| reason.contains(needle)), "{needle}");
        }
    }

    #[test]
    fn vision2_rejects_each_resource_budget_overrun() {
        let overruns: &[(&str, fn(&mut Vision2Measurements))] = &[
            ("native-allocate", |values| values.native_allocate_1080p_us = 1_001),
            ("native-reuse", |values| values.native_reuse_1080p_us = 401),
            ("python-allocate", |values| values.python_allocate_1080p_us = 1_201),
            ("python-reuse", |values| values.python_reuse_1080p_us = 401),
            ("peak-host-memory", |values| {
                values.peak_host_memory_bytes = 64 * 1024 * 1024 + 1;
            }),
            ("steady-state-allocations", |values| values.steady_state_allocations = 1),
            ("worker-threads", |values| values.worker_threads = 13),
            ("gpu-upload", |values| {
                values.gpu_host_to_device_bytes = 3840 * 2160 * 4 + 1;
            }),
            ("gpu-resident-readback", |values| values.gpu_device_to_host_bytes = 4),
        ];
        for &(budget_id, mutate) in overruns {
            let mut evidence = passing();
            mutate(&mut evidence.measurements);
            let decision = Vision2ReleaseGate::evaluate(&evidence);
            assert!(!decision.allowed, "{budget_id}");
            assert!(
                decision.reasons.iter().any(|reason| reason.contains(budget_id)),
                "{budget_id}: {:?}",
                decision.reasons
            );
        }
    }
}
