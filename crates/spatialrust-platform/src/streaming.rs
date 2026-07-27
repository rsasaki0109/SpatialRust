//! SpatialRust 1.2 bounded-streaming fail-closed release gate.

use std::fmt::Write;

use crate::{
    BudgetKind, ConformanceReport, ConformanceStatus, LtsPolicy, PerformanceBudget,
    PerformanceBudgetReport, ReleaseGate, ReleaseGateDecision, SecurityChecklist,
    StabilityRegistry,
};

const REQUIRED_CASES: &[&str] = &[
    "streaming-linux",
    "streaming-windows",
    "streaming-macos",
    "streaming-memory-fail-closed",
    "streaming-cancellation-cleanup",
    "streaming-format-roundtrip",
    "streaming-copc-range",
    "streaming-deterministic-voxel",
    "streaming-rust-cli",
    "streaming-python-iterator",
    "streaming-unsafe-audit",
];

const REQUIRED_RECEIPTS: &[&str] = &[
    "epic121-streaming-contract",
    "epic122-bounded-record-stream",
    "epic123-streaming-io",
    "epic124-chunk-ops",
    "epic125-streaming-e2e",
];

const REQUIRED_EXAMPLES: &[&str] = &[
    "streaming_receipt",
    "bounded_record_stream",
    "bounded_pcd_to_ply",
    "bounded_voxel",
    "spatialrust-stream",
    "streaming_1_2_release_gate",
];

const MAX_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SPOOL_LIMIT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_OPEN_SPILL_FILES: u64 = 1025;

/// Typed resource and transfer measurements consumed by the 1.2 gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Streaming12Measurements {
    /// Configured hard tracked-memory ceiling.
    pub memory_budget_bytes: u64,
    /// Peak explicitly tracked resident bytes.
    pub peak_tracked_bytes: u64,
    /// Configured maximum temporary spool extent.
    pub spool_limit_bytes: u64,
    /// Temporary bytes written by the canonical external operation.
    pub spilled_bytes: u64,
    /// Tracked bytes still live after completion or cancellation cleanup.
    pub current_bytes_after_finish: u64,
    /// Host-copy bytes not represented by leased chunks or named IO.
    pub hidden_host_copy_bytes: u64,
    /// Explicit host-to-device bytes in the CPU-only 1.2 workflow.
    pub host_to_device_bytes: u64,
    /// Explicit device-to-host bytes in the CPU-only 1.2 workflow.
    pub device_to_host_bytes: u64,
    /// Cross-chunk/run-size deterministic output mismatches.
    pub determinism_mismatches: u64,
    /// Maximum simultaneously open spool/run files.
    pub max_open_spill_files: u64,
}

/// Evidence gathered by CI and release tooling for SpatialRust 1.2.
#[derive(Clone, Debug)]
pub struct Streaming12ReleaseEvidence {
    /// Required platform, correctness, audit, and workflow cases.
    pub conformance: ConformanceReport,
    /// Satisfied security audit evidence.
    pub security: SecurityChecklist,
    /// Typed memory, spill, cleanup, copy, and determinism measurements.
    pub measurements: Streaming12Measurements,
    /// Dated Epic receipt identifiers.
    pub passed_receipts: Vec<String>,
    /// Cargo/Python/CLI examples exercised by the candidate.
    pub verified_examples: Vec<String>,
    /// Migration policy identifier; must equal `bounded-streaming-1.2`.
    pub migration_policy: String,
}

/// Mandatory SpatialRust 1.2 bounded-streaming release policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Streaming12ReleaseGate;

impl Streaming12ReleaseGate {
    /// Returns conformance ids that must be present exactly once with `Pass`.
    pub const fn required_conformance_cases() -> &'static [&'static str] {
        REQUIRED_CASES
    }

    /// Returns implementation receipt ids required by the candidate.
    pub const fn required_receipts() -> &'static [&'static str] {
        REQUIRED_RECEIPTS
    }

    /// Returns runnable workflow ids required by the candidate.
    pub const fn required_examples() -> &'static [&'static str] {
        REQUIRED_EXAMPLES
    }

    /// Evaluates every mandatory item and returns all denial reasons.
    pub fn evaluate(evidence: &Streaming12ReleaseEvidence) -> ReleaseGateDecision {
        let base = ReleaseGate {
            stability: Some(StabilityRegistry::bounded_streaming_v1_2_surface()),
            conformance: Some(evidence.conformance.clone()),
            security: Some(evidence.security.clone()),
            lts: Some(LtsPolicy::spatialrust_v1()),
            budgets: Some(streaming_budgets(evidence.measurements)),
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
        if evidence.migration_policy != "bounded-streaming-1.2" {
            decision
                .reasons
                .push("migration policy `bounded-streaming-1.2` was not acknowledged".into());
        }
        let values = evidence.measurements;
        if values.peak_tracked_bytes > values.memory_budget_bytes {
            decision.reasons.push(format!(
                "streaming peak {} exceeds configured memory budget {}",
                values.peak_tracked_bytes, values.memory_budget_bytes
            ));
        }
        if values.spilled_bytes > values.spool_limit_bytes {
            decision.reasons.push(format!(
                "streaming spill {} exceeds configured spool limit {}",
                values.spilled_bytes, values.spool_limit_bytes
            ));
        }
        decision.allowed = decision.reasons.is_empty();
        decision
    }

    /// Generates the auditable Markdown release receipt.
    #[must_use]
    pub fn render_markdown(evidence: &Streaming12ReleaseEvidence) -> String {
        let decision = Self::evaluate(evidence);
        let mut output = String::from("# SpatialRust 1.2 bounded-streaming release receipt\n\n");
        let _ = writeln!(
            output,
            "Decision: **{}**\n",
            if decision.allowed { "allowed" } else { "denied" }
        );
        output.push_str("| Measurement | Observed | Ceiling |\n");
        output.push_str("| --- | ---: | ---: |\n");
        for (label, observed, ceiling) in measurement_rows(evidence.measurements) {
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
                reasons.push(format!("required conformance `{required}` is {:?}", case.status));
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

fn measurement_rows(values: Streaming12Measurements) -> [(&'static str, u64, u64); 10] {
    [
        ("configured memory budget (bytes)", values.memory_budget_bytes, MAX_MEMORY_BUDGET_BYTES),
        ("peak tracked memory (bytes)", values.peak_tracked_bytes, MAX_MEMORY_BUDGET_BYTES),
        ("configured spool limit (bytes)", values.spool_limit_bytes, MAX_SPOOL_LIMIT_BYTES),
        ("spilled bytes", values.spilled_bytes, MAX_SPOOL_LIMIT_BYTES),
        ("live bytes after finish", values.current_bytes_after_finish, 0),
        ("hidden host-copy bytes", values.hidden_host_copy_bytes, 0),
        ("host-to-device bytes", values.host_to_device_bytes, 0),
        ("device-to-host bytes", values.device_to_host_bytes, 0),
        ("determinism mismatches", values.determinism_mismatches, 0),
        ("open spill files", values.max_open_spill_files, MAX_OPEN_SPILL_FILES),
    ]
}

fn streaming_budgets(values: Streaming12Measurements) -> PerformanceBudgetReport {
    let kinds = [
        BudgetKind::MemoryBytes,
        BudgetKind::MemoryBytes,
        BudgetKind::MemoryBytes,
        BudgetKind::MemoryBytes,
        BudgetKind::MemoryBytes,
        BudgetKind::BytesCopied,
        BudgetKind::BytesCopied,
        BudgetKind::BytesCopied,
        BudgetKind::AllocationCount,
        BudgetKind::AllocationCount,
    ];
    let ids = [
        "streaming-configured-memory-budget-bytes",
        "streaming-peak-tracked-memory-bytes",
        "streaming-configured-spool-limit-bytes",
        "streaming-spilled-bytes",
        "streaming-live-bytes-after-finish",
        "streaming-hidden-host-copy-bytes",
        "streaming-host-to-device-bytes",
        "streaming-device-to-host-bytes",
        "streaming-determinism-mismatches",
        "streaming-open-spill-files",
    ];
    let mut report = PerformanceBudgetReport::new();
    for (((_, observed, ceiling), kind), id) in
        measurement_rows(values).into_iter().zip(kinds).zip(ids)
    {
        report.declare(PerformanceBudget { id: id.into(), kind, ceiling });
        report.sample(id, observed);
    }
    report
}

#[cfg(test)]
mod tests {
    use super::{Streaming12Measurements, Streaming12ReleaseEvidence, Streaming12ReleaseGate};
    use crate::{ConformanceReport, ConformanceStatus, SecurityChecklist};

    fn passing() -> Streaming12ReleaseEvidence {
        let mut conformance = ConformanceReport::new();
        for &id in Streaming12ReleaseGate::required_conformance_cases() {
            conformance.record(id, ConformanceStatus::Pass, Some("CI receipt".into()));
        }
        Streaming12ReleaseEvidence {
            conformance,
            security: SecurityChecklist::north_star_baseline_satisfied(),
            measurements: Streaming12Measurements {
                memory_budget_bytes: 1024 * 1024,
                peak_tracked_bytes: 64 * 1024,
                spool_limit_bytes: 1024 * 1024,
                spilled_bytes: 4096,
                current_bytes_after_finish: 0,
                hidden_host_copy_bytes: 0,
                host_to_device_bytes: 0,
                device_to_host_bytes: 0,
                determinism_mismatches: 0,
                max_open_spill_files: 4,
            },
            passed_receipts: Streaming12ReleaseGate::required_receipts()
                .iter()
                .map(ToString::to_string)
                .collect(),
            verified_examples: Streaming12ReleaseGate::required_examples()
                .iter()
                .map(ToString::to_string)
                .collect(),
            migration_policy: "bounded-streaming-1.2".into(),
        }
    }

    #[test]
    fn streaming_complete_evidence_is_allowed_and_rendered() {
        let evidence = passing();
        assert!(Streaming12ReleaseGate::evaluate(&evidence).allowed);
        let markdown = Streaming12ReleaseGate::render_markdown(&evidence);
        assert!(markdown.contains("Decision: **allowed**"));
        assert!(markdown.contains("epic125-streaming-e2e"));
    }

    #[test]
    fn streaming_rejects_missing_skipped_and_duplicate_evidence() {
        let mut evidence = passing();
        let mut conformance = ConformanceReport::new();
        for &id in Streaming12ReleaseGate::required_conformance_cases() {
            if id == "streaming-macos" {
                conformance.record(id, ConformanceStatus::Skip, None);
            } else if id == "streaming-rust-cli" {
                conformance.record(id, ConformanceStatus::Pass, None);
                conformance.record(id, ConformanceStatus::Pass, None);
            } else if id != "streaming-python-iterator" {
                conformance.record(id, ConformanceStatus::Pass, None);
            }
        }
        evidence.conformance = conformance;
        evidence.passed_receipts.pop();
        evidence.verified_examples.push("streaming_1_2_release_gate".into());
        evidence.migration_policy = "vision-2".into();
        let decision = Streaming12ReleaseGate::evaluate(&evidence);
        assert!(!decision.allowed);
        for needle in [
            "streaming-python-iterator",
            "streaming-macos",
            "streaming-rust-cli",
            "epic125-streaming-e2e",
            "streaming_1_2_release_gate",
            "migration policy",
        ] {
            assert!(decision.reasons.iter().any(|reason| reason.contains(needle)), "{needle}");
        }
    }

    #[test]
    fn streaming_rejects_each_resource_budget_overrun() {
        let overruns: &[(&str, fn(&mut Streaming12Measurements))] = &[
            ("configured-memory", |v| v.memory_budget_bytes = 256 * 1024 * 1024 + 1),
            ("peak-tracked", |v| v.peak_tracked_bytes = 256 * 1024 * 1024 + 1),
            ("configured-spool", |v| v.spool_limit_bytes = 2 * 1024 * 1024 * 1024 + 1),
            ("spilled", |v| v.spilled_bytes = 2 * 1024 * 1024 * 1024 + 1),
            ("live-bytes", |v| v.current_bytes_after_finish = 1),
            ("hidden-host-copy", |v| v.hidden_host_copy_bytes = 1),
            ("host-to-device", |v| v.host_to_device_bytes = 1),
            ("device-to-host", |v| v.device_to_host_bytes = 1),
            ("determinism", |v| v.determinism_mismatches = 1),
            ("open-spill-files", |v| v.max_open_spill_files = 1026),
        ];
        for &(budget_id, mutate) in overruns {
            let mut evidence = passing();
            mutate(&mut evidence.measurements);
            let decision = Streaming12ReleaseGate::evaluate(&evidence);
            assert!(!decision.allowed, "{budget_id}");
            assert!(
                decision.reasons.iter().any(|reason| reason.contains(budget_id)),
                "{budget_id}: {:?}",
                decision.reasons
            );
        }
    }

    #[test]
    fn streaming_rejects_peak_and_spill_above_configured_limits() {
        let mut evidence = passing();
        evidence.measurements.peak_tracked_bytes = evidence.measurements.memory_budget_bytes + 1;
        evidence.measurements.spilled_bytes = evidence.measurements.spool_limit_bytes + 1;
        let decision = Streaming12ReleaseGate::evaluate(&evidence);
        assert!(!decision.allowed);
        assert!(decision.reasons.iter().any(|reason| reason.contains("configured memory budget")));
        assert!(decision.reasons.iter().any(|reason| reason.contains("configured spool limit")));
    }
}
