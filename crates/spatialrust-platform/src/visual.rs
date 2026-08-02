//! SpatialRust Visual fail-closed conformance and release gate.

use std::fmt::Write;

use crate::{
    BudgetKind, ConformanceReport, ConformanceStatus, LtsPolicy, PerformanceBudget,
    PerformanceBudgetReport, ReleaseGate, ReleaseGateDecision, SecurityChecklist,
    StabilityRegistry,
};

const REQUIRED_CASES: &[&str] = &[
    "visual-headless-linux",
    "visual-headless-windows",
    "visual-headless-macos",
    "visual-native-smoke",
    "visual-web-wasm",
    "visual-browser-smoke",
    "visual-python-38",
    "visual-python-current",
    "visual-jupyter-notebook",
    "visual-lod-budgets",
    "visual-transfer-ledger",
    "visual-docs",
    "visual-unsafe-audit",
];

const REQUIRED_RECEIPTS: &[&str] = &[
    "visual-viz-contracts",
    "visual-wgpu-renderer",
    "visual-native-debug",
    "visual-scene-rgbd",
    "visual-bounded-lod",
    "visual-web-viewer",
    "visual-python-jupyter",
];

const REQUIRED_EXAMPLES: &[&str] = &["visual_release_gate"];
const MAX_RECEIPT_AGE_DAYS: u64 = 30;

/// Typed canonical measurements consumed by the Visual release gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualMeasurements {
    /// Pixel/channel mismatches in strict canonical headless fixtures.
    pub headless_pixel_mismatches: u64,
    /// Maximum absolute channel delta in canonical headless fixtures.
    pub headless_max_channel_delta: u64,
    /// Explicit host-to-device geometry bytes for the canonical point source.
    pub geometry_upload_bytes: u64,
    /// Explicit render-uniform bytes for the canonical frame.
    pub render_uniform_upload_bytes: u64,
    /// Device-to-host bytes before caller-requested readback.
    pub unexpected_readback_bytes: u64,
    /// Caller-requested RGBA screenshot readback bytes.
    pub screenshot_readback_bytes: u64,
    /// Peak accounted host bytes in the canonical LOD run.
    pub peak_lod_host_bytes: u64,
    /// Peak accounted device bytes in the canonical LOD run.
    pub peak_lod_gpu_bytes: u64,
    /// Maximum concurrent LOD chunk requests.
    pub inflight_lod_chunks: u64,
    /// Total admitted HTTP Range bytes in the browser fixture.
    pub browser_requested_bytes: u64,
    /// Explicit bytes copied by the canonical Python fixture.
    pub python_copy_bytes: u64,
    /// Portable state mismatches across native, Web, Python, and Jupyter.
    pub state_roundtrip_mismatches: u64,
}

/// One dated Visual evidence receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisualReceiptEvidence {
    /// Stable receipt identifier.
    pub id: String,
    /// UTC day containing the evidence, expressed as Unix epoch days.
    pub captured_unix_days: u64,
}

/// Evidence gathered by CI and release tooling for a Visual candidate.
#[derive(Clone, Debug)]
pub struct VisualReleaseEvidence {
    /// Required image, platform, adapter, audit, and documentation cases.
    pub conformance: ConformanceReport,
    /// Satisfied security audit evidence.
    pub security: SecurityChecklist,
    /// Typed image, transfer, residency, request, and state values.
    pub measurements: VisualMeasurements,
    /// Candidate UTC day, expressed as Unix epoch days.
    pub candidate_unix_days: u64,
    /// Required dated implementation receipts.
    pub receipts: Vec<VisualReceiptEvidence>,
    /// Cargo examples compiled and exercised by the candidate.
    pub verified_examples: Vec<String>,
    /// Migration policy identifier; must equal `visual-1`.
    pub migration_policy: String,
}

/// Mandatory SpatialRust Visual release policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct VisualReleaseGate;

impl VisualReleaseGate {
    /// Returns conformance ids that must be present exactly once with `Pass`.
    pub const fn required_conformance_cases() -> &'static [&'static str] {
        REQUIRED_CASES
    }

    /// Returns fresh receipt ids required by the release candidate.
    pub const fn required_receipts() -> &'static [&'static str] {
        REQUIRED_RECEIPTS
    }

    /// Returns runnable examples required by the release candidate.
    pub const fn required_examples() -> &'static [&'static str] {
        REQUIRED_EXAMPLES
    }

    /// Maximum accepted receipt age in whole days.
    pub const fn max_receipt_age_days() -> u64 {
        MAX_RECEIPT_AGE_DAYS
    }

    /// Evaluates every mandatory item and returns all denial reasons.
    pub fn evaluate(evidence: &VisualReleaseEvidence) -> ReleaseGateDecision {
        let base = ReleaseGate {
            stability: Some(StabilityRegistry::visual_surface()),
            conformance: Some(evidence.conformance.clone()),
            security: Some(evidence.security.clone()),
            lts: Some(LtsPolicy::spatialrust_v1()),
            budgets: Some(visual_budgets(evidence.measurements)),
            reject_experimental: true,
        };
        let mut decision = base.evaluate();
        require_passing_cases(&mut decision.reasons, &evidence.conformance);
        require_fresh_receipts(
            &mut decision.reasons,
            evidence.candidate_unix_days,
            &evidence.receipts,
        );
        require_names(
            &mut decision.reasons,
            "example",
            REQUIRED_EXAMPLES,
            &evidence.verified_examples,
        );
        if evidence.migration_policy != "visual-1" {
            decision.reasons.push("migration policy `visual-1` was not acknowledged".into());
        }
        decision.allowed = decision.reasons.is_empty();
        decision
    }

    /// Generates the auditable Markdown receipt embedded in release docs.
    #[must_use]
    pub fn render_markdown(evidence: &VisualReleaseEvidence) -> String {
        let decision = Self::evaluate(evidence);
        let mut output = String::from("# Visual release receipt\n\n");
        let _ = writeln!(
            output,
            "Decision: **{}**\n",
            if decision.allowed { "allowed" } else { "denied" }
        );
        let _ = writeln!(output, "Candidate Unix day: `{}`\n", evidence.candidate_unix_days);
        output.push_str("| Measurement | Observed | Ceiling |\n");
        output.push_str("| --- | ---: | ---: |\n");
        for (label, observed, ceiling) in measurement_rows(evidence.measurements) {
            let _ = writeln!(output, "| {label} | {observed} | {ceiling} |");
        }
        output.push_str("\nRequired receipts:\n\n");
        for receipt in REQUIRED_RECEIPTS {
            let matching = evidence.receipts.iter().find(|item| item.id == *receipt);
            let present = matching.is_some();
            let age = matching
                .and_then(|item| evidence.candidate_unix_days.checked_sub(item.captured_unix_days))
                .map_or_else(|| "n/a".into(), |days| format!("{days} day(s)"));
            let _ = writeln!(output, "- [{}] `{receipt}` ({age})", if present { "x" } else { " " });
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

fn require_fresh_receipts(
    reasons: &mut Vec<String>,
    candidate_unix_days: u64,
    receipts: &[VisualReceiptEvidence],
) {
    if candidate_unix_days == 0 {
        reasons.push("candidate Unix day must be non-zero".into());
    }
    for required in REQUIRED_RECEIPTS {
        let matching =
            receipts.iter().filter(|receipt| receipt.id == *required).collect::<Vec<_>>();
        match matching.as_slice() {
            [receipt] if receipt.captured_unix_days > candidate_unix_days => {
                reasons.push(format!("required receipt `{required}` is dated after the candidate"))
            }
            [receipt]
                if candidate_unix_days - receipt.captured_unix_days > MAX_RECEIPT_AGE_DAYS =>
            {
                reasons.push(format!(
                    "required receipt `{required}` is stale (older than {MAX_RECEIPT_AGE_DAYS} days)"
                ));
            }
            [_] => {}
            [] => reasons.push(format!("required receipt `{required}` is missing")),
            _ => reasons.push(format!("required receipt `{required}` is duplicated")),
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

fn measurement_rows(values: VisualMeasurements) -> [(&'static str, u64, u64); 12] {
    [
        ("headless pixel mismatches", values.headless_pixel_mismatches, 0),
        ("headless maximum channel delta", values.headless_max_channel_delta, 0),
        ("canonical geometry upload (bytes)", values.geometry_upload_bytes, 8 * 1024 * 1024),
        ("render uniform upload (bytes)", values.render_uniform_upload_bytes, 112),
        ("unexpected readback (bytes)", values.unexpected_readback_bytes, 0),
        ("screenshot readback (bytes)", values.screenshot_readback_bytes, 64 * 64 * 4),
        ("peak LOD host memory (bytes)", values.peak_lod_host_bytes, 64 * 1024 * 1024),
        ("peak LOD GPU memory (bytes)", values.peak_lod_gpu_bytes, 64 * 1024 * 1024),
        ("in-flight LOD chunks", values.inflight_lod_chunks, 8),
        ("browser requested bytes", values.browser_requested_bytes, 8 * 1024 * 1024),
        ("Python explicit copy (bytes)", values.python_copy_bytes, 12 * 1024 * 1024),
        ("state round-trip mismatches", values.state_roundtrip_mismatches, 0),
    ]
}

fn visual_budgets(values: VisualMeasurements) -> PerformanceBudgetReport {
    let kinds = [
        BudgetKind::AllocationCount,
        BudgetKind::AllocationCount,
        BudgetKind::BytesCopied,
        BudgetKind::BytesCopied,
        BudgetKind::BytesCopied,
        BudgetKind::BytesCopied,
        BudgetKind::MemoryBytes,
        BudgetKind::MemoryBytes,
        BudgetKind::ThreadCount,
        BudgetKind::BytesCopied,
        BudgetKind::BytesCopied,
        BudgetKind::AllocationCount,
    ];
    let ids = [
        "visual-headless-pixel-mismatches",
        "visual-headless-channel-delta",
        "visual-geometry-upload-bytes",
        "visual-render-uniform-upload-bytes",
        "visual-unexpected-readback-bytes",
        "visual-screenshot-readback-bytes",
        "visual-peak-lod-host-bytes",
        "visual-peak-lod-gpu-bytes",
        "visual-inflight-lod-chunks",
        "visual-browser-requested-bytes",
        "visual-python-copy-bytes",
        "visual-state-roundtrip-mismatches",
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
    use super::{
        VisualMeasurements, VisualReceiptEvidence, VisualReleaseEvidence, VisualReleaseGate,
    };
    use crate::{ConformanceReport, ConformanceStatus, SecurityChecklist};

    fn passing() -> VisualReleaseEvidence {
        let mut conformance = ConformanceReport::new();
        for &id in VisualReleaseGate::required_conformance_cases() {
            conformance.record(id, ConformanceStatus::Pass, Some("CI receipt".into()));
        }
        VisualReleaseEvidence {
            conformance,
            security: SecurityChecklist::north_star_baseline_satisfied(),
            measurements: VisualMeasurements {
                headless_pixel_mismatches: 0,
                headless_max_channel_delta: 0,
                geometry_upload_bytes: 40,
                render_uniform_upload_bytes: 112,
                unexpected_readback_bytes: 0,
                screenshot_readback_bytes: 64 * 64 * 4,
                peak_lod_host_bytes: 24 * 1024 * 1024,
                peak_lod_gpu_bytes: 20 * 1024 * 1024,
                inflight_lod_chunks: 4,
                browser_requested_bytes: 2 * 1024 * 1024,
                python_copy_bytes: 3 * 1024 * 1024,
                state_roundtrip_mismatches: 0,
            },
            candidate_unix_days: 20_662,
            receipts: VisualReleaseGate::required_receipts()
                .iter()
                .map(|id| VisualReceiptEvidence { id: (*id).into(), captured_unix_days: 20_662 })
                .collect(),
            verified_examples: VisualReleaseGate::required_examples()
                .iter()
                .map(ToString::to_string)
                .collect(),
            migration_policy: "visual-1".into(),
        }
    }

    #[test]
    fn complete_visual_evidence_is_allowed_and_rendered() {
        let evidence = passing();
        assert!(VisualReleaseGate::evaluate(&evidence).allowed);
        let markdown = VisualReleaseGate::render_markdown(&evidence);
        assert!(markdown.contains("Decision: **allowed**"));
        assert!(markdown.contains("visual-python-jupyter"));
        assert!(markdown.contains("(0 day(s))"));
    }

    #[test]
    fn rejects_missing_skipped_duplicate_and_wrong_migration_evidence() {
        let mut evidence = passing();
        let mut conformance = ConformanceReport::new();
        for &id in VisualReleaseGate::required_conformance_cases() {
            if id == "visual-headless-macos" {
                conformance.record(id, ConformanceStatus::Skip, None);
            } else if id == "visual-docs" {
                conformance.record(id, ConformanceStatus::Pass, None);
                conformance.record(id, ConformanceStatus::Pass, None);
            } else if id != "visual-browser-smoke" {
                conformance.record(id, ConformanceStatus::Pass, None);
            }
        }
        evidence.conformance = conformance;
        evidence.receipts.pop();
        evidence.verified_examples.push("visual_release_gate".into());
        evidence.migration_policy = "visual-0".into();
        let decision = VisualReleaseGate::evaluate(&evidence);
        assert!(!decision.allowed);
        for needle in [
            "visual-headless-macos",
            "visual-docs",
            "visual-browser-smoke",
            "visual-python-jupyter",
            "visual_release_gate",
            "migration policy",
        ] {
            assert!(
                decision.reasons.iter().any(|reason| reason.contains(needle)),
                "{needle}: {:?}",
                decision.reasons
            );
        }
    }

    #[test]
    fn rejects_stale_future_duplicate_and_undated_receipts() {
        let mut stale = passing();
        stale.receipts[0].captured_unix_days =
            stale.candidate_unix_days - VisualReleaseGate::max_receipt_age_days() - 1;
        assert!(VisualReleaseGate::evaluate(&stale)
            .reasons
            .iter()
            .any(|reason| reason.contains("stale")));

        let mut future = passing();
        future.receipts[0].captured_unix_days = future.candidate_unix_days + 1;
        assert!(VisualReleaseGate::evaluate(&future)
            .reasons
            .iter()
            .any(|reason| reason.contains("after the candidate")));

        let mut duplicate = passing();
        duplicate.receipts.push(duplicate.receipts[0].clone());
        assert!(VisualReleaseGate::evaluate(&duplicate)
            .reasons
            .iter()
            .any(|reason| reason.contains("duplicated")));

        let mut undated = passing();
        undated.candidate_unix_days = 0;
        assert!(VisualReleaseGate::evaluate(&undated)
            .reasons
            .iter()
            .any(|reason| reason.contains("non-zero")));
    }

    #[test]
    fn rejects_each_visual_budget_overrun() {
        let overruns: &[(&str, fn(&mut VisualMeasurements))] = &[
            ("pixel-mismatches", |v| v.headless_pixel_mismatches = 1),
            ("channel-delta", |v| v.headless_max_channel_delta = 1),
            ("geometry-upload", |v| v.geometry_upload_bytes = 8 * 1024 * 1024 + 1),
            ("render-uniform-upload", |v| v.render_uniform_upload_bytes = 113),
            ("unexpected-readback", |v| v.unexpected_readback_bytes = 1),
            ("screenshot-readback", |v| v.screenshot_readback_bytes = 64 * 64 * 4 + 1),
            ("peak-lod-host", |v| v.peak_lod_host_bytes = 64 * 1024 * 1024 + 1),
            ("peak-lod-gpu", |v| v.peak_lod_gpu_bytes = 64 * 1024 * 1024 + 1),
            ("inflight-lod", |v| v.inflight_lod_chunks = 9),
            ("browser-requested", |v| v.browser_requested_bytes = 8 * 1024 * 1024 + 1),
            ("python-copy", |v| v.python_copy_bytes = 12 * 1024 * 1024 + 1),
            ("state-roundtrip", |v| v.state_roundtrip_mismatches = 1),
        ];
        for &(budget_id, mutate) in overruns {
            let mut evidence = passing();
            mutate(&mut evidence.measurements);
            let decision = VisualReleaseGate::evaluate(&evidence);
            assert!(!decision.allowed, "{budget_id}");
            assert!(
                decision.reasons.iter().any(|reason| reason.contains(budget_id)),
                "{budget_id}: {:?}",
                decision.reasons
            );
        }
    }
}
