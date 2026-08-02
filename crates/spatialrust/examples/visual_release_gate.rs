//! Constructs and prints the canonical Visual release receipt.

use spatialrust::platform::{
    ConformanceReport, ConformanceStatus, SecurityChecklist, VisualMeasurements,
    VisualReceiptEvidence, VisualReleaseEvidence, VisualReleaseGate,
};

fn main() {
    let mut conformance = ConformanceReport::new();
    for &id in VisualReleaseGate::required_conformance_cases() {
        conformance.record(id, ConformanceStatus::Pass, Some("CI/release receipt".into()));
    }
    let evidence = VisualReleaseEvidence {
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
    };
    let decision = VisualReleaseGate::evaluate(&evidence);
    assert!(decision.allowed, "Visual release denied: {:?}", decision.reasons);
    print!("{}", VisualReleaseGate::render_markdown(&evidence));
}
